# prune_model.py
"""ベースモデルの構造的レイヤー剪定 (depth pruning)。

ShortGPT の Block Influence (BI) 指標を使い、各Transformerブロックの
入出力hidden stateがどれだけ変化しているかを小説データでキャリブレーションして測定。
BIが低い(=そのブロックがほぼ恒等変換=冗長)レイヤーから順に削除し、
VRAM/推論コストを削減した小さいベースモデルを保存する。

削除後は表現がずれるため、そのまま使わず train_lora.py で軽く
LoRA fine-tune (healing) してから使うことを推奨。

Usage:
    python prune_model.py --base-model Qwen/Qwen2.5-7B --num-layers-to-prune 4 \
        --output-dir ./pruned_out-qwen25-7b
"""
import argparse
import glob
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import pick_device, pick_dtype


def load_calibration_texts(data_glob: str, num_samples: int, seed: int) -> list[str]:
    paths = sorted(glob.glob(data_glob))
    if not paths:
        raise FileNotFoundError(f"キャリブレーション用データが見つかりません: {data_glob}")
    rng = random.Random(seed)
    rng.shuffle(paths)
    texts = []
    for path in paths:
        if len(texts) >= num_samples:
            break
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text")
                if text:
                    texts.append(text)
                break  # 1ファイル(1作品)につき1レコード
    if not texts:
        raise ValueError("有効な text フィールドを持つレコードが見つかりませんでした")
    return texts[:num_samples]


@torch.no_grad()
def compute_block_influence(
    model, tokenizer, texts: list[str], seq_len: int, device: str
) -> list[float]:
    """各レイヤーiのBlock Influence = 1 - avg cos_sim(h_{i-1}, h_i) を計算。値が低いほど冗長。"""
    num_layers = model.config.num_hidden_layers
    sim_sums = [0.0] * num_layers
    token_counts = [0] * num_layers

    model.eval()
    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=seq_len
        ).to(device)
        if enc["input_ids"].shape[1] < 2:
            continue
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states  # tuple: len == num_layers + 1
        for i in range(num_layers):
            h_in = hidden_states[i].float()
            h_out = hidden_states[i + 1].float()
            cos = torch.nn.functional.cosine_similarity(h_in, h_out, dim=-1)
            sim_sums[i] += cos.sum().item()
            token_counts[i] += cos.numel()

    block_influence = []
    for i in range(num_layers):
        avg_cos = sim_sums[i] / max(token_counts[i], 1)
        block_influence.append(1.0 - avg_cos)
    return block_influence


def prune_layers(model, drop_indices: set[int]) -> None:
    """model.model.layers から指定indexのレイヤーを取り除き、layer_idxを振り直す。"""
    layers = model.model.layers
    num_layers = len(layers)
    keep_idx = [i for i in range(num_layers) if i not in drop_indices]
    kept = [layers[i] for i in keep_idx]
    for new_idx, layer in enumerate(kept):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_idx
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx
    model.model.layers = torch.nn.ModuleList(kept)
    model.config.num_hidden_layers = len(kept)

    # Qwen3等のhybrid attention構成は config.layer_types のような「層ごとの設定リスト」を
    # 持つことがある。num_hidden_layers だけ更新すると長さ不一致でsave_pretrainedが
    # 検証エラーになるため、元のレイヤー数と同じ長さのlist属性は同じ index で間引く。
    for attr, value in list(vars(model.config).items()):
        if isinstance(value, list) and len(value) == num_layers:
            setattr(model.config, attr, [value[i] for i in keep_idx])


def main():
    ap = argparse.ArgumentParser(description="ベースモデルのレイヤー剪定 (ShortGPT BI法)")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--output-dir", default="./pruned_out-model")
    ap.add_argument("--data-glob", default="./data/*.jsonl")
    ap.add_argument("--calib-samples", type=int, default=16, help="キャリブレーションに使う作品数")
    ap.add_argument("--calib-seq-len", type=int, default=1024)
    ap.add_argument("--num-layers-to-prune", type=int, default=0, help="削除するレイヤー数")
    ap.add_argument("--prune-ratio", type=float, default=None, help="削除割合を指定する場合(num-layers-to-pruneより優先)")
    ap.add_argument("--keep-first", type=int, default=2, help="剪定対象から除外する先頭レイヤー数")
    ap.add_argument("--keep-last", type=int, default=2, help="剪定対象から除外する末尾レイヤー数")
    ap.add_argument("--device", default=None, help="cuda/mps/cpuを明示指定(省略時は自動選択)")
    ap.add_argument("--dtype", default=None, help="float16/bfloat16/float32を明示指定")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="重要度ランキングを表示するだけで保存しない")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    print(f"[prune_model] device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, trust_remote_code=args.trust_remote_code
    )
    model.to(device)

    num_layers = model.config.num_hidden_layers
    print(f"[prune_model] 剪定前レイヤー数: {num_layers}, パラメータ数: {model.num_parameters():,}")

    texts = load_calibration_texts(args.data_glob, args.calib_samples, args.seed)
    print(f"[prune_model] キャリブレーション作品数: {len(texts)}")

    bi = compute_block_influence(model, tokenizer, texts, args.calib_seq_len, device)

    candidates = [
        i for i in range(num_layers)
        if args.keep_first <= i < num_layers - args.keep_last
    ]
    ranked = sorted(candidates, key=lambda i: bi[i])  # BIが低い(冗長)順

    if args.prune_ratio is not None:
        n_prune = max(0, min(len(candidates), round(num_layers * args.prune_ratio)))
    else:
        n_prune = max(0, min(len(candidates), args.num_layers_to_prune))

    drop_indices = set(ranked[:n_prune])

    print("[prune_model] レイヤーごとのBlock Influence (低いほど冗長):")
    for i in range(num_layers):
        mark = " <- 削除候補" if i in drop_indices else ""
        print(f"  layer {i:3d}: BI={bi[i]:.6f}{mark}")

    if n_prune == 0:
        print("[prune_model] 削除レイヤー数が0のため何もしません (--num-layers-to-prune か --prune-ratio を指定)")
        return

    print(f"[prune_model] 削除対象レイヤー ({n_prune}層): {sorted(drop_indices)}")

    if args.dry_run:
        print("[prune_model] --dry-run のため保存はスキップします")
        return

    prune_layers(model, drop_indices)
    print(f"[prune_model] 剪定後レイヤー数: {model.config.num_hidden_layers}, パラメータ数: {model.num_parameters():,}")
    print("[prune_model] 注意: 剪定直後は表現がずれています。train_lora.py で軽くLoRA fine-tune(healing)してください。")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[prune_model] 保存先: {args.output_dir}")


if __name__ == "__main__":
    main()
