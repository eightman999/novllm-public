# train_lora.py
import argparse
import glob as glob_module
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from common import pick_device, pick_dtype
from control_format import build_training_text

ap = argparse.ArgumentParser(description="小説データでベースモデルをLoRA fine-tune")
ap.add_argument("--base-model", default=os.environ.get("NOVLLM_BASE_MODEL", "Qwen/Qwen3-8B-Base"),
                 help="剪定済みモデルを使う場合は prune_model.py の --output-dir を指定")
ap.add_argument("--data-glob", default="./data/*.jsonl", help="コーパス全体を使う場合は glob パターン")
ap.add_argument("--output-dir", default="./lora_out-novel")
ap.add_argument("--max-seq-len", type=int, default=2048)
ap.add_argument("--max-memory", default=os.environ.get("NOVLLM_MAX_MEMORY"),
                help="device_map='auto' のGPU別上限。偏り矯正用。例: '0=9GiB,1=13GiB'")


def _env_int(name):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else None


ap.add_argument("--layers-device", type=int, default=_env_int("NOVLLM_LAYERS_DEVICE"),
                help="decoder層を載せるGPU番号。--head-device と併用で明示device_mapになる。"
                     "混成GPU構成ではFlashAttentionが使える側(Ampere以上)を指定する")
ap.add_argument("--head-device", type=int, default=_env_int("NOVLLM_HEAD_DEVICE"),
                help="embed_tokens/norm/lm_head を載せるGPU番号。logits+CEもここに乗るため"
                     "VRAMに余裕のある側を指定する")
ap.add_argument("--layers-on-head-device", type=int,
                default=int(os.environ.get("NOVLLM_LAYERS_ON_HEAD_DEVICE", "0")),
                help="先頭から何層を --head-device 側へ載せるか(既定0)。層配分の微調整用")
ap.add_argument("--cast-head-dtype", action="store_true",
                default=os.environ.get("NOVLLM_CAST_HEAD_DTYPE", "0") == "1",
                help="prepare_model_for_kbit_training がfp32へ上げた embed/lm_head を "
                     "--dtype に戻す。4096では約3.5GiB効く")
ap.add_argument("--device", default=None, help="cuda/mps/cpuを明示指定(省略時は自動選択)")
ap.add_argument("--dtype", default=None)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--batch-size", type=int, default=1)
ap.add_argument("--grad-accum", type=int, default=8)
ap.add_argument("--max-steps", type=int, default=-1, help="スモークテスト用に総ステップ数を制限する場合に指定")
ap.add_argument("--save-steps", type=int, default=500)
ap.add_argument("--logging-steps", type=int, default=20)
ap.add_argument("--save-total-limit", type=int, default=3, help="ディスク保護のため古いcheckpointを自動削除")
ap.add_argument("--resume", action="store_true", help="output-dir内の最新checkpointから再開する")
ap.add_argument("--dataset-num-proc", type=int, default=8,
                 help="コーパス全体を1つのArrow writerで処理すると offset overflow "
                      "(pyarrow int32上限)でクラッシュするため、複数プロセスに分割する")
ap.add_argument("--batch-source-log", default=None,
                help="各マイクロバッチの作品IDを記録するJSONL "
                     "(既定: OUTPUT_DIR/batch_sources.jsonl)")
ap.add_argument("--dataset-dir", default=None,
                help="dataset_v2/ のパス。指定するとVERSION.json/manifestsのSHA-256を"
                     "学習ログ(OUTPUT_DIR/dataset_provenance.json)へ記録する")
ap.add_argument("--val-data-glob", default=None,
                help="検証用shardのglob。指定すると--data-globとの間でncode(作品ID)が"
                     "重複していないかを学習開始前に検証する(作品単位split前提の安全チェック)")
ap.add_argument("--strict-token-len", action="store_true",
                help="max-seq-lenを超えるレコードが見つかった場合、除外せずエラー終了する")
ap.add_argument("--allow-unknown-ncode", action="store_true",
                help="meta.ncodeが欠落/unknownなレコードを許可する(既定は拒否)")
ap.add_argument("--learning-rate", type=float, default=2e-4,
                help="LoRAなら1e-4〜3e-4くらいから。発散したらまずここを下げる")
ap.add_argument("--max-grad-norm", type=float, default=1.0,
                help="勾配クリッピングの閾値(transformersの既定と同じ1.0)。"
                     "明示しておくと、発散時に効いていたのかログで確認できる")
ap.add_argument("--archive-every-saves", type=int, default=0,
                help="N回保存するごとに、そのcheckpointをアーカイブ先へ複製する。"
                     "save-total-limit で消える前に途中経過を残すための保険。0で無効")
ap.add_argument("--archive-dir", default=None,
                help="アーカイブ先。既定は OUTPUT_DIR/archive")
ap.add_argument("--no-amp", action="store_true",
                help="Trainer側のAMP(bf16/fp16フラグ)を無効にする。モデルを --dtype で "
                     "既に目的のdtypeでロードしている場合、AMPは二重になるうえ、"
                     "fp16 AMPのGradScalerがbf16勾配に当たって "
                     "NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda "
                     "を出すことがある(7/27実測)")
ap.add_argument("--attn-implementation", default=None,
                help="eager/sdpa/flash_attention_2。Ampere未満のGPUが混ざる構成では eager を指定する")
ap.add_argument("--control-prefix", action="store_true",
                help="control_tagsを条件行として本文の先頭に差し込む(条件付き生成の学習)。"
                     "書式は control_format.py が単一の正本。generate.py も同じ関数を使う")
args = ap.parse_args()


def compute_dataset_provenance(dataset_dir):
    """dataset_v2/ のVERSION.jsonとmanifests配下ファイルのSHA-256を集計する。
    (由来: 指示書8節 - dataset versionとmanifest SHA-256を学習ログへ記録する要件)"""
    provenance = {"dataset_dir": os.path.abspath(dataset_dir), "manifest_sha256": {}}
    version_path = os.path.join(dataset_dir, "VERSION.json")
    if os.path.exists(version_path):
        with open(version_path, "r", encoding="utf-8") as f:
            provenance["version"] = json.load(f)
    manifests_dir = os.path.join(dataset_dir, "manifests")
    if os.path.isdir(manifests_dir):
        for name in sorted(os.listdir(manifests_dir)):
            path = os.path.join(manifests_dir, name)
            if not os.path.isfile(path):
                continue
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            provenance["manifest_sha256"][name] = h.hexdigest()
    return provenance


def collect_ncodes(data_glob, num_proc):
    """globが指すJSONL群からmeta.ncodeの集合だけを軽量に集める
    (train/val重複チェック用。本文はロードしない)。"""
    ncodes = set()
    for path in sorted(glob_module.glob(data_glob)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ncode = (rec.get("meta") or {}).get("ncode")
                if ncode:
                    ncodes.add(ncode)
    return ncodes


class BatchSourceLoggingCollator:
    """学習用collatorへ渡す直前に、各マイクロバッチの作品IDを追記記録する。"""

    def __init__(self, base_collator, log_path, grad_accum):
        self.base_collator = base_collator
        self.log_path = log_path
        self.grad_accum = grad_accum
        self.micro_batch = 0
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def __call__(self, features):
        self.micro_batch += 1
        source_ids = []
        clean_features = []
        for feature in features:
            meta = feature.get("meta") or {}
            source_ids.append(meta.get("ncode", "unknown"))
            clean_features.append({
                key: value
                for key, value in feature.items()
                if key in {"input_ids", "labels", "seq_lengths"}
            })

        record = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "micro_batch": self.micro_batch,
            "optimizer_step_in_run": (
                (self.micro_batch - 1) // self.grad_accum + 1
            ),
            "accumulation_index": (
                (self.micro_batch - 1) % self.grad_accum + 1
            ),
            "source_ids": source_ids,
        }
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        return self.base_collator(clean_features)


def find_last_checkpoint(output_dir):
    """checkpoint-* を新しい順に見て、trainer_state.json/adapter重みが揃っている
    (中断で壊れていない)最初のものを返す。壊れたcheckpointを掴んで再起動を
    繰り返すことを避けるため。"""
    import glob as _glob
    ckpts = _glob.glob(os.path.join(output_dir, "checkpoint-*"))
    ckpts.sort(key=lambda p: int(p.rsplit("-", 1)[-1]), reverse=True)
    for ckpt in ckpts:
        has_state = os.path.exists(os.path.join(ckpt, "trainer_state.json"))
        has_adapter = os.path.exists(os.path.join(ckpt, "adapter_model.safetensors"))
        if has_state and has_adapter:
            return ckpt
        print(f"[train_lora] checkpoint不完全のためスキップ: {ckpt}")
    return None

class CheckpointArchiver(TrainerCallback):
    """save-total-limit で消える前に、N回に1回のcheckpointを別ディレクトリへ複製する。

    「細かく保存して古いものは消す」運用だと途中経過が一切残らないので、
    間引いてアーカイブする。`on_save` は Trainer が保存した直後に呼ばれるため、
    削除される前に確実に掴める（ポーリングで拾おうとすると削除と競合する）。

    optimizer.pt と rng_state は再開用であってアーカイブには要らないので複製しない
    （optimizer.pt だけで1つ175MBある）。アーカイブから学習を再開することは想定せず、
    「その時点の重みを取り出して生成を試す」用途に絞る。
    """

    def __init__(self, output_dir, archive_dir, every_saves):
        self.output_dir = output_dir
        self.archive_dir = archive_dir
        self.every_saves = every_saves
        self.saves = 0

    def on_save(self, sft_args, state, control, **kwargs):
        self.saves += 1
        if self.every_saves <= 0 or self.saves % self.every_saves != 0:
            return
        source = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(source):
            print(f"[train_lora] アーカイブ対象が見つかりません: {source}")
            return
        destination = os.path.join(self.archive_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(destination):
            return
        os.makedirs(self.archive_dir, exist_ok=True)
        temporary = destination + ".tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.copytree(source, temporary,
                        ignore=shutil.ignore_patterns("optimizer.pt", "rng_state*"))
        os.replace(temporary, destination)  # 中途半端なディレクトリを残さない
        print(f"[train_lora] アーカイブ: {destination}")


device = pick_device(args.device)
dtype = pick_dtype(device, args.dtype)
use_bnb = device == "cuda"  # bitsandbytes 4bit(QLoRA)はCUDA専用。MPS/CPUでは通常LoRAにフォールバック
print(f"[train_lora] device={device} dtype={dtype} qlora={use_bnb}")

ds = load_dataset("json", data_files=args.data_glob, split="train")

# --- control_tagsを条件行として本文に差し込む --------------------------------
# SFTConfig(dataset_text_field="text") は text しか見ないため、ここで text 自体を
# 作り直さないと control_tags は学習に一切伝わらない。
# トークン長の検証より前に適用する(条件行の分だけ長くなるので、後に回すと
# max-seq-len の判定がずれる)。
if args.control_prefix:
    sample_before = ds[0]["text"][:40]
    ds = ds.map(lambda example: {"text": build_training_text(example)},
                num_proc=args.dataset_num_proc)
    print(f"[train_lora] control_tagsを条件行として付与しました\n"
          f"  付与前: {sample_before!r}...\n"
          f"  付与後: {ds[0]['text'][:120]!r}...")

tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"

# --- 作品ID不明レコードの拒否 (既定。--allow-unknown-ncodeで解除) --------------------
def _has_valid_ncode(example):
    ncode = (example.get("meta") or {}).get("ncode")
    return bool(ncode) and ncode != "unknown"


if not args.allow_unknown_ncode:
    before_count = len(ds)
    ds = ds.filter(_has_valid_ncode, num_proc=args.dataset_num_proc)
    rejected_count = before_count - len(ds)
    if rejected_count:
        print(f"[train_lora] meta.ncode不明のため除外: {rejected_count}/{before_count}件")
    if len(ds) == 0:
        raise SystemExit("[train_lora] 全レコードがncode不明で除外されました。--allow-unknown-ncodeで許可するか、"
                          "データを確認してください。")

# --- token上限検証 (既定は超過レコードを除外、--strict-token-lenでエラー終了) ----------
# prompt/completion 形式（build_context_view.py の出力）なら、損失は completion 側だけに
# かける。文脈側にも損失をかけると「文脈を丸暗記する」方を学んでしまい、
# 肝心の「続きを書く」能力が育たない。TRL は prompt/completion 列を持つデータセットを
# 自動で認識し、completion_only_loss を既定で有効にする。
is_prompt_completion = "prompt" in ds.column_names and "completion" in ds.column_names
if is_prompt_completion:
    print(f"[train_lora] prompt/completion 形式を検出（損失は completion 側のみ）\n"
          f"  prompt例    : {ds[0]['prompt'][:110]!r}...\n"
          f"  completion例: {ds[0]['completion'][:80]!r}...")


def _add_token_len(example):
    text = (example["prompt"] + example["completion"]) if is_prompt_completion \
        else example["text"]
    return {"__token_len": len(tokenizer(text, add_special_tokens=False)["input_ids"])}


ds = ds.map(_add_token_len, num_proc=args.dataset_num_proc)
over_limit_count = len(ds.filter(lambda ex: ex["__token_len"] > args.max_seq_len,
                                  num_proc=args.dataset_num_proc))
if over_limit_count:
    if args.strict_token_len:
        raise SystemExit(f"[train_lora] max-seq-len({args.max_seq_len})超過レコードが{over_limit_count}件"
                          "あります(--strict-token-len指定のため中断)。chunker側のchunk化設定を確認してください。")
    print(f"[train_lora] max-seq-len({args.max_seq_len})超過のため除外: {over_limit_count}件")
    ds = ds.filter(lambda ex: ex["__token_len"] <= args.max_seq_len, num_proc=args.dataset_num_proc)
ds = ds.remove_columns(["__token_len"])

# --- train/validation間のncode重複チェック (--val-data-glob指定時のみ) -----------------
if args.val_data_glob:
    train_ncodes = collect_ncodes(args.data_glob, args.dataset_num_proc)
    val_ncodes = collect_ncodes(args.val_data_glob, args.dataset_num_proc)
    overlap = train_ncodes & val_ncodes
    if overlap:
        raise SystemExit(
            f"[train_lora] train/validation間で{len(overlap)}件のncodeが重複しています"
            f"(作品単位split前提が破られています): {sorted(overlap)[:10]}..."
        )
    print(f"[train_lora] train/validation ncode重複チェックOK (train={len(train_ncodes)}作品 "
          f"val={len(val_ncodes)}作品、重複0件)")

# --- dataset provenance (version/manifest SHA-256) の記録 ---------------------------
dataset_provenance = None
if args.dataset_dir:
    dataset_provenance = compute_dataset_provenance(args.dataset_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    provenance_path = os.path.join(args.output_dir, "dataset_provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(dataset_provenance, f, ensure_ascii=False, indent=2)
    print(f"[train_lora] dataset_provenance={provenance_path} "
          f"(schema_version={dataset_provenance.get('version', {}).get('schema_version')})")

model_kwargs = dict(dtype=dtype, trust_remote_code=True)
# attention実装。既定のsdpaはFlashAttentionバックエンドを選ぶことがあり、
# Ampere未満のGPUでは実行時に
# `RuntimeError: FlashAttention only supports Ampere GPUs or newer` で落ちる。
# 実測(7/27): RTX 3060(Ampere) と Tesla P100(Pascal) の2枚構成で device_map="auto"
# を使うと、P100に載った層のforwardでこれが起きた。混成環境では eager を選ぶ。
if args.attn_implementation:
    model_kwargs["attn_implementation"] = args.attn_implementation
if use_bnb:
    from transformers import BitsAndBytesConfig

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        # 計算dtypeは --dtype に追従させる。ここをbf16に固定すると、fp16を選んだ際に
        # AMPのGradScalerがbf16パラメータに当たって
        # NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not
        # implemented for 'BFloat16' で落ちる（7/27実測）。
        bnb_4bit_compute_dtype=dtype,
    )
    model_kwargs["device_map"] = "auto"
    # device_map="auto" のGPU間配分は偏る。2026-07-30 に max-seq-len 4096 で実測すると
    # GPU0(3060 12GiB)が9,834MiBで2.4GiB余っているのにGPU1(P100 16GiB)が15,504MiBで
    # OOMした（expandable_segments を足しても2.30GiB要求に対し2.09GiB空きで約210MiB不足）。
    # 偏りを人手で矯正できるようにする。書式は "0=9GiB,1=13GiB"。
    if args.max_memory:
        model_kwargs["max_memory"] = {
            (int(k) if k.strip().isdigit() else k.strip()): v.strip()
            for k, v in (pair.split("=", 1) for pair in args.max_memory.split(","))
        }
        print(f"[train_lora] max_memory={model_kwargs['max_memory']}")

    # --- 混成GPU向けの明示device_map（2026-07-30 実測で決定）---
    # max_memory は accelerate の見積もりが非量子化dtype基準で膨らむため制御しづらく
    # （0=8GiB,1=4GiB を渡すと "Some modules are dispatched on the CPU" で弾かれる）、
    # "auto"(=balanced) は P100 側に層とlm_headを寄せてしまい 4096 で必ずOOMした。
    #
    # 実測で判った要点:
    #  - torch は「現在のCUDAデバイス(=GPU0, Ampere)」の capability で FlashAttention を
    #    選ぶため、P100 に層が載っていると実行時に
    #    `FlashAttention only supports Ampere GPUs or newer` で落ちる。
    #    → **全ての decoder 層を Ampere 側に置けば sdpa/flash がそのまま使える。**
    #  - P100(sm_60) の SDPA は mem-efficient が fp16 のみ、bf16 は MATH に落ちて
    #    seq 4096 でピーク 8.28GiB を食う（3060 の flash は 0.31GiB）。
    #  - logits+CE は lm_head と同じデバイスに乗る。4096×151,936 は大きいので
    #    VRAMに余裕のある側（P100 16GiB）へ寄せる。
    #
    # 結果: 全36層=3060 / embed+norm+lm_head=P100 で bf16・4096 が通り、
    # GPU0 peak 7.51GiB(/11.63)、GPU1 peak 10.54GiB(/15.89) と両方に余裕が出た。
    if args.layers_device is not None and args.head_device is not None:
        from transformers import AutoConfig

        n_layers = AutoConfig.from_pretrained(
            args.base_model, trust_remote_code=True).num_hidden_layers
        n_on_head = max(0, min(args.layers_on_head_device, n_layers))
        explicit = {
            "model.embed_tokens": args.head_device,
            "model.rotary_emb": args.head_device,
            "model.norm": args.head_device,
            "lm_head": args.head_device,
        }
        for i in range(n_layers):
            explicit[f"model.layers.{i}"] = (
                args.head_device if i < n_on_head else args.layers_device)
        model_kwargs["device_map"] = explicit
        model_kwargs.pop("max_memory", None)
        print(f"[train_lora] 明示device_map: 層{n_on_head}〜{n_layers - 1}→GPU{args.layers_device} / "
              f"層0〜{n_on_head - 1}とembed/norm/lm_head→GPU{args.head_device}"
              if n_on_head else
              f"[train_lora] 明示device_map: 全{n_layers}層→GPU{args.layers_device} / "
              f"embed/norm/lm_head→GPU{args.head_device}")

model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

if use_bnb:
    model = prepare_model_for_kbit_training(model)
    # prepare_model_for_kbit_training は非量子化パラメータを全部fp32へ上げる。
    # Qwen3-8B では embed_tokens(622M) と lm_head(622M) がそれに当たり、
    # 2つで +2.5GiB、さらに logits まで fp32 になって 4096 では致命的になる。
    # QLoRA の標準構成どおり compute dtype へ戻すと、実測で lm_head 側のGPUの
    # ピークが 15.01GiB → 11.49GiB（-3.5GiB）に下がった（2026-07-30）。
    if args.cast_head_dtype:
        model.get_input_embeddings().to(dtype)
        if getattr(model, "lm_head", None) is not None:
            model.lm_head.to(dtype)
        print(f"[train_lora] embed / lm_head を {dtype} に戻した")
else:
    model.to(device)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],  # Qwen/Mistral/Llamaで大体OK
)

model = get_peft_model(model, lora_cfg)

# 作品IDと各学習batchの対応を監査可能にするためpackingは無効化する。
# packingを有効にすると複数作品が1シーケンスへ結合され、TRLの前処理で
# 元作品メタデータも破棄されるため、正確なsource logを残せない。
use_packing = False

sft_config = SFTConfig(
    output_dir=args.output_dir,
    # prompt/completion 形式のときは dataset_text_field を渡さない。
    # 渡すとTRLが単一テキストとして扱い、completion_only_loss が効かなくなる。
    **({"completion_only_loss": True} if is_prompt_completion
       else {"dataset_text_field": "text"}),
    # TRL 1.9.0 の既定は loss_type="chunked_nll" だが、QLoRA経路とは噛み合わない。
    # prepare_model_for_kbit_training を通すと model.forward が functools.partial に
    # 差し替わるため、TRL 側の _patch_chunked_ce_lm_head が
    # `inspect.signature(original_forward.__func__)` で
    # AttributeError: 'functools.partial' object has no attribute '__func__' を出す。
    # TRL 自身が同ファイル内の別分岐で案内している回避策が loss_type="nll" への切り替え。
    # chunked_nll は lm_head のロジットをチャンクしてメモリを節約する最適化にすぎず、
    # 損失の定義は nll と同じなので学習結果は変わらない。
    loss_type="nll",
    max_length=args.max_seq_len,
    packing=use_packing,
    num_train_epochs=args.epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    learning_rate=args.learning_rate,
    max_grad_norm=args.max_grad_norm,
    weight_decay=0.0,
    warmup_ratio=0.03,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    dataset_num_proc=args.dataset_num_proc,
    # BatchSourceLoggingCollatorがmetaを受け取り、モデル用キーだけを内側の
    # collatorへ渡す。Trainerによる事前のmeta列削除を防ぐ。
    remove_unused_columns=False,
    # fp16=TrueはTrainer側のAMP(GradScaler)経路を有効にする指定で、CUDA前提。
    # MPS/CPUではモデル自体を既にfp16 dtypeでロード済みなので、二重キャストの
    # あいまいさを避けるためTrainer側のfp16フラグはCUDA限定にする。
    # bf16 と fp16 は同時にTrueにできない（transformersが起動時に弾く）。
    # dtype で選ばれた方だけを立てる。Pascal(P100等)は bf16 を持たず
    # エミュレーションで極端に遅くなるため、--dtype float16 を選べる必要がある。
    bf16=(not args.no_amp and device == "cuda" and dtype != torch.float16),
    fp16=(not args.no_amp and device == "cuda" and dtype == torch.float16),
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    args=sft_config,
)

if args.archive_every_saves > 0:
    archive_dir = args.archive_dir or os.path.join(args.output_dir, "archive")
    trainer.add_callback(CheckpointArchiver(args.output_dir, archive_dir,
                                           args.archive_every_saves))
    print(f"[train_lora] {args.archive_every_saves}回保存ごとに {archive_dir} へアーカイブします")

batch_source_log = (
    args.batch_source_log
    or os.path.join(args.output_dir, "batch_sources.jsonl")
)
trainer.data_collator = BatchSourceLoggingCollator(
    trainer.data_collator,
    batch_source_log,
    args.grad_accum,
)
print(f"[train_lora] batch_source_log={batch_source_log}")

resume_ckpt = find_last_checkpoint(args.output_dir) if args.resume else None
if args.resume:
    print(f"[train_lora] resume_from_checkpoint={resume_ckpt}")
trainer.train(resume_from_checkpoint=resume_ckpt)
trainer.model.save_pretrained(os.path.join(args.output_dir, "adapter"))
tokenizer.save_pretrained(args.output_dir)
print("saved:", args.output_dir)
