#!/usr/bin/env python3
"""4bit量子化がモデルの品質をどれだけ落としているかを held-out データで測る。

生成文を読んで「崩れている/いない」を判定すると読み手の主観が入るので、
**未学習の作品（val シャード）に対する perplexity** で測る。
低いほど日本語をよく予測できている＝崩れにくい、と読める。

val シャードは作品単位で train から分離されている（build_dataset.py の split）ので、
学習で覚えた文章を当てているわけではない。

使い方（同じコマンドを --no-4bit の有無で2回流して比べる）:
  .venv/bin/python tools/eval_quality.py --adapter <path> --limit 40
  .venv/bin/python tools/eval_quality.py --adapter <path> --limit 40 --no-4bit
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from control_format import build_training_text  # noqa: E402  学習時と同じ書式で測る


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--chunks",
                        default="./data/chunks_v2/val-*.jsonl")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    kwargs = {"dtype": torch.bfloat16, "trust_remote_code": True,
              "attn_implementation": args.attn_implementation, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    label = "4bit(nf4)" if args.load_in_4bit else "bf16"
    print(f"[eval_quality] {label} で読み込み中…", file=sys.stderr)
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"[eval_quality] 読み込み {time.time() - started:.1f}s", file=sys.stderr)

    # val から先頭N件。作品が偏らないよう、シャードを順に舐めて拾う。
    texts = []
    for path in sorted(glob.glob(args.chunks)):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                texts.append(build_training_text(json.loads(line)))
                if len(texts) >= args.limit:
                    break
        if len(texts) >= args.limit:
            break
    if not texts:
        raise SystemExit(f"val チャンクが見つかりません: {args.chunks}")

    total_nll, total_tokens = 0.0, 0
    started = time.time()
    with torch.no_grad():
        for index, text in enumerate(texts, 1):
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=args.max_length).input_ids.to(model.device)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1  # 次トークン予測なので1つ減る
            total_nll += out.loss.item() * n
            total_tokens += n
            if index % 10 == 0:
                print(f"  {index}/{len(texts)}", file=sys.stderr)

    mean_nll = total_nll / total_tokens
    print(f"\n=== {label} / adapter={args.adapter or 'なし'} ===")
    print(f"評価チャンク数 : {len(texts)}")
    print(f"評価トークン数 : {total_tokens:,}")
    print(f"平均loss(nats) : {mean_nll:.4f}")
    print(f"perplexity     : {math.exp(mean_nll):.3f}")
    print(f"所要           : {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
