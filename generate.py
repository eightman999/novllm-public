# generate.py
"""ベースモデル(または剪定済みモデル) + LoRAアダプタでテキスト生成する。

Usage:
    python generate.py --base-model Qwen/Qwen2.5-7B --adapter ./lora_out-novel/adapter \
        --prompt "森の奥に光る扉があった。" --max-new-tokens 400
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import pick_device, pick_dtype
from control_format import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True, help="ベースモデル、または prune_model.py の出力ディレクトリ")
    ap.add_argument("--adapter", default=None, help="train_lora.py の adapter ディレクトリ(省略時はベースモデルのみ)")
    ap.add_argument("--prompt", required=True)
    # --- 条件付き生成。train_lora.py --control-prefix で学習したアダプタ用 ---
    # 書式は control_format.py が単一の正本。学習時と同じ関数を通すのでずれない。
    ap.add_argument("--viewpoint", default=None,
                    help="first_person / third_person / unknown")
    ap.add_argument("--protagonist-gender", default=None,
                    help="male / female / multiple / non_human / unknown")
    ap.add_argument("--setting", default=None, help="カンマ区切り(例: isekai,school)")
    ap.add_argument("--genre", default=None, help="カンマ区切り(例: romance,military)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    print(f"[generate] device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype, trust_remote_code=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()

    def split_list(value):
        return [item.strip() for item in value.split(",") if item.strip()] if value else None

    conditions = (args.viewpoint, args.protagonist_gender, args.setting, args.genre)
    if any(conditions):
        prompt = build_prompt(
            viewpoint=args.viewpoint, protagonist_gender=args.protagonist_gender,
            setting=split_list(args.setting), genre=split_list(args.genre), body=args.prompt,
        )
        print(f"[generate] 条件行: {prompt.splitlines()[0]}")
    else:
        prompt = args.prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n----- output -----")
    print(text)


if __name__ == "__main__":
    main()
