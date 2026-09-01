#!/usr/bin/env python3
"""条件付き生成を試すための小さなHTTPサーバ（PLAN_20260727.md C-4）。

**なぜサーバなのか**: モデルを載せるGPU機と操作画面を開く端末を分けられる。
X転送を張るより、リモートでHTTPを立ててブラウザから触る方が軽い。
追加依存を入れないよう、標準ライブラリの http.server だけで書いてある。

条件行は必ず control_format.build_prompt を通す。学習時と1文字でも違うと
条件付けは効かないので、書式の正本を1箇所に保つ（B-4と同じ理由）。

起動:
  .venv/bin/python tools/gen_server.py \
    --base-model Qwen/Qwen3-8B-Base \
    --adapter ./lora_out-novel/adapter \
    --port 8760
別端末から: http://<GPU機のIPアドレス>:8760/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # noqa: E402

from control_format import build_prompt  # noqa: E402  書式の正本

STATE: dict = {"model": None, "tokenizer": None, "adapter": None, "lock": threading.Lock()}
PAGE_PATH = Path(__file__).with_name("gen_ui.html")


def load_model(args) -> None:
    print(f"[gen_server] 読み込み中: base={args.base_model} adapter={args.adapter}", file=sys.stderr)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    kwargs = {"dtype": torch.bfloat16, "trust_remote_code": True,
              "attn_implementation": args.attn_implementation,
              # 4bitでない場合も device_map は要る。付けないとCPUに載って桁違いに遅くなる。
              "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    STATE.update({"model": model, "tokenizer": tokenizer, "adapter": args.adapter or "(なし)"})
    print(f"[gen_server] 準備完了 {time.time() - started:.1f}s", file=sys.stderr)


def generate_stream(payload: dict):
    """1トークンずつ yield する。速度計測のため各chunkに経過時間を載せる。"""
    tokenizer, model = STATE["tokenizer"], STATE["model"]
    prompt = build_prompt(
        viewpoint=payload.get("viewpoint") or None,
        protagonist_gender=payload.get("protagonist_gender") or None,
        setting=payload.get("setting") or None,
        genre=payload.get("genre") or None,
        r18=payload.get("r18") or None,
        body=payload.get("body") or "",
    )
    yield {"type": "prompt", "text": prompt}

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        **inputs, streamer=streamer,
        max_new_tokens=int(payload.get("max_new_tokens", 300)),
        do_sample=True,
        temperature=float(payload.get("temperature", 0.9)),
        top_p=float(payload.get("top_p", 0.95)),
        top_k=int(payload.get("top_k", 50)),
        repetition_penalty=float(payload.get("repetition_penalty", 1.05)),
        pad_token_id=tokenizer.eos_token_id,
    )
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    started = time.time()
    thread.start()
    n_tokens = 0
    first_token_at = None
    for piece in streamer:
        if not piece:
            continue
        n_tokens += 1
        now = time.time()
        # **プロンプト処理時間を復号速度に混ぜない。**
        # started から測ると、最初の数トークンの tok/s がプロンプト処理ぶん薄まり、
        # 長い書き出しでは1を切って見える（7/29に実測。速度低下ではなく計測の誤り）。
        # 復号速度は「2トークン目以降の平均」で測る。
        if first_token_at is None:
            first_token_at = now
        decode_elapsed = now - first_token_at
        tps = (n_tokens - 1) / decode_elapsed if decode_elapsed > 0 else 0.0
        yield {"type": "token", "text": piece, "n": n_tokens,
               "elapsed": round(now - started, 2),
               "prompt_s": round(first_token_at - started, 2),
               "tps": round(tps, 2)}
    thread.join()
    now = time.time()
    prompt_s = (first_token_at - started) if first_token_at else 0.0
    decode_elapsed = (now - first_token_at) if first_token_at else 0.0
    yield {"type": "done", "n": n_tokens, "elapsed": round(now - started, 2),
           "prompt_s": round(prompt_s, 2),
           "tps": round((n_tokens - 1) / decode_elapsed, 2) if decode_elapsed > 0 else 0.0,
           "prompt_tokens": int(inputs["input_ids"].shape[1])}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # アクセスログは黙らせる
        pass

    def _send(self, code, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/info":
            body = json.dumps({
                "adapter": STATE["adapter"],
                "ready": STATE["model"] is not None,
            }, ensure_ascii=False).encode()
            self._send(200, body, "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        # GPUは1つしかないので生成は直列化する。並行実行するとOOMする。
        with STATE["lock"]:
            try:
                for event in generate_stream(payload):
                    self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
            except Exception as exc:  # 画面側に理由を出す。黙って切らない
                error = {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                self.wfile.write(f"data: {json.dumps(error, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8760)
    parser.add_argument("--attn-implementation", default="eager",
                        help="P100(Pascal)が混ざる構成では eager が必要")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    args = parser.parse_args()

    load_model(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[gen_server] http://{args.host}:{args.port}/ で待機中", file=sys.stderr)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
