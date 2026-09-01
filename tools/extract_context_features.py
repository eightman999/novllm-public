#!/usr/bin/env python3
"""代表区間JSONLに対してLLM構造化抽出を複数回まわす（PLAN_20260726.md Phase L1）。

同じ入力・temperature=0 でも出力が揺れるかを測るのが目的なので、
1つの入力を --repeats 回まわして、run_index 付きで1行ずつ追記する。
--resume は (ncode, method, run_index) 単位で済みを飛ばす。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any

from extract_work_features import (
    PROMPT_TEMPLATE,
    call_ollama,
    load_schema,
    parse_deadline,
    past_deadline,
    validate,
)


def load_done(out_path: Path) -> set[tuple[str, str, int]]:
    if not out_path.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    with open(out_path, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("ok"):
                    done.add((str(row["ncode"]), str(row["method"]), int(row["run_index"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", required=True, nargs="+",
                        help="sample_work_contexts.py の出力（方式ごとに複数指定可）")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1:11434")
    parser.add_argument("--model", default="general-qwen3-30b")
    parser.add_argument("--num-ctx", type=int, default=65536)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--deadline", help="HH:MM。この時刻を過ぎたら新規に着手しない")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    schema = load_schema(Path(args.schema))
    deadline = parse_deadline(args.deadline)
    out_path = Path(args.out)
    done = load_done(out_path) if args.resume else set()

    items: list[dict[str, Any]] = []
    for path in args.contexts:
        with open(path, encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    items.append(json.loads(line))

    todo = [(item, run_index) for item in items for run_index in range(args.repeats)
            if (item["ncode"], item["method"], run_index) not in done]
    print(f"入力={len(items)}件 × {args.repeats}回 / 済={len(done)} / 今回={len(todo)}", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for index, (item, run_index) in enumerate(todo, 1):
            if past_deadline(deadline):
                print(f"締切 {args.deadline} に達したので停止します", file=sys.stderr)
                break
            record: dict[str, Any] = {
                "ncode": item["ncode"], "method": item["method"], "run_index": run_index,
                "n_chars": len(item["text"]), "sampling": item.get("sampling"),
                "model": args.model, "num_ctx": args.num_ctx,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            started = time.time()
            try:
                features, stats = call_ollama(
                    args.host, args.model, PROMPT_TEMPLATE.format(text=item["text"]),
                    schema, args.num_ctx, args.think, args.timeout,
                )
                problems = validate(features, schema)
                record.update({"ok": True, "features": features, "problems": problems, "stats": stats})
                n_ok += 1
                status = "OK" if not problems else f"OK(問題{len(problems)}件)"
            except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError) as exc:
                record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                n_fail += 1
                status = "FAIL"
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{index}/{len(todo)}] {item['ncode']} {item['method']} run{run_index} "
                  f"{len(item['text']):,}字 {status} {time.time() - started:.1f}s", file=sys.stderr)

    print(f"完了: ok={n_ok} fail={n_fail} -> {out_path}", file=sys.stderr)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
