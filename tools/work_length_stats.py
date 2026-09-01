#!/usr/bin/env python3
"""作品本文長の分布を実測する（PLAN_20260726.md Phase 1-1 / 1-2）。

work_texts_{N}.jsonl は本文を N 文字で打ち切ったキャッシュなので、
len(text) == N の作品は「N文字以上」としか言えない（右側打ち切り）。
短編の閾値（2万字前後）を決めるだけなら打ち切りの影響を受けないので、
このキャッシュで十分。打ち切り件数は必ず一緒に報告する。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def quantile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="work_texts_{N}.jsonl")
    ap.add_argument("--manifest", help="works.jsonl（split等を突き合わせる場合）")
    ap.add_argument("--out", help="集計結果のJSON出力先")
    ap.add_argument(
        "--thresholds",
        default="10000,15000,20000,25000,30000,40000",
        help="短編候補の閾値（文字数）をカンマ区切りで",
    )
    args = ap.parse_args()

    lengths: dict[str, int] = {}
    with open(args.cache, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            lengths[rec["ncode"]] = len(rec["text"])

    if not lengths:
        print("ERROR: キャッシュが空です", file=sys.stderr)
        return 1

    vals = sorted(lengths.values())
    cap = max(vals)  # 打ち切り値（= キャッシュのN）とみなす
    n_capped = sum(1 for v in vals if v >= cap)

    stats = {
        "cache": str(args.cache),
        "n_works": len(vals),
        "cap_chars": cap,
        "n_at_cap": n_capped,
        "quantiles": {f"p{int(q*100)}": quantile(vals, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "min": vals[0],
        "max": vals[-1],
        "thresholds": {},
    }

    for t in (int(x) for x in args.thresholds.split(",")):
        n = sum(1 for v in vals if v <= t)
        stats["thresholds"][str(t)] = {"n_works": n, "pct": round(100.0 * n / len(vals), 2)}

    if args.manifest:
        splits: dict[str, str] = {}
        with open(args.manifest, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                splits[rec["ncode"]] = rec.get("split", "unknown")
        stats["manifest_n"] = len(splits)
        stats["missing_in_cache"] = sum(1 for nc in splits if nc not in lengths)

    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # ヒストグラム（0〜10万字を5千字刻み、それ以上はまとめる）
    bins = list(range(0, 100001, 5000))
    print("\n--- ヒストグラム（文字数） ---")
    for lo, hi in zip(bins, bins[1:]):
        n = sum(1 for v in vals if lo <= v < hi)
        if n:
            print(f"{lo:>7,}-{hi:>7,}: {'#' * min(n, 60)} {n}")
    n = sum(1 for v in vals if v >= bins[-1])
    print(f"{bins[-1]:>7,}+       : {'#' * min(n, 60)} {n}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"stats": stats, "lengths": lengths}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n書き出し: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
