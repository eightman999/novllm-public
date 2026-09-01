#!/usr/bin/env python3
"""長編作品を決定的に選ぶ。

乱数を使わない。同じmanifestと引数なら何度実行しても同じ集合が出る。
継続評価したい作品は--pinnedで明示できる。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="1行1件のncodeリスト")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--min-chunks", type=int, default=60)
    parser.add_argument("--pinned", default="",
                        help="必ず含めるncode（カンマ区切り）")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_ncode = {row["ncode"]: row for row in rows}
    requested = [item.strip() for item in args.pinned.split(",") if item.strip()]
    pinned = [ncode for ncode in requested if ncode in by_ncode]
    missing = [ncode for ncode in requested if ncode not in by_ncode]
    if missing:
        print(f"警告: 固定指定が manifest に無い: {missing}")

    pool = sorted((row for row in rows
                   if row["chunk_count"] >= args.min_chunks and row["ncode"] not in pinned),
                  key=lambda row: row["ncode"])
    need = args.count - len(pinned)
    if need < 0:
        raise SystemExit(f"--pinned の件数 {len(pinned)} が --count {args.count} を超えています")
    if need > len(pool):
        raise SystemExit(f"候補不足: pool={len(pool)} < 必要={need}")
    # 等間隔インデックス。ncode順に並べた母集団を偏りなく拾う（乱数を使わないため）。
    picked = [pool[round(i * (len(pool) - 1) / (need - 1))] for i in range(need)] if need > 1 else pool[:need]

    selected = pinned + [row["ncode"] for row in picked]
    Path(args.out).write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"{'ncode':<14}{'chunks':>8}{'episodes':>10}  {'r18':<6}{'site':<10}タグ")
    for ncode in selected:
        row = by_ncode[ncode]
        tags = ",".join(row.get("normalized_keywords") or [])[:40]
        print(f"{ncode:<14}{row['chunk_count']:>8}{row['actual_episode_count']:>10}  "
              f"{str(row['is_r18']):<6}{row['site_type']:<10}{tags}")
    print(f"\n書き出し: {args.out}（{len(selected)}件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
