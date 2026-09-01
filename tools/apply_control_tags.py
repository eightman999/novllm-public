#!/usr/bin/env python3
"""チャンクの control_tags を作り直す（PLAN_20260727.md B-2 / B-3）。

**元のシャードには一切書き込まない。** 新しいディレクトリへ書き出す。
5.5GB に対しディスクは2.6T空いているので、退避より「触らない」方を選ぶ。

control_tags は作品単位で決まり本文に依存しない（build_dataset.py では
`control_tags_base` を作品ごとに1つ作って全チャンクへ複製している）ので、
本文を作り直さずに作品IDで引いて差し替えるだけでよい。
--verify はそれを実際に確認する（control_tags 以外が1バイトでも変われば失敗扱い）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_dataset import build_control_tags  # noqa: E402  既存のマッピングを再利用する


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="入力シャードのglob")
    parser.add_argument("--classification", required=True, help="build_work_classification.py の出力")
    parser.add_argument("--manifest", required=True, help="works.jsonl（申告フラグ）")
    parser.add_argument("--out-dir", help="出力先ディレクトリ。--verify 時は不要")
    parser.add_argument("--verify", action="store_true",
                        help="書き出さず、control_tags 以外が変化しないことだけ確認する")
    parser.add_argument("--limit-shards", type=int, default=0, help="0で全部。B-2の1シャード確認用")
    args = parser.parse_args()
    if not args.verify and not args.out_dir:
        raise SystemExit("--out-dir か --verify のどちらかが要ります")

    classification = {}
    for line in Path(args.classification).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            classification[row["ncode"]] = row

    flags_by_work = {}
    with open(args.manifest, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                flags_by_work[row["ncode"]] = row.get("control_flags") or {}

    # 作品ごとに1回だけ計算する（チャンク数分まわすと無駄）
    tags_cache: dict[str, dict] = {}

    def tags_for(ncode: str) -> dict | None:
        if ncode not in tags_cache:
            if ncode not in flags_by_work:
                return None
            tags_cache[ncode] = build_control_tags(flags_by_work[ncode], classification.get(ncode))
        return tags_cache[ncode]

    paths = sorted(glob.glob(args.chunks))
    if args.limit_shards:
        paths = paths[: args.limit_shards]
    if not paths:
        raise SystemExit(f"シャードが見つかりません: {args.chunks}")
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    for index, path in enumerate(paths, 1):
        rows_out = []
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                ncode = str((row.get("meta") or {}).get("ncode") or "")
                new_tags = tags_for(ncode)
                if new_tags is None:
                    counters["作品未登録(据え置き)"] += 1
                    rows_out.append(row)
                    continue
                old_tags = row.get("control_tags")
                # control_tags 以外が変化しないことの確認（B-2の本体）
                before = {k: v for k, v in row.items() if k != "control_tags"}
                row["control_tags"] = new_tags
                after = {k: v for k, v in row.items() if k != "control_tags"}
                if before != after:
                    raise SystemExit(f"想定外: control_tags 以外が変化しました ({path} {ncode})")
                counters["変更なし" if old_tags == new_tags else "control_tagsを更新"] += 1
                rows_out.append(row)
        if out_dir:
            target = out_dir / Path(path).name
            temporary = target.with_suffix(".tmp")
            with open(temporary, "w", encoding="utf-8") as out:
                for row in rows_out:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(temporary, target)  # 中断しても中途半端なファイルを残さない
        print(f"[{index}/{len(paths)}] {Path(path).name} {len(rows_out):,}行", file=sys.stderr)

    print(f"\n{'(検証のみ・書き出しなし)' if args.verify else f'書き出し: {out_dir}'}")
    total = sum(counters.values())
    for key, value in counters.most_common():
        print(f"  {key:<22}{value:>10,} ({value / total:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
