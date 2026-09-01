#!/usr/bin/env python3
"""学習が実際に読む最小の view を作る（PLAN_20260727.md C-2）。

**なぜ要るか（7/27実測）**:
`load_dataset("json", ...)` は shard ごとに型を推論して結合するため、
`control_tags.genre` が**全行空の shard が6/163ある**と `list<null>` と推論され、
他の shard の `list<string>` と衝突して
`TypeError: Couldn't cast array of type string to null` で落ちる。
これは chunks_v2 で作り込んだものではなく**旧 chunks にも同じだけ存在する**
（むしろ setting は 1shard → 0shard に改善した）。本番学習が一度も走って
いなかった原因の一つ。

対策として、学習に要る列だけの平坦なJSONLへ落とす:
- `text`        … 条件行を差し込み済みの本文（`control_format` を通す＝書式の正本は1つのまま）
- `meta.ncode`  … train_lora.py の作品ID検証と batch source log が使う

こうすると全 shard で型が一意に決まり、推論の余地が無くなる。
副次的に、モデルが実際に見る文字列をそのまま目で読める形で残せる。

このviewに対しては train_lora.py へ `--control-prefix` を**付けない**
（既に差し込み済みのため。二重に付くと条件行が2行になる）。
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

from control_format import build_training_text  # noqa: E402  書式の正本はここだけ


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="入力シャードのglob")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit-shards", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.chunks))
    if args.limit_shards:
        paths = paths[: args.limit_shards]
    if not paths:
        raise SystemExit(f"シャードが見つかりません: {args.chunks}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    prefixes = Counter()

    for index, path in enumerate(paths, 1):
        target = out_dir / Path(path).name
        temporary = target.with_suffix(".tmp")
        rows = 0
        with open(path, encoding="utf-8") as source, \
                open(temporary, "w", encoding="utf-8") as out:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                ncode = (record.get("meta") or {}).get("ncode")
                if not ncode or ncode == "unknown":
                    counters["ncode不明のため除外"] += 1
                    continue
                text = build_training_text(record)
                prefixes[text.split("\n", 1)[0]] += 1
                out.write(json.dumps({"text": text, "meta": {"ncode": ncode}},
                                     ensure_ascii=False) + "\n")
                rows += 1
                counters["書き出し"] += 1
        os.replace(temporary, target)
        print(f"[{index}/{len(paths)}] {Path(path).name} {rows:,}行", file=sys.stderr)

    print(f"\n書き出し: {out_dir}")
    for key, value in counters.most_common():
        print(f"  {key:<22}{value:>10,}")
    print(f"\n条件行の種類: {len(prefixes)}種（上位5）")
    for prefix, count in prefixes.most_common(5):
        print(f"  {count:>9,}  {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
