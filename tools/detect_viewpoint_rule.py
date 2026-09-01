#!/usr/bin/env python3
"""地の文の一人称代名詞比で視点人称を判定する。

LLMより先にこれを作る理由:
- 効くならGPUを1秒も使わずに済む（`viewpoint` は810,533チャンク全部が unknown）。
- 効かなくてもLLM出力の**独立した検証手段**になる。

原理: 三人称の地の文は一人称代名詞をほぼ使わないため、会話文を除いた本文の
一人称代名詞密度を判定材料にする。閾値は対象コーパスで検証して調整する。

限界（承知の上で使う）:
- 一人称 / 三人称の2値までしか出せない。third_limited と third_omniscient は区別できない。
- 「まえがき・あとがき」が本文として混入すると作者の「私」を拾い、三人称作品を
  一人称側へ誤判定する可能性がある。
- 空白帯（--low〜--high）に落ちた作品はunknownにして、推測で埋めない。
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# 鉤括弧内（会話文）を落とす。入れ子は繰り返し適用で処理する。
DIALOGUE = re.compile(r"[「『][^「『」』]*[」』]")
PRONOUNS = ["俺", "私", "僕", "あたし", "儂", "わし", "拙者", "吾輩", "我輩",
            "オレ", "ワタシ", "ボク", "わたし", "ぼく", "おれ"]
PRONOUN_PATTERN = re.compile("|".join(PRONOUNS))


def strip_dialogue(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = DIALOGUE.sub("", text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="train-*.jsonl のglob")
    parser.add_argument("--out", required=True, help="1作品1行のJSONL")
    parser.add_argument("--low", type=float, default=0.5,
                        help="千字あたりの一人称がこれ未満なら三人称")
    parser.add_argument("--high", type=float, default=1.0,
                        help="千字あたりの一人称がこれ以上なら一人称。間は unknown")
    parser.add_argument("--min-narration-chars", type=int, default=2000,
                        help="地の文がこれ未満の作品は判定しない")
    args = parser.parse_args()
    if args.low > args.high:
        raise SystemExit("--low は --high 以下にしてください")

    narration_chars: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    per_pronoun: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lines_total: dict[str, int] = defaultdict(int)
    lines_hit: dict[str, int] = defaultdict(int)

    paths = sorted(glob.glob(args.chunks))
    if not paths:
        raise SystemExit(f"チャンクが見つかりません: {args.chunks}")
    for shard_index, path in enumerate(paths, 1):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                ncode = str((row.get("meta") or {}).get("ncode") or "")
                if not ncode:
                    continue
                narration = strip_dialogue(str(row.get("text") or ""))
                narration_chars[ncode] += len(narration)
                for match in PRONOUN_PATTERN.findall(narration):
                    hits[ncode] += 1
                    per_pronoun[ncode][match] += 1
                for raw in narration.split("\n"):
                    stripped = raw.strip()
                    if len(stripped) > 25:
                        lines_total[ncode] += 1
                        if PRONOUN_PATTERN.search(stripped):
                            lines_hit[ncode] += 1
        print(f"[{shard_index}/{len(paths)}] {Path(path).name} 作品数={len(narration_chars)}",
              file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    with open(out_path, "w", encoding="utf-8") as out:
        for ncode in sorted(narration_chars):
            chars = narration_chars[ncode]
            rate = hits[ncode] / chars * 1000 if chars else 0.0
            if chars < args.min_narration_chars:
                viewpoint = "unknown"
            elif rate >= args.high:
                viewpoint = "first_person"
            elif rate < args.low:
                viewpoint = "third_person"
            else:
                viewpoint = "unknown"  # 空白帯。推測で埋めない
            counts[viewpoint] += 1
            out.write(json.dumps({
                "ncode": ncode, "viewpoint": viewpoint,
                "narration_chars": chars, "first_person_hits": hits[ncode],
                "rate_per_1000": round(rate, 4),
                "line_rate": round(lines_hit[ncode] / lines_total[ncode], 4) if lines_total[ncode] else 0.0,
                "pronouns": dict(sorted(per_pronoun[ncode].items(), key=lambda item: -item[1])),
                "method": "narration_first_person_rate",
                "thresholds": {"low": args.low, "high": args.high},
            }, ensure_ascii=False) + "\n")

    total = sum(counts.values())
    print(f"\n判定: {dict(counts)} / 計{total}作品", file=sys.stderr)
    for key, value in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {key:<14}{value:>6} ({value / total:.1%})", file=sys.stderr)
    print(f"書き出し: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
