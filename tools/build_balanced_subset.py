#!/usr/bin/env python3
"""条件を均等化した学習サブセットを train_view から作る（PLAN_20260727.md C-3準備）。

**なぜ要るか（7/27実測）**:
- 1,000ステップ×8シーケンス＝**8,000チャンクしか消費しない**（全790,897件の約1%）。
  1エポックは93.4秒/stepで107日かかり非現実的なので、はじめから使う分だけ作る。
- 全globのままでも Trainer は RandomSampler なので順序の偏りは出ないが、
  **条件の偏りはそのまま乗る**: viewpoint は first 73.3% / third 22.8%、
  protagonist_gender は male 71.7% / female 25.6% / non_human 0.02%。
  107種の組み合わせのうち45種が0.1%未満しかない。
  条件付き生成を学習させたいのに、指定したい値を2割しか見せないのは不利。
- **作品ごとの偏りも大きい**: 1作品あたり最小1 / p50=258 / 最大11,559チャンクで、
  上位10作品だけで全体の7.5%。素直にランダムに引くと長編に支配される。

やること:
1. 条件（条件行そのもの）でグループ分けし、**できるだけ等量**になるよう配分する。
   目標に満たないグループは全部使い、余りを容量のあるグループへ再配分する。
2. グループ内は**作品をラウンドロビン**で回して、1作品が占有しないようにする。
3. 1作品から複数取るときは**位置均等**に取る。チャンク順＝物語順なので、
   先頭から取ると導入部しか学習しない（ends3サンプラで踏んだのと同じ罠）。

train_view は行順を保っているので、(シャード, 行番号) を索引にした2パスで処理する。
本文を全部メモリに載せない（790,897行×約1.2KBで1GB超になるため）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def even_indices(length: int, count: int) -> list[int]:
    """0..length-1 から count 個を位置均等に選ぶ。"""
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted({round(i * (length - 1) / (count - 1)) for i in range(count)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, help="train_view のglob")
    parser.add_argument("--out", required=True, help="出力JSONL（1ファイル）")
    parser.add_argument("--target", type=int, default=16000, help="目標チャンク数")
    parser.add_argument("--max-per-work", type=int, default=100,
                        help="1作品から取る上限。長編の占有を防ぐ")
    parser.add_argument("--report", help="配分結果のJSON")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.view))
    if not paths:
        raise SystemExit(f"見つかりません: {args.view}")

    # --- パス1: 条件と作品だけを索引化する（本文は読み捨てる） ---
    index: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    total = 0
    for shard_index, path in enumerate(paths):
        with open(path, encoding="utf-8") as source:
            for line_index, line in enumerate(source):
                if not line.strip():
                    continue
                row = json.loads(line)
                # 旧形式は text、新形式（build_context_view.py）は prompt に条件行がある。
                source_text = row.get("text") or row.get("prompt") or ""
                condition = source_text.split("\n", 1)[0]
                ncode = (row.get("meta") or {}).get("ncode") or "unknown"
                index[condition][ncode].append((shard_index, line_index))
                total += 1
        print(f"[索引 {shard_index + 1}/{len(paths)}] {Path(path).name}", file=sys.stderr)
    print(f"総チャンク {total:,} / 条件 {len(index)}種", file=sys.stderr)

    # --- 配分: できるだけ等量にし、余りを容量のあるグループへ回す ---
    capacity = {
        condition: sum(min(len(v), args.max_per_work) for v in works.values())
        for condition, works in index.items()
    }
    quota = {condition: 0 for condition in index}
    remaining = args.target
    open_groups = set(index)
    while remaining > 0 and open_groups:
        share = max(1, remaining // len(open_groups))
        progressed = False
        for condition in sorted(open_groups):
            room = capacity[condition] - quota[condition]
            if room <= 0:
                open_groups.discard(condition)
                continue
            take = min(share, room, remaining)
            if take <= 0:
                continue
            quota[condition] += take
            remaining -= take
            progressed = True
            if quota[condition] >= capacity[condition]:
                open_groups.discard(condition)
            if remaining <= 0:
                break
        if not progressed:
            break

    # --- 選択: グループ内は作品をラウンドロビン、作品内は位置均等 ---
    selected: set[tuple[int, int]] = set()
    per_condition = Counter()
    per_work = Counter()
    for condition, works in index.items():
        want = quota[condition]
        if want <= 0:
            continue
        names = sorted(works)
        # 各作品から何個取るかをラウンドロビンで決める
        take_count = {name: 0 for name in names}
        left = want
        while left > 0:
            moved = False
            for name in names:
                room = min(len(works[name]), args.max_per_work) - take_count[name]
                if room <= 0:
                    continue
                take_count[name] += 1
                left -= 1
                moved = True
                if left <= 0:
                    break
            if not moved:
                break
        for name, count in take_count.items():
            if count <= 0:
                continue
            positions = works[name]
            for i in even_indices(len(positions), count):
                selected.add(positions[i])
                per_condition[condition] += 1
                per_work[name] += 1

    # --- パス2: 選んだ行だけを書き出す ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(".tmp")
    written = 0
    with open(temporary, "w", encoding="utf-8") as out:
        for shard_index, path in enumerate(paths):
            with open(path, encoding="utf-8") as source:
                for line_index, line in enumerate(source):
                    if (shard_index, line_index) in selected:
                        out.write(line if line.endswith("\n") else line + "\n")
                        written += 1
    os.replace(temporary, out_path)

    print(f"\n書き出し: {out_path}（{written:,}チャンク / {len(per_work):,}作品）")
    print(f"条件: {len(per_condition)}種 使用")
    counts = sorted(per_condition.values())
    print(f"  1条件あたり: 最小{counts[0]} p50={counts[len(counts) // 2]} 最大{counts[-1]}")
    work_counts = sorted(per_work.values())
    print(f"  1作品あたり: 最小{work_counts[0]} p50={work_counts[len(work_counts) // 2]} "
          f"最大{work_counts[-1]}")
    print("\n  多い条件5:")
    for condition, count in per_condition.most_common(5):
        print(f"    {count:>6}  {condition}")
    print("  少ない条件5:")
    for condition, count in per_condition.most_common()[-5:]:
        print(f"    {count:>6}  {condition}")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "target": args.target, "written": written,
            "max_per_work": args.max_per_work,
            "n_conditions": len(per_condition), "n_works": len(per_work),
            "per_condition": dict(per_condition),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n配分レポート: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
