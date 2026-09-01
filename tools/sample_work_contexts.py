#!/usr/bin/env python3
"""長編の代表チャンクを選び、1作品1行のLLM入力JSONLを作る（PLAN_20260726.md Phase L1）。

3方式を同じ総文字数予算で作り、どれが安定した特徴を出すかを比較できるようにする。

- uniform9    : 位置を均等に9点（7/26の基準線）
- stratified9 : 冒頭3・転機候補3・終盤3の位置層化9点。確定属性（結末・関係の帰着）は
                端に出やすいという想定を、位置の割り当てだけで表現する
- ends3       : 冒頭・中盤・終盤の3点のみ。1点あたりを長く取り、断片化を減らす

**2026-07-27の変更**: 以前は「選んだチャンクを頭から詰めて予算が尽きたら打ち切り」
だったため、後半の区間が丸ごと落ちることがあった。方式間で条件を揃える必要があるので、
**1区間あたりの予算 = max_chars // 区間数** に改めた。旧挙動は git 履歴を参照。
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

# 位置は「作品全体を 0.0〜1.0 に正規化したときの開始点」。
METHODS: dict[str, list[float]] = {
    "uniform9": [i / 8 for i in range(9)],
    # 冒頭(0.00/0.03/0.07) / 転機候補(0.22/0.45/0.68) / 終盤(0.90/0.96/1.00)
    "stratified9": [0.00, 0.03, 0.07, 0.22, 0.45, 0.68, 0.90, 0.96, 1.00],
    "ends3": [0.0, 0.5, 1.0],
}


def build_context(rows: list[dict], positions: list[float], max_chars: int) -> tuple[str, list[dict]]:
    """各位置から連続チャンクを区間予算いっぱいまで取り、区間を連結する。

    区間同士は重複させない。開始点が既に使われている場合は空きチャンクまで前に送る。

    前方に取り切れなかった分は後方へ伸ばす。position=1.0 の区間は開始点が
    最終チャンクなので、前方だけだと1チャンクしか取れず「終盤」が実質空になる
    （実測: ends3 の総量が 24,000字予算に対し 17,000字台まで落ちた）。
    """
    rows.sort(key=lambda row: (int(row["meta"]["episode_no"]), int(row["meta"]["chunk_index"])))
    total = len(rows)
    per_segment = max(1, max_chars // len(positions))
    used: set[int] = set()
    parts: list[str] = []
    sources: list[dict] = []

    for position in positions:
        start = round(position * (total - 1)) if total > 1 else 0
        while start < total and start in used:
            start += 1
        if start >= total:
            continue
        budget = per_segment
        segment_rows: list[dict] = []
        index = start
        while index < total and index not in used and budget > 0:
            text = str(rows[index].get("text") or "")[:budget]
            if not text:
                break
            segment_rows.append({"row": rows[index], "text": text})
            used.add(index)
            budget -= len(text)
            index += 1
        # 前方で足りなければ後方へ伸ばす（終端位置の区間を痩せさせないため）
        index = start - 1
        while index >= 0 and index not in used and budget > 0:
            text = str(rows[index].get("text") or "")
            if not text:
                break
            text = text[max(0, len(text) - budget):]  # 後方は末尾側を残す（続きが繋がる）
            segment_rows.insert(0, {"row": rows[index], "text": text})
            used.add(index)
            budget -= len(text)
            index -= 1
        if not segment_rows:
            continue
        start = min(start, index + 1)
        head, tail = segment_rows[0]["row"]["meta"], segment_rows[-1]["row"]["meta"]
        label = (f"[位置{position:.2f} / 第{head['episode_no']}話・チャンク{head['chunk_index']}"
                 f"〜第{tail['episode_no']}話・チャンク{tail['chunk_index']}]\n")
        body = "".join(item["text"] for item in segment_rows)
        parts.append(label + body)
        sources.append({
            "position": position,
            "start_chunk": start,
            "n_chunks": len(segment_rows),
            "episode_from": head["episode_no"],
            "episode_to": tail["episode_no"],
            "n_chars": len(body),
        })
    return "\n\n--- 代表区間 ---\n\n".join(parts), sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="train-*.jsonl のglob")
    parser.add_argument("--ncodes", required=True,
                        help="カンマ区切りの作品ID、またはIDを1行1件で書いたファイルのパス")
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", default="uniform9", choices=sorted(METHODS))
    parser.add_argument("--max-chars", type=int, default=24000)
    args = parser.parse_args()

    ncodes_path = Path(args.ncodes)
    raw = ncodes_path.read_text(encoding="utf-8").split() if ncodes_path.exists() \
        else args.ncodes.split(",")
    wanted = {value.strip() for value in raw if value.strip()}

    positions = METHODS[args.method]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(args.chunks)):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                ncode = str((row.get("meta") or {}).get("ncode") or "")
                if ncode in wanted:
                    grouped[ncode].append(row)
    missing = wanted - set(grouped)
    if missing:
        raise SystemExit(f"対象作品が見つかりません: {', '.join(sorted(missing))}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for ncode in sorted(wanted):
            text, sources = build_context(grouped[ncode], positions, args.max_chars)
            out.write(json.dumps({"ncode": ncode, "method": args.method, "text": text, "sampling": {
                "method": args.method, "positions": positions, "max_chars": args.max_chars,
                "per_segment_chars": max(1, args.max_chars // len(positions)),
                "total_chunks": len(grouped[ncode]), "segments": len(sources), "sources": sources,
            }}, ensure_ascii=False) + "\n")
            print(f"{ncode}: {len(grouped[ncode])}チャンクから{len(sources)}区間・{len(text):,}字")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
