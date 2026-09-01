#!/usr/bin/env python3
"""「直前の文脈と計画を踏まえて続きを書く」形の学習サンプルを作る。

**なぜ作り直すのか（7/29実測）**:
今の学習サンプルは「条件行＋物語の途中の断片（中央値1,523トークン＝2,148字）」で、
前後の文脈も計画も入っていない。作品の中央値は147,711字なので、モデルは
**連続した物語を2,100字までしか見たことがない**。生成が前後と噛み合わないのは
学習量の不足ではなく、この構造が原因。ステップを増やしても直らない。

新形式（prompt / completion に分け、completion 側だけに損失をかける）:
    prompt =
      [viewpoint=..][protagonist_gender=..][setting=..][genre=..][r18=..][dialogue=..][sentence=..]
      【作品】{タイトル}
      【話】{episode_title。裸のナンバリングは unknown}
      【語】{TF-IDFで抽出した作品固有語}
      【ここまで】
      {直前の文脈}
      【続き】
    completion = {本文}

損失を completion 側に限る理由: 文脈側にも損失をかけると「文脈を丸暗記する」方を
学んでしまい、肝心の「続きを書く」能力が育たない。

文体軸（dialogue / sentence）の境界は決め打ちにせず、**このコーパスの三分位**を
実データから算出して `style_buckets.json` に残す。分布とずれた境界を切ると
片方の値ばかりになり、条件として機能しなくなる（ending_type が open 90% で
κ=0.259 になったのと同じ失敗）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_format import bucket, build_sample  # noqa: E402  書式の正本

# 「第107話」「1204」「ep3」のように中身の無いタイトルを弾く。実測で10.8%ある。
NUMBERING = re.compile(
    r"^[\s　]*(?:第?\s*[0-9０-９]+\s*(?:話|章|節|回|部)?|[Ee][Pp]\.?\s*[0-9]+|[0-9０-９]+)[\s　.、\-—–]*")


def informative_title(title: str | None) -> str | None:
    if not title:
        return None
    rest = NUMBERING.sub("", title).strip("　 .-—–")
    return title.strip() if len(rest) > 1 else None


def quantiles(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    n = len(values)
    return (values[n // 3], values[n * 2 // 3])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--terms", help="extract_work_terms.py の出力")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-chars", type=int, default=2400,
                        help="【ここまで】に入れる直前文脈の字数")
    parser.add_argument("--continuation-chars", type=int, default=2900,
                        help="【続き】＝損失をかける本文の字数")
    parser.add_argument("--stride", type=int, default=2,
                        help="何チャンクごとにサンプルを作るか。1で全チャンク")
    args = parser.parse_args()

    terms_by_work: dict[str, list[str]] = {}
    if args.terms:
        for line in Path(args.terms).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                terms_by_work[row["ncode"]] = row.get("terms") or []

    paths = sorted(glob.glob(args.chunks))
    if not paths:
        raise SystemExit(f"見つかりません: {args.chunks}")

    # --- パス1: 文体指標の分布から三分位を出す（決め打ちにしない）---
    dialogue_values, sentence_values = [], []
    for path in paths:
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                metrics = (json.loads(line)["meta"].get("style_metrics") or {})
                if metrics.get("dialogue_ratio") is not None:
                    dialogue_values.append(float(metrics["dialogue_ratio"]))
                if metrics.get("avg_sentence_length") is not None:
                    sentence_values.append(float(metrics["avg_sentence_length"]))
    dialogue_edges = quantiles(dialogue_values)
    sentence_edges = quantiles(sentence_values)
    print(f"文体の三分位: dialogue_ratio {dialogue_edges} / "
          f"avg_sentence_length {sentence_edges}", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "style_buckets.json").write_text(json.dumps({
        "dialogue_ratio_edges": dialogue_edges,
        "avg_sentence_length_edges": sentence_edges,
        "n_samples_used": len(dialogue_values),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- パス2: 作品ごとに読み順で並べ、(直前文脈 → 続き) を作る ---
    written = 0
    no_title = 0
    for shard_index, path in enumerate(paths, 1):
        by_work: dict[str, list[dict]] = defaultdict(list)
        with open(path, encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    by_work[row["meta"]["ncode"]].append(row)

        target = out_dir / Path(path).name
        temporary = target.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as out:
            for ncode, rows in by_work.items():
                rows.sort(key=lambda r: (r["meta"]["episode_no"], r["meta"]["chunk_index"]))
                for i in range(1, len(rows), args.stride):
                    head = rows[i]["meta"]
                    # 直前の文脈は「読み順で前にあるチャンクの末尾」を使う。
                    # 先頭から取ると続きと繋がらない。
                    context = ""
                    j = i - 1
                    while j >= 0 and len(context) < args.context_chars:
                        context = str(rows[j].get("text") or "") + context
                        j -= 1
                    context = context[-args.context_chars:]

                    continuation = "".join(
                        str(r.get("text") or "") for r in rows[i:i + 3]
                    )[: args.continuation_chars]
                    if len(continuation) < 400 or len(context) < 200:
                        continue

                    metrics = head.get("style_metrics") or {}
                    tags = dict(rows[i].get("control_tags") or {})
                    tags["r18"] = "yes" if head.get("is_r18") else "no"
                    tags["dialogue"] = bucket(metrics.get("dialogue_ratio"), dialogue_edges)
                    tags["sentence"] = bucket(
                        metrics.get("avg_sentence_length"), sentence_edges,
                        ("short", "mid", "long"))

                    episode_title = informative_title(head.get("episode_title"))
                    if episode_title is None:
                        no_title += 1
                    prompt, completion = build_sample(
                        tags, head.get("title"), episode_title,
                        terms_by_work.get(ncode), context, continuation)
                    out.write(json.dumps({
                        "prompt": prompt, "completion": completion,
                        "meta": {"ncode": ncode, "episode_no": head["episode_no"]},
                    }, ensure_ascii=False) + "\n")
                    written += 1
        os.replace(temporary, target)
        print(f"[{shard_index}/{len(paths)}] {Path(path).name} 累計={written:,}", file=sys.stderr)

    print(f"\n書き出し: {out_dir}（{written:,}件）", file=sys.stderr)
    print(f"  話タイトルが無情報だったもの: {no_title:,} ({no_title / max(1, written):.1%})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
