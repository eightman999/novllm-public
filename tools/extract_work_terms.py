#!/usr/bin/env python3
"""作品ごとの固有語（人名・地名・固有名詞）をLLMなしで抽出する。

**LLMを使わない理由**: 固有名詞は「その作品に頻出だが他作品ではほぼ出ない語」という
統計だけで十分に取れる。Macでの要約が1件37秒（実測）なのに対し、こちらは全作品でも数分。
捏造も混ざらない（本文に実在する文字列しか出さない）。

方法:
1. カタカナ2文字以上の連続、および漢字2〜4文字の連続を候補にする
   （日本語の人名・地名・作中用語はこの2つでほぼ拾える）
2. その作品での出現回数 × log(全作品数 / その語が出る作品数) で採点する（TF-IDF）
   一般語（「自分」「魔法」等）は多くの作品に出るので自動的に下がる
3. 上位N件を返す

出力は1作品1行のJSONL。build_context_view.py が条件の【語】欄に使う。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

KATAKANA = re.compile(r"[ァ-ヴー]{2,}")
KANJI = re.compile(r"[一-龥]{2,4}")
# どの作品にも出る語は固有名詞ではない。明らかな一般語は先に落としておく。
STOPWORDS = {
    "自分", "相手", "本当", "今日", "明日", "昨日", "時間", "場所", "世界", "人間",
    "言葉", "気持", "様子", "感じ", "普通", "問題", "必要", "以上", "以下", "一人",
    "二人", "全部", "最初", "最後", "今回", "前回", "程度", "状態", "場合", "理由",
    "方法", "結果", "関係", "意味", "存在", "可能", "無理", "大丈夫", "一緒", "一体",
}


def candidates(text: str) -> list[str]:
    words = [w for w in KATAKANA.findall(text) if len(w) >= 2]
    words += [w for w in KANJI.findall(text) if w not in STOPWORDS]
    return words


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="シャードのglob")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=8, help="1作品あたりの語数")
    parser.add_argument("--min-count", type=int, default=5,
                        help="作品内でこの回数以上出る語だけを候補にする")
    parser.add_argument("--max-chars-per-work", type=int, default=200_000,
                        help="1作品から読む上限。長編で時間を食いすぎないため")
    args = parser.parse_args()

    per_work: dict[str, Counter] = defaultdict(Counter)
    chars_read: Counter = Counter()
    paths = sorted(glob.glob(args.chunks))
    if not paths:
        raise SystemExit(f"見つかりません: {args.chunks}")
    for index, path in enumerate(paths, 1):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                ncode = row["meta"]["ncode"]
                if chars_read[ncode] >= args.max_chars_per_work:
                    continue
                text = str(row.get("text") or "")
                chars_read[ncode] += len(text)
                per_work[ncode].update(candidates(text))
        print(f"[{index}/{len(paths)}] {Path(path).name} 作品={len(per_work)}", file=sys.stderr)

    # 語がいくつの作品に出るか（文書頻度）
    doc_freq: Counter = Counter()
    for counts in per_work.values():
        doc_freq.update({w for w, c in counts.items() if c >= args.min_count})
    n_works = len(per_work)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for ncode, counts in sorted(per_work.items()):
            scored = []
            for word, count in counts.items():
                if count < args.min_count:
                    continue
                df = doc_freq.get(word, 1)
                if df > n_works * 0.3:   # 3割超の作品に出る語は固有名詞ではない
                    continue
                scored.append((count * math.log(n_works / df), word, count))
            scored.sort(reverse=True)
            terms = [w for _, w, _ in scored[: args.top]]
            out.write(json.dumps({"ncode": ncode, "terms": terms}, ensure_ascii=False) + "\n")

    print(f"\n書き出し: {out_path}（{n_works}作品）", file=sys.stderr)
    shown = 0
    for line in open(out_path, encoding="utf-8"):
        row = json.loads(line)
        if row["terms"] and shown < 8:
            print(f"  {row['ncode']}: {'、'.join(row['terms'])}", file=sys.stderr)
            shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
