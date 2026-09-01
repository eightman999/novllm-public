#!/usr/bin/env python3
"""学習データ側で【語】行が信号になりうるかを測る（GPU不要／PLAN_20260801.md 手順1）。

【語】に並べた語が completion にそもそも現れないなら、その行は信号ではなく飾りで、
モデルは「無視する」ことを学ぶ。**モデルを測る前にデータを測る**。

同時に、7/31に見つかった2件の再発検知も兼ねる:
  - prompt末尾と completion先頭の重複（overlap_tokens=128 由来の丸写し）
  - 合計トークン長の分布（字数基準の切り出しによる「余らせながら溢れる」）
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter

TERMS_LINE = re.compile(r"^【語】(.*)$", re.MULTILINE)
MARK = "【続き】"
CTX_HEAD = "【ここまで】"

# 作者あとがき・活動報告の混入検知（2026-08-01 発見）。
# Web小説は本文とあとがきが同じ話に入っており、チャンク化でそのまま学習データへ流れ込む。
# checkpoint-400 の生成が本文の途中で「お久しぶりです。実は先週は病気で…」という
# あとがきに化けたのが発覚の端緒。**本文の続きを書く学習の妨げになるので落とす。**
# 曖昧な語（よろしくお願いします / ポイントを / お久しぶりです 等）は**入れない**。
# 会話文で普通に出るため誤検知になる（実測: よろしくお願いします 284件中109件が会話文内）。
# ここは「あとがき以外ではまず出ない」語だけに絞る。
AFTERWORD = re.compile(
    r"あとがき|活動報告|お読みいただき|読んでいただき|ご感想|感想をお待ち|感想をいただ"
    r"|評価をいただ|評価してくださ|次回予告|誤字報告|誤字脱字|ブックマーク|投稿します"
    r"|更新が遅れ")


def context_body(prompt: str) -> str:
    """**見出しを除いた**文脈本体だけを返す。

    見出しには【語】行そのものが入っているので、prompt全体を「文脈」として扱うと
    語は必ず文脈に含まれてしまい、「文脈に無い語が出たか」を測れなくなる。
    """
    body = prompt.rsplit(MARK, 1)[0]
    return body.split(CTX_HEAD, 1)[1] if CTX_HEAD in body else body


def terms_of(prompt: str) -> list[str]:
    m = TERMS_LINE.search(prompt)
    if not m:
        return []
    body = m.group(1).strip()
    if body in ("", "none"):
        return []
    return [t.strip() for t in re.split(r"[、,]", body) if t.strip()]


def overlap_len(ctx: str, comp: str, floor: int = 20, cap: int = 1200) -> int:
    n = min(len(ctx), len(comp), cap)
    for k in range(n, floor - 1, -1):
        if ctx[-k:] == comp[:k]:
            return k
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="./data/train_ctx_subset/balanced-00000.jsonl")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--tokenize", action="store_true",
                    help="トークン長も測る（遅い。--limit を効かせること）")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    args = ap.parse_args()

    recs = []
    with open(args.data, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            recs.append(json.loads(line))
    print(f"対象 {len(recs):,}件  ({args.data})")

    # --- 1. 【語】が信号になっているか -------------------------------------
    n_terms, in_ctx, in_comp = [], [], []
    no_terms = 0
    all_terms: Counter[str] = Counter()
    for r in recs:
        terms = terms_of(r["prompt"])
        if not terms:
            no_terms += 1
            continue
        ctx = context_body(r["prompt"])
        comp = r["completion"]
        n_terms.append(len(terms))
        in_ctx.append(sum(1 for t in terms if t in ctx))
        in_comp.append(sum(1 for t in terms if t in comp))
        all_terms.update(terms)

    n = len(n_terms)
    print(f"\n--- 【語】 ---")
    print(f"【語】が無いサンプル: {no_terms:,}件")
    if n:
        print(f"語数              中央値 {st.median(n_terms):.0f} / 平均 {st.mean(n_terms):.1f}")
        print(f"うち文脈に出現    中央値 {st.median(in_ctx):.0f} / 平均 {st.mean(in_ctx):.2f}")
        print(f"うち続きに出現    中央値 {st.median(in_comp):.0f} / 平均 {st.mean(in_comp):.2f}")
        hit = sum(1 for x in in_comp if x > 0)
        print(f"続きに1語でも出現: {hit:,}/{n:,} ({100 * hit / n:.1f}%)")
        kata = sum(1 for t in all_terms if re.fullmatch(r"[ァ-ヶー]+", t))
        kanji = sum(1 for t in all_terms if re.fullmatch(r"[一-龥]+", t))
        print(f"異なり語 {len(all_terms):,}  カタカナのみ {100 * kata / len(all_terms):.1f}% / "
              f"漢字のみ {100 * kanji / len(all_terms):.1f}%")

    # --- 2. 丸写しの再発検知 -----------------------------------------------
    ov = []
    for r in recs:
        ctx = context_body(r["prompt"]).rstrip("\n")
        ov.append(overlap_len(ctx, r["completion"].lstrip("\n")))
    nz = [x for x in ov if x > 0]
    print(f"\n--- 文脈と続きの重複（0であるべき）---")
    print(f"重複あり {len(nz):,}/{len(ov):,} ({100 * len(nz) / len(ov):.2f}%)")
    if nz:
        print(f"重複字数 中央値 {st.median(nz):.0f} / 最大 {max(nz)}")
        print("  ** 0%でなければ overlap_tokens 由来の丸写しが残っている **")

    # --- 3. 作者あとがきの混入（0であるべき）--------------------------------
    aw_comp = sum(1 for r in recs if AFTERWORD.search(r["completion"]))
    aw_ctx = sum(1 for r in recs if AFTERWORD.search(context_body(r["prompt"])))
    print(f"\n--- 作者あとがき・活動報告の混入（0であるべき）---")
    print(f"続きに混入 {aw_comp:,}/{len(recs):,} ({100 * aw_comp / len(recs):.2f}%)")
    print(f"文脈に混入 {aw_ctx:,}/{len(recs):,} ({100 * aw_ctx / len(recs):.2f}%)")
    if aw_comp:
        print("  ** 損失をかける側にあとがきが入っている。"
              "モデルが本文の途中であとがきを書き出す原因 **")

    # --- 4. トークン長の分布 ------------------------------------------------
    if args.tokenize:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.base_model)
        texts = [r["prompt"] + r["completion"] for r in recs]
        lens = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
        s = sorted(lens)
        over = sum(1 for x in lens if x > 4096)
        print(f"\n--- 合計トークン長 ---")
        print(f"中央値 {st.median(lens):.0f} / p99 {s[int(len(s) * 0.99)]} / 最大 {max(lens)}")
        print(f"4096超過 {over:,}件 ({100 * over / len(lens):.2f}%)")
        print(f"4096に対する使用率（中央値）: {100 * st.median(lens) / 4096:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
