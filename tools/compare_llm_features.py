#!/usr/bin/env python3
"""LLM抽出項目を特徴量にした分類器を、TF-IDF単独と比較する（PLAN_20260726.md Phase 2-1）。

同じ作品集合・同じタグ集合の上で、3つの特徴量を突き合わせる:

  tfidf  … 既存の文字n-gram TF-IDF
  llm    … extract_work_features.py が出した構造化項目のみ
  both   … 上記2つを連結

評価は `classify_works.eval_tag` をそのまま使う。指標の定義や交差検証の切り方が
本番と変わってしまうと比較にならないため、自前で書き直さない。

対象は短編サブセット（LLM抽出が済んでいる作品）のみ。1520作品全体の話ではないので、
ここでの AUC を本番runの AUC と直接比べないこと。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from classify_works import (  # noqa: E402
    eval_tag,
    raw_keyword_tags,
    silver_tags,
)

# classify_works.vectorize / extract_work_texts はあえて使わない。
# vectorize は (ngram, max_features, chars_per_work) だけをキャッシュキーにするので、
# 163件のサブセットを渡しても 1520件ぶんの行列を読み込んでしまう
# （関数自身のコメントが警告している事故そのもの）。ここでは毎回計算する。

# --- LLM抽出項目 → 特徴量 ----------------------------------------------------
# カテゴリ値は one-hot、強度は数値、複数値を取るものは multi-hot にする。
CATEGORICAL = {
    "protagonist_gender": ["male", "female", "multiple", "non_human_or_none", "unclear"],
    "viewpoint_person": ["first", "third_limited", "third_omniscient", "second",
                         "mixed", "unclear"],
    "setting_world": ["contemporary_real", "historical_real", "near_future",
                      "far_future_sf", "secondary_world_fantasy", "modern_fantasy",
                      "post_apocalypse", "virtual_or_game", "other", "unclear"],
    "world_transfer": ["reincarnation", "transfer", "returned", "none", "unclear"],
    "ending_type": ["happy", "bittersweet", "tragic", "open", "cyclical",
                    "unresolved", "unclear"],
    "protagonist_role": ["chosen_hero", "reluctant_hero", "ordinary_person",
                         "villain_or_antihero", "noble_or_ruler", "soldier_or_knight",
                         "merchant_or_artisan", "servant_or_slave", "scholar_or_mage",
                         "observer_narrator", "other"],
}
RELATIONSHIP_KINDS = [
    "romance_heterosexual", "romance_male_male", "romance_female_female",
    "harem_multiple_partners", "family", "master_servant", "mentor_student",
    "comrades", "rivalry", "antagonism", "friendship", "other",
]
STORY_SHAPES = [
    "coming_of_age", "revenge", "quest_adventure", "romance_courtship",
    "slice_of_life", "mystery_investigation", "war_campaign", "political_intrigue",
    "survival", "tragedy_downfall", "rise_to_power", "workplace_or_craft",
    "horror_confrontation", "comedy", "other",
]
SALIENCE_WEIGHT = {"central": 1.0, "major": 0.6, "minor": 0.3}


def feature_names() -> list[str]:
    names: list[str] = []
    for field, values in CATEGORICAL.items():
        names += [f"{field}={v}" for v in values]
    names += [f"rel:{k}" for k in RELATIONSHIP_KINDS]
    names += [f"shape:{s}" for s in STORY_SHAPES]
    names += ["protagonist_count", "sexual_present", "sexual_explicitness",
              "violence_present", "violence_explicitness", "confidence"]
    return names


def featurize(feat: dict[str, Any]) -> list[float]:
    row: list[float] = []
    for field, values in CATEGORICAL.items():
        v = feat.get(field)
        row += [1.0 if v == value else 0.0 for value in values]

    rels = {r.get("kind"): SALIENCE_WEIGHT.get(r.get("salience"), 0.3)
            for r in (feat.get("main_relationships") or []) if isinstance(r, dict)}
    row += [rels.get(k, 0.0) for k in RELATIONSHIP_KINDS]

    shapes = feat.get("story_shape") or []
    # 先頭ほど重要という並びなので重みを付ける
    shape_w = {s: 1.0 - 0.25 * i for i, s in enumerate(shapes) if isinstance(s, str)}
    row += [shape_w.get(s, 0.0) for s in STORY_SHAPES]

    sexual = feat.get("sexual_content") or {}
    violence = feat.get("violence_content") or {}
    row += [
        float(feat.get("protagonist_count") or 0),
        1.0 if sexual.get("present") else 0.0,
        float(sexual.get("explicitness") or 0),
        1.0 if violence.get("present") else 0.0,
        float(violence.get("explicitness") or 0),
        float(feat.get("confidence") or 0.0),
    ]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, help="extract_work_features.py の出力JSONL")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--texts", required=True,
                    help="cache/work_texts_{N}.jsonl。本文はここから読む")
    ap.add_argument("--ngram-min", type=int, default=2)
    ap.add_argument("--ngram-max", type=int, default=3)
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=3.0)
    ap.add_argument("--min-pos", type=int, default=8,
                    help="短編サブセットは163件程度しかないので既定を小さくしてある")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", help="比較結果のJSONL出力先")
    args = ap.parse_args()

    # --- LLM抽出結果 ---
    feats: dict[str, dict] = {}
    n_lines = n_skipped = 0
    with open(args.features, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            rec = json.loads(line)
            if not rec.get("ok"):
                n_skipped += 1
                continue
            feats[rec["ncode"]] = rec["features"]
    print(f"LLM抽出: {n_lines}行 / 失敗 {n_skipped} / 有効 {len(feats)}件", file=sys.stderr)

    # --- manifest ---
    works: dict[str, dict] = {}
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                works[rec["ncode"]] = rec

    ncodes = sorted(nc for nc in feats if nc in works)
    if not ncodes:
        print("ERROR: LLM抽出結果と manifest の突き合わせが0件です", file=sys.stderr)
        return 1
    print(f"比較対象: {len(ncodes)}作品", file=sys.stderr)

    # --- TF-IDF（サブセット専用に毎回計算する。キャッシュは使わない） ---
    texts_all: dict[str, str] = {}
    with open(args.texts, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                texts_all[r["ncode"]] = r["text"]
    missing = [nc for nc in ncodes if nc not in texts_all]
    if missing:
        print(f"WARNING: 本文が取れない作品 {len(missing)}件を除外します", file=sys.stderr)
        ncodes = [nc for nc in ncodes if nc in texts_all]
    texts = [texts_all[nc] for nc in ncodes]
    vec = TfidfVectorizer(analyzer="char",
                          ngram_range=(args.ngram_min, args.ngram_max),
                          max_features=args.max_features, min_df=args.min_df,
                          sublinear_tf=True, dtype=np.float32)
    X_tfidf = vec.fit_transform(texts)

    # --- LLM特徴 ---
    X_llm = sparse.csr_matrix(np.array([featurize(feats[nc]) for nc in ncodes],
                                       dtype=np.float64))
    X_both = sparse.hstack([X_tfidf, X_llm]).tocsr()
    print(f"次元数: tfidf={X_tfidf.shape[1]} llm={X_llm.shape[1]}", file=sys.stderr)

    # --- 対象タグ（サブセット内で正例が min-pos 以上のもの） ---
    tag_pos: dict[str, list[int]] = {}
    for i, nc in enumerate(ncodes):
        for tag in silver_tags(works[nc]) | raw_keyword_tags(works[nc]):
            tag_pos.setdefault(tag, []).append(i)
    tags = sorted((t for t, idx in tag_pos.items() if len(idx) >= args.min_pos),
                  key=lambda t: -len(tag_pos[t]))
    print(f"評価タグ: {len(tags)}個 (正例 >= {args.min_pos})", file=sys.stderr)
    if not tags:
        print("ERROR: 条件を満たすタグがありません。--min-pos を下げてください",
              file=sys.stderr)
        return 1

    matrices = {"tfidf": X_tfidf, "llm": X_llm, "both": X_both}
    rows = []
    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    header = (f"{'タグ':<14}" + "".join(f"{k+' AUC':>11}{k+' AP':>10}" for k in matrices)
              + f"{'Δllm':>9}{'Δboth':>9}")
    print("\n" + header)
    try:
        for tag in tags:
            y = np.zeros(len(ncodes), dtype=np.int64)
            y[tag_pos[tag]] = 1
            row: dict[str, Any] = {"tag": tag, "n_pos": int(y.sum()),
                                   "n_total": len(ncodes)}
            line = f"{tag:<14}"
            for name, X in matrices.items():
                r = eval_tag(X, y, args.folds, args.seed, args.C, calibrate=False)
                row[name] = {k: v for k, v in r.items() if not k.startswith("_")}
                line += f"{r['auc']:>11.3f}{r['ap']:>10.3f}"
            row["auc_delta_llm"] = round(row["llm"]["auc"] - row["tfidf"]["auc"], 4)
            row["auc_delta_both"] = round(row["both"]["auc"] - row["tfidf"]["auc"], 4)
            line += f"{row['auc_delta_llm']:>+9.3f}{row['auc_delta_both']:>+9.3f}"
            print(line, flush=True)
            rows.append(row)
            if out_f:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if out_f:
            out_f.close()

    # --- 集計 ---
    print("\n--- 平均 (対象タグ全体) ---")
    for name in matrices:
        aucs = [r[name]["auc"] for r in rows]
        aps = [r[name]["ap"] for r in rows]
        print(f"  {name:<6} AUC {np.mean(aucs):.4f}   AP {np.mean(aps):.4f}")
    wins = {name: sum(1 for r in rows
                      if r[name]["auc"] == max(r[k]["auc"] for k in matrices))
            for name in matrices}
    print(f"\n--- AUC が最良だったタグ数 ---\n  {wins}")

    # LLM特徴がどのタグに効いているかを、目視ではなく差分で出す。
    # 平均だけ見ると「効かない」で終わるが、効く相手と効かない相手が
    # はっきり分かれている可能性がある。
    by_delta = sorted(rows, key=lambda r: -r["auc_delta_llm"])
    print("\n--- LLM特徴が効いたタグ (Δ AUC 上位10) ---")
    print(f"  {'タグ':<14}{'正例':>5}{'tfidf':>9}{'llm':>9}{'Δ':>9}")
    for r in by_delta[:10]:
        print(f"  {r['tag']:<14}{r['n_pos']:>5}{r['tfidf']['auc']:>9.3f}"
              f"{r['llm']['auc']:>9.3f}{r['auc_delta_llm']:>+9.3f}")
    print("\n--- LLM特徴が効かなかったタグ (Δ AUC 下位10) ---")
    print(f"  {'タグ':<14}{'正例':>5}{'tfidf':>9}{'llm':>9}{'Δ':>9}")
    for r in by_delta[-10:]:
        print(f"  {r['tag']:<14}{r['n_pos']:>5}{r['tfidf']['auc']:>9.3f}"
              f"{r['llm']['auc']:>9.3f}{r['auc_delta_llm']:>+9.3f}")

    n_llm_up = sum(1 for r in rows if r["auc_delta_llm"] > 0)
    n_both_up = sum(1 for r in rows if r["auc_delta_both"] > 0)
    print("\n--- TF-IDF を上回ったタグ数 ---")
    print(f"  llm  が上回った: {n_llm_up}/{len(rows)}")
    print(f"  both が上回った: {n_both_up}/{len(rows)}"
          "   ← ここが多ければ「連結して足す」方針が成り立つ")
    if args.out:
        print(f"\n書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
