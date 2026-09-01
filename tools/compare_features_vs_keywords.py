#!/usr/bin/env python3
"""LLM抽出項目と作者申告キーワードの一致率を見る（PLAN_20260726.md Phase 1-5）。

注意: 作者申告タグは positive-only（申告が無い＝そのタグが当てはまらない、とは限らない）。
したがって「申告=False かつ LLM=True」は誤りとは限らず、未申告の可能性がある。
本スクリプトは混同行列をそのまま出し、
  - recall  = 申告Trueのうち LLM も True と言った割合  … ここが低いとLLMが読めていない
  - precision = LLM Trueのうち 申告も True だった割合  … 未申告の影響で低く出るのは想定内
の両方を並べて報告する。判断に使うのは主に recall。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

# 作者申告 control_flags のキー -> LLM抽出項目からその真偽を導く関数
PREDICATES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "男主人公": lambda f: f.get("protagonist_gender") == "male",
    "女主人公": lambda f: f.get("protagonist_gender") == "female",
    "群像劇": lambda f: (f.get("protagonist_count") or 0) >= 3,
    "異世界転生": lambda f: f.get("world_transfer") == "reincarnation",
    "異世界転移": lambda f: f.get("world_transfer") == "transfer",
    "ボーイズラブ": lambda f: _has_rel(f, "romance_male_male"),
    "ガールズラブ": lambda f: _has_rel(f, "romance_female_female"),
    "ハーレム": lambda f: _has_rel(f, "harem_multiple_partners"),
    "恋愛": lambda f: _has_rel(f, "romance_heterosexual", "romance_male_male",
                             "romance_female_female", "harem_multiple_partners")
    or _has_shape(f, "romance_courtship"),
    "残酷な描写あり": lambda f: (f.get("violence_content") or {}).get("explicitness", 0) >= 3,
    "R15": lambda f: (f.get("sexual_content") or {}).get("explicitness", 0) >= 2
    or (f.get("violence_content") or {}).get("explicitness", 0) >= 3,
    "現代": lambda f: f.get("setting_world") in ("contemporary_real", "modern_fantasy"),
    "歴史": lambda f: f.get("setting_world") == "historical_real",
    "SF": lambda f: f.get("setting_world") in ("far_future_sf", "near_future"),
    "ホラー": lambda f: _has_shape(f, "horror_confrontation"),
    "ミステリー": lambda f: _has_shape(f, "mystery_investigation"),
    "戦争": lambda f: _has_shape(f, "war_campaign"),
    "ミリタリー": lambda f: _has_shape(f, "war_campaign"),
    "オリジナル戦記": lambda f: _has_shape(f, "war_campaign"),
    "架空戦記": lambda f: _has_shape(f, "war_campaign"),
}


def _has_rel(f: dict[str, Any], *kinds: str) -> bool:
    return any(r.get("kind") in kinds for r in (f.get("main_relationships") or []))


def _has_shape(f: dict[str, Any], *shapes: str) -> bool:
    return any(s in shapes for s in (f.get("story_shape") or []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="extract_work_features.py の出力JSONL")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", help="集計JSONの出力先")
    args = ap.parse_args()

    meta: dict[str, dict] = {}
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                meta[rec["ncode"]] = rec

    feats: dict[str, dict] = {}
    n_lines = n_bad = 0
    with open(args.features, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            rec = json.loads(line)
            if not rec.get("ok") or rec.get("problems"):
                n_bad += 1
                if not rec.get("ok"):
                    continue
            feats[rec["ncode"]] = rec["features"]

    print(f"LLM出力: {n_lines}行 / スキーマ問題あり {n_bad}行 / 有効 {len(feats)}件")

    results: dict[str, Any] = {}
    for tag, pred in PREDICATES.items():
        tp = fp = fn = tn = 0
        for ncode, feat in feats.items():
            declared = bool((meta.get(ncode, {}).get("control_flags") or {}).get(tag, False))
            predicted = bool(pred(feat))
            if declared and predicted:
                tp += 1
            elif declared and not predicted:
                fn += 1
            elif not declared and predicted:
                fp += 1
            else:
                tn += 1
        n_pos = tp + fn
        recall = tp / n_pos if n_pos else None
        precision = tp / (tp + fp) if (tp + fp) else None
        results[tag] = {
            "declared_pos": n_pos, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "precision": precision,
        }

    print(f"\n{'タグ':<14}{'申告+':>6}{'TP':>5}{'FN':>5}{'FP':>5}{'recall':>9}{'prec':>8}")
    for tag, r in sorted(results.items(), key=lambda kv: -kv[1]["declared_pos"]):
        rc = f"{r['recall']:.3f}" if r["recall"] is not None else "  -  "
        pr = f"{r['precision']:.3f}" if r["precision"] is not None else "  -  "
        print(f"{tag:<14}{r['declared_pos']:>6}{r['tp']:>5}{r['fn']:>5}{r['fp']:>5}{rc:>9}{pr:>8}")

    covered = [r for r in results.values() if r["declared_pos"] > 0]
    if covered:
        micro_tp = sum(r["tp"] for r in covered)
        micro_pos = sum(r["declared_pos"] for r in covered)
        print(f"\nマイクロ平均 recall = {micro_tp}/{micro_pos} = {micro_tp / micro_pos:.3f}")
        print(f"申告+が1件以上あるタグ: {len(covered)}/{len(results)}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"n_works": len(feats), "tags": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
