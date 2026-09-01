#!/usr/bin/env python3
"""Phase L1/L2: サンプリング方式ごとに特徴量の安定性を測る。

作品全体の曖昧な一致率ではなく、フィールド単位で見る（PLAN_20260726.md の原則）。

2種類の「安定性」を分けて出す。ここを混ぜると判断を誤る:

1. **再実行安定性**（同じ入力を2回）
   temperature=0 かつ同一プロンプトなので、貪欲デコードでは原理的にほぼ一致する。
   実測でも run1 はプロンプトキャッシュに当たって数秒で返る。つまりこれは
   数値的な非決定性のサニティチェックであって、特徴量の信頼性の指標ではない。
2. **方式間一致率**（見せる箇所を変えたとき同じ値が出るか）
   原則にある「サンプル位置を変えたときの値の安定性」はこちら。
   ここが低いフィールドは、値が作品の属性ではなく**たまたま見せた区間の属性**を
   反映しているということなので、本番展開してはいけない。

任意で --manual に人手確認の正解を渡すと一致率も出す。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SCALAR_FIELDS = ["protagonist_gender", "protagonist_count", "viewpoint_person", "setting_world",
                 "setting_era", "world_transfer", "ending_type", "protagonist_role"]
CONTENT_FIELDS = ["sexual_content", "violence_content"]
SET_FIELDS = ["story_shape", "main_relationships"]
ALL_FIELDS = SCALAR_FIELDS + CONTENT_FIELDS + SET_FIELDS


def as_set(field: str, value: Any) -> set[str]:
    if field == "story_shape":
        return {str(item) for item in (value or [])}
    return {str((item or {}).get("kind")) for item in (value or []) if (item or {}).get("kind")}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def field_agreement(field: str, a: dict, b: dict) -> float:
    """1.0=完全一致。集合フィールドは Jaccard を返すので中間値を取りうる。"""
    if field in SCALAR_FIELDS:
        left, right = a.get(field), b.get(field)
        if field == "setting_era":  # 自由記述。表記ゆれを吸収して比較する
            left = str(left or "").strip().lower()
            right = str(right or "").strip().lower()
        return 1.0 if left == right else 0.0
    if field in CONTENT_FIELDS:
        left, right = a.get(field) or {}, b.get(field) or {}
        return 1.0 if (left.get("present"), left.get("explicitness")) == \
                      (right.get("present"), right.get("explicitness")) else 0.0
    return jaccard(as_set(field, a.get(field)), as_set(field, b.get(field)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="extract_context_features.py の出力")
    parser.add_argument("--manual", help="人手確認 JSONL: {ncode, fields:{...}}")
    parser.add_argument("--out", help="集計JSONの書き出し先")
    args = parser.parse_args()

    runs: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for line in Path(args.results).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok"):
            runs[(row["ncode"], row["method"])][int(row["run_index"])] = row

    methods = sorted({method for _, method in runs})
    summary: dict[str, Any] = {"methods": {}}

    print(f"{'方式':<13}{'作品':>5}{'安定Fld':>9}{'平均安定率':>11}{'平均conf':>10}{'秒/作品':>9}{'字/作品':>10}")
    for method in methods:
        pairs = [(ncode, by_run) for (ncode, m), by_run in runs.items()
                 if m == method and len(by_run) >= 2]
        if not pairs:
            continue
        per_field: dict[str, list[float]] = defaultdict(list)
        confidences, durations, sizes, all_stable_flags = [], [], [], []
        for _, by_run in pairs:
            a, b = by_run[0]["features"], by_run[1]["features"]
            scores = {field: field_agreement(field, a, b) for field in ALL_FIELDS}
            for field, score in scores.items():
                per_field[field].append(score)
            all_stable_flags.append(all(score == 1.0 for score in scores.values()))
            confidences.append((float(a.get("confidence") or 0) + float(b.get("confidence") or 0)) / 2)
            durations.append(sum(float((by_run[i].get("stats") or {}).get("total_duration_s") or 0)
                                 for i in (0, 1)) / 2)
            sizes.append(by_run[0]["n_chars"])

        field_rates = {field: sum(values) / len(values) for field, values in per_field.items()}
        n_perfect = sum(1 for rate in field_rates.values() if rate == 1.0)
        mean_rate = sum(field_rates.values()) / len(field_rates)
        summary["methods"][method] = {
            "n_works": len(pairs), "field_rates": field_rates,
            "n_fields_always_stable": n_perfect, "mean_stability": round(mean_rate, 4),
            "mean_confidence": round(sum(confidences) / len(confidences), 4),
            "mean_seconds": round(sum(durations) / len(durations), 1),
            "mean_chars": round(sum(sizes) / len(sizes)),
            "n_works_fully_stable": sum(all_stable_flags),
            # confidence が安定性の指標として使えるか（使えるなら前者 > 後者になるはず）
            "conf_when_fully_stable": round(
                sum(c for c, f in zip(confidences, all_stable_flags) if f)
                / max(1, sum(all_stable_flags)), 4),
            "conf_when_unstable": round(
                sum(c for c, f in zip(confidences, all_stable_flags) if not f)
                / max(1, len(all_stable_flags) - sum(all_stable_flags)), 4),
        }
        entry = summary["methods"][method]
        print(f"{method:<13}{entry['n_works']:>5}{n_perfect:>6}/{len(ALL_FIELDS):<3}"
              f"{mean_rate:>11.3f}{entry['mean_confidence']:>10.3f}"
              f"{entry['mean_seconds']:>9.1f}{entry['mean_chars']:>10,}")

    print(f"\n--- フィールド別の安定率（1.000 = 全作品で2回とも同じ値）---")
    print(f"  {'フィールド':<22}" + "".join(f"{m:>14}" for m in methods))
    for field in ALL_FIELDS:
        cells = "".join(f"{summary['methods'][m]['field_rates'][field]:>14.3f}"
                        if m in summary["methods"] else f"{'-':>14}" for m in methods)
        print(f"  {field:<22}{cells}")

    print("\n--- confidence は安定性の指標になるか ---")
    for method in methods:
        entry = summary["methods"].get(method)
        if entry is None:  # --repeats 1 の方式は再実行安定性を測れない
            print(f"  {method:<13} 再実行が1回のみのため測定対象外")
            continue
        print(f"  {method:<13} 全項目安定={entry['n_works_fully_stable']}/{entry['n_works']}作品  "
              f"conf(安定)={entry['conf_when_fully_stable']:.3f} "
              f"conf(不安定)={entry['conf_when_unstable']:.3f}")

    # --- 方式間一致率 -------------------------------------------------------
    # 各作品で「見せる区間を変えても同じ値が出るか」。本命の指標。
    print("\n--- 方式間一致率（run0 同士。1.000 = どの方式でも同じ値）---")
    pair_names = [(methods[i], methods[j])
                  for i in range(len(methods)) for j in range(i + 1, len(methods))]
    cross: dict[str, dict[str, list[float]]] = {f"{a}|{b}": defaultdict(list) for a, b in pair_names}
    consensus: dict[str, list[float]] = defaultdict(list)
    ncodes = sorted({ncode for ncode, _ in runs})
    for ncode in ncodes:
        available = {method: runs[(ncode, method)][0]["features"]
                     for method in methods
                     if (ncode, method) in runs and 0 in runs[(ncode, method)]}
        for left, right in pair_names:
            if left in available and right in available:
                for field in ALL_FIELDS:
                    cross[f"{left}|{right}"][field].append(
                        field_agreement(field, available[left], available[right]))
        if len(available) == len(methods):
            for field in ALL_FIELDS:
                scores = [field_agreement(field, available[a], available[b]) for a, b in pair_names]
                consensus[field].append(sum(scores) / len(scores))

    keys = list(cross)
    print(f"  {'フィールド':<22}" + "".join(f"{k:>26}" for k in keys) + f"{'3方式平均':>12}")
    for field in ALL_FIELDS:
        cells = "".join(
            f"{sum(cross[k][field]) / len(cross[k][field]):>26.3f}" if cross[k][field] else f"{'-':>26}"
            for k in keys)
        mean = sum(consensus[field]) / len(consensus[field]) if consensus[field] else float("nan")
        print(f"  {field:<22}{cells}{mean:>12.3f}")
    summary["cross_method"] = {
        "pairs": {k: {field: round(sum(v) / len(v), 4) for field, v in fields.items() if v}
                  for k, fields in cross.items()},
        "consensus": {field: round(sum(v) / len(v), 4) for field, v in consensus.items() if v},
    }
    stable = sorted((field for field in ALL_FIELDS if consensus.get(field)),
                    key=lambda f: -sum(consensus[f]) / len(consensus[f]))
    print("\n  方式に依存しない（＝作品の属性として拾えている）フィールド順:")
    for field in stable:
        print(f"    {field:<24}{sum(consensus[field]) / len(consensus[field]):.3f}")

    # --- 偶然一致の補正 -----------------------------------------------------
    # 一致率だけ見ると「いつも同じ値を返すだけのフィールド」が優秀に見えてしまう。
    # 実測: ending_type は30回中27回が open。方式間一致 0.867 は「読めている」の
    # 証拠ではなく、単に定数を返しているだけ。κ = (実測一致 - 偶然一致)/(1 - 偶然一致)
    # で補正する。κ<=0 は情報を持っていないということ。
    print("\n--- 偶然一致の補正（run0 全30観測の値分布から）---")
    print(f"  {'フィールド':<22}{'異なり値':>9}{'最頻値の割合':>13}{'偶然一致':>10}{'方式間一致':>11}{'κ':>9}  最頻値")
    summary["chance_corrected"] = {}
    for field in SCALAR_FIELDS + CONTENT_FIELDS:
        observed_values = []
        for (ncode, method), by_run in runs.items():
            if 0 not in by_run:
                continue
            value = by_run[0]["features"].get(field)
            if field in CONTENT_FIELDS:
                value = f"{(value or {}).get('present')}/{(value or {}).get('explicitness')}"
            observed_values.append(str(value).strip().lower())
        if not observed_values:
            continue
        counts: dict[str, int] = defaultdict(int)
        for value in observed_values:
            counts[value] += 1
        total = len(observed_values)
        shares = [count / total for count in counts.values()]
        expected = sum(share * share for share in shares)
        agreement = (sum(consensus[field]) / len(consensus[field])) if consensus.get(field) else 0.0
        kappa = (agreement - expected) / (1 - expected) if expected < 1 else 0.0
        top_value, top_count = max(counts.items(), key=lambda item: item[1])
        summary["chance_corrected"][field] = {
            "n_distinct": len(counts), "majority_share": round(top_count / total, 3),
            "expected_agreement": round(expected, 3), "observed_agreement": round(agreement, 3),
            "kappa": round(kappa, 3), "majority_value": top_value,
        }
        print(f"  {field:<22}{len(counts):>9}{top_count / total:>13.3f}{expected:>10.3f}"
              f"{agreement:>11.3f}{kappa:>9.3f}  {top_value[:24]}")
    print("  ※ κ が低いフィールドは、方式を変えても一致するのではなく"
          "「いつも同じ値」なので特徴量にならない")

    if args.manual:
        manual = {row["ncode"]: row["fields"] for row in
                  (json.loads(line) for line in
                   Path(args.manual).read_text(encoding="utf-8").splitlines() if line.strip())}
        print("\n--- 人手確認との一致率（run0 を使用）---")
        summary["manual"] = {}
        for method in methods:
            hits: dict[str, list[float]] = defaultdict(list)
            for (ncode, m), by_run in runs.items():
                if m != method or ncode not in manual or 0 not in by_run:
                    continue
                truth = manual[ncode]
                for field, value in truth.items():
                    if value in ("", None, [], {}):  # 未記入は集計から外す
                        continue
                    hits[field].append(field_agreement(field, by_run[0]["features"], truth))
            if not hits:
                continue
            rates = {field: sum(values) / len(values) for field, values in hits.items()}
            summary["manual"][method] = rates
            overall = sum(rates.values()) / len(rates)
            detail = " ".join(f"{field}={rate:.2f}" for field, rate in sorted(rates.items()))
            print(f"  {method:<13} 平均={overall:.3f}  {detail}")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
