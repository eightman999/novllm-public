#!/usr/bin/env python3
"""Phase A の抽出結果を build_control_tags() が食える形にまとめる（PLAN_20260727.md B-1）。

供給するのは**L1のκとA-3のrecallで裏付けが取れた項目だけ**。
不採用にした項目（setting_world κ=0.387 / ending_type 0.259 / story_shape /
main_relationships / protagonist_count）は、値を持っていても意図的に供給しない。

- `narrative.viewpoint`  … ルールベース（A-0）。LLMは使わない。
  A-0の実測でLLMは10作品中2件を誤っており、ルールの方が正確だった。
- `protagonist.gender`   … LLM（A-2）。ただし作者申告フラグがある作品では
  build_control_tags() 側でフラグが優先されるので、実際に効くのは未申告作品のみ。
- `setting.world`        … 異世界枠のみ。**申告が無い作品にだけ**付ける。
  B-5の決定により転生/転移には分けず `isekai` に統合する。
  A-3実測で二分の的中は52%しかないが、統合すれば0.798まで上がるため。

genre / era / tone は供給しない（裏付けが無い）。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# LLMのenum -> control_tags の語彙。unclear は unknown に寄せる。
GENDER_MAP = {
    "male": "male", "female": "female", "multiple": "multiple",
    "non_human_or_none": "non_human", "unclear": "unknown",
}
ISEKAI_VALUES = {"reincarnation", "transfer"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="phase_a2_features.jsonl")
    parser.add_argument("--viewpoint", required=True, help="viewpoint_rule.jsonl")
    parser.add_argument("--manifest", required=True, help="works.jsonl（申告フラグの参照用）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    features = {row["ncode"]: row["features"] for row in load_jsonl(Path(args.features)) if row.get("ok")}
    viewpoints = {row["ncode"]: row["viewpoint"] for row in load_jsonl(Path(args.viewpoint))}
    works = load_jsonl(Path(args.manifest))

    stats: dict[str, Counter] = {"viewpoint": Counter(), "gender": Counter(), "setting": Counter()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for work in works:
            ncode = work["ncode"]
            flags = work.get("control_flags") or {}
            feature = features.get(ncode) or {}

            viewpoint = viewpoints.get(ncode, "unknown")
            gender = GENDER_MAP.get(feature.get("protagonist_gender"), "unknown")

            # 異世界枠は「申告が無い作品」にだけ付ける。申告済みは正本が優先されるので触らない。
            setting: dict[str, str] = {}
            declared_isekai = bool(flags.get("異世界転生") or flags.get("異世界転移"))
            if not declared_isekai and feature.get("world_transfer") in ISEKAI_VALUES:
                setting["world"] = "isekai"  # B-5(b): 転生/転移に分けない

            stats["viewpoint"][viewpoint] += 1
            stats["gender"][gender] += 1
            stats["setting"]["isekai付与" if setting else
                             ("申告あり(触らない)" if declared_isekai else "なし")] += 1

            out.write(json.dumps({
                "ncode": ncode,
                "genre": [],
                "setting": setting,
                "protagonist": {"gender": gender},
                "narrative": {"viewpoint": viewpoint},
                "tone": [],
                "source": {"viewpoint": "rule_narration_first_person",
                           "protagonist_gender": "llm_ends3_12k",
                           "setting": "llm_ends3_12k_merged_isekai"},
            }, ensure_ascii=False) + "\n")

    print(f"書き出し: {out_path}（{len(works)}作品）")
    for key, counter in stats.items():
        total = sum(counter.values())
        print(f"  {key}: " + " / ".join(f"{k}={v}({v / total:.1%})" for k, v in counter.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
