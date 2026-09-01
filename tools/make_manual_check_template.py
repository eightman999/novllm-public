#!/usr/bin/env python3
"""Phase L1 の人手確認シートの雛形を作る。

原則（PLAN_20260726.md）に従い、**このシートには作者申告タグ・R指定・完結状態を入れない**。
それらは取得済みのサイトメタデータであり、別軸で扱う。ここに書くと記入者が
それに引きずられて、LLM評価の正解として汚染される。

同じ理由でLLMの出力値も入れない。空欄のまま人が本文を読んで埋める。
埋めたものを compare_sampling_methods.py --manual に渡す。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 人手で確認する最小限の項目。値の書式は work_features.schema.json の enum に合わせる。
BLANK_FIELDS = {
    "viewpoint_person": "",       # first / third_limited / third_omniscient / second / mixed / unclear
    "protagonist_gender": "",     # male / female / multiple / non_human_or_none / unclear
    "main_relationships": [],     # [{"kind": "...", "salience": "central|major|minor"}]
    "story_shape": [],            # 重要な順に最大3件
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncodes", required=True, help="1行1件のncodeリスト")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    wanted = [line.strip() for line in
              Path(args.ncodes).read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = {}
    with open(args.manifest, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if row["ncode"] in wanted:
                    meta[row["ncode"]] = row

    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"既に存在します（記入済みを壊さないため上書きしません）: {out_path}")
    with open(out_path, "w", encoding="utf-8") as out:
        for ncode in wanted:
            row = meta.get(ncode, {})
            out.write(json.dumps({
                "ncode": ncode,
                "title": row.get("title"),
                "_note": "fields を本文を読んで埋める。空欄の項目は集計から自動的に除外される",
                "fields": dict(BLANK_FIELDS),
            }, ensure_ascii=False) + "\n")
    print(f"雛形を書き出しました（{len(wanted)}件）: {out_path}")
    print("記入後: tools/compare_sampling_methods.py --manual でこのファイルを渡す")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
