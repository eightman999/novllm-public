#!/usr/bin/env python3
"""長編を章単位でLLM構造化抽出し、作品単位に集約する。

抽出結果は章ごとにJSONLへ即時追記する。--resume を付ければ成功済み章は
再送しないため、SSH切断・モデル停止・締切で中断しても安全に再開できる。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from extract_work_features import (
    PROMPT_TEMPLATE,
    call_ollama,
    load_schema,
    parse_deadline,
    past_deadline,
    validate,
)


def load_episodes(pattern: str, limit_works: int, max_chars: int) -> list[dict[str, Any]]:
    """チャンクを (作品, 章) ごとに元の順序で連結する。"""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                meta = row.get("meta") or {}
                ncode, episode_no = meta.get("ncode"), meta.get("episode_no")
                if ncode is not None and episode_no is not None:
                    grouped[(str(ncode), int(episode_no))].append(row)

    selected_codes = sorted({ncode for ncode, _ in grouped})[:limit_works] if limit_works else None
    episodes: list[dict[str, Any]] = []
    for (ncode, episode_no), rows in sorted(grouped.items()):
        if selected_codes is not None and ncode not in selected_codes:
            continue
        rows.sort(key=lambda row: int((row.get("meta") or {}).get("chunk_index", 0)))
        meta = rows[0]["meta"]
        # チャンクの重複領域も残す。章を切り落とすより、根拠を欠かさないことを優先する。
        text = "\n\n--- chunk boundary ---\n\n".join(str(row.get("text") or "") for row in rows)
        episodes.append({
            "ncode": ncode,
            "episode_no": episode_no,
            "episode_title": meta.get("episode_title"),
            "n_chars": len(text),
            "text": text,
            "too_long": len(text) > max_chars,
        })
    return episodes


def existing_keys(out_path: Path, retry_failed: bool) -> set[tuple[str, int]]:
    if not out_path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with open(out_path, encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
                if row.get("ok") or not retry_failed:
                    done.add((str(row["ncode"]), int(row["episode_no"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def run_extract(args: argparse.Namespace) -> int:
    schema = load_schema(Path(args.schema))
    deadline = parse_deadline(args.deadline)
    out_path = Path(args.out)
    done = existing_keys(out_path, args.retry_failed) if args.resume else set()
    episodes = load_episodes(args.chunks, args.limit_works, args.max_episode_chars)
    todo = [episode for episode in episodes if (episode["ncode"], episode["episode_no"]) not in done]
    print(f"章={len(episodes)} / 済={len(done)} / 今回={len(todo)}", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_fail = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for index, episode in enumerate(todo, 1):
            if past_deadline(deadline):
                print(f"締切 {args.deadline} に達したので停止します", file=sys.stderr)
                break
            record: dict[str, Any] = {
                "ncode": episode["ncode"], "episode_no": episode["episode_no"],
                "episode_title": episode["episode_title"], "n_chars": episode["n_chars"],
                "model": args.model, "num_ctx": args.num_ctx,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            started = time.time()
            if episode["too_long"]:
                record.update({"ok": False, "skipped": "episode_too_long"})
                n_skip += 1
                status = "SKIP(long)"
            else:
                try:
                    features, stats = call_ollama(
                        args.host, args.model, PROMPT_TEMPLATE.format(text=episode["text"]), schema,
                        args.num_ctx, args.think, args.timeout,
                    )
                    problems = validate(features, schema)
                    record.update({"ok": True, "features": features, "problems": problems, "stats": stats})
                    n_ok += 1
                    status = "OK" if not problems else f"OK(問題{len(problems)})"
                except Exception as exc:  # API失敗も行単位で記録し、--retry-failedで再試行できる。
                    record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                    n_fail += 1
                    status = "FAIL"
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{index}/{len(todo)}] {episode['ncode']} e{episode['episode_no']} "
                  f"{episode['n_chars']:,}字 {status} {time.time() - started:.1f}s", file=sys.stderr)
    print(f"完了: ok={n_ok} skip={n_skip} fail={n_fail} -> {out_path}", file=sys.stderr)
    return 0 if n_fail == 0 else 1


def weighted_mode(values: list[tuple[Any, float]], fallback: Any) -> Any:
    if not values:
        return fallback
    scores: dict[Any, float] = defaultdict(float)
    for value, weight in values:
        scores[value] += weight
    return max(scores, key=lambda value: (scores[value], str(value)))


def aggregate_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """章ごとの特徴を信頼度加重の最頻値・出現頻度で集約する。"""
    features = [row["features"] for row in rows]
    weights = [float(feature.get("confidence") or 0.0) or 1.0 for feature in features]
    categorical = ["protagonist_gender", "viewpoint_person", "setting_world", "setting_era",
                   "world_transfer", "ending_type", "protagonist_role"]
    result = {field: weighted_mode([(feature.get(field), weight) for feature, weight in zip(features, weights)
                                    if feature.get(field) is not None], "unclear") for field in categorical}
    result["protagonist_count"] = round(sum(float(f.get("protagonist_count") or 0) * w
                                          for f, w in zip(features, weights)) / sum(weights))
    for content_field in ("sexual_content", "violence_content"):
        values = [feature.get(content_field) or {} for feature in features]
        result[content_field] = {
            "present": any(value.get("present") for value in values),
            "explicitness": max((int(value.get("explicitness") or 0) for value in values), default=0),
        }
    shape_scores: dict[str, float] = defaultdict(float)
    relationship_scores: dict[str, float] = defaultdict(float)
    salience = {"central": 1.0, "major": 0.6, "minor": 0.3}
    for feature, weight in zip(features, weights):
        for rank, shape in enumerate(feature.get("story_shape") or []):
            shape_scores[shape] += weight * (1.0 - rank * 0.25)
        for relation in feature.get("main_relationships") or []:
            if relation.get("kind"):
                relationship_scores[relation["kind"]] += weight * salience.get(relation.get("salience"), 0.3)
    result["story_shape"] = [shape for shape, _ in sorted(shape_scores.items(), key=lambda item: -item[1])[:3]] or ["other"]
    result["main_relationships"] = [
        {"kind": kind, "salience": "central" if score >= sum(weights) * 0.75 else "major"}
        for kind, score in sorted(relationship_scores.items(), key=lambda item: -item[1])[:4]
    ]
    result["confidence"] = round(sum(float(f.get("confidence") or 0.0) for f in features) / len(features), 4)
    return result


def run_aggregate(args: argparse.Namespace) -> int:
    by_work: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(args.episodes, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if row.get("ok") and not row.get("problems"):
                    by_work[str(row["ncode"])].append(row)
    out_path = Path(args.out)
    done = set()
    if args.resume and out_path.exists():
        done = {json.loads(line)["ncode"] for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    with open(out_path, "a", encoding="utf-8") as out:
        for ncode, rows in sorted(by_work.items()):
            if ncode in done:
                continue
            out.write(json.dumps({"ncode": ncode, "ok": True, "chapter_count": len(rows),
                                  "features": aggregate_features(rows),
                                  "source": "chapter_weighted_aggregation"}, ensure_ascii=False) + "\n")
            out.flush()
    print(f"集約完了: {len(by_work)}作品 -> {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--chunks", required=True, help="train-*.jsonl のglob")
    extract.add_argument("--schema", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--host", default="127.0.0.1:11434")
    extract.add_argument("--model", default="general-qwen3-30b")
    extract.add_argument("--num-ctx", type=int, default=65536)
    extract.add_argument("--max-episode-chars", type=int, default=30000)
    extract.add_argument("--limit-works", type=int, default=20, help="0で全作品")
    extract.add_argument("--think", action="store_true")
    extract.add_argument("--timeout", type=int, default=1800)
    extract.add_argument("--deadline")
    extract.add_argument("--resume", action="store_true")
    extract.add_argument("--retry-failed", action="store_true")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--episodes", required=True)
    aggregate.add_argument("--out", required=True)
    aggregate.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return run_extract(args) if args.command == "extract" else run_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
