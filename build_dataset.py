#!/usr/bin/env python3
# build_dataset.py
"""dataset_v2 統合オーケストレータ。

正本DB(読み取り専用) -> pipeline/ 配下の各モジュール -> dataset_v2.tmp/ へ
shard/manifest/state/VERSION.jsonを生成する。全検証成功後にのみ dataset_v2/ へ
atomic renameする(--finalizeオプション、本スクリプト単体では常に .tmp 側にのみ書く)。

既存の data/ ・ 正本DB・checkpointには一切書き込まない。
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import random
import shutil
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import db_audit
from pipeline.chunker import chunk_episode_text, find_duplicate_indices
from pipeline.normalize_meta import (
    determine_r18,
    extract_control_flags,
    normalize_keywords,
    parse_source_keywords,
)
from pipeline.schemas import (
    SCHEMA_VERSION,
    ChunkMeta,
    ChunkRecord,
    ClassificationMeta,
    default_chunk_classification,
)
from pipeline.style_stats import compute_style_metrics
from pipeline.text_normalize import normalize_episode_body

# --- 22 control flag -> 学習用control_tags(英語スラッグ)へのマッピング ------------
# 指示書は正確なマッピング表を規定していないため、ここで定義し dataset_stats.json /
# VERSION.json 経由で監査可能にする。R15・残酷描写等の内容評価軸はgenreに含めず
# meta.is_r18 / meta.content 側で表現する。
_GENRE_FLAG_TO_TAG = {
    "オリジナル戦記": "original_war_chronicle",
    "架空戦記": "fictional_war_chronicle",
    "IF戦記": "if_war_chronicle",
    "ミリタリー": "military",
    "戦争": "war",
    "ハーレム": "harem",
    "恋愛": "romance",
    "学園": "school",
    "現代": "contemporary",
    "歴史": "historical",
    "SF": "sf",
    "ホラー": "horror",
    "ミステリー": "mystery",
    "ボーイズラブ": "bl",
    "ガールズラブ": "gl",
    "TS": "ts",
    "群像劇": "ensemble_cast",
}
_SETTING_FLAG_TO_TAG = {
    "異世界転生": "isekai_reincarnation",
    "異世界転移": "isekai_transfer",
    "歴史": "historical",
    "現代": "contemporary",
    "学園": "school",
}


def build_control_tags(flags: dict, work_classification: dict | None) -> dict:
    genre = [tag for flag, tag in _GENRE_FLAG_TO_TAG.items() if flags.get(flag)]
    if work_classification:
        for g in work_classification.get("genre") or []:
            if g not in genre:
                genre.append(g)

    setting = [tag for flag, tag in _SETTING_FLAG_TO_TAG.items() if flags.get(flag)]
    if work_classification:
        wc_setting = work_classification.get("setting") or {}
        for key in ("world", "era"):
            v = wc_setting.get(key)
            if v and v != "unknown" and v not in setting:
                setting.append(v)

    if flags.get("男主人公") and not flags.get("女主人公"):
        protagonist_gender = "male"
    elif flags.get("女主人公") and not flags.get("男主人公"):
        protagonist_gender = "female"
    elif work_classification:
        protagonist_gender = (work_classification.get("protagonist") or {}).get("gender", "unknown")
    else:
        protagonist_gender = "unknown"

    viewpoint = "unknown"
    tone: list = []
    if work_classification:
        viewpoint = (work_classification.get("narrative") or {}).get("viewpoint", "unknown")
        tone = list(work_classification.get("tone") or [])

    return {
        "genre": genre,
        "setting": setting,
        "protagonist_gender": protagonist_gender,
        "viewpoint": viewpoint,
        "tone": tone,
    }


class ShardWriter:
    """NDJSON shardを一時ファイル->os.replaceでatomicに確定させる。

    中断・再開対応: out_dir に既存の {prefix}-NNNNN.jsonl があれば、その続き
    (最後のshardの末尾)から追記する。既存の完成済みshardは書き換えない
    (最後の未満杯shardのみ、既存内容をtmpへコピーしてから追記して確定させる)。
    """

    def __init__(self, out_dir: str, prefix: str, max_records: int = 5000):
        self.out_dir = out_dir
        self.prefix = prefix
        self.max_records = max_records
        self.total_written = 0
        self._fh = None
        self._tmp_path = None
        self._final_path = None
        os.makedirs(out_dir, exist_ok=True)

        existing = sorted(glob.glob(os.path.join(out_dir, f"{prefix}-[0-9][0-9][0-9][0-9][0-9].jsonl")))
        self._resume_from_path = None
        if existing:
            last = existing[-1]
            self.shard_index = int(os.path.splitext(os.path.basename(last))[0].rsplit("-", 1)[-1])
            with open(last, "r", encoding="utf-8") as f:
                self.count_in_shard = sum(1 for _ in f)
            if self.count_in_shard < max_records:
                self._resume_from_path = last
            else:
                self.shard_index += 1
                self.count_in_shard = 0
        else:
            self.shard_index = 0
            self.count_in_shard = 0

    def _open_shard(self) -> None:
        self._final_path = os.path.join(self.out_dir, f"{self.prefix}-{self.shard_index:05d}.jsonl")
        self._tmp_path = self._final_path + ".tmp"
        if self._resume_from_path and os.path.exists(self._resume_from_path):
            shutil.copyfile(self._resume_from_path, self._tmp_path)
            self._fh = open(self._tmp_path, "a", encoding="utf-8")
        else:
            self._fh = open(self._tmp_path, "w", encoding="utf-8")
        self._resume_from_path = None

    def write(self, record: dict) -> None:
        if self._fh is None:
            self._open_shard()
        elif self.count_in_shard >= self.max_records:
            self._close_shard()
            self.shard_index += 1
            self.count_in_shard = 0
            self._open_shard()
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count_in_shard += 1
        self.total_written += 1

    def _close_shard(self) -> None:
        if self._fh is not None:
            self._fh.close()
            os.replace(self._tmp_path, self._final_path)
            self._fh = None

    def close(self) -> None:
        self._close_shard()


def recompute_full_dataset_stats(out_dir: str, target_works_count: int) -> dict:
    """manifests/works.jsonl と chunks/*.jsonl を全走査して累積統計を再計算する。

    中断・再開で複数回に分けて実行された場合、直近の呼び出し分だけの差分統計を
    dataset_stats.json として残すと不正確になるため、常にこの関数で
    「今この時点でout_dirに存在する全データ」から作り直す。
    """
    manifests_dir = os.path.join(out_dir, "manifests")
    chunks_dir = os.path.join(out_dir, "chunks")
    works_path = os.path.join(manifests_dir, "works.jsonl")

    stats = {
        "target_works": target_works_count,
        "adopted_works": 0,
        "excluded_works_zero_episode": 0,
        "total_episodes": 0,
        "total_chunks": 0,
        "total_chars": 0,
        "total_tokens": 0,
        "train_chunks": 0,
        "val_chunks": 0,
        "genre_distribution": defaultdict(int),
        "protagonist_gender_distribution": defaultdict(int),
        "viewpoint_distribution": defaultdict(int),
        "is_r18_distribution": defaultdict(int),
        "content_intensity_distribution": defaultdict(int),
    }

    if os.path.exists(works_path):
        with open(works_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                stats["total_episodes"] += rec.get("actual_episode_count", 0)
                if rec.get("chunk_count", 0) > 0:
                    stats["adopted_works"] += 1
                elif rec.get("actual_episode_count", 0) == 0:
                    stats["excluded_works_zero_episode"] += 1

    for shard_path in sorted(glob.glob(os.path.join(chunks_dir, "*.jsonl"))):
        is_val = os.path.basename(shard_path).startswith("val-")
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                stats["total_chunks"] += 1
                stats["total_chars"] += len(rec.get("text", ""))
                meta = rec.get("meta", {})
                stats["total_tokens"] += (meta.get("style_metrics", {}) or {}).get("token_count") or 0
                stats["train_chunks" if not is_val else "val_chunks"] += 1
                for g in rec.get("control_tags", {}).get("genre", []):
                    stats["genre_distribution"][g] += 1
                stats["protagonist_gender_distribution"][
                    rec.get("control_tags", {}).get("protagonist_gender", "unknown")
                ] += 1
                stats["viewpoint_distribution"][
                    rec.get("control_tags", {}).get("viewpoint", "unknown")
                ] += 1
                stats["is_r18_distribution"]["r18" if meta.get("is_r18") else "general"] += 1
                content = meta.get("content", {}) or {}
                max_intensity = max([v for v in content.values() if isinstance(v, int)], default=0)
                stats["content_intensity_distribution"][str(max_intensity)] += 1

    for key in (
        "genre_distribution", "protagonist_gender_distribution",
        "viewpoint_distribution", "is_r18_distribution", "content_intensity_distribution",
    ):
        stats[key] = dict(stats[key])
    return stats


def fetch_target_works(conn: sqlite3.Connection, ncodes: list[str] | None, limit: int | None):
    cur = conn.cursor()
    query = (
        "SELECT ncode, title, author, main_tag, sub_tag, rating, sub_site, site_type, "
        "total_ep, general_all_no FROM novels_descs"
    )
    params: tuple = ()
    if ncodes:
        placeholders = ",".join("?" for _ in ncodes)
        query += f" WHERE ncode IN ({placeholders})"
        params = tuple(ncodes)
    query += " ORDER BY ncode"
    if limit is not None and not ncodes:
        query += f" LIMIT {int(limit)}"
    cur.execute(query, params)
    return cur.fetchall()


def fetch_url_is_r18(conn: sqlite3.Connection, ncode: str):
    cur = conn.cursor()
    cur.execute("SELECT is_r18 FROM url_entity WHERE ncode=?", (ncode,))
    row = cur.fetchone()
    return row[0] if row else None


def fetch_episodes(conn: sqlite3.Connection, ncode: str):
    cur = conn.cursor()
    cur.execute(
        "SELECT episode_no, body, e_title FROM episodes WHERE ncode=? "
        "ORDER BY CAST(episode_no AS INTEGER)",
        (ncode,),
    )
    return cur.fetchall()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="novllm dataset_v2 統合オーケストレータ")
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", default="./dataset_v2.tmp")
    ap.add_argument("--limit", type=int, default=None, help="対象作品数を絞る(デバッグ用)")
    ap.add_argument("--ncodes", default=None, help="特定作品だけ処理(カンマ区切りncode)")
    ap.add_argument("--enable-lmstudio", action="store_true", help="既定オフ。LM Studio分類を有効化")
    ap.add_argument("--enable-api-refresh", action="store_true", help="既定オフ。なろうAPIリフレッシュを有効化")
    ap.add_argument("--val-split-by-author", action="store_true", help="作者単位でtrain/val分割する")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-tokens", type=int, default=1536)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--overlap-tokens", type=int, default=128)
    ap.add_argument("--dedup-threshold", type=float, default=0.9)
    ap.add_argument("--shard-max-records", type=int, default=5000)
    ap.add_argument("--base-model", default=os.environ.get("NOVLLM_BASE_MODEL", "Qwen/Qwen3-8B-Base"))
    ap.add_argument("--skip-audit", action="store_true", help="既存manifests/rejected.jsonlを使い回す")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    ncode_filter = [c.strip() for c in args.ncodes.split(",") if c.strip()] if args.ncodes else None

    chunks_dir = os.path.join(args.out_dir, "chunks")
    manifests_dir = os.path.join(args.out_dir, "manifests")
    state_dir = os.path.join(args.out_dir, "state")
    for d in (chunks_dir, manifests_dir, state_dir):
        os.makedirs(d, exist_ok=True)

    if not args.skip_audit:
        print("[build_dataset] Phase1 DB監査を実行中...")
        audit_report = db_audit.run_audit(args.db, manifests_dir)
        print(
            f"[build_dataset] audit: novels={audit_report.novels_total} "
            f"episodes={audit_report.episodes_total} "
            f"empty_body={audit_report.empty_body_episode_count} "
            f"encoding_anomaly={audit_report.encoding_anomaly_episode_count}"
        )

    conn = db_audit.read_only_connect(args.db)
    works = fetch_target_works(conn, ncode_filter, args.limit)
    print(f"[build_dataset] 対象作品数: {len(works)}")
    if not works:
        print("[build_dataset] 対象作品が0件のため終了します。")
        conn.close()
        return

    # --- Pass 1: 決定的メタデータ正規化(episode本文は読まない、軽量) -----------------
    work_meta: dict[str, dict] = {}
    for ncode, title, author, main_tag, sub_tag, rating, sub_site, site_type, total_ep, general_all_no in works:
        url_is_r18 = fetch_url_is_r18(conn, ncode)
        src_kw = parse_source_keywords(main_tag, sub_tag)
        norm_kw = normalize_keywords(src_kw)
        flags = extract_control_flags(norm_kw)
        is_r18, r18_source = determine_r18(sub_site, rating, url_is_r18)
        work_meta[ncode] = {
            "title": title, "author": author, "site_type": site_type,
            "source_keywords": src_kw, "normalized_keywords": norm_kw, "flags": flags,
            "is_r18": is_r18, "r18_source": r18_source,
            "total_ep_db": total_ep, "general_all_no_db": general_all_no,
            "api_genre": None, "api_biggenre": None,
        }

    # --- Pass 1.5: なろうAPIリフレッシュ(既定オフ、site_type=1のみ) -------------------
    if args.enable_api_refresh:
        from pipeline.api_refresh import run_refresh

        api_out_path = os.path.join(manifests_dir, "api_refresh.jsonl")
        api_state_path = os.path.join(state_dir, "api_refresh_state.json")
        targets = [
            (ncode, work_meta[ncode]["is_r18"])
            for ncode in work_meta
            if work_meta[ncode]["site_type"] == 1
        ]
        print(f"[build_dataset] Phase3 なろうAPIリフレッシュ対象: {len(targets)}件")
        summary = run_refresh(targets, api_out_path, api_state_path)
        print(f"[build_dataset] api_refresh: {summary}")
        if os.path.exists(api_out_path):
            with open(api_out_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    ncode = rec.get("ncode")
                    payload = rec.get("payload")
                    if ncode in work_meta and payload:
                        work_meta[ncode]["api_genre"] = payload.get("genre")
                        work_meta[ncode]["api_biggenre"] = payload.get("biggenre")

    # --- tokenizer (chunker用、学習対象ベースモデルと同一) -----------------------------
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print(f"[build_dataset] tokenizer読み込み中: {args.base_model}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)

    # --- train/validation分割(作品単位、既定。--val-split-by-authorで作者単位) --------
    rng = random.Random(args.seed)
    if args.val_split_by_author:
        authors = sorted({m["author"] for m in work_meta.values()})
        author_is_val = {a: (rng.random() < args.val_ratio) for a in authors}
        is_val_for = lambda ncode: author_is_val[work_meta[ncode]["author"]]  # noqa: E731
    else:
        ncodes_sorted = sorted(work_meta.keys())
        ncode_is_val = {n: (rng.random() < args.val_ratio) for n in ncodes_sorted}
        is_val_for = lambda ncode: ncode_is_val[ncode]  # noqa: E731

    train_writer = ShardWriter(chunks_dir, "train", max_records=args.shard_max_records)
    val_writer = ShardWriter(chunks_dir, "val", max_records=args.shard_max_records)

    works_manifest_path = os.path.join(manifests_dir, "works.jsonl")
    classification_manifest_path = os.path.join(manifests_dir, "classification.jsonl")

    # --- 中断・再開: 既にworks.jsonlに記録済みのncodeは再処理しない --------------------
    already_done_ncodes: set[str] = set()
    if os.path.exists(works_manifest_path):
        with open(works_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    already_done_ncodes.add(json.loads(line)["ncode"])
                except (json.JSONDecodeError, KeyError):
                    continue
    skipped_already_done = [n for n in work_meta if n in already_done_ncodes]
    if skipped_already_done:
        print(f"[build_dataset] 既に処理済みのため再処理をスキップ: {len(skipped_already_done)}作品")
        for n in skipped_already_done:
            del work_meta[n]

    stats = {
        "target_works": len(works),
        "skipped_already_done_works": len(skipped_already_done),
        "adopted_works": 0,
        "excluded_works_zero_episode": 0,
        "total_episodes": 0,
        "skipped_empty_body_episodes": 0,
        "skipped_duplicate_chunks": 0,
        "total_chunks": 0,
        "total_chars": 0,
        "total_tokens": 0,
        "train_chunks": 0,
        "val_chunks": 0,
        "genre_distribution": defaultdict(int),
        "protagonist_gender_distribution": defaultdict(int),
        "viewpoint_distribution": defaultdict(int),
        "is_r18_distribution": defaultdict(int),
        "content_intensity_distribution": defaultdict(int),
    }

    lmstudio_enabled = args.enable_lmstudio
    classify_work = classify_chunk = should_send_work_to_lmstudio = should_send_chunk_to_lmstudio = None
    if lmstudio_enabled:
        from pipeline.lmstudio_classify import (
            classify_chunk,
            classify_work,
            should_send_chunk_to_lmstudio,
            should_send_work_to_lmstudio,
        )

    with open(works_manifest_path, "a", encoding="utf-8") as works_f, \
         open(classification_manifest_path, "a", encoding="utf-8") as classification_f:

        for ncode, meta in work_meta.items():
            episodes = fetch_episodes(conn, ncode)
            if not episodes:
                stats["excluded_works_zero_episode"] += 1
                continue

            work_classification = None
            if lmstudio_enabled and should_send_work_to_lmstudio(meta["normalized_keywords"]):
                idxs = sorted({0, len(episodes) // 2, len(episodes) - 1})
                rep_texts = []
                for i in idxs:
                    _ep_no, ep_body, _ep_title = episodes[i]
                    rep_norm = normalize_episode_body(ep_body)
                    rep_texts.append(rep_norm.text[:1500])
                work_classification = classify_work(rep_texts)
                classification_f.write(json.dumps(
                    {"ncode": ncode, "level": "work", "result": work_classification},
                    ensure_ascii=False,
                ) + "\n")

            control_tags_base = build_control_tags(meta["flags"], work_classification)
            is_val_work = is_val_for(ncode)
            work_chunk_count = 0

            for episode_no_raw, body, e_title in episodes:
                stats["total_episodes"] += 1
                if not body or not body.strip():
                    stats["skipped_empty_body_episodes"] += 1
                    continue

                norm = normalize_episode_body(body)
                if not norm.text.strip():
                    stats["skipped_empty_body_episodes"] += 1
                    continue

                try:
                    episode_no = int(episode_no_raw)
                except (TypeError, ValueError):
                    episode_no = -1

                ep_chunks = chunk_episode_text(
                    ncode, episode_no, norm.text, tokenizer,
                    target_tokens=args.target_tokens, max_tokens=args.max_tokens,
                    overlap_tokens=args.overlap_tokens,
                )
                if not ep_chunks:
                    continue

                # episode内の完全/近似重複チャンクを検出しスキップする(全チャンク総当たりは
                # 長編でO(n^2)になり非現実的なため、境界が閉じるepisode単位に限定する)。
                dup_map = find_duplicate_indices(
                    [c.text for c in ep_chunks], threshold=args.dedup_threshold
                )

                for c in ep_chunks:
                    if c.chunk_index in dup_map:
                        stats["skipped_duplicate_chunks"] += 1
                        continue

                    chunk_classification = None
                    if lmstudio_enabled and should_send_chunk_to_lmstudio(0.0):
                        chunk_classification = classify_chunk(c.text)
                        classification_f.write(json.dumps(
                            {"ncode": ncode, "level": "chunk", "chunk_id": c.chunk_id,
                             "result": chunk_classification},
                            ensure_ascii=False,
                        ) + "\n")

                    content = (chunk_classification or default_chunk_classification())["content"]
                    style_metrics = compute_style_metrics(c.text, token_count=c.token_count)

                    method = "rules+lmstudio" if (work_classification or chunk_classification) else "rules"
                    confidence = 0.0
                    if chunk_classification:
                        confidence = chunk_classification.get("confidence", 0.0)
                    elif work_classification:
                        confidence = work_classification.get("confidence", 0.0)

                    record = ChunkRecord(
                        id=c.chunk_id,
                        text=c.text,
                        control_tags=control_tags_base,
                        meta=ChunkMeta(
                            ncode=ncode, title=meta["title"], author=meta["author"],
                            site_type=("syosetu" if meta["site_type"] == 1 else "kakuyomu"),
                            is_r18=meta["is_r18"], episode_no=episode_no, episode_title=e_title,
                            chunk_index=c.chunk_index, chunk_count_in_episode=len(ep_chunks),
                            source_keywords=meta["source_keywords"],
                            api_genre=meta["api_genre"], api_biggenre=meta["api_biggenre"],
                            content=content, style_metrics=dataclasses.asdict(style_metrics),
                            classification=ClassificationMeta(
                                method=method,
                                model=(os.environ.get("NOVLLM_LMSTUDIO_MODEL", "qwen3-14b-mlx-4bit")
                                       if lmstudio_enabled else None),
                                schema_version=SCHEMA_VERSION, confidence=confidence,
                            ).to_dict(),
                            source_sha256=norm.normalized_sha256,
                        ),
                    )
                    rec_dict = record.to_dict()
                    (val_writer if is_val_work else train_writer).write(rec_dict)

                    stats["total_chunks"] += 1
                    work_chunk_count += 1
                    stats["total_chars"] += len(c.text)
                    stats["total_tokens"] += c.token_count
                    stats["train_chunks" if not is_val_work else "val_chunks"] += 1
                    for g in control_tags_base["genre"]:
                        stats["genre_distribution"][g] += 1
                    stats["protagonist_gender_distribution"][control_tags_base["protagonist_gender"]] += 1
                    stats["viewpoint_distribution"][control_tags_base["viewpoint"]] += 1
                    stats["is_r18_distribution"]["r18" if meta["is_r18"] else "general"] += 1
                    max_intensity = max(
                        [v for k, v in content.items() if isinstance(v, int)], default=0
                    )
                    stats["content_intensity_distribution"][str(max_intensity)] += 1

            if work_chunk_count > 0:
                stats["adopted_works"] += 1

            works_f.write(json.dumps({
                "ncode": ncode, "title": meta["title"], "author": meta["author"],
                "site_type": ("syosetu" if meta["site_type"] == 1 else "kakuyomu"),
                "is_r18": meta["is_r18"], "r18_source": meta["r18_source"],
                "source_keywords": meta["source_keywords"],
                "normalized_keywords": meta["normalized_keywords"],
                "control_flags": meta["flags"],
                "total_ep_db": meta["total_ep_db"], "general_all_no_db": meta["general_all_no_db"],
                "actual_episode_count": len(episodes), "chunk_count": work_chunk_count,
                "split": "val" if is_val_work else "train",
            }, ensure_ascii=False) + "\n")

    train_writer.close()
    val_writer.close()
    conn.close()

    for key in (
        "genre_distribution", "protagonist_gender_distribution",
        "viewpoint_distribution", "is_r18_distribution", "content_intensity_distribution",
    ):
        stats[key] = dict(stats[key])

    # dataset_stats.json は「この回の差分」ではなく「out_dirに存在する全データ」から
    # 常に作り直す(中断・再開を重ねても正しい累積値になるようにする)。
    all_target_ncodes = already_done_ncodes | {w[0] for w in works}
    full_stats = recompute_full_dataset_stats(args.out_dir, len(all_target_ncodes))
    full_stats["skipped_empty_body_episodes_this_run"] = stats["skipped_empty_body_episodes"]
    full_stats["skipped_duplicate_chunks_this_run"] = stats["skipped_duplicate_chunks"]
    stats_path = os.path.join(manifests_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(full_stats, f, ensure_ascii=False, indent=2)

    version_info = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_model_tokenizer": args.base_model,
        "target_tokens": args.target_tokens,
        "max_tokens": args.max_tokens,
        "overlap_tokens": args.overlap_tokens,
        "dedup_threshold": args.dedup_threshold,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "val_split_by_author": args.val_split_by_author,
        "enable_lmstudio": args.enable_lmstudio,
        "enable_api_refresh": args.enable_api_refresh,
        "db_path": os.path.abspath(args.db),
        "is_partial_run": bool(args.limit or ncode_filter),
    }
    with open(os.path.join(args.out_dir, "VERSION.json"), "w", encoding="utf-8") as f:
        json.dump(version_info, f, ensure_ascii=False, indent=2)

    print("[build_dataset] 今回の実行分:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("[build_dataset] out_dir累積(dataset_stats.jsonに保存):")
    print(json.dumps(full_stats, ensure_ascii=False, indent=2))
    print(f"[build_dataset] 完了。out_dir={args.out_dir}")
    if version_info["is_partial_run"]:
        print("[build_dataset] 注意: --limit/--ncodesによる部分実行です。dataset_v2/への昇格は行っていません。")


if __name__ == "__main__":
    main()
