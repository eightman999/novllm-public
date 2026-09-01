# db_audit.py
"""Phase 1: 正本DBのスキーマ・整合性監査。読み取り専用URIでのみ接続する。
不整合は自動修正せず、rejected.jsonl と集計JSONへ記録するだけに留める。"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sqlite3
import unicodedata
from collections import Counter
from typing import Any, Iterator

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "novels_descs": {
        "ncode", "title", "author", "Synopsis", "main_tag", "sub_tag", "rating",
        "total_ep", "general_all_no", "site_type", "sub_site", "noveltype",
        "length", "updated_at", "last_update_date",
    },
    "episodes": {"ncode", "episode_no", "body", "e_title", "update_time"},
    "url_entity": {"ncode", "api_url", "url", "is_r18"},
    "episode_mapping": {"ncode", "episode_no", "kakuyomu_episode_id"},
}

# 監査目的の簡易HTML除去(本文正規化の本実装は text_normalize.py 側にある)
_TAG_RE = re.compile(r"<[^>]+>")
_REPLACEMENT_CHAR = "�"
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def read_only_connect(db_path: str) -> sqlite3.Connection:
    abs_path = os.path.abspath(db_path)
    uri = f"file:{abs_path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON;")
    return conn


def _strip_tags_for_measurement(text: str) -> str:
    return _TAG_RE.sub("", text)


@dataclasses.dataclass
class AuditReport:
    required_tables_missing: list[str] = dataclasses.field(default_factory=list)
    required_columns_missing: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    novels_total: int = 0
    novels_duplicate_ncode: int = 0
    episodes_total: int = 0
    episodes_duplicate_key: int = 0
    orphan_episode_ncodes: list[str] = dataclasses.field(default_factory=list)
    episode_count_mismatch_general_all_no: int = 0
    episode_count_mismatch_total_ep: int = 0
    empty_body_episode_count: int = 0
    encoding_anomaly_episode_count: int = 0
    html_ratio_anomaly_episode_count: int = 0
    zero_episode_novels: int = 0
    site_type_breakdown: dict[str, int] = dataclasses.field(default_factory=dict)
    sub_site_breakdown: dict[str, int] = dataclasses.field(default_factory=dict)
    r18_source_breakdown: dict[str, int] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def check_required_schema(conn: sqlite3.Connection) -> tuple[list[str], dict[str, list[str]]]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = {row[0] for row in cur.fetchall()}

    missing_tables = [t for t in REQUIRED_COLUMNS if t not in existing_tables]
    missing_columns: dict[str, list[str]] = {}
    for table, required in REQUIRED_COLUMNS.items():
        if table in missing_tables:
            continue
        cur.execute(f"PRAGMA table_info(`{table}`);")
        existing_cols = {row[1] for row in cur.fetchall()}
        missing = sorted(required - existing_cols)
        if missing:
            missing_columns[table] = missing
    return missing_tables, missing_columns


def _iter_rejected_records(conn: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    cur = conn.cursor()

    # 孤立episode (novels_descsに存在しないncode)
    cur.execute(
        """
        SELECT DISTINCT e.ncode
        FROM episodes e
        LEFT JOIN novels_descs n ON e.ncode = n.ncode
        WHERE n.ncode IS NULL
        """
    )
    for (ncode,) in cur.fetchall():
        yield {"reason": "orphan_episode", "ncode": ncode}

    # 本文が空の話
    cur.execute("SELECT ncode, episode_no FROM episodes WHERE body IS NULL OR trim(body) = ''")
    for ncode, episode_no in cur.fetchall():
        yield {"reason": "empty_body", "ncode": ncode, "episode_no": episode_no}


def _iter_encoding_and_html_anomalies(conn: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT ncode, episode_no, body FROM episodes")
    while True:
        rows = cur.fetchmany(2000)
        if not rows:
            break
        for ncode, episode_no, body in rows:
            if not body:
                continue
            anomalies = []
            if _REPLACEMENT_CHAR in body:
                anomalies.append("replacement_char")
            if _CONTROL_CHAR_RE.search(body):
                anomalies.append("control_char")
            try:
                body.encode("utf-8", errors="strict")
            except UnicodeEncodeError:
                anomalies.append("unpaired_surrogate")
            if anomalies:
                yield {
                    "reason": "encoding_anomaly",
                    "ncode": ncode,
                    "episode_no": episode_no,
                    "detail": anomalies,
                }

            raw_len = len(body)
            if raw_len >= 200:
                stripped_len = len(_strip_tags_for_measurement(body))
                ratio = stripped_len / raw_len
                if ratio < 0.10:
                    yield {
                        "reason": "html_ratio_anomaly",
                        "ncode": ncode,
                        "episode_no": episode_no,
                        "raw_len": raw_len,
                        "stripped_len": stripped_len,
                        "ratio": round(ratio, 4),
                    }


def run_audit(db_path: str, out_dir: str) -> AuditReport:
    os.makedirs(out_dir, exist_ok=True)
    rejected_path = os.path.join(out_dir, "rejected.jsonl")

    conn = read_only_connect(db_path)
    report = AuditReport()

    missing_tables, missing_columns = check_required_schema(conn)
    report.required_tables_missing = missing_tables
    report.required_columns_missing = missing_columns

    if missing_tables:
        # 必須テーブルが無ければこれ以上の集計は無意味なので打ち切る
        with open(rejected_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"reason": "missing_required_tables", "tables": missing_tables},
                                 ensure_ascii=False) + "\n")
        conn.close()
        return report

    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM novels_descs")
    report.novels_total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) - COUNT(DISTINCT ncode) FROM novels_descs")
    report.novels_duplicate_ncode = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM episodes")
    report.episodes_total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) - COUNT(DISTINCT ncode || char(31) || episode_no) FROM episodes")
    report.episodes_duplicate_key = cur.fetchone()[0]

    cur.execute(
        """
        SELECT n.ncode, n.general_all_no, n.total_ep, COUNT(e.episode_no) AS actual
        FROM novels_descs n LEFT JOIN episodes e ON n.ncode = e.ncode
        GROUP BY n.ncode
        """
    )
    mismatch_general = 0
    mismatch_total = 0
    zero_episode = 0
    for ncode, general_all_no, total_ep, actual in cur.fetchall():
        if actual == 0:
            zero_episode += 1
        if actual != general_all_no:
            mismatch_general += 1
        if actual != total_ep:
            mismatch_total += 1
    report.episode_count_mismatch_general_all_no = mismatch_general
    report.episode_count_mismatch_total_ep = mismatch_total
    report.zero_episode_novels = zero_episode

    cur.execute("SELECT COUNT(*) FROM episodes WHERE body IS NULL OR trim(body) = ''")
    report.empty_body_episode_count = cur.fetchone()[0]

    cur.execute("SELECT site_type, COUNT(*) FROM novels_descs GROUP BY site_type")
    report.site_type_breakdown = {str(k): v for k, v in cur.fetchall()}
    cur.execute("SELECT sub_site, COUNT(*) FROM novels_descs GROUP BY sub_site")
    report.sub_site_breakdown = {str(k): v for k, v in cur.fetchall()}

    cur.execute(
        """
        SELECT
            SUM(CASE WHEN n.sub_site = 2 THEN 1 ELSE 0 END) AS by_sub_site,
            SUM(CASE WHEN n.sub_site != 2 AND n.rating = 1 THEN 1 ELSE 0 END) AS by_rating,
            SUM(CASE WHEN n.sub_site != 2 AND n.rating != 1 AND u.is_r18 = 1 THEN 1 ELSE 0 END) AS by_url_entity
        FROM novels_descs n LEFT JOIN url_entity u ON n.ncode = u.ncode
        """
    )
    by_sub_site, by_rating, by_url_entity = cur.fetchone()
    report.r18_source_breakdown = {
        "sub_site_eq_2": by_sub_site or 0,
        "rating_eq_1": by_rating or 0,
        "url_entity_is_r18": by_url_entity or 0,
    }

    encoding_count = 0
    html_ratio_count = 0
    with open(rejected_path, "w", encoding="utf-8") as fh:
        for record in _iter_rejected_records(conn):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        for record in _iter_encoding_and_html_anomalies(conn):
            if record["reason"] == "encoding_anomaly":
                encoding_count += 1
            elif record["reason"] == "html_ratio_anomaly":
                html_ratio_count += 1
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    report.encoding_anomaly_episode_count = encoding_count
    report.html_ratio_anomaly_episode_count = html_ratio_count

    conn.close()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="novllm dataset_v2 DB監査 (Phase1)")
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", default="./dataset_v2.tmp/manifests")
    args = ap.parse_args()

    report = run_audit(args.db, args.out_dir)
    report_path = os.path.join(args.out_dir, "db_audit_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(f"\n[db_audit] report written: {report_path}")
    print(f"[db_audit] rejected records written: {os.path.join(args.out_dir, 'rejected.jsonl')}")


if __name__ == "__main__":
    main()
