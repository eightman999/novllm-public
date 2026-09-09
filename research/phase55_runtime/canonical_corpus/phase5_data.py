"""Build fixed, non-overlapping source pools for the Phase 5 tokenizer lab.

The source artifacts are immutable.  This module only writes compact Phase 5
pool JSONL/manifest files below a caller supplied output directory.  Web uses
the Phase 3 work split but deliberately does not reuse its 30M character
selection budget.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .source import CorpusError, dumps

POOL_VERSION = "phase5-fixed-pools-1.0"
WEB_PER_WORK_CAP = 200_000
AOZORA_EVAL_MODULUS = 100
AOZORA_EVAL_BUCKETS = frozenset({0})
AA_EVAL_MODULUS = 20
AA_EVAL_BUCKETS = frozenset({0})
_AOZORA_UNIT = re.compile(r"^[0-9]{6}$")
ORTHOGRAPHY = {
    "新字新仮名": "shin_shin",
    "新字旧仮名": "shin_kyu",
    "旧字旧仮名": "kyu_kyu",
    "旧字新仮名": "other",
}


class Phase5DataError(CorpusError):
    pass


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _bucket(value: str, modulus: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % modulus


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise Phase5DataError(f"non-object JSONL row: {path}")
                yield row


def _write_pool(root: Path, name: str, rows: Iterable[Mapping[str, Any]], *,
                source: Mapping[str, Any], exclusions: Mapping[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    shard = root / "pool-00000.jsonl"
    count = chars = utf8 = 0
    strata: Counter[str] = Counter()
    works: set[str] = set()
    seen_ids: set[str] = set()
    with shard.open("x", encoding="utf-8", newline="\n") as fh:
        for item in rows:
            row = dict(item)
            text = row.get("text")
            record_id = row.get("record_id")
            work_id = row.get("work_id")
            if not isinstance(text, str) or not text or not isinstance(record_id, str):
                raise Phase5DataError(f"invalid {name} pool row")
            if record_id in seen_ids:
                raise Phase5DataError(f"duplicate {name} record_id: {record_id}")
            seen_ids.add(record_id)
            row["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            row["characters"] = len(text)
            row["utf8_bytes"] = len(text.encode("utf-8"))
            fh.write(dumps(row) + "\n")
            count += 1; chars += row["characters"]; utf8 += row["utf8_bytes"]
            strata[str(row.get("stratum", "unclassified"))] += row["characters"]
            if isinstance(work_id, str):
                works.add(work_id)
    manifest = {
        "format": "novllm-phase5-source-pool", "version": POOL_VERSION,
        "pool": name, "status": "success", "valid": count > 0,
        "records": count, "works": len(works), "source_chars": chars,
        "utf8_bytes": utf8, "stratum_chars": dict(sorted(strata.items())),
        "shards": [{"path": shard.name, "rows": count, "bytes": shard.stat().st_size,
                    "sha256": _sha_file(shard)}],
        "source": dict(source), "exclusions": dict(exclusions),
    }
    (root / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = _sha_file(root / "manifest.json")
    return manifest


def validate_pool(root: str | os.PathLike) -> dict[str, Any]:
    path = Path(root); manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise Phase5DataError("pool manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "novllm-phase5-source-pool" or not manifest.get("valid"):
        raise Phase5DataError("invalid Phase 5 pool manifest")
    rows = chars = utf8 = 0; seen: set[str] = set()
    for shard in manifest.get("shards", []):
        p = path / str(shard.get("path", ""))
        if p.is_symlink() or not p.is_file() or _sha_file(p) != shard.get("sha256"):
            raise Phase5DataError(f"pool shard hash mismatch: {p}")
        for row in _jsonl(p):
            text = row.get("text"); rid = row.get("record_id")
            if not isinstance(text, str) or not isinstance(rid, str) or rid in seen:
                raise Phase5DataError("invalid or duplicate pool row")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != row.get("text_sha256"):
                raise Phase5DataError("pool text hash mismatch")
            seen.add(rid); rows += 1; chars += len(text); utf8 += len(text.encode("utf-8"))
    if (rows, chars, utf8) != (manifest.get("records"), manifest.get("source_chars"), manifest.get("utf8_bytes")):
        raise Phase5DataError("pool aggregate mismatch")
    return {**manifest, "manifest_sha256": _sha_file(manifest_path)}


def _web_eligibility(corpus_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(corpus_root.glob("eligibility-*.jsonl")):
        for row in _jsonl(path):
            rid = row.get("episode_revision_id")
            if isinstance(rid, str) and row.get("eligible") and row.get("split") == "train":
                result[rid] = row
    if not result:
        raise Phase5DataError("no Phase 3 train eligibility records")
    return result


def build_web_pool(derived_root: str | os.PathLike, phase3_corpus: str | os.PathLike,
                   output: str | os.PathLike, *, per_work_cap: int = WEB_PER_WORK_CAP) -> dict[str, Any]:
    """Build ordinary-Web train pool without the old 30M global cap."""
    derived = Path(derived_root); corpus = Path(phase3_corpus)
    eligible = _web_eligibility(corpus)
    used: defaultdict[str, int] = defaultdict(int)
    skipped = Counter()

    def rows() -> Iterator[dict[str, Any]]:
        for path in sorted(derived.glob("derived_views-*.jsonl")):
            for row in _jsonl(path):
                rid = row.get("episode_revision_id")
                meta = eligible.get(rid) if isinstance(rid, str) else None
                if meta is None:
                    skipped["not_train_or_ineligible"] += 1; continue
                strata = meta.get("strata") if isinstance(meta.get("strata"), Mapping) else {}
                if strata.get("aa_candidate") is True:
                    skipped["aa_separated"] += 1; continue
                text = row.get("text")
                work = meta.get("stable_work_key")
                if not isinstance(text, str) or not text or not isinstance(work, str):
                    skipped["missing_text_or_work"] += 1; continue
                remaining = per_work_cap - used[work]
                if remaining <= 0:
                    skipped["work_cap"] += 1; continue
                take = min(len(text), remaining)
                used[work] += take
                yield {"record_id": rid, "work_id": work, "stratum": "web_novel",
                       "text": text[:take], "source_offset": 0, "source_characters": len(text),
                       "author": strata.get("author_raw"), "phase3_split": "train",
                       "eligibility_reason": meta.get("reason"), "aa_candidate": False}

    source = {"derived_root": str(derived), "derived_manifest_sha256": _sha_file(derived / "manifest.json"),
              "phase3_corpus": str(corpus), "phase3_manifest_sha256": _sha_file(corpus / "manifest.json")}
    manifest = _write_pool(Path(output), "web_pool", rows(), source=source,
                           exclusions={"split": "train_only", "validation_probe": True,
                                       "aa_separated": True, "html_fallback": True,
                                       "per_work_cap": per_work_cap, "old_global_cap_reused": False})
    manifest["audit_skipped"] = dict(sorted(skipped.items()))
    manifest_path = Path(output) / "manifest.json"
    persisted = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    manifest_path.write_text(dumps(persisted) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = _sha_file(manifest_path)
    return manifest


def _aozora_orthography(raw_row: Mapping[str, Any]) -> str:
    contributors = raw_row.get("contributors")
    if isinstance(contributors, list):
        for contributor in contributors:
            if isinstance(contributor, Mapping) and contributor.get("role") == "著者":
                value = contributor.get("orthography")
                if isinstance(value, str) and value:
                    return ORTHOGRAPHY.get(value, "other")
        for contributor in contributors:
            if isinstance(contributor, Mapping):
                value = contributor.get("orthography")
                if isinstance(value, str) and value:
                    return ORTHOGRAPHY.get(value, "other")
    return "other"


def build_aozora_pools(source_root: str | os.PathLike, train_output: str | os.PathLike,
                       eval_output: str | os.PathLike) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(source_root); units = root / "units"
    train_rows: list[dict[str, Any]] = []; eval_rows: list[dict[str, Any]] = []
    for unit in sorted(x for x in units.iterdir() if x.is_dir() and _AOZORA_UNIT.fullmatch(x.name)):
        unit_manifest = json.loads((unit / "unit_manifest.json").read_text(encoding="utf-8"))
        if not unit_manifest.get("complete"):
            continue
        derived = json.loads((unit / "derived.json").read_text(encoding="utf-8"))
        raw_path = unit / "aozora" / "raw-00000.jsonl"
        raw = next(_jsonl(raw_path))
        text = derived.get("text_raw")
        if not isinstance(text, str) or not text:
            raise Phase5DataError(f"missing Aozora derived text: {unit.name}")
        stratum = _aozora_orthography(raw)
        author_ids = sorted({str(c.get("person_id")) for c in raw.get("contributors", [])
                             if isinstance(c, Mapping) and c.get("role") == "著者" and c.get("person_id")})
        item = {"record_id": str(derived["record_id"]), "work_id": unit.name,
                "stratum": stratum, "text": text, "author_ids": author_ids,
                "rights": raw.get("rights"), "source_url": raw.get("source_url")}
        (eval_rows if _bucket(unit.name, AOZORA_EVAL_MODULUS) in AOZORA_EVAL_BUCKETS else train_rows).append(item)
    source = {"root": str(root), "bulk_manifest_sha256": _sha_file(root / "manifest.json"),
              "successful_units": len(train_rows) + len(eval_rows), "bulk_complete": False,
              "quarantine_preserved": True}
    train = _write_pool(Path(train_output), "aozora_pool", train_rows, source=source,
                        exclusions={"deterministic_eval_modulus": AOZORA_EVAL_MODULUS,
                                    "deterministic_eval_buckets": sorted(AOZORA_EVAL_BUCKETS),
                                    "quarantine_non_blocking": True})
    evaluation = _write_pool(Path(eval_output), "aozora_eval", eval_rows, source=source,
                             exclusions={"training_eligible": False})
    return train, evaluation


def build_aa_pools(source_root: str | os.PathLike, train_output: str | os.PathLike,
                   eval_output: str | os.PathLike) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(source_root); manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    train_rows: list[dict[str, Any]] = []; eval_rows: list[dict[str, Any]] = []
    for shard in manifest.get("shards", []):
        for row in _jsonl(root / shard["path"]):
            text = row.get("text_raw"); work = row.get("stable_work_key")
            if not isinstance(text, str) or not text or not isinstance(work, str):
                continue
            item = {"record_id": str(row["record_id"]), "work_id": work, "stratum": "aa",
                    "text": text, "aa_label_confirmed": bool(row.get("aa_label_confirmed")),
                    "heuristic": bool(row.get("heuristic")), "redistribution": row.get("redistribution")}
            (eval_rows if _bucket(work, AA_EVAL_MODULUS) in AA_EVAL_BUCKETS else train_rows).append(item)
    source = {"root": str(root), "source_manifest_sha256": _sha_file(root / "manifest.json"),
              "heuristic_only": True, "private": True}
    train = _write_pool(Path(train_output), "aa_pool", train_rows, source=source,
                        exclusions={"deterministic_eval_modulus": AA_EVAL_MODULUS,
                                    "deterministic_eval_buckets": sorted(AA_EVAL_BUCKETS),
                                    "ordinary_web_overlap_prevented_by_record_id": True})
    evaluation = _write_pool(Path(eval_output), "aa_eval", eval_rows, source=source,
                             exclusions={"training_eligible": False})
    return train, evaluation


def pool_record_ids(root: str | os.PathLike) -> set[str]:
    path = Path(root); manifest = validate_pool(path)
    return {str(row["record_id"]) for shard in manifest["shards"] for row in _jsonl(path / shard["path"])}


def validate_disjoint(*roots: str | os.PathLike) -> dict[str, Any]:
    sets = [(str(root), pool_record_ids(root)) for root in roots]
    overlaps = []
    for i, (left, a) in enumerate(sets):
        for right, b in sets[i + 1:]:
            common = a & b
            if common:
                overlaps.append({"left": left, "right": right, "records": len(common)})
    if overlaps:
        raise Phase5DataError(f"Phase 5 pool overlap: {overlaps}")
    return {"status": "ok", "pools": len(sets), "overlaps": []}


__all__ = ["build_web_pool", "build_aozora_pools", "build_aa_pools", "validate_pool",
           "validate_disjoint", "pool_record_ids", "Phase5DataError"]
