"""One immutable, source-accounted LM dataset for every Phase 5.5 tokenizer.

This does not reuse N/J tokenizer-training recipes. Work, record and exact
prefix checks are distinct from tokenizer-training exposure and near duplicates.
No source corpus or tokenizer artifact is modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .phase5_data import validate_pool

VERSION = "phase55-common-lm-data-1.0"
CATEGORIES = ("web_novel", "aozora_shin_shin", "aozora_shin_kyu", "aozora_kyu_kyu", "aa",
              "historical_kana", "old_orthography", "kanbun", "kakikudashi", "gyaru",
              "technical", "whitespace", "unicode_edge")
MAJOR_CATEGORIES = CATEGORIES[:5]
DEFAULT_RECIPE = {"web_novel": .70, "aozora_shin_shin": .13, "aozora_shin_kyu": .065,
                  "aozora_kyu_kyu": .065, "aa": .04}


class Phase55DataError(ValueError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("rb") as fh:
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise Phase55DataError(f"non-object source row: {path}")
                yield offset, row


def _resolve(reference: str | Path, root: Path) -> Path:
    path = Path(reference)
    if path.is_absolute():
        return path.resolve()
    for base in (root, *root.parents, Path.cwd()):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    raise Phase55DataError(f"source reference missing: {reference}")


def _text(row: Mapping[str, Any], field: str = "text") -> str:
    text = row.get(field)
    if not isinstance(text, str) or not text:
        raise Phase55DataError("empty or non-text source record")
    if field == "text" and row.get("text_sha256") not in (None, _sha(text)):
        raise Phase55DataError("source text hash mismatch")
    return text


class _PrefixIndex:
    """Exact equality or complete-string prefix, never approximate matching."""
    def __init__(self, texts: Iterable[str]):
        self.groups: dict[str, list[str]] = defaultdict(list)
        for text in set(texts):
            if text:
                self.groups[text[0]].append(text)

    def overlaps(self, text: str) -> bool:
        return any(text.startswith(other) or other.startswith(text)
                   for other in self.groups.get(text[:1], ()))


def _record(rid: str, work: str | None, category: str, text: str, *, source: str,
            source_record_id: str, full_text: str, source_path: Path,
            identity_level: str, start: int = 0, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if full_text[start:start + len(text)] != text:
        raise Phase55DataError("selection is not an exact source span")
    return {"id": rid, "document_id": work, "category": category, "text": text,
            "text_sha256": _sha(text), "source_chars": len(text),
            "source_bytes": len(text.encode("utf-8")), "source": source,
            "lineage": {"source_path": str(source_path), "source_record_id": source_record_id,
                        "work_id": work, "identity_level": identity_level,
                        "full_source_text_sha256": _sha(full_text),
                        "full_source_chars": len(full_text),
                        "span": {"start": start, "end": start + len(text), "unit": "unicode_codepoints"},
                        **dict(extra or {})}}


def _rank(seed: int, kind: str, rid: str) -> str:
    return _sha(f"{seed}\0{kind}\0{rid}")


def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = {}
    for category in sorted({r["category"] for r in rows}):
        selected = [r for r in rows if r["category"] == category]
        groups[category] = {"records": len(selected), "source_chars": sum(r["source_chars"] for r in selected),
                            "source_bytes": sum(r["source_bytes"] for r in selected)}
    return {"records": len(rows), "source_chars": sum(r["source_chars"] for r in rows),
            "source_bytes": sum(r["source_bytes"] for r in rows),
            "unique_documents": len({r["document_id"] for r in rows if r["document_id"] is not None}),
            "categories": groups}


def _selected_audit(train: list[dict[str, Any]], evaluation: list[dict[str, Any]]) -> dict[str, Any]:
    works = {r["document_id"] for r in evaluation if r["lineage"]["identity_level"] == "work"}
    records = {r["lineage"]["source_record_id"] for r in evaluation}
    full_hashes = {r["lineage"]["full_source_text_sha256"] for r in evaluation}
    hashes = {r["text_sha256"] for r in evaluation}
    prefixes = _PrefixIndex(r["text"] for r in evaluation)
    result = {"work_overlap_records": sum(r["document_id"] in works for r in train),
              "record_overlap_records": sum(r["lineage"]["source_record_id"] in records for r in train),
              "full_source_hash_overlap_records": sum(r["lineage"]["full_source_text_sha256"] in full_hashes for r in train),
              "selected_hash_overlap_records": sum(r["text_sha256"] in hashes for r in train),
              "selected_prefix_overlap_records": sum(prefixes.overlaps(r["text"]) for r in train)}
    if any(result.values()):
        raise Phase55DataError(f"train/eval contamination: {result}")
    return {"status": "pass", **result}


def build_dataset(phase5_root: Path, output_dir: Path, *, train_chars: int = 100_000,
                  eval_chars_per_category: int = 4096, seed: int = 20260908,
                  recipe: Mapping[str, float] | None = None,
                  phase3_corpus: Path | None = None, web_heldout: Path | None = None) -> dict[str, Any]:
    """Freeze a deterministic shared raw-text subset, or validate/reuse it.

    Major categories use real heldout prose only. All declared 13 categories are
    required, with fixed probes retained for diagnostics. Limits are codepoints;
    a short diagnostic category is retained in full without repetition.
    """
    root, output = Path(phase5_root).resolve(), Path(output_dir).resolve()
    weights = dict(DEFAULT_RECIPE if recipe is None else recipe)
    if set(weights) != set(MAJOR_CATEGORIES) or any(not math.isfinite(v) or v <= 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise Phase55DataError("recipe requires five positive major-category weights summing to 1")
    if train_chars < len(MAJOR_CATEGORIES) or eval_chars_per_category < 1:
        raise Phase55DataError("positive train/eval character budgets required")
    if "runs" not in output.parts or output == root or root in output.parents:
        raise Phase55DataError("dataset payload must use a separate ignored runs directory")
    pool_roots = {name: root / "pools" / name for name in ("web", "aozora-train", "aa-train", "aozora-eval", "aa-eval")}
    # The small manifest is sufficient to resolve references before streaming.
    web_manifest = json.loads((pool_roots["web"] / "manifest.json").read_text())
    corpus = _resolve(phase3_corpus or web_manifest["source"]["phase3_corpus"], root)
    heldout = _resolve(web_heldout or corpus.parent / "eval_heldout_clean_v2.jsonl", root)
    config = {"version": VERSION, "phase5_root": str(root), "train_chars": train_chars,
              "eval_chars_per_category": eval_chars_per_category, "seed": seed,
              "recipe": weights, "phase3_corpus": str(corpus), "web_heldout": str(heldout),
              "selection": "sha256(seed, category, record id), exact prefix, no repeats",
              "major_eval_source": "real_heldout_only", "categories": list(CATEGORIES)}
    if output.exists():
        _, _, previous = load_dataset(output)
        if previous["config"] != config:
            raise Phase55DataError("frozen dataset arguments differ; choose a new output directory")
        for item in previous["inputs"]:
            if _sha_file(Path(item["path"])) != item["sha256"]:
                raise Phase55DataError("frozen source input changed")
        return previous

    inputs: dict[str, dict[str, Any]] = {}
    def remember(path: Path, expected: str | None = None) -> None:
        if not path.is_file() or path.is_symlink():
            raise Phase55DataError(f"missing or symlink input: {path}")
        digest = _sha_file(path)
        if expected is not None and digest != expected:
            raise Phase55DataError(f"input hash mismatch: {path}")
        inputs[str(path)] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}

    manifests = {}
    for name, directory in pool_roots.items():
        manifests[name] = validate_pool(directory)
        remember(directory / "manifest.json")
        for shard in manifests[name]["shards"]:
            remember(directory / shard["path"], shard["sha256"])
    expanded_path = root / "eval" / "expanded_eval.json"
    remember(expanded_path)
    expanded = json.loads(expanded_path.read_text())
    if expanded.get("training_eligible") is not False or _sha(_json(expanded["probes"])) != expanded.get("probe_sha256"):
        raise Phase55DataError("invalid expanded probe manifest/hash")
    source_hashes = expanded.get("inputs_sha256", {})
    for name in ("aozora-eval", "aa-eval"):
        expected = source_hashes.get(name.replace("-", "_") + "_manifest")
        if expected is None or expected != inputs[str(pool_roots[name] / "manifest.json")]["sha256"]:
            raise Phase55DataError("expanded eval source pool hash mismatch")
    remember(heldout, source_hashes.get("web_heldout"))
    remember(corpus / "manifest.json", web_manifest["source"].get("phase3_manifest_sha256"))
    corpus_manifest = json.loads((corpus / "manifest.json").read_text())
    reserved_works: set[str] = set()
    reserved_records: set[str] = set()
    full_eval_texts: list[str] = []
    eval_pool_index: dict[str, tuple[dict, Path]] = {}
    for name in ("aozora-eval", "aa-eval"):
        for shard in manifests[name]["shards"]:
            path = pool_roots[name] / shard["path"]
            for _, row in _rows(path):
                rid = row["record_id"]
                if rid in eval_pool_index:
                    raise Phase55DataError("duplicate eval source record identity")
                eval_pool_index[rid] = (row, path)
                reserved_works.add(row["work_id"]); reserved_records.add(rid)
                full_eval_texts.append(_text(row))
    for shard in corpus_manifest.get("shards", []):
        if shard.get("kind") not in ("validation", "tokenizer_probe"):
            continue
        path = corpus / shard["path"]
        remember(path, shard["sha256"])
        for _, row in _rows(path):
            boundary = row.get("boundary", {})
            if not boundary.get("stable_work_key") or not boundary.get("episode_revision_id"):
                raise Phase55DataError("Phase 3 heldout work identity missing")
            reserved_works.add(boundary["stable_work_key"])
            reserved_records.add(boundary["episode_revision_id"])
            full_eval_texts.append(_text(row))
    web_index = {}
    for _, row in _rows(heldout):
        if row.get("split") not in ("validation", "tokenizer_probe", "val", "test", "probe"):
            raise Phase55DataError("Web heldout is not an evaluation split")
        if not row.get("source_work_key") or not row.get("episode_revision_id"):
            raise Phase55DataError("Web heldout work identity missing")
        if row["id"] in web_index:
            raise Phase55DataError("duplicate Web heldout identity")
        web_index[row["id"]] = row
        reserved_works.add(row["source_work_key"]); reserved_records.add(row["episode_revision_id"])
        full_eval_texts.append(_text(row))

    # Expanded probes are the frozen diagnostic inputs. Their record ids are
    # retained even where that artifact no longer carries a work-level identity.
    eval_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for probe in expanded["probes"]:
        text = _text(probe)
        full_eval_texts.append(text)
        category, pid = probe["category"], probe["id"]
        if category not in CATEGORIES:
            continue
        if category in MAJOR_CATEGORIES and probe.get("source") != "real_heldout":
            continue
        source = str(probe.get("source", "unknown"))
        if category == "web_novel":
            original = web_index.get(pid.removeprefix("phase5:web:"))
            if original is None:
                raise Phase55DataError("Web probe failed heldout metadata join")
            full = _text(original)
            record = _record(pid, original["source_work_key"], category, text, source=source,
                             source_record_id=original["episode_revision_id"], full_text=full,
                             source_path=heldout, identity_level="work",
                             extra={"raw_sha256": original.get("raw_sha256"), "split": original["split"]})
        elif category in MAJOR_CATEGORIES:
            rid = pid.removeprefix("phase5:" + category + ":")
            item = eval_pool_index.get(rid)
            if item is None:
                raise Phase55DataError("major probe failed eval pool metadata join")
            original, path = item
            record = _record(pid, original["work_id"], category, text, source=source,
                             source_record_id=rid, full_text=_text(original), source_path=path,
                             identity_level="work")
        else:
            rid = pid.split(":", 2)[-1] if pid.startswith("phase5:") else pid
            reserved_records.add(rid)
            record = _record(pid, None, category, text, source=source, source_record_id=rid,
                             full_text=text, source_path=expanded_path,
                             identity_level="synthetic" if source == "fixed" else "record_only",
                             extra={"full_text_scope": "frozen_probe_only", "work_identity_unknown": source != "fixed"})
        eval_candidates[category].append(record)
    missing = set(CATEGORIES) - eval_candidates.keys()
    if missing:
        raise Phase55DataError(f"missing evaluation categories: {sorted(missing)}")
    evaluation = []
    for category in CATEGORIES:
        remaining = eval_chars_per_category
        for row in sorted(eval_candidates[category], key=lambda r: _rank(seed, category, r["id"])):
            if remaining <= 0:
                break
            selected = dict(row)
            text = row["text"][:remaining]
            selected.update(text=text, text_sha256=_sha(text), source_chars=len(text), source_bytes=len(text.encode()))
            selected["lineage"] = {**row["lineage"], "span": {"start": 0, "end": len(text), "unit": "unicode_codepoints"},
                                   "expanded_probe_id": row["id"], "expanded_probe_text_sha256": row["text_sha256"]}
            evaluation.append(selected); remaining -= len(text)
    full_eval_texts.extend(r["text"] for r in evaluation)
    reserved_hashes = {_sha(t) for t in full_eval_texts}
    prefixes = _PrefixIndex(full_eval_texts)
    exclusions: dict[str, Counter] = defaultdict(Counter)
    eligible: dict[str, list[dict]] = defaultdict(list)
    excluded_work_ids: dict[str, set[str]] = defaultdict(set)
    seen_train_ids: set[str] = set()
    # Store file offsets instead of copying the > 1 GB raw pools into RAM.
    for name in ("web", "aozora-train", "aa-train"):
        for shard in manifests[name]["shards"]:
            path = pool_roots[name] / shard["path"]
            for offset, row in _rows(path):
                text = _text(row); rid, work = row["record_id"], row.get("work_id")
                if not isinstance(work, str) or not work:
                    raise Phase55DataError("train work identity missing")
                if rid in seen_train_ids:
                    raise Phase55DataError("train pools overlap by record identity")
                seen_train_ids.add(rid)
                category = "web_novel" if name == "web" else "aa" if name == "aa-train" else "aozora_" + row["stratum"]
                reasons = []
                if category not in weights: reasons.append("outside_recipe")
                if work in reserved_works:
                    reasons.append("heldout_work"); excluded_work_ids[name].add(work)
                if rid in reserved_records: reasons.append("heldout_record")
                if row["text_sha256"] in reserved_hashes: reasons.append("heldout_full_text_hash")
                if prefixes.overlaps(text): reasons.append("heldout_exact_prefix")
                if reasons:
                    exclusions[name]["excluded_records"] += 1; exclusions[name]["excluded_chars"] += len(text)
                    exclusions[name].update(reasons)
                    continue
                eligible[category].append({"path": path, "offset": offset, "record_id": rid,
                                           "rank": _rank(seed, category, rid), "pool": name})
    # Largest remainder allocation preserves the exact requested source budget.
    budgets = {c: math.floor(train_chars * weights[c]) for c in MAJOR_CATEGORIES}
    for category in sorted(MAJOR_CATEGORIES, key=lambda c: (-(train_chars * weights[c] - budgets[c]), c))[:train_chars - sum(budgets.values())]:
        budgets[category] += 1
    train = []
    for category in MAJOR_CATEGORIES:
        remaining = budgets[category]
        if remaining < 1:
            raise Phase55DataError("train budget too small to retain every recipe category")
        for item in sorted(eligible[category], key=lambda r: r["rank"]):
            if remaining <= 0: break
            with item["path"].open("rb") as fh:
                fh.seek(item["offset"]); original = json.loads(fh.readline())
            full = _text(original); text = full[:remaining]
            if prefixes.overlaps(text):
                exclusions[item["pool"]]["selection_prefix_collision"] += 1
                continue
            row = _record("phase55:train:" + item["record_id"], original["work_id"], category, text,
                          source="phase5_pool:" + item["pool"], source_record_id=item["record_id"],
                          full_text=full, source_path=item["path"], identity_level="work",
                          extra={"pool": item["pool"], "source_offset": original.get("source_offset", 0),
                                 "original_source_chars": original.get("source_characters", len(full)),
                                 "full_text_scope": "phase5_pool_row", "source_file_byte_offset": item["offset"]})
            train.append(row); remaining -= len(text)
        if remaining:
            raise Phase55DataError(f"insufficient disjoint source characters for {category}: missing {remaining}")
    train.sort(key=lambda r: _rank(seed, "shared_train_order", r["id"]))
    audit = _selected_audit(train, evaluation)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".phase55-data-", dir=output.parent))
    shards = []
    for split, rows in (("train", train), ("eval", evaluation)):
        path = temp / (split + ".jsonl")
        path.write_text("".join(_json(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
        shards.append({"split": split, "path": path.name, "sha256": _sha_file(path), "bytes": path.stat().st_size,
                       "rows": len(rows)})
    manifest = {"format": "novllm-phase55-common-lm-dataset", "version": VERSION, "config": config,
                "shards": shards, "inputs": sorted(inputs.values(), key=lambda x: x["path"]),
                "probe_sha256": expanded["probe_sha256"], "train": _totals(train), "eval": _totals(evaluation),
                "selection_budgets": budgets, "candidate_specific_lm_recipe": False,
                "lm_training_contamination": {**audit, "reservation_scope": "all Phase5 eval pools, Phase3 heldout shards, web heldout, all expanded probes",
                    "reserved_works": len(reserved_works), "reserved_records": len(reserved_records),
                    "source_pool_exclusions": {name: {**dict(counts), "excluded_work_ids": sorted(excluded_work_ids[name])}
                                               for name, counts in sorted(exclusions.items())}},
                "tokenizer_training_exposure": {"status": "not_certified_clean", "hard_freeze_gate_pass": False,
                    "reason": "Phase5 tokenizer recipes may have seen heldout works; LM exclusions do not undo tokenizer fitting exposure"},
                "limitations": ["Exact checks do not establish absence of paraphrases, near duplicates or translated equivalents.",
                    "Diagnostic source work identities absent from expanded_eval remain unknown; record identity and frozen text are retained.",
                    "Full source text hashes refer to preserved pool/heldout rows, which may already be source prefixes.",
                    "This dataset certifies measured LM split checks only, not the full no-contamination freeze gate."],
                "readiness": {"probe_lm_dataset_ready": True, "novtokenizer_freeze_ready": False,
                    "balanced_cultural_pretraining_ready": False, "aozora_bulk_complete": manifests["aozora-train"]["source"].get("bulk_complete"),
                    "quarantine_non_blocking_for_probe_lm": True}}
    manifest["dataset_sha256"] = _sha(_json(manifest))
    path = temp / "manifest.json"
    path.write_text(_json(manifest) + "\n", encoding="utf-8")
    (temp / "manifest.sha256").write_text(_sha_file(path) + "\n", encoding="ascii")
    load_dataset(temp)
    os.rename(temp, output)
    return manifest


def load_dataset(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate a self-contained frozen dataset without opening source corpora."""
    root = Path(output_dir)
    path = root / "manifest.json"
    if not path.is_file() or path.is_symlink() or not (root / "manifest.sha256").is_file():
        raise Phase55DataError("frozen dataset manifest missing")
    if _sha_file(path) != (root / "manifest.sha256").read_text().strip():
        raise Phase55DataError("dataset manifest hash mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "novllm-phase55-common-lm-dataset" or manifest.get("version") != VERSION:
        raise Phase55DataError("unsupported dataset format")
    body = {k: v for k, v in manifest.items() if k != "dataset_sha256"}
    if _sha(_json(body)) != manifest.get("dataset_sha256"):
        raise Phase55DataError("dataset content hash mismatch")
    splits: dict[str, list[dict]] = {}
    for shard in manifest["shards"]:
        split, name = shard["split"], shard["path"]
        if split not in ("train", "eval") or split in splits or Path(name).name != name:
            raise Phase55DataError("invalid dataset shard declaration")
        p = root / name
        if p.is_symlink() or not p.is_file() or _sha_file(p) != shard["sha256"] or p.stat().st_size != shard["bytes"]:
            raise Phase55DataError("dataset shard hash/size mismatch")
        rows = [r for _, r in _rows(p)]
        ids = set()
        for row in rows:
            text = _text(row); lineage = row["lineage"]
            if row["id"] in ids or row["category"] not in CATEGORIES:
                raise Phase55DataError("duplicate row or unknown category")
            ids.add(row["id"])
            span = lineage["span"]
            if row["source_chars"] != len(text) or row["source_bytes"] != len(text.encode()) or span["end"] - span["start"] != len(text) or span["start"] < 0 or span["end"] > lineage["full_source_chars"]:
                raise Phase55DataError("source span/accounting mismatch")
            if lineage["identity_level"] == "work" and row["document_id"] != lineage["work_id"]:
                raise Phase55DataError("source work identity mismatch")
        if len(rows) != shard["rows"] or _totals(rows) != manifest[split]:
            raise Phase55DataError("dataset aggregate mismatch")
        splits[split] = rows
    if set(splits) != {"train", "eval"} or set(manifest["eval"]["categories"]) != set(CATEGORIES):
        raise Phase55DataError("required dataset split/category missing")
    observed = _selected_audit(splits["train"], splits["eval"])
    if any(manifest["lm_training_contamination"].get(k) != v for k, v in observed.items()):
        raise Phase55DataError("stored contamination audit mismatch")
    return splits["train"], splits["eval"], manifest
