"""Lossless, source-isolated cultural corpus storage and lineage metadata."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from typing import Iterable, Mapping, Any
from .cultural_registry import CulturalRegistry, RegistryError

class StoreError(ValueError): pass

def _sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def _json(value: Any) -> bytes: return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()

def write_source_records(records: Iterable[Mapping[str, Any]], root: str | Path, *, source_id: str,
                         registry: CulturalRegistry | None = None, shard_rows: int = 10000,
                         maxbytes: int | None = None, stage: str = "raw") -> dict:
    """Write canonical raw JSONL shards for one source, retaining ``text_raw`` exactly."""
    if shard_rows <= 0 or (maxbytes is not None and maxbytes <= 0): raise StoreError("shard limits must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", source_id) or source_id in {".", ".."} or "\x00" in source_id:
        raise StoreError("unsafe source_id")
    if stage not in {"raw", "derived", "tokenizer", "training"}: raise StoreError("invalid stage")
    if registry is not None: registry.get(source_id)
    base = Path(root) / source_id
    if base.exists(): raise StoreError("source output already exists")
    base.mkdir(parents=True)
    shards, rows, chars, byte_count, seen_ids = [], 0, 0, 0, set()
    record_digest = hashlib.sha256()
    stream = None
    try:
        for item in records:
            row = dict(item)
            if row.get("source_id", source_id) != source_id: raise StoreError("mixed source records")
            text = row.get("text_raw")
            if not isinstance(text, str): raise StoreError("record requires text_raw string")
            row["source_id"] = source_id
            expected_raw = _sha(text.encode("utf-8"))
            for key, expected in (("raw_sha256", expected_raw), ("raw_char_count", len(text)), ("raw_utf8_bytes", len(text.encode("utf-8")))):
                if key in row and row[key] != expected: raise StoreError("record raw metadata mismatch: " + key)
                row[key] = expected
            rid = row.get("record_id")
            if rid is None:
                rid = "cultural_" + _sha((source_id + "\0" + expected_raw).encode("utf-8"))
                row["record_id"] = rid
            if not isinstance(rid, str) or rid in seen_ids: raise StoreError("duplicate or invalid record_id")
            seen_ids.add(rid)
            data = _json(row)
            record_digest.update(data)
            if stream is None or stream["rows"] >= shard_rows or (maxbytes is not None and stream["bytes"] + len(data) > maxbytes and stream["rows"]):
                if stream: stream["fh"].close(); shards.append(_finish(stream))
                name = f"raw-{len(shards):05d}.jsonl"
                stream = {"name": name, "fh": (base / name).open("xb"), "rows": 0, "bytes": 0, "hash": hashlib.sha256()}
            stream["fh"].write(data); stream["hash"].update(data); stream["rows"] += 1; stream["bytes"] += len(data)
            rows += 1; chars += len(text); byte_count += len(text.encode("utf-8"))
    finally:
        if stream: stream["fh"].close(); shards.append(_finish(stream))
    rights = None
    if registry is not None:
        src = registry.get(source_id)
        rights = {"copyright_status": src.copyright_status, "license": src.license, "training_basis": src.training_basis,
                  "redistribution": src.redistribution, "storage_policy": src.storage_policy, "citation": src.citation}
    manifest = {"format": "novllm-cultural-canonical-raw", "schema_version": "1.1.0", "stage": stage, "source_id": source_id,
                "complete": True, "counts": {"records": rows, "characters": chars, "utf8_bytes": byte_count},
                "record_sha256": record_digest.hexdigest(), "rights": rights, "shards": shards,
                "lineage": {"source_id": source_id, "canonical": "raw"}}
    (base / "manifest.json").write_bytes(_json(manifest))
    return manifest

def _finish(stream):
    return {"path": stream["name"], "rows": stream["rows"], "bytes": stream["bytes"], "sha256": stream["hash"].hexdigest()}

def iter_source_records(root: str | Path, *, source_id: str | None = None):
    base = Path(root) / source_id if source_id else Path(root)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "novllm-cultural-canonical-raw" or manifest.get("complete") is not True: raise StoreError("invalid cultural raw manifest")
    if manifest.get("schema_version") not in {"1.0.0", "1.1.0"}: raise StoreError("unsupported cultural raw schema")
    expected = manifest.get("source_id")
    for shard in manifest.get("shards", []):
        path = base / shard["path"]
        if path.is_symlink() or path.resolve().parent != base.resolve(): raise StoreError("unsafe shard path")
        data = path.read_bytes()
        if _sha(data) != shard.get("sha256") or len(data) != shard.get("bytes"): raise StoreError("shard integrity failure")
        count = 0
        for line in data.splitlines():
            count += 1
            row = json.loads(line.decode("utf-8"))
            if row.get("source_id") != expected: raise StoreError("source isolation violation")
            yield row
        if manifest.get("schema_version") == "1.1.0" and count != shard.get("rows"):
            raise StoreError("shard row count mismatch")

def validate_source_store(root: str | Path, *, source_id: str | None = None) -> dict:
    base = Path(root) / source_id if source_id else Path(root)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True: raise StoreError("incomplete cultural manifest")
    rows = 0; chars = 0; byte_count = 0; seen = set(); digest = hashlib.sha256()
    for row in iter_source_records(root, source_id=source_id):
        rid = row.get("record_id")
        if not isinstance(rid, str) or rid in seen: raise StoreError("duplicate record_id")
        seen.add(rid); text = row.get("text_raw")
        if not isinstance(text, str): raise StoreError("record requires text_raw")
        if manifest.get("schema_version") == "1.1.0":
            if row.get("raw_sha256") != _sha(text.encode("utf-8")): raise StoreError("raw_sha256 mismatch")
            if row.get("raw_char_count") != len(text) or row.get("raw_utf8_bytes") != len(text.encode("utf-8")): raise StoreError("raw count metadata mismatch")
        digest.update(_json(row)); rows += 1; chars += len(text); byte_count += len(text.encode("utf-8"))
    counts = {"records": rows, "characters": chars, "utf8_bytes": byte_count}
    if counts != manifest.get("counts"): raise StoreError("manifest counts mismatch")
    if manifest.get("schema_version") == "1.1.0" and manifest.get("record_sha256") != digest.hexdigest(): raise StoreError("record stream hash mismatch")
    return {"valid": True, "counts": counts, "source_id": manifest.get("source_id"), "stage": manifest.get("stage", "raw")}

def build_lineage(*, source_id: str, source_sha256: str, canonical_sha256: str, derived_sha256: str | None = None,
                  tokenizer_sha256: str | None = None, training_sha256: str | None = None,
                  run_id: str | None = None, checkpoint_id: str | None = None,
                  parent_ids: list[str] | None = None, source_ids: list[str] | None = None) -> dict:
    """Return metadata-only lineage; deletion of payload never removes this record."""
    return {"format": "novllm-cultural-lineage", "version": "1.1.0", "source_id": source_id, "source_ids": source_ids or [source_id], "parent_ids": parent_ids or [source_id], "source_sha256": source_sha256, "canonical_sha256": canonical_sha256,
            "derived_sha256": derived_sha256, "tokenizer_sha256": tokenizer_sha256,
            "training_sha256": training_sha256, "run_id": run_id, "checkpoint_id": checkpoint_id}

class LineageGraph:
    def __init__(self, records=()): self.records = list(records)
    def add(self, *, node_id, kind, sha256=None, source_id=None, parent_id=None, recipe=None, characters=None, tokens=None, **meta):
        parent_ids = meta.pop("parent_ids", None)
        if parent_ids is None: parent_ids = [] if parent_id is None else [parent_id]
        if parent_id is not None and parent_id not in parent_ids: parent_ids.append(parent_id)
        row = {"node_id": node_id, "kind": kind, "sha256": sha256, "source_id": source_id, "parent_ids": parent_ids,
               "recipe": recipe, "characters": characters, "tokens": tokens, **meta}
        if any(r["node_id"] == node_id for r in self.records): raise StoreError("duplicate lineage node")
        self.records.append(row); return row
    def descendants(self, source_id):
        roots = {r["node_id"] for r in self.records if r.get("source_id") == source_id or r.get("node_id") == source_id}
        out = []
        while roots:
            current = roots.pop()
            for row in self.records:
                parents = row.get("parent_ids", [row.get("parent_id")] if row.get("parent_id") else [])
                if current in parents and row not in out:
                    out.append(row); roots.add(row["node_id"])
        return out
    def validate(self):
        ids = {r.get("node_id") for r in self.records}
        if None in ids or len(ids) != len(self.records): raise StoreError("duplicate or missing lineage node")
        edges = {r["node_id"]: list(r.get("parent_ids", [r.get("parent_id")] if r.get("parent_id") else [])) for r in self.records}
        missing = sorted({p for parents in edges.values() for p in parents if p not in ids})
        if missing: raise StoreError("missing lineage parents: " + ",".join(missing))
        visiting, done = set(), set()
        def visit(node):
            if node in visiting: raise StoreError("lineage cycle")
            if node in done or node not in edges: return
            visiting.add(node)
            for parent in edges[node]: visit(parent)
            visiting.remove(node); done.add(node)
        for node in edges: visit(node)
        source_ids = {r.get("source_id") for r in self.records if r.get("source_id")}
        for row in self.records: source_ids.update(row.get("source_ids", []))
        return {"valid": True, "source_ids": sorted(source_ids)}
    def dump(self, path):
        self.validate(); Path(path).write_bytes(_json({"format":"novllm-cultural-lineage","version":"1.1.0","nodes":self.records}))
    @classmethod
    def load(cls, path):
        data=json.loads(Path(path).read_text(encoding="utf-8"));
        if data.get("format") != "novllm-cultural-lineage": raise StoreError("invalid lineage")
        graph = cls(data.get("nodes", [])); graph.validate(); return graph

def deletion_plan(registry: CulturalRegistry, source_id: str, root: str | Path, *, lineage: LineageGraph | None = None) -> dict:
    source = registry.get(source_id)
    base = Path(root).resolve()
    configured = []
    if source.payload_path: configured.append(base / source.payload_path)
    extensions = source.extensions or {}
    if extensions.get("payload_root"): configured.append(base / extensions["payload_root"])
    payload_roles = {"original", "source", "raw", "derived", "derived_payload", "canonical_shard", "sample", "tokenizer", "training"}
    for artifact in extensions.get("artifacts", []):
        if artifact.get("role") in payload_roles and artifact.get("path"):
            configured.append(base / artifact["path"])
    if not configured: configured.append(base / source_id)
    descendants = lineage.descendants(source_id) if lineage is not None else []
    mixed = [r for r in descendants if ((set(r.get("source_ids", [])) | ({r.get("source_id")} if r.get("source_id") else set())) - {source_id})]
    rebuild = [r for r in descendants if r not in mixed and r.get("node_id") != source_id and r.get("kind") not in {"checkpoint", "metadata"}]
    protected = set()
    for node in mixed + [r for r in descendants if r.get("kind") in {"checkpoint", "metadata"}]:
        for item in node.get("payload_paths", []):
            candidate = (base / item)
            if candidate.is_symlink() or any(parent.is_symlink() for parent in [candidate, *candidate.parents] if parent != base):
                raise StoreError("symlink deletion target")
            resolved = candidate.resolve()
            if base not in resolved.parents: raise StoreError("unsafe lineage payload path")
            protected.add(resolved)
    for node in rebuild:
        configured.extend(base / p for p in node.get("payload_paths", []))
    metadata_names = {"manifest.json", "registry.json", "lineage.json"}
    paths = []
    for item in configured:
        if item.is_symlink() or any(parent.is_symlink() for parent in [item, *item.parents] if parent != base): raise StoreError("symlink deletion target")
        target = item.resolve()
        if base not in target.parents and target != base: raise StoreError("unsafe deletion target")
        if target.is_file():
            if target.name not in metadata_names: paths.append(str(target))
        elif target.is_dir():
            for child in target.rglob("*"):
                if child.is_symlink(): raise StoreError("symlink deletion target")
                resolved_child = child.resolve()
                if child.is_file() and child.name not in metadata_names and resolved_child not in protected:
                    paths.append(str(resolved_child))
    paths = sorted(set(paths))
    hashes = {}
    for item in paths: hashes[item] = _sha(Path(item).read_bytes())
    return {"action": "delete_payload", "source_id": source_id, "payload_root": str(configured[0].resolve()) if configured else str(base / source_id), "paths": paths, "payload_paths": paths, "path_sha256": hashes,
            "source_sha256": source.sha256, "metadata_retained": True, "requires_apply": True,
            "mixed_source_descendants": mixed, "rebuild_required": bool(rebuild or mixed), "rebuild_nodes": rebuild,
            "weights_unlearning": False}

def apply_deletion(plan: Mapping[str, Any], *, confirm: bool = False, root: str | Path | None = None, fixture_root: str | Path | None = None) -> None:
    if not confirm: raise StoreError("payload deletion requires explicit confirm")
    root = Path(root or fixture_root).resolve() if (root or fixture_root) else None
    if root is None or not plan.get("paths"): raise StoreError("root and planned paths required")
    paths = [Path(x) for x in plan["paths"]]
    metadata_names = {"manifest.json", "registry.json", "lineage.json"}
    for raw_target in paths:
        if raw_target.is_symlink(): raise StoreError("symlink deletion target")
        target = raw_target.resolve()
        if root not in target.parents or target.name in metadata_names or "checkpoint" in target.parts: raise StoreError("unsafe deletion target")
        expected = plan.get("path_sha256", {}).get(str(target))
        if not expected or not target.is_file() or _sha(target.read_bytes()) != expected: raise StoreError("stale deletion hash")
    for item in plan["paths"]:
        Path(item).unlink()

__all__ = ["StoreError", "write_source_records", "iter_source_records", "validate_source_store", "build_lineage", "LineageGraph", "deletion_plan", "apply_deletion"]
