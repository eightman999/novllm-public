"""Rights-aware registry for Phase 4.5 cultural sources.

The registry contains metadata only.  Payloads are addressed by hash and are
never embedded in this file.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

SOURCE_TYPES = {"web_novel", "aozora", "wikisource", "waka", "classical", "kanbun", "jhpt", "gyaru", "aa", "technical"}
TRAINING_BASIS = {"public_domain", "open_license", "JP_Copyright_Act_30_4", "permission", "unknown"}
REDISTRIBUTION = {"allowed", "allowed_with_conditions", "not_allowed", "unknown"}
STORAGE_POLICY = {"permanent", "private", "licensed_local", "train_only", "ephemeral", "quarantine"}
COPYRIGHT_STATUS = {"public_domain", "copyrighted", "unknown", "mixed", "permission"}

class RegistryError(ValueError):
    pass

def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

@dataclass(frozen=True)
class CulturalSource:
    source_id: str
    source_name: str
    source_type: str
    origin_url: str
    revision: str
    retrieved_at: str | None
    sha256: str | None
    copyright_status: str
    license: str
    training_basis: str
    redistribution: str
    git_commit_allowed: bool
    storage_policy: str
    citation: str
    domain: str
    language_period: str
    orthography: str
    payload_path: str | None = None
    extensions: dict[str, Any] | None = None

    def __post_init__(self):
        if not self.source_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.source_id) or self.source_id in {".", ".."} or "\x00" in self.source_id:
            raise RegistryError("source_id must be a non-empty path-safe identifier")
        if self.source_type not in SOURCE_TYPES: raise RegistryError("unknown source_type")
        if self.training_basis not in TRAINING_BASIS: raise RegistryError("unknown training_basis")
        if self.redistribution not in REDISTRIBUTION: raise RegistryError("unknown redistribution")
        if self.storage_policy not in STORAGE_POLICY: raise RegistryError("unknown storage_policy")
        if self.copyright_status not in COPYRIGHT_STATUS: raise RegistryError("unknown copyright_status")
        if not isinstance(self.git_commit_allowed, bool): raise RegistryError("git_commit_allowed must be boolean")
        if self.sha256 is not None and (len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256.lower())):
            raise RegistryError("sha256 must be a hexadecimal SHA-256")
        if self.payload_path is not None:
            p = Path(self.payload_path)
            if p.is_absolute() or ".." in p.parts or "\\" in self.payload_path:
                raise RegistryError("payload_path must be a relative POSIX path without traversal")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CulturalSource":
        names = {f.name for f in fields(cls)}
        optional = {"payload_path", "extensions"}
        missing = names - set(value) - optional
        if missing: raise RegistryError("missing source fields: " + ",".join(sorted(missing)))
        unknown = set(value) - names
        if unknown:
            ext = dict(value.get("extensions") or {})
            ext.update({k: value[k] for k in unknown})
        else: ext = value.get("extensions")
        data = {k: value[k] for k in names if k in value and k != "extensions"}
        data["extensions"] = ext
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def unknown(cls, source_id: str, source_name: str, source_type: str, *, sha256: str) -> "CulturalSource":
        return cls(source_id, source_name, source_type, "", "", _now(), sha256,
                   "unknown", "unknown", "unknown", "unknown", False, "quarantine", "", "", "", "", None, {})

class CulturalRegistry:
    def __init__(self, sources: Iterable[CulturalSource] = ()):
        self._sources: dict[str, CulturalSource] = {}
        for source in sources: self.add(source)

    def add(self, source: CulturalSource) -> None:
        if source.source_id in self._sources: raise RegistryError("duplicate source_id: " + source.source_id)
        self._sources[source.source_id] = source

    def get(self, source_id: str) -> CulturalSource:
        try: return self._sources[source_id]
        except KeyError as exc: raise RegistryError("unknown source_id: " + source_id) from exc

    def __iter__(self): return iter(self._sources.values())
    def __len__(self): return len(self._sources)
    def source_ids(self): return tuple(self._sources)
    def eligible(self, source_id: str) -> bool:
        s = self.get(source_id)
        return s.training_basis != "unknown" and s.redistribution != "unknown" and s.storage_policy != "quarantine"

    def dump(self, path: str | Path) -> None:
        data = {"format": "novllm-cultural-source-registry", "schema_version": "1.0.0",
                "sources": [s.to_dict() for s in sorted(self, key=lambda x: x.source_id)]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CulturalRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "novllm-cultural-source-registry" or data.get("schema_version") != "1.0.0": raise RegistryError("invalid registry format/schema version")
        return cls(CulturalSource.from_dict(x) for x in data.get("sources", []))

def sha256_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()

__all__ = ["CulturalSource", "CulturalRegistry", "RegistryError", "SOURCE_TYPES", "TRAINING_BASIS", "REDISTRIBUTION", "STORAGE_POLICY", "COPYRIGHT_STATUS", "sha256_bytes"]
