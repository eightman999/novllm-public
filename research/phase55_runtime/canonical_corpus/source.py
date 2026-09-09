"""Strictly non-writing access to quiescent SQLite files.

immutable disables SQLite sidecar writes, so nonempty WAL/journals are refused.
Callers must stop writers; file identity/stat/hash checks detect observed drift.
This does not snapshot a live database and never checkpoints it.
"""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import base64


class CorpusError(ValueError):
    """Input cannot be exported or output does not match the source."""


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def stable_id(kind, *parts):
    return kind + "_" + hashlib.sha256(dumps(parts).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_state(path):
    result = {}
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(path) + suffix)
        try:
            s = p.stat()
        except FileNotFoundError:
            result[suffix] = None
            continue
        if suffix in ("-wal", "-journal") and s.st_size:
            raise CorpusError(f"Nonempty {suffix}: provide a quiescent, checkpointed DB")
        result[suffix] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if result[""] is None:
        raise CorpusError("Source DB does not exist")
    return result


@contextmanager
def read_only_source(path):
    path = Path(path).resolve(strict=True)
    before = file_state(path)
    conn = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("BEGIN")
        yield conn
        if file_state(path) != before:
            raise CorpusError("Source or sidecar changed during read")
    finally:
        conn.close()


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def schema_info(conn):
    schema = [dict(r) for r in conn.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name")]
    tables = [r["name"] for r in schema if r["type"] == "table"]
    return schema, sorted(tables)


def encode_value(value):
    """JSON scalar values are unchanged; BLOB/nonfinite REAL use tagged objects."""
    if isinstance(value, bytes):
        return {"$sqlite_blob_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        return {"$sqlite_real": value.hex()}
    return value


def raw_row(row):
    return {k: encode_value(v) for k, v in dict(row).items()}
