"""Deterministic, source-accounted datasets for the Phase 4 probe LM.

The dataset deliberately keeps episode boundaries. Packing pads final partial
blocks and masks their padding, so validation and source accounting do not
drop tokens. ``source_chars``/``source_bytes`` refer to original source text,
never to tokenizer decoded text.
"""
from __future__ import annotations

import hashlib
import json
import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


def token_source_prefixes(adapter: Any, text: str) -> list[dict[str, Any]]:
    """Map reversible SentencePiece proto pieces to exact source prefixes.

    Wrapper escapes are treated as indivisible atoms (CRLF and its ``ESC R``
    representation are one atom). A piece ending inside an atom is marked
    ``complete=False`` and its accounting reports the preceding complete source
    prefix. This is suitable for same-token-budget partial-chunk accounting.
    """
    processor = getattr(adapter, "processor", None)
    escape = getattr(adapter, "_escape", None)
    if processor is None or not callable(escape):
        raise TypeError("source offsets require a reversible SentencePiece adapter")
    escaped = escape(text)
    atoms: list[tuple[int, int, int]] = []
    source_chars = source_bytes = escaped_bytes = 0
    i = 0
    esc = getattr(adapter, "ESC", "\ue000")
    while i < len(text):
        if text.startswith("\r\n", i): atom = "\r\n"; i += 2
        else: atom = text[i]; i += 1
        encoded_atom = escape(atom)
        escaped_bytes += len(encoded_atom.encode("utf-8"))
        source_chars += len(atom); source_bytes += len(atom.encode("utf-8"))
        atoms.append((escaped_bytes, source_chars, source_bytes))
    if escaped_bytes != len(escaped.encode("utf-8")):
        raise ValueError("escaped atom byte accounting mismatch")
    proto = processor.encode(escaped, out_type="proto")
    pieces = list(getattr(proto, "pieces", proto))
    result = []
    ends = [x[0] for x in atoms]
    for index, piece in enumerate(pieces):
        end = int(piece.end)
        k = bisect.bisect_right(ends, end) - 1
        boundary_complete = end == 0 or (k >= 0 and ends[k] == end)
        # SentencePiece byte-fallback pieces can end at the preceding atom's
        # boundary while still representing only part of the next UTF-8 atom.
        # Preserve that distinction for exact partial-token accounting.
        try:
            is_byte = bool(processor.is_byte(int(piece.id)))
        except (AttributeError, TypeError, ValueError):
            is_byte = str(piece.piece).startswith("<0x")
        partial_byte = is_byte and not getattr(piece, "surface", "")
        complete = boundary_complete and not partial_byte
        if k < 0:
            chars = bytes_ = 0
        else:
            chars, bytes_ = atoms[k][1], atoms[k][2]
        result.append({"token_index": index, "piece": piece.piece, "begin_byte": int(piece.begin),
                       "end_byte": end, "source_chars": chars, "source_bytes": bytes_,
                       "complete": complete, "partial_atom": None if complete else k + 1})
    if pieces and int(pieces[-1].end) != len(escaped.encode("utf-8")):
        raise ValueError("SentencePiece proto does not cover escaped input")
    return result


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class TokenizedEpisode:
    episode_id: str
    text: str
    ids: tuple[int, ...]
    source_chars: int
    source_bytes: int


@dataclass(frozen=True)
class PackedBatch:
    input_ids: tuple[tuple[int, ...], ...]
    labels: tuple[tuple[int, ...], ...]
    loss_mask: tuple[tuple[bool, ...], ...]
    source_chars: int
    source_bytes: int
    token_count: int


class ProbeLMDataset:
    """Stream corpus records, preserving episode and source accounting."""

    def __init__(self, tokenizer: Any, records: Iterable[dict[str, Any]], *, bos_id: int,
                 eos_id: int, pad_id: int | None = None):
        self.tokenizer = tokenizer
        self.records = records
        self.bos_id, self.eos_id = int(bos_id), int(eos_id)
        self.pad_id = int(eos_id if pad_id is None else pad_id)

    @staticmethod
    def read_jsonl(directory: str | Path, *, split: str = "train") -> Iterator[dict[str, Any]]:
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        entries = [x for x in manifest.get("shards", []) if x.get("kind") in (split, None) and x["path"].startswith(split + "-")]
        for entry in sorted(entries, key=lambda x: x["path"]):
            path = root / entry["path"]
            digest = hashlib.sha256()
            rows = 0
            with path.open("rb") as fh:
                for line in fh:
                    digest.update(line); rows += 1
                    row = json.loads(line.decode("utf-8"))
                    if row.get("record_type") != "tokenizer_episode" or not isinstance(row.get("text"), str):
                        raise ValueError(f"invalid tokenizer episode: {path}")
                    yield row
            if rows != entry.get("rows") or digest.hexdigest() != entry.get("sha256"):
                raise ValueError(f"corpus shard integrity failure: {path}")

    def episodes(self) -> Iterator[TokenizedEpisode]:
        for n, row in enumerate(self.records):
            text = row["text"]
            encoded = self.tokenizer.encode(text)
            ids = tuple(int(x) for x in getattr(encoded, "ids", encoded))
            eid = str(row.get("boundary", {}).get("episode_revision_id", row.get("episode_revision_id", n)))
            # BOS/EOS are model framing tokens, so source accounting excludes them.
            yield TokenizedEpisode(eid, text, ids, len(text), len(text.encode("utf-8")))

    def source_totals(self) -> tuple[int, int]:
        """Return exact original character/UTF-8 byte totals for this stream.

        Packed blocks intentionally do not infer byte offsets from token counts;
        callers must use this episode-level total for source throughput metrics.
        """
        chars = bytes_ = 0
        for ep in self.episodes():
            chars += ep.source_chars; bytes_ += ep.source_bytes
        return chars, bytes_

    def packed(self, *, sequence_length: int, batch_size: int = 1,
               add_bos_eos: bool = True) -> Iterator[PackedBatch]:
        if sequence_length < 2 or batch_size < 1:
            raise ValueError("sequence_length >= 2 and batch_size >= 1 required")
        blocks: list[tuple[list[int], int, int, int]] = []
        for ep in self.episodes():
            ids = ([self.bos_id] if add_bos_eos else []) + list(ep.ids) + ([self.eos_id] if add_bos_eos else [])
            # Episode boundaries are explicit; no cross-episode text is joined.
            for off in range(0, len(ids) - 1, sequence_length):
                chunk = ids[off:off + sequence_length + 1]
                actual = len(chunk) - 1
                # Pad the final partial chunk.  Its loss mask excludes padding,
                # while source totals are attached once, on the episode's last
                # chunk, so totals remain exact without token->byte guessing.
                padded = chunk + [self.pad_id] * (sequence_length + 1 - len(chunk))
                is_last = off + sequence_length + 1 >= len(ids)
                blocks.append((padded, actual, ep.source_chars if is_last else 0, ep.source_bytes if is_last else 0))
                if len(blocks) == batch_size:
                    yield self._batch(blocks, sequence_length)
                    blocks.clear()
        if blocks:
            yield self._batch(blocks, sequence_length)

    def _batch(self, blocks: list[tuple[list[int], int, int, int]], length: int) -> PackedBatch:
        ins = tuple(tuple(x[:-1]) for x, _, _, _ in blocks)
        labs = tuple(tuple(x[1:]) for x, _, _, _ in blocks)
        masks = tuple(tuple(i < n for i in range(length)) for _, n, _, _ in blocks)
        return PackedBatch(ins, labs, masks, sum(c for _, _, c, _ in blocks), sum(b for _, _, _, b in blocks), sum(n for _, n, _, _ in blocks))


def load_records(directory: str | Path, split: str) -> list[dict[str, Any]]:
    return list(ProbeLMDataset.read_jsonl(directory, split=split))
