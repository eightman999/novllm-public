# chunker.py
"""Phase5: エピソード本文のチャンク化。
優先境界は 1) episode境界(この関数はエピソード単位で呼ばれるため自動的に満たされる)
2) 空行による段落境界 3) 文末境界 4) 最後の手段としてtoken境界、の順。
文字数からの推測は行わず、必ずtokenizerで実測してtoken数を数える。"""
from __future__ import annotations

import dataclasses
import difflib
import hashlib
import re
from typing import Any

from pipeline.schemas import make_chunk_id

PARA_SEP = "\n\n"

# 空行(連続する改行、間に空白のみを含む場合も許容)による段落境界。
_PARA_SPLIT_RE = re.compile(r"\n[ \t　]*\n[\n \t　]*")

# 文末記号(句点・感嘆符・疑問符・三点リーダ等)+ 直後に続く閉じ括弧/閉じ引用符をひとまとまりの境界とする。
_SENTENCE_BOUNDARY_RE = re.compile(r"([。!?！？♪]+[」』】”’\"')\]]*)")


@dataclasses.dataclass(frozen=True)
class Chunk:
    chunk_index: int
    text: str
    token_count: int
    chunk_id: str


# --- 内部ユーティリティ ---------------------------------------------------

def _token_len(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_paragraphs(text: str) -> list[str]:
    """空行で段落に分割する。段落内部の単一改行はそのまま保持する(行そのものは境界にしない)。"""
    stripped = text.strip("\n")
    if stripped == "":
        return []
    parts = _PARA_SPLIT_RE.split(stripped)
    return [p for p in parts if p != ""]


def _split_sentences(paragraph: str) -> list[str]:
    """文末境界で分割する。区切り記号・直後の閉じ括弧は直前の文に残す。"""
    if paragraph == "":
        return []
    parts = _SENTENCE_BOUNDARY_RE.split(paragraph)
    sentences: list[str] = []
    buf = ""
    for i, part in enumerate(parts):
        if part == "":
            continue
        buf += part
        if i % 2 == 1:  # 奇数indexは区切り記号(+閉じ括弧)側 = 文の終端
            sentences.append(buf)
            buf = ""
    if buf:
        sentences.append(buf)
    return sentences


def _split_by_tokens(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    """文末記号を持たない巨大な文に対する最後の手段: token境界で強制分割する。
    再エンコード時のBPE境界ずれに備えて安全マージンを設ける。"""
    if text == "":
        return []
    safe_max = max(1, max_tokens - 8)
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    if not offsets:
        return [text]
    pieces: list[str] = []
    i = 0
    n = len(offsets)
    while i < n:
        j = min(i + safe_max, n)
        start_char = offsets[i][0]
        end_char = offsets[j - 1][1]
        piece = text[start_char:end_char]
        if piece != "":
            pieces.append(piece)
        i = j
    return pieces if pieces else [text]


def _build_units(text: str, tokenizer: Any, max_tokens: int) -> list[dict[str, Any]]:
    """text全体を、切断してよい最小単位(unit)の列に分解する。
    各unitは {'text', 'sep', 'boundary', 'tokens'} を持つ。
    'boundary' はそのunitの手前にある境界の強さ: 'paragraph' > 'sentence' > 'token'。
    'sep' はチャンク内で前のunitと連結する際に手前へ挿入する文字列。"""
    units: list[dict[str, Any]] = []
    paragraphs = _split_paragraphs(text)
    for para in paragraphs:
        sentences = _split_sentences(para) or [para]
        for s_idx, sent in enumerate(sentences):
            if sent == "":
                continue
            tok = _token_len(tokenizer, sent)
            pieces = _split_by_tokens(sent, tokenizer, max_tokens) if tok > max_tokens else [sent]
            for k, piece in enumerate(pieces):
                if not units:
                    boundary = "paragraph"
                    sep = ""
                elif k == 0 and s_idx == 0:
                    boundary = "paragraph"
                    sep = PARA_SEP
                elif k == 0:
                    boundary = "sentence"
                    sep = ""
                else:
                    boundary = "token"
                    sep = ""
                units.append({
                    "text": piece,
                    "sep": sep,
                    "boundary": boundary,
                    "tokens": _token_len(tokenizer, piece),
                })
    return units


def _render_units(unit_slice: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, u in enumerate(unit_slice):
        parts.append(u["text"] if i == 0 else u["sep"] + u["text"])
    return "".join(parts)


def _accumulate_chunk(units: list[dict[str, Any]], start: int, target_tokens: int, max_tokens: int) -> int:
    """start位置からmax_tokensを超えない範囲でunitを積み、target_tokensに最も近い
    良い境界(paragraph優先、次点sentence)で切る。戻り値はチャンクの終端index(exclusive)。"""
    n = len(units)
    cum = 0
    end = start
    para_candidates: list[tuple[int, int]] = []
    sent_candidates: list[tuple[int, int]] = []
    while end < n:
        nxt = cum + units[end]["tokens"]
        if nxt > max_tokens and end > start:
            break
        cum = nxt
        end += 1
        boundary = units[end]["boundary"] if end < n else "paragraph"
        if boundary == "paragraph":
            para_candidates.append((end, cum))
        elif boundary == "sentence":
            sent_candidates.append((end, cum))

    def pick(cands: list[tuple[int, int]]) -> tuple[int, int] | None:
        if not cands:
            return None
        good = [c for c in cands if c[1] >= target_tokens]
        if good:
            return min(good, key=lambda c: c[1])
        return max(cands, key=lambda c: c[1])

    chosen = pick(para_candidates) or pick(sent_candidates)
    if chosen is not None:
        return chosen[0]
    return max(end, start + 1)


def _backoff_start(units: list[dict[str, Any]], end: int, overlap_tokens: int) -> int:
    if overlap_tokens <= 0:
        return end
    cum = 0
    idx = end
    while idx > 0:
        idx -= 1
        cum += units[idx]["tokens"]
        if cum >= overlap_tokens:
            break
    return idx


def _merge_short_tail(
    units: list[dict[str, Any]],
    boundaries: list[tuple[int, int]],
    tokenizer: Any,
    target_tokens: int,
) -> list[tuple[int, int]]:
    """極端に短い末尾チャンク(目安: 目標の20%未満)を直前チャンクと統合する。"""
    if len(boundaries) < 2:
        return boundaries
    threshold = target_tokens * 0.2
    s, e = boundaries[-1]
    tail_tokens = _token_len(tokenizer, _render_units(units[s:e]))
    if tail_tokens < threshold:
        prev_s, _prev_e = boundaries[-2]
        return boundaries[:-2] + [(prev_s, e)]
    return boundaries


# --- 公開API ---------------------------------------------------------------

def chunk_episode_text(
    ncode: str,
    episode_no: int,
    normalized_text: str,
    tokenizer: Any,
    target_tokens: int = 1536,
    max_tokens: int = 2048,
    overlap_tokens: int = 128,
) -> list[Chunk]:
    if target_tokens <= 0 or max_tokens <= 0:
        raise ValueError("target_tokens/max_tokens must be positive")
    if target_tokens > max_tokens:
        raise ValueError("target_tokens must not exceed max_tokens")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")

    text = normalized_text or ""
    if text.strip() == "":
        return []

    units = _build_units(text, tokenizer, max_tokens)
    if not units:
        return []
    n = len(units)

    boundaries: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = _accumulate_chunk(units, start, target_tokens, max_tokens)
        end = max(end, start + 1)
        end = min(end, n)
        boundaries.append((start, end))
        if end >= n:
            break
        next_start = _backoff_start(units, end, overlap_tokens)
        if next_start <= start:
            next_start = start + 1
        start = next_start

    boundaries = _merge_short_tail(units, boundaries, tokenizer, target_tokens)

    chunks: list[Chunk] = []
    for idx, (s, e) in enumerate(boundaries):
        chunk_text = _render_units(units[s:e])
        token_count = _token_len(tokenizer, chunk_text)
        chunks.append(Chunk(
            chunk_index=idx,
            text=chunk_text,
            token_count=token_count,
            chunk_id=make_chunk_id(ncode, episode_no, idx),
        ))
    return chunks


def compute_text_sha256(text: str) -> str:
    """完全重複検出用のSHA-256ハッシュ。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def is_near_duplicate(text_a: str, text_b: str, threshold: float = 0.9) -> bool:
    """近似重複判定。正規化(空白除去)後の文字5-gram Jaccard類似度と
    difflib.SequenceMatcher比率の大きい方を採用し、閾値以上なら重複とみなす。"""
    norm_a = _normalize_for_dedup(text_a)
    norm_b = _normalize_for_dedup(text_b)
    if norm_a == norm_b:
        return True
    if not norm_a or not norm_b:
        return False

    ngrams_a = _char_ngrams(norm_a)
    ngrams_b = _char_ngrams(norm_b)
    union = ngrams_a | ngrams_b
    jaccard = len(ngrams_a & ngrams_b) / len(union) if union else 0.0

    ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

    similarity = max(jaccard, ratio)
    return similarity >= threshold


def find_duplicate_indices(texts: list[str], threshold: float = 0.9) -> dict[int, int]:
    """textsの中から完全重複(SHA-256一致)・近似重複(is_near_duplicate)を検出する。
    戻り値は {重複側のindex: 最初に現れた側のindex}。呼び出し側はこれを使って
    除外するかフラグ付けするかを選べる(この関数自体はtextsを書き換えない)。"""
    seen_hash: dict[str, int] = {}
    kept: list[int] = []
    duplicates: dict[int, int] = {}
    for i, text in enumerate(texts):
        h = compute_text_sha256(text)
        if h in seen_hash:
            duplicates[i] = seen_hash[h]
            continue
        match = next((j for j in kept if is_near_duplicate(text, texts[j], threshold=threshold)), None)
        if match is not None:
            duplicates[i] = match
            continue
        seen_hash[h] = i
        kept.append(i)
    return duplicates
