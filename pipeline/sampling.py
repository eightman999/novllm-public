# sampling.py
"""Section7: 層化サンプリング。単純な全チャンク均等学習を避けるため、
作品単位・ジャンル単位・作者単位・R18比率・作品規模(チャンク数)による層化を
決定論的に適用する。

各 chunk は以下の形の dict であることを前提とする(最低限のキー):
    {
        "id": str,
        "meta": {
            "ncode": str,
            "author": str,
            "genre": list[str],
            "is_r18": bool,
            "work_chunk_count": int,   # そのchunkが属する作品の総チャンク数
        },
    }
`meta` に他のキーが含まれていてもよい(無視する)。

再現性: random モジュールのグローバル状態には一切依存せず、
random.Random(config.seed) のローカルインスタンスのみを使用する。
同一 seed・同一入力なら、出力の順序・選択結果は常に完全一致する。
"""
from __future__ import annotations

import dataclasses
import logging
import random
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    seed: int
    max_chunks_per_work: int | None = None
    genre_min_max: dict[str, tuple[int, int | None]] | None = None
    max_author_share: float | None = None
    r18_ratio: float | None = None
    length_strata: dict[str, tuple[int, int | None]] | None = None


@dataclasses.dataclass(frozen=True)
class SamplingResult:
    """stratified_sample の戻り値。selected が実際にサンプリングされたchunk一覧。
    それ以外のフィールドは、何がどれだけ切り捨てられたかを可視化するための記録
    (サイレントに消さないための必須情報)。"""

    selected: list[dict[str, Any]]
    dropped_by_max_chunks_per_work: int = 0
    dropped_by_max_author_share: int = 0
    dropped_by_r18_ratio: int = 0
    dropped_by_genre_min_max: int = 0
    total_input: int = 0
    total_selected: int = 0


def _get_genres(chunk: dict[str, Any]) -> list[str]:
    return list(chunk.get("meta", {}).get("genre") or [])


def _get_author(chunk: dict[str, Any]) -> str:
    return chunk.get("meta", {}).get("author", "")


def _get_ncode(chunk: dict[str, Any]) -> str:
    return chunk.get("meta", {}).get("ncode", "")


def _is_r18(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("meta", {}).get("is_r18", False))


def _apply_max_chunks_per_work(
    chunks: list[dict[str, Any]], max_chunks_per_work: int, rng: random.Random
) -> tuple[list[dict[str, Any]], int]:
    by_work: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_work[_get_ncode(c)].append(c)

    kept: list[dict[str, Any]] = []
    dropped = 0
    # 作品(ncode)のイテレーション順を安定させるため sorted を使う
    for ncode in sorted(by_work.keys()):
        work_chunks = by_work[ncode]
        if len(work_chunks) <= max_chunks_per_work:
            kept.extend(work_chunks)
            continue
        # 決定論的に max_chunks_per_work 件をサンプリング(超過分は切り捨て)
        indices = list(range(len(work_chunks)))
        rng.shuffle(indices)
        chosen_indices = sorted(indices[:max_chunks_per_work])
        kept.extend(work_chunks[i] for i in chosen_indices)
        dropped += len(work_chunks) - max_chunks_per_work

    return kept, dropped


def _apply_r18_ratio(
    chunks: list[dict[str, Any]], r18_ratio: float, rng: random.Random
) -> tuple[list[dict[str, Any]], int]:
    if not 0.0 <= r18_ratio <= 1.0:
        raise ValueError(f"r18_ratio must be within [0, 1], got {r18_ratio}")

    r18_chunks = [c for c in chunks if _is_r18(c)]
    non_r18_chunks = [c for c in chunks if not _is_r18(c)]

    total = len(chunks)
    if total == 0:
        return [], 0

    target_r18_count = round(total * r18_ratio)
    target_r18_count = min(target_r18_count, len(r18_chunks))

    # non_r18側の必要件数から全体件数を決める(r18が不足している場合は全体を縮小)
    if r18_ratio > 0:
        target_total = round(target_r18_count / r18_ratio) if target_r18_count > 0 else 0
    else:
        target_total = len(non_r18_chunks)
    target_total = min(target_total, total)
    target_non_r18_count = target_total - target_r18_count
    target_non_r18_count = min(target_non_r18_count, len(non_r18_chunks))

    r18_indices = list(range(len(r18_chunks)))
    rng.shuffle(r18_indices)
    selected_r18 = [r18_chunks[i] for i in sorted(r18_indices[:target_r18_count])]

    non_r18_indices = list(range(len(non_r18_chunks)))
    rng.shuffle(non_r18_indices)
    selected_non_r18 = [non_r18_chunks[i] for i in sorted(non_r18_indices[:target_non_r18_count])]

    kept = selected_r18 + selected_non_r18
    dropped = total - len(kept)
    return kept, dropped


def _apply_max_author_share(
    chunks: list[dict[str, Any]], max_author_share: float, rng: random.Random
) -> tuple[list[dict[str, Any]], int]:
    if not 0.0 < max_author_share <= 1.0:
        raise ValueError(f"max_author_share must be within (0, 1], got {max_author_share}")

    total = len(chunks)
    if total == 0:
        return [], 0

    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_author[_get_author(c)].append(c)

    max_allowed = int(total * max_author_share)
    kept: list[dict[str, Any]] = []
    dropped = 0
    keep_n = max(max_allowed, 0)
    for author in sorted(by_author.keys()):
        author_chunks = by_author[author]
        if len(author_chunks) <= keep_n:
            kept.extend(author_chunks)
            continue
        indices = list(range(len(author_chunks)))
        rng.shuffle(indices)
        chosen_indices = sorted(indices[:keep_n])
        kept.extend(author_chunks[i] for i in chosen_indices)
        dropped += len(author_chunks) - keep_n

    return kept, dropped


def _apply_genre_min_max(
    chunks: list[dict[str, Any]],
    genre_min_max: dict[str, tuple[int, int | None]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    """genreごとの上限(max)のみを切り捨てで適用する。min(下限)はここでは充足を
    保証できない(母集団不足の場合があるため)。不足はログに警告として残す。"""
    by_genre: dict[str, list[int]] = defaultdict(list)  # genre -> chunk indices
    for idx, c in enumerate(chunks):
        for g in _get_genres(c):
            if g in genre_min_max:
                by_genre[g].append(idx)

    drop_indices: set[int] = set()
    for genre, (min_count, max_count) in genre_min_max.items():
        idxs = by_genre.get(genre, [])
        if max_count is not None and len(idxs) > max_count:
            shuffled = list(idxs)
            rng.shuffle(shuffled)
            excess = sorted(shuffled[max_count:])
            drop_indices.update(excess)
        if len(idxs) < min_count:
            logger.warning(
                "genre_min_max: genre=%s の候補チャンク数 %d が下限 %d を満たせません",
                genre,
                len(idxs),
                min_count,
            )

    kept = [c for i, c in enumerate(chunks) if i not in drop_indices]
    return kept, len(drop_indices)


def stratified_sample(chunks: list[dict[str, Any]], config: SamplingConfig) -> SamplingResult:
    """層化サンプリングを適用し、選択されたchunkと切り捨て件数を含む
    SamplingResult を返す。

    同一 chunks・同一 config(同一 seed含む)なら、出力(selected の順序含む)は
    常に完全に再現する。random モジュールのグローバル状態は使用しない。

    適用順序: max_chunks_per_work -> genre_min_max(max) -> max_author_share
    -> r18_ratio。各ステップは直前のステップの出力を入力として受け取る。
    """
    rng = random.Random(config.seed)
    total_input = len(chunks)

    current = list(chunks)

    dropped_max_chunks = 0
    if config.max_chunks_per_work is not None:
        current, dropped_max_chunks = _apply_max_chunks_per_work(
            current, config.max_chunks_per_work, rng
        )

    dropped_genre = 0
    if config.genre_min_max:
        current, dropped_genre = _apply_genre_min_max(current, config.genre_min_max, rng)

    dropped_author = 0
    if config.max_author_share is not None:
        current, dropped_author = _apply_max_author_share(current, config.max_author_share, rng)

    dropped_r18 = 0
    if config.r18_ratio is not None:
        current, dropped_r18 = _apply_r18_ratio(current, config.r18_ratio, rng)

    # 最終的な順序も seed に対して決定論的にシャッフルしておく
    # (層化ステップを経ない場合、元の入力順そのままになるのを避ける)
    final_indices = list(range(len(current)))
    rng.shuffle(final_indices)
    selected = [current[i] for i in final_indices]

    if dropped_max_chunks:
        logger.info("max_chunks_per_work により %d 件切り捨て", dropped_max_chunks)
    if dropped_genre:
        logger.info("genre_min_max(max) により %d 件切り捨て", dropped_genre)
    if dropped_author:
        logger.info("max_author_share により %d 件切り捨て", dropped_author)
    if dropped_r18:
        logger.info("r18_ratio 調整により %d 件切り捨て", dropped_r18)

    return SamplingResult(
        selected=selected,
        dropped_by_max_chunks_per_work=dropped_max_chunks,
        dropped_by_max_author_share=dropped_author,
        dropped_by_r18_ratio=dropped_r18,
        dropped_by_genre_min_max=dropped_genre,
        total_input=total_input,
        total_selected=len(selected),
    )
