# test_sampling.py
"""Section7 stratified_sample のテスト。"""
from __future__ import annotations

import random

from pipeline.sampling import SamplingConfig, stratified_sample


def _make_chunk(cid: str, ncode: str, author: str, genre: list[str], is_r18: bool, work_chunk_count: int) -> dict:
    return {
        "id": cid,
        "meta": {
            "ncode": ncode,
            "author": author,
            "genre": genre,
            "is_r18": is_r18,
            "work_chunk_count": work_chunk_count,
        },
    }


def _build_sample_chunks() -> list[dict]:
    chunks = []
    # 作品A(著者X, ファンタジー, 非R18): 10チャンク
    for i in range(10):
        chunks.append(_make_chunk(f"A-{i}", "nA", "authorX", ["fantasy"], False, 10))
    # 作品B(著者X, 恋愛, 非R18): 8チャンク (同一著者Xの2作目)
    for i in range(8):
        chunks.append(_make_chunk(f"B-{i}", "nB", "authorX", ["romance"], False, 8))
    # 作品C(著者Y, ファンタジー, R18): 15チャンク
    for i in range(15):
        chunks.append(_make_chunk(f"C-{i}", "nC", "authorY", ["fantasy"], True, 15))
    # 作品D(著者Z, SF, 非R18): 5チャンク
    for i in range(5):
        chunks.append(_make_chunk(f"D-{i}", "nD", "authorZ", ["sf"], False, 5))
    return chunks


def test_reproducibility_same_seed_same_input():
    chunks = _build_sample_chunks()
    config = SamplingConfig(
        seed=42,
        max_chunks_per_work=6,
        max_author_share=0.5,
        r18_ratio=0.3,
    )
    result1 = stratified_sample(chunks, config)
    result2 = stratified_sample(chunks, config)

    ids1 = [c["id"] for c in result1.selected]
    ids2 = [c["id"] for c in result2.selected]
    assert ids1 == ids2
    assert result1.dropped_by_max_chunks_per_work == result2.dropped_by_max_chunks_per_work
    assert result1.dropped_by_max_author_share == result2.dropped_by_max_author_share
    assert result1.dropped_by_r18_ratio == result2.dropped_by_r18_ratio


def test_reproducibility_independent_of_global_random_state():
    chunks = _build_sample_chunks()
    config = SamplingConfig(seed=7, max_chunks_per_work=4)

    random.seed(1)
    result1 = stratified_sample(chunks, config)
    random.seed(999999)
    for _ in range(50):
        random.random()
    result2 = stratified_sample(chunks, config)

    ids1 = [c["id"] for c in result1.selected]
    ids2 = [c["id"] for c in result2.selected]
    assert ids1 == ids2


def test_max_chunks_per_work_applied():
    chunks = _build_sample_chunks()
    config = SamplingConfig(seed=1, max_chunks_per_work=6)
    result = stratified_sample(chunks, config)

    counts_by_work: dict[str, int] = {}
    for c in result.selected:
        ncode = c["meta"]["ncode"]
        counts_by_work[ncode] = counts_by_work.get(ncode, 0) + 1

    for ncode, count in counts_by_work.items():
        assert count <= 6, f"{ncode} has {count} chunks, exceeds max_chunks_per_work=6"

    # A:10->6(4切捨て), B:8->6(2切捨て), C:15->6(9切捨て), D:5->5(0切捨て)
    assert result.dropped_by_max_chunks_per_work == 4 + 2 + 9 + 0
    assert counts_by_work["nD"] == 5  # 上限未満の作品はそのまま残る


def test_max_chunks_per_work_none_means_no_limit():
    chunks = _build_sample_chunks()
    config = SamplingConfig(seed=1, max_chunks_per_work=None)
    result = stratified_sample(chunks, config)
    assert result.total_selected == len(chunks)
    assert result.dropped_by_max_chunks_per_work == 0


def test_r18_ratio_close_to_target():
    chunks = _build_sample_chunks()
    config = SamplingConfig(seed=123, r18_ratio=0.3)
    result = stratified_sample(chunks, config)

    total = len(result.selected)
    r18_count = sum(1 for c in result.selected if c["meta"]["is_r18"])
    assert total > 0
    actual_ratio = r18_count / total
    assert abs(actual_ratio - 0.3) < 0.05


def test_max_author_share_applied():
    chunks = _build_sample_chunks()
    config = SamplingConfig(seed=5, max_author_share=0.4)
    result = stratified_sample(chunks, config)

    counts_by_author: dict[str, int] = {}
    for c in result.selected:
        author = c["meta"]["author"]
        counts_by_author[author] = counts_by_author.get(author, 0) + 1

    # max_author_share は、このステップに入力された件数(=このテストでは
    # フィルタが他にないため result.total_input)を基準に上限を計算する。
    max_allowed = int(result.total_input * 0.4)
    for author, count in counts_by_author.items():
        assert count <= max_allowed, f"{author}: {count} > max_allowed {max_allowed}"
    assert result.dropped_by_max_author_share >= 0


def test_dropped_counts_are_reported_not_silent():
    chunks = _build_sample_chunks()
    config = SamplingConfig(seed=1, max_chunks_per_work=3)
    result = stratified_sample(chunks, config)

    assert result.total_input == len(chunks)
    assert result.total_selected == len(result.selected)
    assert result.dropped_by_max_chunks_per_work > 0
    # 入力件数 = 選択件数 + 切り捨て件数(このテストではmax_chunks_per_workのみ適用)
    assert result.total_input - result.dropped_by_max_chunks_per_work == result.total_selected


def test_empty_input_returns_empty_result():
    config = SamplingConfig(seed=1, max_chunks_per_work=5, r18_ratio=0.2, max_author_share=0.5)
    result = stratified_sample([], config)
    assert result.selected == []
    assert result.total_input == 0
    assert result.total_selected == 0


def test_genre_min_max_upper_bound_applied():
    chunks = _build_sample_chunks()
    config = SamplingConfig(seed=1, genre_min_max={"fantasy": (0, 5)})
    result = stratified_sample(chunks, config)

    fantasy_count = sum(1 for c in result.selected if "fantasy" in c["meta"]["genre"])
    assert fantasy_count <= 5
    assert result.dropped_by_genre_min_max > 0
