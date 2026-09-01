# test_chunker.py
"""Phase5 chunker.py のテスト。実際にQwen3-8B-Baseのtokenizerをロードして検証する。"""
from __future__ import annotations

import os
import re

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pipeline.chunker import (
    chunk_episode_text,
    compute_text_sha256,
    find_duplicate_indices,
    is_near_duplicate,
)

NCODE = "n0000aa"
EPISODE_NO = 100


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B-Base", use_fast=True, trust_remote_code=True
    )


def _tok_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# --- 短いepisode: 1チャンクに収まる ----------------------------------------

def test_short_episode_single_chunk(tokenizer):
    text = (
        "　小さな村の片隅で、彼女はひとり空を見上げていた。\n"
        "「今日もいい天気ね」\n"
        "\n"
        "そう呟くと、彼女は歩き出した。目的地はまだ遠い。\n"
        "だが、焦る必要はない。時間はたっぷりあるのだから。"
    )
    chunks = chunk_episode_text(NCODE, EPISODE_NO, text, tokenizer)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_index == 0
    assert chunk.chunk_id == f"{NCODE}:e{EPISODE_NO}:c0"
    assert chunk.token_count == _tok_len(tokenizer, chunk.text)
    assert chunk.token_count > 0
    assert chunk.token_count <= 2048
    # 内容が失われていないか(代表的な断片が残っているか)確認
    assert "小さな村の片隅で" in chunk.text
    assert "時間はたっぷりある" in chunk.text


# --- 長いepisode: 複数チャンクに分割され、overlapが効く --------------------

def test_long_episode_splits_with_overlap(tokenizer):
    n_paragraphs = 40
    paragraphs = [
        f"第{i}段落。これは内容{i}についての説明の文です。テスト用の文章{i}をここに書きます。"
        for i in range(n_paragraphs)
    ]
    text = "\n\n".join(paragraphs)

    target_tokens = 150
    max_tokens = 200
    overlap_tokens = 50

    chunks = chunk_episode_text(
        NCODE, EPISODE_NO, text, tokenizer,
        target_tokens=target_tokens, max_tokens=max_tokens, overlap_tokens=overlap_tokens,
    )

    assert len(chunks) >= 3

    # chunk_index / chunk_id が連番になっている
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.chunk_id == f"{NCODE}:e{EPISODE_NO}:c{idx}"
        assert c.token_count == _tok_len(tokenizer, c.text)

    # 末尾統合が発生していない限り、各チャンクはmax_tokensを大きく超えない
    for c in chunks[:-1]:
        assert c.token_count <= max_tokens

    # overlapが機能していること: 隣接チャンク間で段落マーカーが重複出現する
    marker_re = re.compile(r"第(\d+)段落")
    overlap_found = False
    for i in range(len(chunks) - 1):
        markers_a = set(marker_re.findall(chunks[i].text))
        markers_b = set(marker_re.findall(chunks[i + 1].text))
        if markers_a & markers_b:
            overlap_found = True
            break
    assert overlap_found, "隣接チャンク間でoverlapによる段落の重複が見つからない"

    # 全段落マーカーが少なくとも1回はどこかのチャンクに出現している(内容欠落がない)
    all_text = "".join(c.text for c in chunks)
    all_markers = set(marker_re.findall(all_text))
    assert all_markers == {str(i) for i in range(n_paragraphs)}


# --- 極端に短い末尾チャンクが直前チャンクと統合される -----------------------

def test_short_tail_chunk_is_merged(tokenizer):
    filler_sentence = "これはテストのための文章です。"
    tail_sentence = "うん。"

    filler_tokens = _tok_len(tokenizer, filler_sentence)
    tail_tokens = _tok_len(tokenizer, tail_sentence)

    # filler_sentenceをk回繰り返した段落のtoken数がp1_tokensになるようにkを決める
    k = 20
    paragraph1 = filler_sentence * k
    p1_tokens = _tok_len(tokenizer, paragraph1)

    # tail_sentenceはp1_tokensの20%未満であることを前提として組み立てる
    assert tail_tokens < p1_tokens * 0.2

    text = paragraph1 + "\n\n" + tail_sentence

    target_tokens = p1_tokens
    # max_tokensはparagraph1全体は収まるが、tail_sentenceまでは収まらない値にする
    max_tokens = p1_tokens + tail_tokens - 1

    chunks = chunk_episode_text(
        NCODE, EPISODE_NO, text, tokenizer,
        target_tokens=target_tokens, max_tokens=max_tokens, overlap_tokens=0,
    )

    # 分割された場合でも、短い末尾は直前チャンクへ統合され、
    # 結果として全文が1チャンックにまとまっているはず
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].token_count == _tok_len(tokenizer, text)


# --- is_near_duplicate ------------------------------------------------------

def test_is_near_duplicate_exact_match():
    text = "これは同じ文章です。テストのために書きました。"
    assert is_near_duplicate(text, text) is True


def test_is_near_duplicate_completely_different():
    a = "これは全く関係のない文章です。空の話をしています。"
    b = "宇宙船が爆発して、主人公は絶望した。誰も助けに来ない。"
    assert is_near_duplicate(a, b, threshold=0.9) is False


def test_is_near_duplicate_minor_edit():
    a = "彼女はゆっくりと扉を開けて、部屋の中に入っていった。誰もいない静かな部屋だった。"
    b = "彼女はゆっくりと扉を開けて、部屋の中に入っていった。誰もいない静かな部屋であった。"
    assert is_near_duplicate(a, b, threshold=0.8) is True


def test_compute_text_sha256_stable_and_sensitive():
    a = "同じ文章"
    b = "同じ文章"
    c = "違う文章"
    assert compute_text_sha256(a) == compute_text_sha256(b)
    assert compute_text_sha256(a) != compute_text_sha256(c)


def test_find_duplicate_indices():
    texts = [
        "これはオリジナルの文章です。ここに内容が書かれています。",
        "これは別の文章です。全く違う内容が書かれています。",
        "これはオリジナルの文章です。ここに内容が書かれています。",  # 完全重複(index 0)
        "これはオリジナルの文章です。ここに内容が書かれています!",  # 近似重複(index 0)
    ]
    duplicates = find_duplicate_indices(texts, threshold=0.9)
    assert duplicates.get(2) == 0
    assert duplicates.get(3) == 0
    assert 1 not in duplicates
