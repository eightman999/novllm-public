# test_normalize_meta.py
"""Phase2 normalize_meta.py のテスト。"""
from __future__ import annotations

from pipeline.normalize_meta import (
    CONTROL_FLAG_NAMES,
    determine_r18,
    extract_control_flags,
    normalize_keywords,
    normalize_whitespace,
    parse_source_keywords,
)


# --- normalize_whitespace / parse_source_keywords ------------------------------

def test_normalize_whitespace_collapses_runs_and_strips():
    assert normalize_whitespace("  a   b　　c  ") == "a b c"
    assert normalize_whitespace("") == ""
    assert normalize_whitespace(None) == ""  # type: ignore[arg-type]


def test_parse_source_keywords_matches_real_db_shape():
    # novel_database_backup の実データ形式(先頭スペース・複数タグ)を模した例
    main_tag = "ファンタジー"
    sub_tag = " R15 残酷な描写あり 異世界転生 男主人公 "
    result = parse_source_keywords(main_tag, sub_tag)
    assert result == ["ファンタジー", "R15", "残酷な描写あり", "異世界転生", "男主人公"]


def test_parse_source_keywords_preserves_original_tokens_unnormalized():
    # 表記揺れをそのまま残す(正規化はしない) = 元のまま保持されることの検証
    result = parse_source_keywords("恋愛", "BL gl ts")
    assert result == ["恋愛", "BL", "gl", "ts"]


def test_parse_source_keywords_empty_tags_returns_empty_list():
    assert parse_source_keywords("", "") == []
    assert parse_source_keywords(None, None) == []  # type: ignore[arg-type]


# --- normalize_keywords ----------------------------------------------------------

def test_normalize_keywords_unifies_notation_variants():
    source = ["BL", "gl", "ts", "Ｒ１５", "military", "mystery"]
    result = normalize_keywords(source)
    assert result == ["ボーイズラブ", "ガールズラブ", "TS", "R15", "ミリタリー", "ミステリー"]


def test_normalize_keywords_dedupes_while_preserving_first_occurrence_order():
    source = ["BL", "ボーイズラブ", "bl", "現代", "現代もの", "現代"]
    result = normalize_keywords(source)
    assert result == ["ボーイズラブ", "現代"]


def test_normalize_keywords_keeps_unmapped_tokens_as_is():
    source = ["ファンタジー", "追放", "ざまぁ"]
    result = normalize_keywords(source)
    assert result == ["ファンタジー", "追放", "ざまぁ"]


def test_normalize_keywords_empty_input_returns_empty_list():
    assert normalize_keywords([]) == []


# --- extract_control_flags -------------------------------------------------------

def test_extract_control_flags_true_for_present_tags_only():
    normalized = ["R15", "異世界転生", "男主人公"]
    flags = extract_control_flags(normalized)
    assert set(flags.keys()) == set(CONTROL_FLAG_NAMES)
    assert flags["R15"] is True
    assert flags["異世界転生"] is True
    assert flags["男主人公"] is True
    # それ以外は全てFalse
    for name in CONTROL_FLAG_NAMES:
        if name not in ("R15", "異世界転生", "男主人公"):
            assert flags[name] is False


def test_extract_control_flags_no_tags_all_false():
    flags = extract_control_flags([])
    assert set(flags.keys()) == set(CONTROL_FLAG_NAMES)
    assert all(value is False for value in flags.values())


def test_extract_control_flags_end_to_end_from_raw_variants():
    # main_tag/sub_tag -> source_keywords -> normalize_keywords -> extract_control_flags
    source_keywords = parse_source_keywords("ファンタジー", " R15 異世界転移 男主人公 ハーレム ")
    normalized = normalize_keywords(source_keywords)
    flags = extract_control_flags(normalized)
    assert flags["R15"] is True
    assert flags["異世界転移"] is True
    assert flags["男主人公"] is True
    assert flags["ハーレム"] is True
    assert flags["女主人公"] is False
    assert flags["ボーイズラブ"] is False


# --- determine_r18: 優先順位3パターン --------------------------------------------

def test_determine_r18_priority_sub_site_wins_first():
    # sub_site==2 が最優先。ratingやurl_entityが矛盾していても sub_site を採用
    is_r18, source = determine_r18(sub_site=2, rating=2, url_entity_is_r18=0)
    assert (is_r18, source) == (True, "sub_site")


def test_determine_r18_priority_rating_used_when_sub_site_not_2():
    is_r18, source = determine_r18(sub_site=1, rating=1, url_entity_is_r18=0)
    assert (is_r18, source) == (True, "rating")


def test_determine_r18_priority_url_entity_used_as_last_resort():
    is_r18, source = determine_r18(sub_site=1, rating=2, url_entity_is_r18=1)
    assert (is_r18, source) == (True, "url_entity")


def test_determine_r18_none_when_no_signal():
    is_r18, source = determine_r18(sub_site=1, rating=2, url_entity_is_r18=0)
    assert (is_r18, source) == (False, "none")

    is_r18, source = determine_r18(sub_site=0, rating=5, url_entity_is_r18=None)
    assert (is_r18, source) == (False, "none")


def test_determine_r18_matches_real_db_known_case():
    # sub_siteとratingがともにR18を示すケースでも、優先順位はsub_siteになる。
    is_r18, source = determine_r18(sub_site=2, rating=1, url_entity_is_r18=None)
    assert (is_r18, source) == (True, "sub_site")
