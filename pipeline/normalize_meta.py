# normalize_meta.py
"""Phase2: メタデータ正規化。
novels_descs.main_tag / sub_tag を結合してsource_keywordsとして扱う(指示書Phase2そのまま)。
main_tagを正式ジャンルとして使わない(main_tag/sub_tagはいずれも「ユーザーが付けたタグ列」として等価に扱う)。

このモジュールは pipeline/schemas.py を参照してよいが変更はしない。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

# --- 空白正規化 ----------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """連続する空白(全角スペース\\u3000含む)を単一の半角スペースに畳み込み、前後を trim する。"""
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def parse_source_keywords(main_tag: str, sub_tag: str) -> list[str]:
    """novels_descs.main_tag と sub_tag を結合し、空白区切りでトークン化する(指示書Phase2の式そのまま)。
    raw_keywords = normalize_whitespace(f"{main_tag} {sub_tag}").split()
    ここでは表記揺れの正規化・重複除去は行わない(元のトークンをそのまま返す = normalize_keywordsの入力用)。
    """
    raw_keywords = normalize_whitespace(f"{main_tag or ''} {sub_tag or ''}").split()
    return raw_keywords


# --- 表記揺れ正規化マップ --------------------------------------------------------
# 正規タグ(値側)は22項目の指示書列挙に基づく control flag 名と一致させる。
# キー側(表記揺れ)は NFKC 正規化 + casefold 済みの状態で照合する(下の _build_lookup 参照)。

NORMALIZED_TAG_MAP: dict[str, str] = {
    # R15
    "R15": "R15",
    "Ｒ１５": "R15",
    "R-15": "R15",
    "r15": "R15",
    # 残酷な描写あり
    "残酷な描写あり": "残酷な描写あり",
    "残酷描写あり": "残酷な描写あり",
    "残酷な描写有り": "残酷な描写あり",
    "残酷シーンあり": "残酷な描写あり",
    # ボーイズラブ
    "ボーイズラブ": "ボーイズラブ",
    "BL": "ボーイズラブ",
    "bl": "ボーイズラブ",
    "Boys Love": "ボーイズラブ",
    "ボーイズラヴ": "ボーイズラブ",
    # ガールズラブ
    "ガールズラブ": "ガールズラブ",
    "GL": "ガールズラブ",
    "gl": "ガールズラブ",
    "Girls Love": "ガールズラブ",
    "百合": "ガールズラブ",
    # 異世界転生
    "異世界転生": "異世界転生",
    "異世界へ転生": "異世界転生",
    "異世界転生もの": "異世界転生",
    "いせかい転生": "異世界転生",
    # 異世界転移
    "異世界転移": "異世界転移",
    "異世界へ転移": "異世界転移",
    "異世界転移もの": "異世界転移",
    "いせかい転移": "異世界転移",
    # 男主人公
    "男主人公": "男主人公",
    "男性主人公": "男主人公",
    "主人公は男性": "男主人公",
    "男性が主人公": "男主人公",
    # 女主人公
    "女主人公": "女主人公",
    "女性主人公": "女主人公",
    "主人公は女性": "女主人公",
    "女性が主人公": "女主人公",
    # 群像劇
    "群像劇": "群像劇",
    "群像劇もの": "群像劇",
    "群像劇形式": "群像劇",
    # オリジナル戦記
    "オリジナル戦記": "オリジナル戦記",
    "オリジナル戦記もの": "オリジナル戦記",
    "オリジナル戦記モノ": "オリジナル戦記",
    # 架空戦記
    "架空戦記": "架空戦記",
    "架空戦記もの": "架空戦記",
    "架空戦記モノ": "架空戦記",
    # IF戦記
    "IF戦記": "IF戦記",
    "if戦記": "IF戦記",
    "ＩＦ戦記": "IF戦記",
    "IF戦記もの": "IF戦記",
    # ミリタリー
    "ミリタリー": "ミリタリー",
    "ミリタリ": "ミリタリー",
    "military": "ミリタリー",
    "Military": "ミリタリー",
    # 戦争
    "戦争": "戦争",
    "戦争もの": "戦争",
    "war": "戦争",
    "War": "戦争",
    # TS
    "TS": "TS",
    "ts": "TS",
    "ＴＳ": "TS",
    "性転換": "TS",
    "ティーエス": "TS",
    # ハーレム
    "ハーレム": "ハーレム",
    "はーれむ": "ハーレム",
    "ハーレムもの": "ハーレム",
    "ハーレム系": "ハーレム",
    # 恋愛
    "恋愛": "恋愛",
    "恋愛もの": "恋愛",
    "ラブロマンス": "恋愛",
    "れんあい": "恋愛",
    # 学園
    "学園": "学園",
    "学園もの": "学園",
    "学校": "学園",
    "学園系": "学園",
    # 現代
    "現代": "現代",
    "現代もの": "現代",
    "現代日本": "現代",
    "現代社会": "現代",
    # 歴史
    "歴史": "歴史",
    "歴史もの": "歴史",
    "史実": "歴史",
    "歴史系": "歴史",
    # SF
    "SF": "SF",
    "sf": "SF",
    "ＳＦ": "SF",
    "サイエンスフィクション": "SF",
    # ホラー
    "ホラー": "ホラー",
    "ほらー": "ホラー",
    "ホラーもの": "ホラー",
    "ホラー系": "ホラー",
    # ミステリー
    "ミステリー": "ミステリー",
    "ミステリ": "ミステリー",
    "推理": "ミステリー",
    "mystery": "ミステリー",
    "Mystery": "ミステリー",
}

# 22項目のcontrol flag名(extract_control_flagsが返すキー集合)。
# NORMALIZED_TAG_MAPの値側(正規タグ)と完全一致させる。
CONTROL_FLAG_NAMES: tuple[str, ...] = (
    "R15",
    "残酷な描写あり",
    "ボーイズラブ",
    "ガールズラブ",
    "異世界転生",
    "異世界転移",
    "男主人公",
    "女主人公",
    "群像劇",
    "オリジナル戦記",
    "架空戦記",
    "IF戦記",
    "ミリタリー",
    "戦争",
    "TS",
    "ハーレム",
    "恋愛",
    "学園",
    "現代",
    "歴史",
    "SF",
    "ホラー",
    "ミステリー",
)


def _normalize_token_for_lookup(token: str) -> str:
    return unicodedata.normalize("NFKC", token).casefold()


_LOOKUP: dict[str, str] = {
    _normalize_token_for_lookup(variant): canonical
    for variant, canonical in NORMALIZED_TAG_MAP.items()
}


def normalize_keywords(source_keywords: list[str]) -> list[str]:
    """表記揺れを NORMALIZED_TAG_MAP で正規タグに統合し、重複を除去する(出現順は維持)。
    マップに存在しないトークンは NFKC 正規化のみ行い、そのまま残す。
    """
    result: list[str] = []
    seen: set[str] = set()
    for token in source_keywords:
        if not token:
            continue
        lookup_key = _normalize_token_for_lookup(token)
        normalized = _LOOKUP.get(lookup_key, unicodedata.normalize("NFKC", token))
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def extract_control_flags(normalized_keywords: list[str]) -> dict[str, bool]:
    """正規化済みキーワード列から22項目のcontrol flag(bool)を抽出する。"""
    keyword_set = set(normalized_keywords)
    return {name: name in keyword_set for name in CONTROL_FLAG_NAMES}


# --- R18判定 --------------------------------------------------------------------

def determine_r18(
    sub_site: int | None,
    rating: int | None,
    url_entity_is_r18: Any,
) -> tuple[bool, str]:
    """R18判定優先順位(指示書のとおり):
    1) novels_descs.sub_site == 2
    2) novels_descs.rating == 1
    3) url_entity.is_r18 == 1 (真値)
    4) いずれも該当しなければ False, "none"
    """
    if sub_site == 2:
        return True, "sub_site"
    if rating == 1:
        return True, "rating"
    if url_entity_is_r18:
        return True, "url_entity"
    return False, "none"
