# text_normalize.py
"""Phase4: episodes.body の本文正規化。
実施してよい処理: HTMLタグ除去(中身のテキストは保持)、HTML entity復号、
改行コード統一(\\r\\n,\\r→\\n)、NBSP(\\xa0)を半角空白へ正規化、
3個以上連続する空行を1個の空行に圧縮、明らかな取得UI断片(HTMLコメント等)の除去。

実施禁止: 誤字脱字修正、伏字化、暴力表現削除、作者独自の記号・改行・ルビ表現の一律削除。

本文フォーマット(実測): 各話は <p id="L1"></p> のような <p id="...">テキスト</p> の
並びで保存されている。空の<p>は空行(段落区切り)として扱う。
一部のルビタグは <rb>/<rt> が欠落し <ruby>...</rb> という壊れた構造になっており、
読みは「（エルフ）」のようにプレーンテキストの丸括弧としてそのまま残っている。
この関数はruby特別処理をせず、汎用的な「タグを消してテキストは残す」実装で対応する。
"""
from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass

# HTMLコメント(取得時に混入し得るUI断片の一種)は中身ごと除去する。
# コメントは可視テキストではないため、内容を残す通常のタグとは扱いを分ける。
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# <p id="...">中身</p> を1行として抽出する。壊れたrubyタグ等で中身が
# 複数行(改行+インデント)にまたがるケースがあるため re.DOTALL を使う。
_P_TAG_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.DOTALL | re.IGNORECASE)

# 残った全タグを除去(中身のテキストは保持)する汎用タグ除去。
_TAG_RE = re.compile(r"<[^>]+>")

# <p>要素の中に残る改行は、正規化前のHTMLがpretty-printされた際に
# 混入した構造的な空白であり、作者が書いた改行ではない(作者の改行は
# 別の<p>要素として表現される)。そのため改行+周囲の半角空白のみを
# 除去し、全角スペース(　, 作者の字下げ)には触れない。
_INNER_STRUCTURAL_NEWLINE_RE = re.compile(r"[ \t]*\n[ \t]*")

_NBSP = "\xa0"


@dataclass(frozen=True)
class NormalizedEpisode:
    text: str
    raw_char_count: int
    normalized_char_count: int
    raw_sha256: str        # 正規化前(生body)のSHA-256 hexdigest
    normalized_sha256: str  # 正規化後テキストのSHA-256 hexdigest


def _process_block(content: str) -> str:
    """<p>要素1個分の中身(生テキスト)を正規化する。"""
    content = _COMMENT_RE.sub("", content)
    content = _TAG_RE.sub("", content)
    content = _INNER_STRUCTURAL_NEWLINE_RE.sub("", content)
    content = html.unescape(content)
    content = content.replace(_NBSP, " ")
    return content


def _compact_blank_runs(lines: list[str]) -> list[str]:
    """3個以上連続する空行を1個の空行に圧縮する(1〜2個の空行はそのまま保持)。"""
    result: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i] == "":
            j = i
            while j < n and lines[j] == "":
                j += 1
            run_len = j - i
            if run_len >= 3:
                result.append("")  # 3個以上の連続空行は1個に圧縮
            else:
                result.extend(lines[i:j])
            i = j
        else:
            result.append(lines[i])
            i += 1
    return result


def normalize_episode_body(raw_body: str) -> NormalizedEpisode:
    raw_char_count = len(raw_body)
    raw_sha256 = hashlib.sha256(raw_body.encode("utf-8")).hexdigest()

    # 改行コード統一(\r\n, \r → \n)
    unified = raw_body.replace("\r\n", "\n").replace("\r", "\n")

    p_matches = _P_TAG_RE.findall(unified)
    if p_matches:
        lines = [_process_block(m) for m in p_matches]
    else:
        # <p>構造を持たない本文へのフォールバック: この場合は\nが唯一の
        # 段落区切りなので、先に分割してから各行を個別に処理する
        # (先に1ブロックとして処理すると本来の改行まで構造的noiseとして
        # 除去されてしまうため)。
        lines = [_process_block(line) for line in unified.split("\n")]

    lines = _compact_blank_runs(lines)
    text = "\n".join(lines)

    normalized_char_count = len(text)
    normalized_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return NormalizedEpisode(
        text=text,
        raw_char_count=raw_char_count,
        normalized_char_count=normalized_char_count,
        raw_sha256=raw_sha256,
        normalized_sha256=normalized_sha256,
    )
