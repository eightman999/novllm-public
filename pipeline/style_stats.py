# style_stats.py
"""Phase6: 決定的文体統計。LLMを一切使わず標準ライブラリ+正規表現のみで計算する。
指示書 Phase6 の仕様通り、metrics_versionは pipeline/schemas.py の STYLE_METRICS_VERSION を使う。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.schemas import STYLE_METRICS_VERSION

# 文末とみなす記号。「」『』の閉じ括弧も文の終端に含める(会話文の後に続く地の文と区切るため)。
# 末尾の終端記号を見て疑問文/感嘆文判定するための正規表現
_QUESTION_END_RE = re.compile(r"？」?』?$")
_EXCLAMATION_END_RE = re.compile(r"！」?』?$")

_DIALOGUE_RE = re.compile(r"「[^」]*」|『[^』]*』")
_KANJI_RE = re.compile(r"[一-鿿]")
_HIRAGANA_RE = re.compile(r"[぀-ゟ]")
_KATAKANA_RE = re.compile(r"[゠-ヿ]")
_PUNCTUATION_RE = re.compile(r"[。、,.！？!?]")
_ELLIPSIS_RE = re.compile(r"…|\.{3,}")
_DASH_RE = re.compile(r"――|—|‐{2,}")
_BRACKET_DIALOGUE_RE = re.compile(r"「[^」]*」")


@dataclass(frozen=True)
class StyleMetrics:
    char_count: int
    token_count: int | None
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    avg_paragraph_length: float
    dialogue_ratio: float
    kanji_ratio: float
    hiragana_ratio: float
    katakana_ratio: float
    punctuation_ratio: float
    question_sentence_ratio: float
    exclamation_sentence_ratio: float
    ellipsis_frequency: float
    dash_frequency: float
    bracket_dialogue_frequency: float
    metrics_version: str


def _zero_metrics(char_count: int, token_count: int | None) -> StyleMetrics:
    return StyleMetrics(
        char_count=char_count,
        token_count=token_count,
        sentence_count=0,
        paragraph_count=0,
        avg_sentence_length=0.0,
        avg_paragraph_length=0.0,
        dialogue_ratio=0.0,
        kanji_ratio=0.0,
        hiragana_ratio=0.0,
        katakana_ratio=0.0,
        punctuation_ratio=0.0,
        question_sentence_ratio=0.0,
        exclamation_sentence_ratio=0.0,
        ellipsis_frequency=0.0,
        dash_frequency=0.0,
        bracket_dialogue_frequency=0.0,
        metrics_version=STYLE_METRICS_VERSION,
    )


_SENTENCE_TOKEN_RE = re.compile(r"[^。！？…」』]*[。！？…」』]+|[^。！？…」』]+$")


def _split_sentences(text: str) -> list[str]:
    """。！？…」』等の終端記号で文を分割する。終端記号の連続(例: ？」)は同じ文にまとめる。"""
    sentences = [m.group(0) for m in _SENTENCE_TOKEN_RE.finditer(text)]
    return [s for s in sentences if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """空行区切りで段落分割する。"""
    raw_paragraphs = re.split(r"\n\s*\n", text)
    return [p for p in raw_paragraphs if p.strip()]


def compute_style_metrics(text: str, token_count: int | None = None) -> StyleMetrics:
    char_count = len(text)
    if char_count == 0:
        return _zero_metrics(char_count, token_count)

    sentences = _split_sentences(text)
    sentence_count = len(sentences)

    paragraphs = _split_paragraphs(text)
    paragraph_count = len(paragraphs)

    avg_sentence_length = (char_count / sentence_count) if sentence_count else 0.0
    avg_paragraph_length = (char_count / paragraph_count) if paragraph_count else 0.0

    dialogue_chars = sum(len(m.group(0)) for m in _DIALOGUE_RE.finditer(text))
    dialogue_ratio = dialogue_chars / char_count

    kanji_count = len(_KANJI_RE.findall(text))
    hiragana_count = len(_HIRAGANA_RE.findall(text))
    katakana_count = len(_KATAKANA_RE.findall(text))
    kanji_ratio = kanji_count / char_count
    hiragana_ratio = hiragana_count / char_count
    katakana_ratio = katakana_count / char_count

    punctuation_count = len(_PUNCTUATION_RE.findall(text))
    punctuation_ratio = punctuation_count / char_count

    if sentence_count:
        question_count = sum(1 for s in sentences if _QUESTION_END_RE.search(s.strip()))
        exclamation_count = sum(1 for s in sentences if _EXCLAMATION_END_RE.search(s.strip()))
        question_sentence_ratio = question_count / sentence_count
        exclamation_sentence_ratio = exclamation_count / sentence_count
    else:
        question_sentence_ratio = 0.0
        exclamation_sentence_ratio = 0.0

    per_mille = char_count / 1000
    ellipsis_count = len(_ELLIPSIS_RE.findall(text))
    dash_count = len(_DASH_RE.findall(text))
    bracket_dialogue_count = len(_BRACKET_DIALOGUE_RE.findall(text))
    ellipsis_frequency = ellipsis_count / per_mille
    dash_frequency = dash_count / per_mille
    bracket_dialogue_frequency = bracket_dialogue_count / per_mille

    return StyleMetrics(
        char_count=char_count,
        token_count=token_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        avg_sentence_length=avg_sentence_length,
        avg_paragraph_length=avg_paragraph_length,
        dialogue_ratio=dialogue_ratio,
        kanji_ratio=kanji_ratio,
        hiragana_ratio=hiragana_ratio,
        katakana_ratio=katakana_ratio,
        punctuation_ratio=punctuation_ratio,
        question_sentence_ratio=question_sentence_ratio,
        exclamation_sentence_ratio=exclamation_sentence_ratio,
        ellipsis_frequency=ellipsis_frequency,
        dash_frequency=dash_frequency,
        bracket_dialogue_frequency=bracket_dialogue_frequency,
        metrics_version=STYLE_METRICS_VERSION,
    )
