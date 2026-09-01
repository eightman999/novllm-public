# test_style_stats.py
"""Phase6 決定的文体統計のテスト。"""
from __future__ import annotations

from pipeline.schemas import STYLE_METRICS_VERSION
from pipeline.style_stats import compute_style_metrics


def test_empty_text_returns_all_zero_ratios():
    metrics = compute_style_metrics("")
    assert metrics.char_count == 0
    assert metrics.token_count is None
    assert metrics.sentence_count == 0
    assert metrics.paragraph_count == 0
    assert metrics.avg_sentence_length == 0.0
    assert metrics.avg_paragraph_length == 0.0
    assert metrics.dialogue_ratio == 0.0
    assert metrics.kanji_ratio == 0.0
    assert metrics.hiragana_ratio == 0.0
    assert metrics.katakana_ratio == 0.0
    assert metrics.punctuation_ratio == 0.0
    assert metrics.question_sentence_ratio == 0.0
    assert metrics.exclamation_sentence_ratio == 0.0
    assert metrics.ellipsis_frequency == 0.0
    assert metrics.dash_frequency == 0.0
    assert metrics.bracket_dialogue_frequency == 0.0
    assert metrics.metrics_version == STYLE_METRICS_VERSION


def test_metrics_version_matches_schemas_constant():
    metrics = compute_style_metrics("これはテストです。")
    assert metrics.metrics_version == STYLE_METRICS_VERSION
    assert STYLE_METRICS_VERSION == "1.0"


def test_token_count_passthrough():
    assert compute_style_metrics("テスト。", token_count=None).token_count is None
    assert compute_style_metrics("テスト。", token_count=42).token_count == 42


def test_sentence_count_basic_split():
    text = "今日は晴れた。彼女は笑った！本当に？"
    metrics = compute_style_metrics(text)
    assert metrics.sentence_count == 3
    assert metrics.char_count == len(text)


def test_paragraph_count_blank_line_split():
    text = "一段落目の文章。\n\n二段落目の文章。\n\n三段落目。"
    metrics = compute_style_metrics(text)
    assert metrics.paragraph_count == 3


def test_dialogue_ratio_counts_bracket_contents_inclusive():
    text = "「こんにちは」と彼は言った。"
    metrics = compute_style_metrics(text)
    dialogue_len = len("「こんにちは」")
    assert metrics.dialogue_ratio == dialogue_len / len(text)


def test_dialogue_ratio_supports_double_brackets():
    text = "『これは書名』を読んだ。"
    metrics = compute_style_metrics(text)
    dialogue_len = len("『これは書名』")
    assert metrics.dialogue_ratio == dialogue_len / len(text)


def test_kanji_hiragana_katakana_ratio():
    text = "漢字ひらがなカタカナ"  # 2 kanji, 4 hiragana, 4 katakana = 10 chars
    metrics = compute_style_metrics(text)
    assert metrics.kanji_ratio == 2 / 10
    assert metrics.hiragana_ratio == 4 / 10
    assert metrics.katakana_ratio == 4 / 10


def test_punctuation_ratio():
    text = "あ、い。う！え？"
    metrics = compute_style_metrics(text)
    punct_count = sum(1 for c in text if c in "、。！？")
    assert metrics.punctuation_ratio == punct_count / len(text)


def test_question_and_exclamation_sentence_ratio():
    text = "普通の文だ。疑問だろうか？驚いた！また普通だ。"
    metrics = compute_style_metrics(text)
    assert metrics.sentence_count == 4
    assert metrics.question_sentence_ratio == 1 / 4
    assert metrics.exclamation_sentence_ratio == 1 / 4


def test_question_end_inside_dialogue_close_bracket():
    text = "「本当？」と聞いた。"
    metrics = compute_style_metrics(text)
    assert metrics.sentence_count >= 1
    # 「本当？」で終わる文がquestionとしてカウントされる
    assert metrics.question_sentence_ratio > 0.0


def test_ellipsis_frequency_dots_and_horizontal_ellipsis():
    text = "彼は……何も言わなかった。それから...考え込んだ。" * 20
    metrics = compute_style_metrics(text)
    assert metrics.ellipsis_frequency > 0.0
    expected_count = len(__import__("re").findall(r"…|\.{3,}", text))
    expected_freq = expected_count / (len(text) / 1000)
    assert metrics.ellipsis_frequency == expected_freq


def test_dash_frequency():
    text = "――そして全てが変わった。彼は去った――永遠に。" * 10
    metrics = compute_style_metrics(text)
    assert metrics.dash_frequency > 0.0


def test_bracket_dialogue_frequency():
    text = "「そうだね」「うん」と二人は頷いた。" * 10
    metrics = compute_style_metrics(text)
    assert metrics.bracket_dialogue_frequency > 0.0


def test_avg_sentence_and_paragraph_length_positive_for_nonempty_text():
    text = "これは最初の文です。これは二番目の文です。\n\nこれは別の段落です。"
    metrics = compute_style_metrics(text)
    assert metrics.avg_sentence_length > 0.0
    assert metrics.avg_paragraph_length > 0.0


def test_no_zero_division_error_on_various_edge_inputs():
    # 文末記号が一切無いテキストでもZeroDivisionErrorを起こさないこと
    text = "文末記号のない文字列"
    metrics = compute_style_metrics(text)
    assert metrics.sentence_count in (0, 1)
    assert metrics.avg_sentence_length >= 0.0


def test_ruby_stripped_realistic_novel_excerpt_ratios_sane():
    text = (
        "「行くぞ」と彼は言った。空は青く澄み渡っていた。\n\n"
        "本当に大丈夫なのか？　少し不安になった。\n\n"
        "――そして、旅は始まった。"
    )
    metrics = compute_style_metrics(text)
    assert 0.0 <= metrics.dialogue_ratio <= 1.0
    assert 0.0 <= metrics.kanji_ratio <= 1.0
    assert 0.0 <= metrics.hiragana_ratio <= 1.0
    assert 0.0 <= metrics.katakana_ratio <= 1.0
    assert metrics.paragraph_count == 3
    assert metrics.metrics_version == STYLE_METRICS_VERSION
