# test_lmstudio_classify.py
"""pipeline/lmstudio_classify.py のユニットテスト。requests.postはモックし、
実際のLM Studioサーバへは接続しない。"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from pipeline.lmstudio_classify import (
    classify_chunk,
    classify_work,
    should_send_chunk_to_lmstudio,
    should_send_work_to_lmstudio,
)
from pipeline.schemas import default_chunk_classification, default_work_classification

_VALID_WORK_JSON = {
    "genre": ["fantasy"],
    "setting": {"world": "isekai", "era": "medieval", "locations": ["王都"]},
    "protagonist": {
        "gender": "male", "age_group": "young_adult", "roles": ["主人公"],
        "group_protagonist": False,
    },
    "narrative": {
        "viewpoint": "first_person", "viewpoint_scope": "limited", "tense": "past",
        "chronology": "linear",
    },
    "tone": ["light"],
    "style": {
        "description_density": "medium", "exposition_density": "medium",
        "inner_monologue": "medium", "vocabulary_register": "plain",
        "pacing": "medium", "scene_transition_frequency": "medium",
        "cliffhanger_frequency": "low",
    },
    "confidence": 0.9,
}

_VALID_CHUNK_JSON = {
    "scene": ["battle"],
    "tone": ["tense"],
    "content": {
        "violence": 1, "gore": 0, "sexual_content": 0, "sexual_violence": 0,
        "coercion": 0, "minor_related": "none", "self_harm": 0, "abuse": 0,
    },
    "confidence": 0.8,
}


def _make_response(content_str: str, status_ok: bool = True) -> MagicMock:
    response = MagicMock()
    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = requests.HTTPError("http error")
    response.json.return_value = {
        "choices": [{"message": {"content": content_str}}],
    }
    return response


class TestClassifyWork:
    @patch("pipeline.lmstudio_classify.requests.post")
    def test_normal_response_returns_parsed_classification(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response(json.dumps(_VALID_WORK_JSON))

        result = classify_work(["代表テキスト1", "代表テキスト2"])

        assert result == _VALID_WORK_JSON
        assert mock_post.call_count == 1
        call_kwargs = mock_post.call_args
        assert call_kwargs.args[0] == "http://localhost:1234/v1/chat/completions"
        payload = call_kwargs.kwargs["json"]
        assert payload["temperature"] == 0.0
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["name"] == "work_classification"
        prompt = payload["messages"][0]["content"]
        assert "third_person=三人称" in prompt
        assert "limited=一人物の知覚・内心に限定" in prompt

    @patch("pipeline.lmstudio_classify.requests.post")
    def test_invalid_json_then_success_retries(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = [
            _make_response("これはJSONではない応答です"),
            _make_response(json.dumps(_VALID_WORK_JSON)),
        ]

        result = classify_work(["代表テキスト"], max_retries=3)

        assert result == _VALID_WORK_JSON
        assert mock_post.call_count == 2

    @patch("pipeline.lmstudio_classify.requests.post")
    def test_all_retries_fail_returns_default(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response("壊れたJSON{{{")

        result = classify_work(["代表テキスト"], max_retries=3)

        assert result == default_work_classification()
        assert mock_post.call_count == 3

    @patch("pipeline.lmstudio_classify.requests.post")
    def test_prompt_too_long_returns_default_without_calling_api(self, mock_post: MagicMock) -> None:
        huge_text = "あ" * 10000

        result = classify_work([huge_text])

        assert result == default_work_classification()
        mock_post.assert_not_called()

    @patch("pipeline.lmstudio_classify.requests.post")
    def test_uses_env_model_override(self, mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NOVLLM_LMSTUDIO_MODEL", "custom-model")
        mock_post.return_value = _make_response(json.dumps(_VALID_WORK_JSON))

        classify_work(["テキスト"])

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "custom-model"

    @patch("pipeline.lmstudio_classify.requests.post")
    def test_explicit_model_overrides_env(self, mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NOVLLM_LMSTUDIO_MODEL", "env-model")
        mock_post.return_value = _make_response(json.dumps(_VALID_WORK_JSON))

        classify_work(["テキスト"], model="explicit-model")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "explicit-model"


class TestClassifyChunk:
    @patch("pipeline.lmstudio_classify.requests.post")
    def test_normal_response_returns_parsed_classification(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response(json.dumps(_VALID_CHUNK_JSON))

        result = classify_chunk("チャンク本文")

        assert result == _VALID_CHUNK_JSON
        payload = mock_post.call_args.kwargs["json"]
        assert payload["response_format"]["json_schema"]["name"] == "chunk_classification"
        assert "minor_related は none=該当なし" in payload["messages"][0]["content"]

    @patch("pipeline.lmstudio_classify.requests.post")
    def test_invalid_json_then_success_retries(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = [
            _make_response("not json"),
            _make_response(json.dumps(_VALID_CHUNK_JSON)),
        ]

        result = classify_chunk("チャンク本文", max_retries=3)

        assert result == _VALID_CHUNK_JSON
        assert mock_post.call_count == 2

    @patch("pipeline.lmstudio_classify.requests.post")
    def test_all_retries_fail_returns_default(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response("{{invalid")

        result = classify_chunk("チャンク本文", max_retries=2)

        assert result == default_chunk_classification()
        assert mock_post.call_count == 2

    @patch("pipeline.lmstudio_classify.requests.post")
    def test_http_error_retries_then_falls_back_to_default(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _make_response("irrelevant", status_ok=False)

        result = classify_chunk("チャンク本文", max_retries=2)

        assert result == default_chunk_classification()
        assert mock_post.call_count == 2

    @patch("pipeline.lmstudio_classify.requests.post")
    def test_custom_session_used_instead_of_module_requests(self, mock_post: MagicMock) -> None:
        custom_session = MagicMock()
        custom_session.post.return_value = _make_response(json.dumps(_VALID_CHUNK_JSON))

        result = classify_chunk("チャンク本文", session=custom_session)

        assert result == _VALID_CHUNK_JSON
        mock_post.assert_not_called()
        custom_session.post.assert_called_once()

    @patch("pipeline.lmstudio_classify.requests.post")
    def test_strips_think_tags_before_parsing(self, mock_post: MagicMock) -> None:
        content_with_think = "<think>内部思考のログ</think>\n" + json.dumps(_VALID_CHUNK_JSON)
        mock_post.return_value = _make_response(content_with_think)

        result = classify_chunk("チャンク本文")

        assert result == _VALID_CHUNK_JSON


class TestShouldSendWorkToLmstudio:
    def test_no_existing_keywords_returns_true(self) -> None:
        assert should_send_work_to_lmstudio([]) is True

    def test_existing_keywords_no_conflict_returns_false(self) -> None:
        assert should_send_work_to_lmstudio(["fantasy"], tags_conflict=False) is False

    def test_existing_keywords_with_conflict_returns_true(self) -> None:
        assert should_send_work_to_lmstudio(["fantasy"], tags_conflict=True) is True


class TestShouldSendChunkToLmstudio:
    def test_below_threshold_returns_true(self) -> None:
        assert should_send_chunk_to_lmstudio(0.3) is True

    def test_above_threshold_returns_false(self) -> None:
        assert should_send_chunk_to_lmstudio(0.9) is False

    def test_custom_threshold(self) -> None:
        assert should_send_chunk_to_lmstudio(0.5, threshold=0.4) is False
        assert should_send_chunk_to_lmstudio(0.3, threshold=0.4) is True
