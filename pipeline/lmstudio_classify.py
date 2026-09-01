# lmstudio_classify.py
"""Phase7: LM Studio(OpenAI互換API)を用いた作品単位/チャンク単位の分類クライアント。
接続先はローカルのLM Studioサーバ(http://localhost:1234/v1)のみ。外部(クラウド)LLMサービスへは
絶対に接続しない。本文の再生成・要約・引用をLLMに要求せず、分類値だけをJSON Schemaで強制して
返させる。パース・応答失敗時はmax_retries回まで再試行し、それでも失敗したら例外を投げずデフォルト
値(confidence=0.0)を返す。"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

from pipeline.schemas import (
    CHUNK_CLASSIFICATION_JSON_SCHEMA,
    WORK_CLASSIFICATION_JSON_SCHEMA,
    default_chunk_classification,
    default_work_classification,
)

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL_ENV = "NOVLLM_LMSTUDIO_MODEL"
DEFAULT_MODEL = "qwen3-14b-mlx-4bit"
DEFAULT_TIMEOUT_SECONDS = 120

# max context 8192相当を想定したプロンプト長チェック。日本語主体のため1文字≒1トークン超になり
# 得るので、安全側に1文字1トークン換算とし、応答用に十分な余白を残す。
MAX_CONTEXT_TOKENS = 8192
MAX_PROMPT_CHARS = 6000

# thinking無効化の指示(モデルのsystemプロンプトで明示する)
_THINKING_DISABLED_DIRECTIVE = "/no_think"

_WORK_FIELD_RUBRIC = """
選択肢の定義:
- protagonist.gender: male/female=本文で明示された性別、other=それ以外、unknown=不明。
- protagonist.age_group: child=子ども、teen=十代、young_adult=若年成人、adult=成人、elderly=高齢者、mixed=複数年齢層、unknown=不明。
- narrative.viewpoint: first_person=一人称、second_person=二人称、third_person=三人称、mixed=混在、unknown=不明。
- narrative.viewpoint_scope: limited=一人物の知覚・内心に限定、omniscient=複数人物の内心や全体を俯瞰、multiple_limited=複数の限定視点、unknown=不明。
- narrative.tense: past=過去形主体、present=現在形主体、mixed=混在、unknown=不明。chronology: linear=時系列順、nonlinear=回想・時間跳躍が主、framed=枠物語、unknown=不明。
- style の low/medium/high は、その要素が作品中に占める頻度または強さ。vocabulary_register は plain=平易、literary=文芸的、colloquial=口語的、archaic=古風、technical=専門的、unknown=不明。
""".strip()

_CHUNK_FIELD_RUBRIC = """
contentの強度は 0=なし、1=示唆・言及のみ、2=明示的な描写あり、3=詳細または中心的な描写。
minor_related は none=該当なし、nonsexual=未成年者に関する非性的内容、ambiguous=年齢または性的文脈が不明、sexual=未成年者を伴う性的内容、unknown=判別不能。
""".strip()

_WORK_SYSTEM_PROMPT = (
    "あなたは小説のメタデータ分類器です。与えられた作品の代表テキスト断片から、"
    "ジャンル・設定・主人公・文体などの分類値のみをJSON Schemaに厳密に従って出力してください。"
    "本文の要約・引用・言い換え・再生成は一切行わず、指定されたフィールドの分類値だけを出力してください。"
    f"\n{_WORK_FIELD_RUBRIC}\n{_THINKING_DISABLED_DIRECTIVE}"
)

_CHUNK_SYSTEM_PROMPT = (
    "あなたは小説チャンクのメタデータ分類器です。与えられたテキストチャンクから、"
    "シーン・トーン・content(暴力/性描写等の強度0-3)などの分類値のみをJSON Schemaに厳密に従って"
    "出力してください。本文の要約・引用・言い換え・再生成は一切行わず、指定されたフィールドの"
    f"分類値だけを出力してください。\n{_CHUNK_FIELD_RUBRIC}\n{_THINKING_DISABLED_DIRECTIVE}"
)


class PromptTooLongError(ValueError):
    """プロンプトがmax context相当を超える場合に送出する。"""


def _resolve_model(model: str | None) -> str:
    if model:
        return model
    return os.environ.get(DEFAULT_MODEL_ENV, DEFAULT_MODEL)


def _resolve_base_url(base_url: str | None) -> str:
    return base_url or DEFAULT_BASE_URL


def _check_prompt_length(text: str) -> None:
    if len(text) > MAX_PROMPT_CHARS:
        raise PromptTooLongError(
            f"prompt length {len(text)} chars exceeds max {MAX_PROMPT_CHARS} chars "
            f"(context budget {MAX_CONTEXT_TOKENS} tokens)"
        )


def _post_chat_completion(
    *,
    session: Any,
    base_url: str,
    model: str,
    system_prompt: str,
    user_content: str,
    json_schema: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    http = session or requests
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema,
        },
    }
    response = http.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _extract_content(response_json: dict[str, Any]) -> str:
    return response_json["choices"][0]["message"]["content"]


def _parse_classification(content: str) -> dict[str, Any]:
    """LLM応答からJSONオブジェクトをパースする。thinkingタグ等の混入に軽く耐性を持たせる。"""
    text = content.strip()
    # 一部モデルは無効化指示を無視して<think>...</think>を出力することがあるため除去を試みる。
    if "<think>" in text and "</think>" in text:
        end = text.rindex("</think>") + len("</think>")
        text = text[end:].strip()
    return json.loads(text)


def classify_work(
    representative_texts: list[str],
    session: Any = None,
    max_retries: int = 3,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """作品単位の分類をLM Studioへ問い合わせる。失敗時はdefault_work_classification()を返す。"""
    resolved_model = _resolve_model(model)
    resolved_base_url = _resolve_base_url(base_url)
    joined = "\n\n---\n\n".join(representative_texts)
    try:
        _check_prompt_length(joined)
    except PromptTooLongError:
        return default_work_classification()

    user_content = (
        "以下は同一作品からの代表テキスト断片です。作品全体のジャンル・設定・主人公・文体などを"
        f"分類してください。\n\n{joined}"
    )

    last_error: Exception | None = None
    for _attempt in range(max_retries):
        try:
            response_json = _post_chat_completion(
                session=session,
                base_url=resolved_base_url,
                model=resolved_model,
                system_prompt=_WORK_SYSTEM_PROMPT,
                user_content=user_content,
                json_schema=WORK_CLASSIFICATION_JSON_SCHEMA,
            )
            content = _extract_content(response_json)
            return _parse_classification(content)
        except (
            requests.RequestException,
            KeyError,
            IndexError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = exc
            continue
    del last_error
    return default_work_classification()


def classify_chunk(
    chunk_text: str,
    session: Any = None,
    max_retries: int = 3,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """チャンク単位の分類をLM Studioへ問い合わせる。失敗時はdefault_chunk_classification()を返す。"""
    resolved_model = _resolve_model(model)
    resolved_base_url = _resolve_base_url(base_url)
    try:
        _check_prompt_length(chunk_text)
    except PromptTooLongError:
        return default_chunk_classification()

    user_content = (
        f"以下のテキストチャンクのシーン・トーン・content強度を分類してください。\n\n{chunk_text}"
    )

    last_error: Exception | None = None
    for _attempt in range(max_retries):
        try:
            response_json = _post_chat_completion(
                session=session,
                base_url=resolved_base_url,
                model=resolved_model,
                system_prompt=_CHUNK_SYSTEM_PROMPT,
                user_content=user_content,
                json_schema=CHUNK_CLASSIFICATION_JSON_SCHEMA,
            )
            content = _extract_content(response_json)
            return _parse_classification(content)
        except (
            requests.RequestException,
            KeyError,
            IndexError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = exc
            continue
    del last_error
    return default_chunk_classification()


def should_send_work_to_lmstudio(existing_keywords: list[str], tags_conflict: bool = False) -> bool:
    """既存タグが0件、または矛盾がある場合のみLM Studioへ送る対象とする。"""
    if tags_conflict:
        return True
    return len(existing_keywords) == 0


def should_send_chunk_to_lmstudio(rule_confidence: float, threshold: float = 0.6) -> bool:
    """ルール判定confidenceが閾値未満の場合のみLM Studioへ送る対象とする。"""
    return rule_confidence < threshold
