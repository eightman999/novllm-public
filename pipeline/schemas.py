# schemas.py
"""dataset_v2 パイプライン全体で共有する契約(JSON Schema / dataclass)。
指示書 5節・7節Phase7の値をそのまま反映する。他モジュールはここを唯一の正典として参照する。"""
from __future__ import annotations

import dataclasses
from typing import Any

SCHEMA_VERSION = 1
STYLE_METRICS_VERSION = "1.0"

# --- Phase3: なろうAPI現存状態 -------------------------------------------------
AVAILABILITY_STATES = (
    "available",
    "not_found",
    "temporarily_unavailable",
    "unknown",
    "not_applicable",
)

# --- Phase7 作品単位分類 enum -------------------------------------------------
GENDER_VALUES = ("male", "female", "other", "unknown")
AGE_GROUP_VALUES = ("child", "teen", "young_adult", "adult", "elderly", "mixed", "unknown")
VIEWPOINT_VALUES = ("first_person", "second_person", "third_person", "mixed", "unknown")
VIEWPOINT_SCOPE_VALUES = ("limited", "omniscient", "multiple_limited", "unknown")
TENSE_VALUES = ("past", "present", "mixed", "unknown")
CHRONOLOGY_VALUES = ("linear", "nonlinear", "framed", "unknown")
DENSITY_VALUES = ("low", "medium", "high", "unknown")
PACING_VALUES = ("slow", "medium", "fast", "unknown")
FREQUENCY_VALUES = ("low", "medium", "high", "unknown")
VOCAB_REGISTER_VALUES = ("plain", "literary", "colloquial", "archaic", "technical", "unknown")

# --- Phase7 チャンク単位分類 enum ---------------------------------------------
MINOR_RELATED_VALUES = ("none", "nonsexual", "ambiguous", "sexual", "unknown")
CONTENT_SEVERITY_MIN = 0
CONTENT_SEVERITY_MAX = 3  # 0:なし 1:示唆・言及 2:明示的 3:詳細・中心的

WORK_CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "name": "work_classification",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "genre", "setting", "protagonist", "narrative", "tone", "style", "confidence",
        ],
        "properties": {
            "genre": {"type": "array", "items": {"type": "string"}},
            "setting": {
                "type": "object",
                "additionalProperties": False,
                "required": ["world", "era", "locations"],
                "properties": {
                    "world": {"type": "string"},
                    "era": {"type": "string"},
                    "locations": {"type": "array", "items": {"type": "string"}},
                },
            },
            "protagonist": {
                "type": "object",
                "additionalProperties": False,
                "required": ["gender", "age_group", "roles", "group_protagonist"],
                "properties": {
                    "gender": {"type": "string", "enum": list(GENDER_VALUES)},
                    "age_group": {"type": "string", "enum": list(AGE_GROUP_VALUES)},
                    "roles": {"type": "array", "items": {"type": "string"}},
                    "group_protagonist": {"type": "boolean"},
                },
            },
            "narrative": {
                "type": "object",
                "additionalProperties": False,
                "required": ["viewpoint", "viewpoint_scope", "tense", "chronology"],
                "properties": {
                    "viewpoint": {"type": "string", "enum": list(VIEWPOINT_VALUES)},
                    "viewpoint_scope": {"type": "string", "enum": list(VIEWPOINT_SCOPE_VALUES)},
                    "tense": {"type": "string", "enum": list(TENSE_VALUES)},
                    "chronology": {"type": "string", "enum": list(CHRONOLOGY_VALUES)},
                },
            },
            "tone": {"type": "array", "items": {"type": "string"}},
            "style": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "description_density", "exposition_density", "inner_monologue",
                    "vocabulary_register", "pacing", "scene_transition_frequency",
                    "cliffhanger_frequency",
                ],
                "properties": {
                    "description_density": {"type": "string", "enum": list(DENSITY_VALUES)},
                    "exposition_density": {"type": "string", "enum": list(DENSITY_VALUES)},
                    "inner_monologue": {"type": "string", "enum": list(DENSITY_VALUES)},
                    "vocabulary_register": {"type": "string", "enum": list(VOCAB_REGISTER_VALUES)},
                    "pacing": {"type": "string", "enum": list(PACING_VALUES)},
                    "scene_transition_frequency": {"type": "string", "enum": list(FREQUENCY_VALUES)},
                    "cliffhanger_frequency": {"type": "string", "enum": list(FREQUENCY_VALUES)},
                },
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
}

CHUNK_CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "name": "chunk_classification",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene", "tone", "content", "confidence"],
        "properties": {
            "scene": {"type": "array", "items": {"type": "string"}},
            "tone": {"type": "array", "items": {"type": "string"}},
            "content": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "violence", "gore", "sexual_content", "sexual_violence",
                    "coercion", "minor_related", "self_harm", "abuse",
                ],
                "properties": {
                    "violence": {"type": "integer", "minimum": 0, "maximum": 3},
                    "gore": {"type": "integer", "minimum": 0, "maximum": 3},
                    "sexual_content": {"type": "integer", "minimum": 0, "maximum": 3},
                    "sexual_violence": {"type": "integer", "minimum": 0, "maximum": 3},
                    "coercion": {"type": "integer", "minimum": 0, "maximum": 3},
                    "minor_related": {"type": "string", "enum": list(MINOR_RELATED_VALUES)},
                    "self_harm": {"type": "integer", "minimum": 0, "maximum": 3},
                    "abuse": {"type": "integer", "minimum": 0, "maximum": 3},
                },
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
}


def default_work_classification() -> dict[str, Any]:
    """全項目unknown/空のデフォルト値(LM Studio未実行・失敗時のフォールバック)。"""
    return {
        "genre": [],
        "setting": {"world": "unknown", "era": "unknown", "locations": []},
        "protagonist": {
            "gender": "unknown", "age_group": "unknown", "roles": [], "group_protagonist": False,
        },
        "narrative": {
            "viewpoint": "unknown", "viewpoint_scope": "unknown", "tense": "unknown",
            "chronology": "unknown",
        },
        "tone": [],
        "style": {
            "description_density": "unknown", "exposition_density": "unknown",
            "inner_monologue": "unknown", "vocabulary_register": "unknown",
            "pacing": "unknown", "scene_transition_frequency": "unknown",
            "cliffhanger_frequency": "unknown",
        },
        "confidence": 0.0,
    }


def default_chunk_classification() -> dict[str, Any]:
    return {
        "scene": [],
        "tone": [],
        "content": {
            "violence": 0, "gore": 0, "sexual_content": 0, "sexual_violence": 0,
            "coercion": 0, "minor_related": "none", "self_harm": 0, "abuse": 0,
        },
        "confidence": 0.0,
    }


# --- 学習用チャンクJSONLレコード ---------------------------------------------

@dataclasses.dataclass(frozen=True)
class ClassificationMeta:
    method: str  # "rules" | "rules+lmstudio"
    model: str | None
    schema_version: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ChunkMeta:
    ncode: str
    title: str
    author: str
    site_type: str  # "syosetu" | "kakuyomu"
    is_r18: bool
    episode_no: int
    episode_title: str
    chunk_index: int
    chunk_count_in_episode: int
    source_keywords: list[str]
    api_genre: str | None
    api_biggenre: str | None
    content: dict[str, Any]
    style_metrics: dict[str, Any]
    classification: dict[str, Any]
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ChunkRecord:
    id: str  # "{ncode}:e{episode_no}:c{chunk_index}"
    text: str
    control_tags: dict[str, Any]
    meta: ChunkMeta

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "control_tags": self.control_tags,
            "meta": self.meta.to_dict(),
        }


def make_chunk_id(ncode: str, episode_no: int, chunk_index: int) -> str:
    return f"{ncode}:e{episode_no}:c{chunk_index}"
