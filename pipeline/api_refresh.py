# api_refresh.py
"""Phase3: なろうAPI (api.syosetu.com) による作品現存状態のリフレッシュ。
DBへは一切書き戻さない。結果は out_jsonl_path へ1行1件で追記し、
state_path に処理済みncode集合を永続化して中断再開できるようにする。"""
from __future__ import annotations

import dataclasses
import gzip
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import requests

from pipeline.schemas import AVAILABILITY_STATES

GENERAL_API_URL = "https://api.syosetu.com/novelapi/api/"
R18_API_URL = "https://api.syosetu.com/novel18api/api/"

# 一時失敗時の指数バックオフ秒数 (1,2,4,8,16)
_BACKOFF_SECONDS = (1, 2, 4, 8, 16)

# payload に含めるフィールド (指示書どおり)
_PAYLOAD_FIELDS = (
    "title", "ncode", "userid", "writer", "story", "biggenre", "genre",
    "keyword", "general_firstup", "general_lastup", "noveltype", "end",
    "general_all_no", "length", "time", "isstop", "isr15", "isbl", "isgl",
    "iszankoku", "istensei", "istenni",
)


def _default_now_fn() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_sleep_fn(seconds: float) -> None:
    time.sleep(seconds)


@dataclasses.dataclass(frozen=True)
class ApiRefreshResult:
    ncode: str
    status: str  # available/not_found/temporarily_unavailable/unknown/not_applicable
    fetched_at: str  # ISO8601文字列
    payload: dict[str, Any] | None
    attempts: int
    error: str | None

    def __post_init__(self) -> None:
        if self.status not in AVAILABILITY_STATES:
            raise ValueError(f"invalid status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _api_url_for(is_r18: bool) -> str:
    return R18_API_URL if is_r18 else GENERAL_API_URL


def _parse_response_body(raw: bytes) -> list[dict[str, Any]]:
    """gzip=5 で圧縮されたレスポンスバイト列を解凍してJSON配列にする。"""
    try:
        decompressed = gzip.decompress(raw)
    except OSError:
        # 万一非圧縮で返ってきた場合はそのままJSONとして扱う
        decompressed = raw
    return json.loads(decompressed.decode("utf-8"))


def _extract_payload(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """1件目はmeta、2件目以降が作品データ。0件応答ならNone。"""
    if len(records) < 2:
        return None
    novel = records[1]
    payload = {field: novel.get(field) for field in _PAYLOAD_FIELDS}
    payload["updated_at"] = None
    return payload


def fetch_novel_status(
    ncode: str,
    is_r18: bool,
    session: "requests.Session | None" = None,
    max_retries: int = 5,
    sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], str] | None = None,
) -> ApiRefreshResult:
    """なろうAPIへ問い合わせて作品の現存状態を判定する。

    5xx・タイムアウト・接続エラーは一時失敗として指数バックオフでmax_retries回まで
    再試行する。それでも失敗したら status="unknown"。
    レスポンスが0件応答(meta only)なら status="not_found"。
    正常に作品データが返れば status="available"。
    """
    sleep_fn = sleep_fn or _default_sleep_fn
    now_fn = now_fn or _default_now_fn
    http = session or requests
    url = _api_url_for(is_r18)
    params = {"out": "json", "gzip": "5", "ncode": ncode}

    attempts = 0
    last_error: str | None = None

    while attempts < max_retries:
        attempts += 1
        try:
            response = http.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempts < max_retries:
                sleep_fn(_BACKOFF_SECONDS[min(attempts - 1, len(_BACKOFF_SECONDS) - 1)])
                continue
            return ApiRefreshResult(
                ncode=ncode, status="unknown", fetched_at=now_fn(),
                payload=None, attempts=attempts, error=last_error,
            )

        status_code = response.status_code
        if status_code >= 500:
            last_error = f"HTTP {status_code}"
            if attempts < max_retries:
                sleep_fn(_BACKOFF_SECONDS[min(attempts - 1, len(_BACKOFF_SECONDS) - 1)])
                continue
            return ApiRefreshResult(
                ncode=ncode, status="unknown", fetched_at=now_fn(),
                payload=None, attempts=attempts, error=last_error,
            )

        if status_code >= 400:
            # 4xxは再試行しても状況が変わらないため即座にunknown扱い(削除済みと断定しない)
            return ApiRefreshResult(
                ncode=ncode, status="unknown", fetched_at=now_fn(),
                payload=None, attempts=attempts, error=f"HTTP {status_code}",
            )

        try:
            records = _parse_response_body(response.content)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempts < max_retries:
                sleep_fn(_BACKOFF_SECONDS[min(attempts - 1, len(_BACKOFF_SECONDS) - 1)])
                continue
            return ApiRefreshResult(
                ncode=ncode, status="unknown", fetched_at=now_fn(),
                payload=None, attempts=attempts, error=last_error,
            )

        payload = _extract_payload(records)
        if payload is None:
            return ApiRefreshResult(
                ncode=ncode, status="not_found", fetched_at=now_fn(),
                payload=None, attempts=attempts, error=None,
            )
        return ApiRefreshResult(
            ncode=ncode, status="available", fetched_at=now_fn(),
            payload=payload, attempts=attempts, error=None,
        )

    # ループを抜けた場合(max_retries<=0など)のフォールバック
    return ApiRefreshResult(
        ncode=ncode, status="unknown", fetched_at=now_fn(),
        payload=None, attempts=attempts, error=last_error,
    )


class ApiRefreshState:
    """state/api_refresh_state.json の読み書き。処理済みncode集合を管理し、
    中断再開できるようにする。"""

    def __init__(self, state_path: str) -> None:
        self.state_path = state_path
        self._done: set[str] = set()

    def load(self) -> None:
        if not os.path.exists(self.state_path):
            self._done = set()
            return
        with open(self.state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._done = set(data.get("done", []))

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"done": sorted(self._done)}, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.state_path)

    def is_done(self, ncode: str) -> bool:
        return ncode in self._done

    def mark_done(self, ncode: str) -> None:
        self._done.add(ncode)


def run_refresh(
    ncodes: Iterable[tuple[str, bool]],
    out_jsonl_path: str,
    state_path: str,
    session: "requests.Session | None" = None,
    sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """未処理ncodeのみ処理してout_jsonl_pathへ1行1件追記し、stateを更新する。

    ncodes: (ncode, is_r18) のタプルを列挙するiterable。
    戻り値: {"processed": int, "skipped": int, "status_counts": dict}
    """
    state = ApiRefreshState(state_path)
    state.load()

    processed = 0
    skipped = 0
    status_counts: dict[str, int] = {}

    os.makedirs(os.path.dirname(out_jsonl_path) or ".", exist_ok=True)
    with open(out_jsonl_path, "a", encoding="utf-8") as out_f:
        for ncode, is_r18 in ncodes:
            if state.is_done(ncode):
                skipped += 1
                continue

            result = fetch_novel_status(
                ncode, is_r18, session=session, sleep_fn=sleep_fn, now_fn=now_fn,
            )
            out_f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            out_f.flush()

            state.mark_done(ncode)
            state.save()

            processed += 1
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

    return {
        "processed": processed,
        "skipped": skipped,
        "status_counts": status_counts,
    }
