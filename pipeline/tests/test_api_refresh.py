# test_api_refresh.py
"""Phase3 api_refresh.py のユニットテスト。ネットワークI/Oは一切行わず、
requests.Session/requests.get を unittest.mock.patch でモックする。"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from pipeline.api_refresh import (
    ApiRefreshResult,
    ApiRefreshState,
    fetch_novel_status,
    run_refresh,
)


def _gzip_json(records: list[dict]) -> bytes:
    return gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"))


def _make_response(status_code: int, records: list[dict] | None = None, body: bytes | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    if body is not None:
        resp.content = body
    elif records is not None:
        resp.content = _gzip_json(records)
    else:
        resp.content = b""
    return resp


META_RECORD = {"allcount": 1}
NOVEL_RECORD = {
    "title": "テスト小説",
    "ncode": "n0000aa",
    "userid": 1,
    "writer": "テスト作者",
    "story": "あらすじ",
    "biggenre": 1,
    "genre": 101,
    "keyword": "テスト",
    "general_firstup": "2020-01-01 00:00:00",
    "general_lastup": "2020-01-02 00:00:00",
    "noveltype": 1,
    "end": 0,
    "general_all_no": 10,
    "length": 12345,
    "time": 30,
    "isstop": 0,
    "isr15": 0,
    "isbl": 0,
    "isgl": 0,
    "iszankoku": 0,
    "istensei": 0,
    "istenni": 0,
}


class FetchNovelStatusTests(unittest.TestCase):
    def test_success_available(self) -> None:
        session = MagicMock()
        session.get.return_value = _make_response(200, [META_RECORD, NOVEL_RECORD])

        result = fetch_novel_status(
            "n0000aa", is_r18=False, session=session,
            sleep_fn=lambda s: None, now_fn=lambda: "2026-07-24T00:00:00+00:00",
        )

        self.assertEqual(result.status, "available")
        self.assertEqual(result.attempts, 1)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.payload)
        self.assertEqual(result.payload["ncode"], "n0000aa")
        self.assertEqual(result.payload["title"], "テスト小説")
        self.assertEqual(result.fetched_at, "2026-07-24T00:00:00+00:00")
        session.get.assert_called_once()
        call_kwargs = session.get.call_args
        self.assertEqual(call_kwargs.args[0], "https://api.syosetu.com/novelapi/api/")
        self.assertEqual(call_kwargs.kwargs["params"]["ncode"], "n0000aa")
        self.assertEqual(call_kwargs.kwargs["params"]["gzip"], "5")

    def test_r18_uses_r18_endpoint(self) -> None:
        session = MagicMock()
        session.get.return_value = _make_response(200, [META_RECORD, NOVEL_RECORD])

        fetch_novel_status(
            "n0000aa", is_r18=True, session=session,
            sleep_fn=lambda s: None, now_fn=lambda: "2026-07-24T00:00:00+00:00",
        )

        call_args = session.get.call_args
        self.assertEqual(call_args.args[0], "https://api.syosetu.com/novel18api/api/")

    def test_zero_hit_response_is_not_found(self) -> None:
        session = MagicMock()
        # 0件応答: meta のみ1件
        session.get.return_value = _make_response(200, [META_RECORD])

        result = fetch_novel_status(
            "n9999zz", is_r18=False, session=session,
            sleep_fn=lambda s: None, now_fn=lambda: "2026-07-24T00:00:00+00:00",
        )

        self.assertEqual(result.status, "not_found")
        self.assertIsNone(result.payload)
        self.assertEqual(result.attempts, 1)

    def test_temporary_failure_then_success_retries(self) -> None:
        session = MagicMock()
        session.get.side_effect = [
            requests.exceptions.ConnectionError("boom"),
            _make_response(503),
            _make_response(200, [META_RECORD, NOVEL_RECORD]),
        ]
        sleeps: list[float] = []

        result = fetch_novel_status(
            "n0000aa", is_r18=False, session=session, max_retries=5,
            sleep_fn=lambda s: sleeps.append(s), now_fn=lambda: "2026-07-24T00:00:00+00:00",
        )

        self.assertEqual(result.status, "available")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(sleeps, [1, 2])  # 指数バックオフで2回待機してから成功
        self.assertEqual(session.get.call_count, 3)

    def test_all_retries_exhausted_returns_unknown(self) -> None:
        session = MagicMock()
        session.get.side_effect = requests.exceptions.Timeout("timed out")
        sleeps: list[float] = []

        result = fetch_novel_status(
            "n0000aa", is_r18=False, session=session, max_retries=3,
            sleep_fn=lambda s: sleeps.append(s), now_fn=lambda: "2026-07-24T00:00:00+00:00",
        )

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.attempts, 3)
        self.assertIsNone(result.payload)
        self.assertIsNotNone(result.error)
        self.assertEqual(sleeps, [1, 2])  # 3回試行=2回待機
        self.assertEqual(session.get.call_count, 3)

    def test_all_5xx_retries_exhausted_returns_unknown(self) -> None:
        session = MagicMock()
        session.get.return_value = _make_response(500)

        result = fetch_novel_status(
            "n0000aa", is_r18=False, session=session, max_retries=2,
            sleep_fn=lambda s: None, now_fn=lambda: "2026-07-24T00:00:00+00:00",
        )

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(session.get.call_count, 2)

    def test_uses_requests_module_when_no_session_given(self) -> None:
        with patch("pipeline.api_refresh.requests") as mock_requests_module:
            mock_requests_module.exceptions = requests.exceptions
            mock_requests_module.get.return_value = _make_response(200, [META_RECORD, NOVEL_RECORD])

            result = fetch_novel_status(
                "n0000aa", is_r18=False, sleep_fn=lambda s: None,
                now_fn=lambda: "2026-07-24T00:00:00+00:00",
            )

            self.assertEqual(result.status, "available")
            mock_requests_module.get.assert_called_once()


class ApiRefreshResultTests(unittest.TestCase):
    def test_rejects_invalid_status(self) -> None:
        with self.assertRaises(ValueError):
            ApiRefreshResult(
                ncode="n0000aa", status="not_a_real_status", fetched_at="2026-07-24T00:00:00+00:00",
                payload=None, attempts=1, error=None,
            )


class ApiRefreshStateTests(unittest.TestCase):
    def test_load_missing_file_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state", "api_refresh_state.json")
            state = ApiRefreshState(state_path)
            state.load()
            self.assertFalse(state.is_done("n0000aa"))

    def test_mark_done_persists_across_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state", "api_refresh_state.json")
            state = ApiRefreshState(state_path)
            state.load()
            state.mark_done("n0000aa")
            state.save()

            reloaded = ApiRefreshState(state_path)
            reloaded.load()
            self.assertTrue(reloaded.is_done("n0000aa"))
            self.assertFalse(reloaded.is_done("n0000bb"))


class RunRefreshTests(unittest.TestCase):
    def test_processes_unprocessed_and_writes_jsonl(self) -> None:
        session = MagicMock()
        session.get.return_value = _make_response(200, [META_RECORD, NOVEL_RECORD])

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "api_refresh.jsonl")
            state_path = os.path.join(tmp, "state", "api_refresh_state.json")

            summary = run_refresh(
                [("n0000aa", False), ("n0000bb", False)],
                out_jsonl_path=out_path,
                state_path=state_path,
                session=session,
                sleep_fn=lambda s: None,
                now_fn=lambda: "2026-07-24T00:00:00+00:00",
            )

            self.assertEqual(summary["processed"], 2)
            self.assertEqual(summary["skipped"], 0)
            self.assertEqual(summary["status_counts"], {"available": 2})

            with open(out_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 2)
            self.assertEqual({r["ncode"] for r in lines}, {"n0000aa", "n0000bb"})

    def test_skips_already_done_ncodes_on_resume(self) -> None:
        session = MagicMock()
        session.get.return_value = _make_response(200, [META_RECORD, NOVEL_RECORD])

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "api_refresh.jsonl")
            state_path = os.path.join(tmp, "state", "api_refresh_state.json")

            # 1回目: n0000aa のみ処理
            run_refresh(
                [("n0000aa", False)], out_jsonl_path=out_path, state_path=state_path,
                session=session, sleep_fn=lambda s: None,
                now_fn=lambda: "2026-07-24T00:00:00+00:00",
            )
            # 2回目: n0000aa は既処理としてスキップされ、n0000bb のみ追記される
            summary = run_refresh(
                [("n0000aa", False), ("n0000bb", False)], out_jsonl_path=out_path, state_path=state_path,
                session=session, sleep_fn=lambda s: None,
                now_fn=lambda: "2026-07-24T00:00:00+00:00",
            )

            self.assertEqual(summary["processed"], 1)
            self.assertEqual(summary["skipped"], 1)

            with open(out_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 2)  # 1回目1件 + 2回目1件、n0000aaは重複しない


if __name__ == "__main__":
    unittest.main()
