#!/usr/bin/env python3
"""Unit tests for scripts/bootstrap_crates_io.py."""

from __future__ import annotations

import argparse
import io
import os
import pathlib
import sys
import unittest
import urllib.request
from unittest import mock

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bootstrap_crates_io as bc  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent


class FakeClock:
    """Deterministic monotonic clock + sleeper pair for RateLimiter tests."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _FakeHttpResponse:
    def __init__(self, status: int, body: str, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self._body = body.encode("utf-8")
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class BootstrapCratesIoTests(unittest.TestCase):
    def test_sparse_index_url_follows_cargo_prefix_rules(self) -> None:
        self.assertEqual(bc.sparse_index_url("a"), "https://index.crates.io/1/a")
        self.assertEqual(bc.sparse_index_url("cc"), "https://index.crates.io/2/cc")
        self.assertEqual(bc.sparse_index_url("syn"), "https://index.crates.io/3/s/syn")
        self.assertEqual(
            bc.sparse_index_url("egraph-core"),
            "https://index.crates.io/eg/ra/egraph-core",
        )
        self.assertEqual(
            bc.sparse_index_url("Serde_JSON"),
            "https://index.crates.io/se/rd/serde_json",
        )

    def test_parse_sparse_index_entries(self) -> None:
        payload = (
            "\n"
            '{"name":"egraph-core","vers":"6.0.0-beta.2","yanked":false}\n'
            'not-valid-json\n'
            '{"name":"egraph-core","vers":"6.0.0-beta.1","yanked":true}\n'
            '{"name":"egraph-core","vers":"6.0.0-beta.3"}\n'
        )
        entries = bc.parse_sparse_index_entries(payload)
        self.assertEqual(
            entries,
            [
                bc.SparseIndexEntry(vers="6.0.0-beta.2", yanked=False),
                bc.SparseIndexEntry(vers="6.0.0-beta.1", yanked=True),
                bc.SparseIndexEntry(vers="6.0.0-beta.3", yanked=False),
            ],
        )

    def test_query_crates_io_state_uses_sparse_index_exclusively(self) -> None:
        unit = bc.PublishUnit(name="egraph-core", version="6.0.0-beta.3", tier=0, deps=())
        seen_urls: list[str] = []

        def fake_get(
            url: str,
            _ua: str,
            max_retries: int = 3,
        ) -> tuple[int, str, dict[str, str]]:
            del max_retries
            seen_urls.append(url)
            return (404, "", {})

        with mock.patch.object(bc, "_http_get_with_retry", side_effect=fake_get):
            self.assertEqual(bc.query_crates_io_state(unit, "ua/1.0"), (False, False))
        self.assertEqual(seen_urls, ["https://index.crates.io/eg/ra/egraph-core"])

        with mock.patch.object(
            bc,
            "_http_get_with_retry",
            return_value=(200, '{"vers":"6.0.0-beta.2","yanked":false}\n', {}),
        ):
            self.assertEqual(bc.query_crates_io_state(unit, "ua/1.0"), (False, True))

        with mock.patch.object(
            bc,
            "_http_get_with_retry",
            return_value=(200, '{"vers":"6.0.0-beta.3","yanked":false}\n', {}),
        ):
            self.assertEqual(bc.query_crates_io_state(unit, "ua/1.0"), (True, True))

        with mock.patch.object(
            bc,
            "_http_get_with_retry",
            return_value=(200, '{"vers":"6.0.0-beta.3","yanked":true}\n', {}),
        ):
            self.assertEqual(bc.query_crates_io_state(unit, "ua/1.0"), (True, True))

        with mock.patch.object(bc, "_http_get_with_retry", return_value=(500, "err", {})):
            with self.assertRaises(RuntimeError):
                bc.query_crates_io_state(unit, "ua/1.0")

    def test_rate_limiter_enforces_one_req_per_sec(self) -> None:
        fake = FakeClock(start=10.0)
        limiter = bc.RateLimiter(min_interval_sec=1.0, clock=fake.clock, sleeper=fake.sleep)

        self.assertEqual(limiter.wait(), 0.0)
        self.assertAlmostEqual(limiter.wait(), 1.0)
        fake.now += 0.3
        self.assertAlmostEqual(limiter.wait(), 0.7)
        fake.now += 1.5
        self.assertEqual(limiter.wait(), 0.0)
        self.assertEqual(len(fake.sleeps), 2)

    def test_http_get_status_guards_user_agent_and_api_rate_limit(self) -> None:
        with self.assertRaises(ValueError):
            bc._http_get_status("https://index.crates.io/eg/ra/egraph-core", "   ")

        fake = FakeClock()
        limiter = bc.RateLimiter(min_interval_sec=1.0, clock=fake.clock, sleeper=fake.sleep)
        with (
            mock.patch.object(bc, "_API_RATE_LIMITER", limiter),
            mock.patch.object(urllib.request, "urlopen", return_value=_FakeHttpResponse(200, "ok")),
        ):
            bc._http_get_status("https://index.crates.io/eg/ra/egraph-core", "ua/1.0")
            bc._http_get_status("https://index.crates.io/eg/ra/egraph-rs", "ua/1.0")
            self.assertEqual(fake.sleeps, [])

            bc._http_get_status("https://crates.io/api/v1/crates/egraph-core", "ua/1.0")
            bc._http_get_status("https://crates.io/api/v1/crates/egraph-rs", "ua/1.0")
            self.assertEqual(fake.sleeps, [1.0])

    def test_existing_crate_429_uses_existing_sleep_seconds(self) -> None:
        unit = bc.PublishUnit(name="egraph-core", version="6.0.0-beta.3", tier=0, deps=())
        publish_calls = iter([(1, "HTTP 429 Too Many Requests"), (0, "ok")])
        sleep_calls: list[tuple[int, str]] = []

        def fake_publish(_u: bc.PublishUnit, _args: argparse.Namespace, _root: str) -> tuple[int, str]:
            return next(publish_calls)

        def fake_sleep(seconds: int, reason: str) -> None:
            sleep_calls.append((seconds, reason))

        with (
            mock.patch.dict(os.environ, {"CARGO_REGISTRY_TOKEN": "dummy-token"}),
            mock.patch.object(bc, "compute_publish_plan", return_value=[unit]),
            mock.patch.object(bc, "query_crates_io_state", return_value=(False, True)),
            mock.patch.object(bc, "_run_cargo_publish", side_effect=fake_publish),
            mock.patch.object(bc, "_sleep_with_progress", side_effect=fake_sleep),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            rc = bc.main(["--repo-root", str(REPO_ROOT), "--existing-sleep-seconds", "65"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(sleep_calls), 1)
        self.assertEqual(sleep_calls[0][0], 65)

    def test_dry_run_offline_and_mocked_online_make_zero_api_calls(self) -> None:
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            rc = bc.main(["--repo-root", str(REPO_ROOT), "--dry-run", "--offline"])
        self.assertEqual(rc, 0)
        self.assertIn("Completed plan (24 crates processed; dry_run=True).", buf.getvalue())

        seen_urls: list[str] = []

        def fake_get(
            url: str,
            _ua: str,
            max_retries: int = 3,
        ) -> tuple[int, str, dict[str, str]]:
            del max_retries
            seen_urls.append(url)
            return (200, '{"vers":"6.0.0-beta.2","yanked":false}\n', {})

        buf_online = io.StringIO()
        with (
            mock.patch.object(bc, "_http_get_with_retry", side_effect=fake_get),
            mock.patch("sys.stdout", buf_online),
        ):
            rc_online = bc.main(["--repo-root", str(REPO_ROOT), "--dry-run"])
        self.assertEqual(rc_online, 0)
        self.assertEqual(len(seen_urls), 24)
        self.assertTrue(all(u.startswith("https://index.crates.io/") for u in seen_urls))


if __name__ == "__main__":
    unittest.main()
