#!/usr/bin/env python3
"""Idempotent, rate-limit-aware topological crates.io publisher/bootstrapper."""

from __future__ import annotations

import argparse
import collections
import datetime
import email.utils
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

USER_AGENT_PREFIX = "egraph-rs-bootstrap"
CRATES_IO_API = "https://crates.io/api/v1/crates"
SPARSE_INDEX_BASE = "https://index.crates.io"
NEW_CRATE_BURST = 5
EXISTING_CRATE_BURST = 30


@dataclass(frozen=True)
class PublishUnit:
    """One publishable workspace crate with topological tier metadata."""

    name: str
    version: str
    tier: int
    deps: tuple[str, ...]


def _is_publishable(pkg: dict[str, Any]) -> bool:
    publish = pkg.get("publish")
    if publish is None:
        return True
    if isinstance(publish, list):
        return len(publish) > 0 and "crates-io" in [str(x) for x in publish]
    return False


def compute_publish_plan(repo_root: str) -> list[PublishUnit]:
    """Compute deterministic Kahn topological tiers across publishable workspace crates."""
    raw = subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=repo_root,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    data: dict[str, Any] = json.loads(raw)
    packages: dict[str, dict[str, Any]] = {}
    for pkg in data.get("packages", []):
        if _is_publishable(pkg):
            packages[str(pkg["name"])] = pkg

    # Edge u -> v means u must be published before v (v depends on u).
    dependents: dict[str, set[str]] = {name: set() for name in packages}
    deps_of: dict[str, set[str]] = {name: set() for name in packages}
    in_degree: dict[str, int] = {name: 0 for name in packages}

    for name, pkg in packages.items():
        for dep in pkg.get("dependencies", []):
            dep_name = str(dep.get("name", ""))
            if dep_name in packages and dep_name != name and dep_name not in deps_of[name]:
                deps_of[name].add(dep_name)
                dependents[dep_name].add(name)
                in_degree[name] += 1

    tier_queue: list[str] = sorted(n for n, deg in in_degree.items() if deg == 0)
    plan: list[PublishUnit] = []
    tier = 0
    while tier_queue:
        next_queue: list[str] = []
        for name in tier_queue:
            ver = str(packages[name]["version"])
            plan.append(PublishUnit(name=name, version=ver, tier=tier, deps=tuple(sorted(deps_of[name]))))
            for child in dependents[name]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_queue.append(child)
        tier_queue = sorted(next_queue)
        tier += 1

    if len(plan) != len(packages):
        remaining = sorted(set(packages) - {u.name for u in plan})
        raise RuntimeError(f"Dependency cycle detected among workspace crates: {remaining}")
    return plan


def sparse_index_url(crate_name: str) -> str:
    """Return Cargo sparse index URL for a crate name."""
    low = crate_name.lower()
    if len(low) == 1:
        rel = f"1/{low}"
    elif len(low) == 2:
        rel = f"2/{low}"
    elif len(low) == 3:
        rel = f"3/{low[0]}/{low}"
    else:
        rel = f"{low[:2]}/{low[2:4]}/{low}"
    return f"{SPARSE_INDEX_BASE}/{rel}"


def _http_get_status(url: str, user_agent: str, timeout: float = 15.0) -> tuple[int, str, dict[str, str]]:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return int(resp.status), body, headers
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        headers = {k.lower(): v for k, v in err.headers.items()}
        return int(err.code), body, headers


def _http_get_with_retry(
    url: str,
    user_agent: str,
    max_retries: int = 3,
    default_retry_sleep: int = 10,
) -> tuple[int, str, dict[str, str]]:
    for attempt in range(max_retries + 1):
        status, body, headers = _http_get_status(url, user_agent)
        if status != 429 or attempt >= max_retries:
            return status, body, headers
        retry_after = headers.get("retry-after", "").strip()
        parts: list[str] = []
        if retry_after:
            if retry_after.isdigit():
                parts.append(f"retry-after: {retry_after}")
            else:
                parts.append(f"Please try again after {retry_after}")
        if body:
            parts.append(body)
        delay = _parse_retry_delay("\n".join(parts), default_retry_sleep)
        _sleep_with_progress(delay, f"HTTP 429 querying {url} (retry {attempt + 1}/{max_retries})")
    raise AssertionError("unreachable")


def query_crates_io_state(unit: PublishUnit, user_agent: str, max_retries: int = 3) -> tuple[bool, bool]:
    """Return (version_published, crate_exists) from crates.io API."""
    ver_url = f"{CRATES_IO_API}/{unit.name}/{unit.version}"
    status, _, _ = _http_get_with_retry(ver_url, user_agent, max_retries=max_retries)
    if status == 200:
        return True, True
    if status != 404:
        raise RuntimeError(f"Unexpected HTTP {status} querying {ver_url}")

    crate_url = f"{CRATES_IO_API}/{unit.name}"
    c_status, _, _ = _http_get_with_retry(crate_url, user_agent, max_retries=max_retries)
    if c_status == 200:
        return False, True
    if c_status == 404:
        return False, False
    raise RuntimeError(f"Unexpected HTTP {c_status} querying {crate_url}")


def wait_for_sparse_index(unit: PublishUnit, user_agent: str, timeout_sec: int) -> None:
    """Poll index.crates.io until the newly published crate version becomes visible."""
    url = sparse_index_url(unit.name)
    deadline = time.monotonic() + timeout_sec
    while True:
        status, body, _ = _http_get_status(url, user_agent, timeout=15.0)
        if status == 200:
            for line in body.splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("vers") == unit.version:
                    return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_sec}s waiting for {unit.name}=={unit.version} in {url}"
            )
        time.sleep(3.0)


def _parse_retry_delay(output_text: str, default_sleep: int) -> int:
    m_sec = re.search(r"retry-after[:\s]+(\d+)", output_text, re.IGNORECASE)
    if m_sec:
        return max(int(m_sec.group(1)) + 5, 10)
    m_date = re.search(
        r"Please try again after\s+([A-Za-z]{3},\s+\d+\s+[A-Za-z]+\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+GMT)",
        output_text,
    )
    if m_date:
        try:
            dt = email.utils.parsedate_to_datetime(m_date.group(1))
            now = datetime.datetime.now(datetime.timezone.utc)
            delta = int((dt - now).total_seconds()) + 5
            if delta > 0:
                return delta
        except (TypeError, ValueError):
            pass
    return default_sleep


def _format_eta(seconds: float) -> str:
    mins, secs = divmod(max(0, int(round(seconds))), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}h{mins:02d}m{secs:02d}s"


def _sleep_with_progress(seconds: int, reason: str) -> None:
    if seconds <= 0:
        return
    resume_at = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)).strftime(
        "%H:%M:%SZ"
    )
    print(f"[cooldown] Sleeping {seconds}s ({reason}; resuming ~{resume_at})...", flush=True)
    remaining = seconds
    while remaining > 0:
        chunk = min(remaining, 60)
        time.sleep(chunk)
        remaining -= chunk
        if remaining > 0:
            print(f"[cooldown] {remaining}s remaining...", flush=True)


def _run_cargo_publish(unit: PublishUnit, args: argparse.Namespace, repo_root: str) -> tuple[int, str]:
    cmd = ["cargo", "publish", "-p", unit.name]
    if args.locked:
        cmd.append("--locked")
    if args.allow_dirty:
        cmd.append("--allow-dirty")
    if args.no_verify:
        cmd.append("--no-verify")
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, combined


def filter_plan(plan: list[PublishUnit], start_from: str | None, only: list[str]) -> list[PublishUnit]:
    """Apply optional --start-from and --only filters while preserving topological order."""
    selected = plan
    if start_from:
        names = [u.name for u in selected]
        if start_from not in names:
            raise ValueError(f"--start-from crate {start_from!r} not found in publishable crates")
        idx = names.index(start_from)
        selected = selected[idx:]
    if only:
        wanted: set[str] = set()
        for item in only:
            for part in item.split(","):
                part = part.strip()
                if part:
                    wanted.add(part)
        known = {u.name for u in plan}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"--only contains unknown/non-publishable crates: {unknown}")
        selected = [u for u in selected if u.name in wanted]
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Topologically publish or bootstrap all 24 publishable egraph-rs crates to crates.io."
    )
    parser.add_argument("--repo-root", default=".", help="Path to repository root (default: .)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without publishing")
    parser.add_argument("--offline", action="store_true", help="Skip crates.io HTTP API queries in --dry-run")
    parser.add_argument("--allow-dirty", action="store_true", help="Pass --allow-dirty to cargo publish")
    parser.add_argument("--locked", action="store_true", help="Pass --locked to cargo publish")
    parser.add_argument("--no-verify", action="store_true", help="Pass --no-verify to cargo publish")
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=610,
        help="Cooldown seconds between brand-new crate creations after the 5-crate burst (default: 610)",
    )
    parser.add_argument(
        "--existing-sleep-seconds",
        type=int,
        default=65,
        help="Cooldown seconds after exhausting the 30-crate existing-version burst (default: 65)",
    )
    parser.add_argument(
        "--index-poll-timeout",
        type=int,
        default=120,
        help="Max seconds to poll index.crates.io between dependency tiers (default: 120)",
    )
    parser.add_argument("--start-from", default=None, help="Resume topological order from this crate")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Publish only specified crate(s) (repeatable or comma-separated)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = os.path.abspath(args.repo_root)
    if args.offline and not args.dry_run:
        print("--offline may only be combined with --dry-run", file=sys.stderr)
        return 2
    if not args.dry_run and not os.environ.get("CARGO_REGISTRY_TOKEN"):
        print("CARGO_REGISTRY_TOKEN must be set when not running with --dry-run", file=sys.stderr)
        return 2

    plan = filter_plan(compute_publish_plan(repo_root), args.start_from, args.only)
    if not plan:
        print("No publishable crates matched the requested filter.")
        return 0

    ua = f"{USER_AGENT_PREFIX}/{plan[0].version} (https://github.com/likr/egraph-rs)"
    total = len(plan)
    tier_published: collections.defaultdict[int, list[PublishUnit]] = collections.defaultdict(list)
    prev_tier: int | None = None
    new_published_count = 0
    existing_published_count = 0

    for idx, unit in enumerate(plan, start=1):
        if prev_tier is not None and unit.tier != prev_tier and not args.dry_run:
            for prev_unit in tier_published[prev_tier]:
                print(
                    f"[index] Waiting for {prev_unit.name}=={prev_unit.version} in sparse index...",
                    flush=True,
                )
                wait_for_sparse_index(prev_unit, ua, args.index_poll_timeout)
        prev_tier = unit.tier

        if args.offline:
            ver_published, crate_exists = False, False
        else:
            ver_published, crate_exists = query_crates_io_state(unit, ua)

        if ver_published:
            print(f"[{idx:2d}/{total}] tier={unit.tier} {unit.name}=={unit.version} -> SKIP (already published)")
            continue

        kind = "existing" if crate_exists else "new"
        remaining_after = total - idx
        est_cooldown = (
            max(0, remaining_after - max(0, NEW_CRATE_BURST - new_published_count - 1)) * args.sleep_seconds
            if not crate_exists
            else 0
        )
        print(
            f"[{idx:2d}/{total}] tier={unit.tier} {unit.name}=={unit.version} "
            f"kind={kind} remaining={remaining_after} est_wait={_format_eta(est_cooldown)}",
            flush=True,
        )
        if args.dry_run:
            if not crate_exists:
                new_published_count += 1
            else:
                existing_published_count += 1
            continue

        if not crate_exists and new_published_count >= NEW_CRATE_BURST:
            _sleep_with_progress(args.sleep_seconds, f"new-crate burst ({NEW_CRATE_BURST}) reached")
        elif crate_exists and existing_published_count >= EXISTING_CRATE_BURST:
            _sleep_with_progress(args.existing_sleep_seconds, f"existing burst ({EXISTING_CRATE_BURST}) reached")
            existing_published_count = 0

        for attempt in range(1, 5):
            rc, out = _run_cargo_publish(unit, args, repo_root)
            if rc == 0:
                print(f"[ok] Published {unit.name}=={unit.version}", flush=True)
                tier_published[unit.tier].append(unit)
                if crate_exists:
                    existing_published_count += 1
                else:
                    new_published_count += 1
                break
            lower_out = out.lower()
            if "already exists" in lower_out or "already uploaded" in lower_out:
                print(f"[ok] {unit.name}=={unit.version} already exists on crates.io", flush=True)
                break
            if ("429" in out or "too many" in lower_out or "rate limit" in lower_out) and attempt < 4:
                delay = _parse_retry_delay(out, args.sleep_seconds)
                _sleep_with_progress(delay, f"HTTP 429 rate-limit retry {attempt}/3")
                continue
            print(out, file=sys.stderr)
            return rc if rc != 0 else 1

    print(f"Completed plan ({total} crates processed; dry_run={args.dry_run}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
