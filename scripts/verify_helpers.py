#!/usr/bin/env python3
"""Shared CLI helpers for scripts/verify.sh and scripts/check_invariants.sh."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
from typing import Any

from packaging_checks import collect_packaging_issues


def cmd_now(_args: argparse.Namespace) -> int:
    print(f"{time.time():.3f}")
    return 0


def cmd_record_step(args: argparse.Namespace) -> int:
    elapsed = round(time.time() - float(args.start_ts), 3)
    record = {
        "name": args.name,
        "status": "pass" if int(args.rc) == 0 else "fail",
        "elapsed_sec": elapsed,
    }
    with pathlib.Path(args.records_file).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return 0


def cmd_write_summary(args: argparse.Namespace) -> int:
    out_path = pathlib.Path(args.out_path)
    records_file = pathlib.Path(args.records_file)

    steps: list[dict[str, Any]] = []
    if records_file.exists():
        for line in records_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                steps.append(json.loads(line))

    branch = subprocess.check_output(
        ["git", "-C", args.repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
        text=True,
    ).strip()
    sha = subprocess.check_output(
        ["git", "-C", args.repo_root, "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()

    payload: dict[str, Any] = {
        "status": args.status,
        "worktree": args.repo_root,
        "branch": branch,
        "head_sha": sha,
        "parallel_workers": int(args.workers),
        "cargo_build_jobs": int(args.jobs),
        "rustc_wrapper": args.wrapper,
        "total_elapsed_sec": round(time.time() - float(args.start_ts), 3),
        "steps": steps,
        "sccache_stats": args.sccache_info,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_detect_changed_crates(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.repo_root)
    changed: set[str] = set()
    for git_cmd in (
        ["git", "-C", str(root), "diff", "--name-only", "HEAD"],
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
    ):
        out = subprocess.check_output(git_cmd, text=True).splitlines()
        changed.update(out)

    pkgs: list[str] = []
    for toml in root.glob("crates/**/Cargo.toml"):
        crate_dir = toml.parent.relative_to(root).as_posix() + "/"
        if any(f.startswith(crate_dir) for f in changed):
            txt = toml.read_text(encoding="utf-8")
            match = re.search(r'^name\s*=\s*"([^"]+)"', txt, re.MULTILINE)
            if match and match.group(1) not in ("egraph-python", "egraph-wasm"):
                pkgs.append(match.group(1))

    for pkg in sorted(set(pkgs)):
        print(pkg)
    return 0


def _cargo_metadata_packages() -> list[str]:
    raw = subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    data: dict[str, Any] = json.loads(raw)
    return [str(pkg["name"]) for pkg in data.get("packages", [])]


def cmd_list_packages(args: argparse.Namespace) -> int:
    excluded = set(args.exclude)
    names = [pkg for pkg in _cargo_metadata_packages() if pkg not in excluded]
    if args.format == "clippy-args":
        print(" ".join(f"-p {name}" for name in names))
    else:
        for name in names:
            print(name)
    return 0


def cmd_has_package(args: argparse.Namespace) -> int:
    try:
        pkgs = set(_cargo_metadata_packages())
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return 1
    return 0 if args.name in pkgs else 1


def cmd_check_packaging(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.repo_root)
    strict_all = bool(args.strict_metadata)
    strict_pub = bool(args.strict_publishable_metadata) or strict_all
    for issue in collect_packaging_issues(root, strict_all=strict_all, strict_pub=strict_pub):
        print(issue)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verification shell script helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    p_now = sub.add_parser("now", help="Print current UNIX epoch timestamp")
    p_now.set_defaults(func=cmd_now)

    p_rec = sub.add_parser("record-step", help="Append one step timing record")
    p_rec.add_argument("records_file")
    p_rec.add_argument("name")
    p_rec.add_argument("rc")
    p_rec.add_argument("start_ts")
    p_rec.set_defaults(func=cmd_record_step)

    p_sum = sub.add_parser("write-summary", help="Write final JSON summary")
    p_sum.add_argument("out_path")
    p_sum.add_argument("status")
    p_sum.add_argument("repo_root")
    p_sum.add_argument("workers")
    p_sum.add_argument("jobs")
    p_sum.add_argument("wrapper")
    p_sum.add_argument("start_ts")
    p_sum.add_argument("records_file")
    p_sum.add_argument("sccache_info")
    p_sum.set_defaults(func=cmd_write_summary)

    p_chg = sub.add_parser("detect-changed-crates", help="List modified Rust library crates")
    p_chg.add_argument("repo_root")
    p_chg.set_defaults(func=cmd_detect_changed_crates)

    p_lst = sub.add_parser("list-packages", help="List workspace Rust package names")
    p_lst.add_argument("--exclude", action="append", default=[])
    p_lst.add_argument("--format", choices=("names", "clippy-args"), default="names")
    p_lst.set_defaults(func=cmd_list_packages)

    p_has = sub.add_parser("has-package", help="Check whether a workspace Rust package exists")
    p_has.add_argument("name")
    p_has.set_defaults(func=cmd_has_package)

    p_pkg = sub.add_parser("check-packaging", help="Check cross-target packaging metadata")
    p_pkg.add_argument("repo_root")
    p_pkg.add_argument("--strict-metadata", action="store_true", help="Check all workspace crates")
    p_pkg.add_argument(
        "--strict-publishable-metadata",
        action="store_true",
        help="Check publishable workspace crates only",
    )
    p_pkg.set_defaults(func=cmd_check_packaging)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Any = args.func
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
