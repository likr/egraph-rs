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

_tomllib: Any
try:
    import tomllib as _tomllib_mod

    _tomllib = _tomllib_mod
except ModuleNotFoundError:
    _tomllib = None

V6_CRATE_SUBDIRS: tuple[str, ...] = (
    "core",
    "ffi",
    "petgraph",
    "dataset",
    "spatial-trees",
    "spatial-fft",
    "spatial-crossings",
    "spatial",
    "metrics-geometry",
    "metrics-cluster",
    "metrics",
    "linalg",
    "linalg-mds",
    "linalg-amg",
    "neighborhood",
    "cluster",
    "layout-constraints",
    "layout-stress",
    "layout-force",
    "layout-neighbor",
    "layout-sgd",
    "layout-multilevel",
    "layout",
    "egraph",
    "cli",
)


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


def _load_toml(path: pathlib.Path) -> dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    if _tomllib is not None:
        loaded: dict[str, Any] = _tomllib.loads(txt)
        return loaded

    # Minimal regex fallback for Python < 3.11 without tomllib.
    data: dict[str, Any] = {"_raw": txt}
    ws_ver = re.search(
        r'^\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"',
        txt,
        re.MULTILINE | re.DOTALL,
    )
    if ws_ver:
        data.setdefault("workspace", {}).setdefault("package", {})["version"] = ws_ver.group(1)
    pkg_sec = re.search(r"^\[package\][^\[]*", txt, re.MULTILINE | re.DOTALL)
    if pkg_sec:
        sec = pkg_sec.group(0)
        if re.search(r"^version\.workspace\s*=\s*true", sec, re.MULTILINE) or re.search(
            r"^version\s*=\s*\{\s*workspace\s*=\s*true",
            sec,
            re.MULTILINE,
        ):
            data.setdefault("package", {})["version"] = {"workspace": True}
        else:
            match = re.search(r'^version\s*=\s*"([^"]+)"', sec, re.MULTILINE)
            if match:
                data.setdefault("package", {})["version"] = match.group(1)
    return data


def _resolve_cargo_version(cargo_path: pathlib.Path, ws_version: str | None) -> str | None:
    data = _load_toml(cargo_path)
    ver = data.get("package", {}).get("version")
    if isinstance(ver, dict) and ver.get("workspace") is True:
        return ws_version
    if isinstance(ver, str):
        return ver
    return None


def _check_python_packaging(
    root: pathlib.Path,
    root_cargo_path: pathlib.Path,
    ws_version: str | None,
    ws_deps: dict[str, Any],
    issues: list[str],
) -> None:
    py_toml = root / "crates" / "python" / "Cargo.toml"
    py_proj = root / "crates" / "python" / "pyproject.toml"
    if not (py_toml.exists() and py_proj.exists()):
        return

    py_cargo_data = _load_toml(py_toml)
    py_cargo_ver = _resolve_cargo_version(py_toml, ws_version)
    if ws_version and py_cargo_ver != ws_version:
        issues.append(
            f"crates/python/Cargo.toml version ({py_cargo_ver}) != workspace version ({ws_version})"
        )

    if _tomllib is not None:
        pyo3_spec = py_cargo_data.get("dependencies", {}).get("pyo3")
        pyo3_features: list[str] = []
        if isinstance(pyo3_spec, dict):
            pyo3_features.extend(str(f) for f in pyo3_spec.get("features", []))
            ws_pyo3 = ws_deps.get("pyo3")
            if pyo3_spec.get("workspace") is True and isinstance(ws_pyo3, dict):
                pyo3_features.extend(str(f) for f in ws_pyo3.get("features", []))
        pyproj_data = _load_toml(py_proj)
        req_py = str(pyproj_data.get("project", {}).get("requires-python", ""))
        proj_deps = [str(d) for d in pyproj_data.get("project", {}).get("dependencies", [])]
    else:
        t_txt = py_toml.read_text(encoding="utf-8")
        r_txt = root_cargo_path.read_text(encoding="utf-8") if root_cargo_path.exists() else ""
        p_txt = py_proj.read_text(encoding="utf-8")
        combined = t_txt + ("\n" + r_txt if "pyo3.workspace" in t_txt else "")
        pyo3_features = re.findall(r"abi3-py\d+", combined)
        m_req = re.search(r'^requires-python\s*=\s*"([^"]+)"', p_txt, re.MULTILINE)
        req_py = m_req.group(1) if m_req else ""
        m_deps = re.search(r"^dependencies\s*=\s*\[([^\]]*)\]", p_txt, re.MULTILINE | re.DOTALL)
        proj_deps = re.findall(r'"([^"]+)"', m_deps.group(1)) if m_deps else []

    if (
        any(feat in pyo3_features for feat in ("abi3-py37", "abi3-py38", "abi3-py39"))
        or "abi3-py310" not in pyo3_features
    ):
        issues.append(f"Python pyo3 abi3 features ({pyo3_features}) must enable abi3-py310")
    if not req_py.startswith(">=3.10"):
        issues.append(
            f"crates/python/pyproject.toml requires-python ({req_py!r}) must be '>=3.10'"
        )
    if not any(re.match(r"^\s*numpy(\b|[<>=!~])", dep) for dep in proj_deps):
        issues.append("crates/python/pyproject.toml omits runtime numpy dependency")


def _check_wasm_packaging(
    root: pathlib.Path,
    ws_version: str | None,
    issues: list[str],
) -> None:
    wasm_toml = root / "crates" / "wasm" / "Cargo.toml"
    wasm_pkg = root / "crates" / "wasm" / "package.json"
    if not (wasm_toml.exists() and wasm_pkg.exists()):
        return

    wasm_cargo_ver = _resolve_cargo_version(wasm_toml, ws_version)
    wasm_pkg_data = json.loads(wasm_pkg.read_text(encoding="utf-8"))
    wasm_pkg_ver = wasm_pkg_data.get("version")
    if ws_version and wasm_cargo_ver != ws_version:
        issues.append(
            f"crates/wasm/Cargo.toml version ({wasm_cargo_ver}) != workspace version ({ws_version})"
        )
    if ws_version and wasm_pkg_ver != ws_version:
        issues.append(
            f"crates/wasm/package.json version ({wasm_pkg_ver}) != workspace version ({ws_version})"
        )
    if wasm_cargo_ver != wasm_pkg_ver:
        issues.append(
            f"WASM Cargo.toml version ({wasm_cargo_ver}) != package.json version ({wasm_pkg_ver})"
        )
    repo_meta = wasm_pkg_data.get("repository")
    if (
        not isinstance(repo_meta, dict)
        or repo_meta.get("type") != "git"
        or repo_meta.get("directory") != "crates/wasm"
    ):
        issues.append(
            "crates/wasm/package.json repository must specify type='git' and "
            "directory='crates/wasm' for npm provenance"
        )
    repo_url = repo_meta.get("url", "") if isinstance(repo_meta, dict) else ""
    if "github.com/likr/egraph-rs" not in str(repo_url):
        issues.append(
            f"crates/wasm/package.json repository.url ({repo_url!r}) must point to "
            "github.com/likr/egraph-rs"
        )


def cmd_check_packaging(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.repo_root)
    issues: list[str] = []

    root_cargo_path = root / "Cargo.toml"
    ws_version: str | None = None
    ws_deps: dict[str, Any] = {}
    if root_cargo_path.exists():
        root_data = _load_toml(root_cargo_path)
        raw_ver = root_data.get("workspace", {}).get("package", {}).get("version")
        ws_version = str(raw_ver) if raw_ver is not None else None
        raw_deps = root_data.get("workspace", {}).get("dependencies", {})
        if isinstance(raw_deps, dict):
            ws_deps = raw_deps

    for sub in V6_CRATE_SUBDIRS:
        sub_toml = root / "crates" / sub / "Cargo.toml"
        if sub_toml.exists() and ws_version:
            sub_ver = _resolve_cargo_version(sub_toml, ws_version)
            if sub_ver != ws_version:
                issues.append(
                    f"crates/{sub}/Cargo.toml version ({sub_ver}) != workspace version ({ws_version})"
                )

    _check_python_packaging(root, root_cargo_path, ws_version, ws_deps, issues)
    _check_wasm_packaging(root, ws_version, issues)

    for issue in issues:
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
    p_pkg.set_defaults(func=cmd_check_packaging)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Any = args.func
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
