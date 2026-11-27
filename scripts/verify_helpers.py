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

NON_PUBLISHABLE_CRATES: frozenset[str] = frozenset({"egraph-ffi", "egraph-python", "egraph-wasm", "egraph-cli"})


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
    ws_sec = re.search(r"^\[workspace\][^\[]*", txt, re.MULTILINE | re.DOTALL)
    if ws_sec:
        m_mem = re.search(r"^members\s*=\s*\[([^\]]*)\]", ws_sec.group(0), re.MULTILINE | re.DOTALL)
        if m_mem:
            data.setdefault("workspace", {})["members"] = re.findall(r'"([^"]+)"', m_mem.group(1))
    ws_pkg_sec = re.search(r"^\[workspace\.package\][^\[]*", txt, re.MULTILINE | re.DOTALL)
    if ws_pkg_sec:
        ws_pkg = data.setdefault("workspace", {}).setdefault("package", {})
        for key in ("version", "repository", "rust-version"):
            m_val = re.search(rf'^{re.escape(key)}\s*=\s*"([^"]+)"', ws_pkg_sec.group(0), re.MULTILINE)
            if m_val:
                ws_pkg[key] = m_val.group(1)
    pkg_sec = re.search(r"^\[package\][^\[]*", txt, re.MULTILINE | re.DOTALL)
    if pkg_sec:
        sec, pkg = pkg_sec.group(0), data.setdefault("package", {})
        for key in ("name", "version", "repository", "rust-version", "description"):
            ek = re.escape(key)
            if re.search(rf"^{ek}(?:\.workspace|\s*=\s*\{{\s*workspace)\s*=\s*true", sec, re.MULTILINE):
                pkg[key] = {"workspace": True}
            elif match := re.search(rf'^{ek}\s*=\s*"([^"]+)"', sec, re.MULTILINE):
                pkg[key] = match.group(1)
        if m_pub := re.search(r"^publish\s*=\s*(true|false)\b", sec, re.MULTILINE):
            pkg["publish"] = m_pub.group(1) == "true"
    return data


def _workspace_member_dirs(root: pathlib.Path, root_data: dict[str, Any]) -> list[pathlib.Path]:
    ws_section = root_data.get("workspace")
    raw_members = ws_section.get("members", []) if isinstance(ws_section, dict) else []
    if not isinstance(raw_members, list):
        return []
    member_dirs: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for entry in raw_members:
        rel = str(entry).strip()
        if not rel:
            continue
        if any(ch in rel for ch in "*?["):
            candidates = sorted(p for p in root.glob(rel) if p.is_dir())
        else:
            candidates = [root / rel]
        for path in candidates:
            if path not in seen:
                seen.add(path)
                member_dirs.append(path)
    return member_dirs


def _resolve_workspace_field(
    pkg_dict: dict[str, Any],
    field: str,
    ws_pkg_dict: dict[str, Any],
) -> str | None:
    val = pkg_dict.get(field)
    if isinstance(val, dict) and val.get("workspace") is True:
        ws_val = ws_pkg_dict.get(field)
        return str(ws_val) if isinstance(ws_val, str) else None
    if isinstance(val, str):
        return val
    return None


def _resolve_cargo_version(cargo_path: pathlib.Path, ws_version: str | None) -> str | None:
    data = _load_toml(cargo_path)
    pkg_dict = data.get("package")
    if not isinstance(pkg_dict, dict):
        return None
    ws_pkg: dict[str, Any] = {"version": ws_version} if ws_version is not None else {}
    return _resolve_workspace_field(pkg_dict, "version", ws_pkg)


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


def _ws_str(ws_pkg_dict: dict[str, Any], key: str) -> str | None:
    val = ws_pkg_dict.get(key)
    return val if isinstance(val, str) else None


def _check_member_metadata(
    rel_dir: str,
    pkg_dict: dict[str, Any],
    ws_pkg_dict: dict[str, Any],
    ws_deps: dict[str, Any],
    strict_all: bool,
    strict_pub: bool,
    issues: list[str],
) -> None:
    pkg_name = pkg_dict.get("name")
    if not isinstance(pkg_name, str) or not pkg_name:
        issues.append(f"{rel_dir}/Cargo.toml is missing package.name")
        return

    ws_version = _ws_str(ws_pkg_dict, "version")
    ws_repository = _ws_str(ws_pkg_dict, "repository")
    ws_rust_version = _ws_str(ws_pkg_dict, "rust-version")
    if ws_version is not None:
        pkg_ver = _resolve_workspace_field(pkg_dict, "version", ws_pkg_dict)
        if pkg_ver != ws_version:
            issues.append(f"{rel_dir}/Cargo.toml version ({pkg_ver}) != workspace ({ws_version})")

    is_non_pub = pkg_name in NON_PUBLISHABLE_CRATES
    if is_non_pub and "publish" in pkg_dict and pkg_dict["publish"] is not False:
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) publish={pkg_dict['publish']!r} != false")

    if ws_rust_version is not None and "rust-version" in pkg_dict:
        pkg_msrv = _resolve_workspace_field(pkg_dict, "rust-version", ws_pkg_dict)
        if pkg_msrv != ws_rust_version:
            issues.append(
                f"{rel_dir}/Cargo.toml rust-version ({pkg_msrv}) != workspace ({ws_rust_version})"
            )

    dep_spec = ws_deps.get(pkg_name)
    if isinstance(dep_spec, dict) and "path" in dep_spec and ws_version is not None:
        dep_ver = dep_spec.get("version")
        if dep_ver is not None and (not isinstance(dep_ver, str) or dep_ver.lstrip("=") != ws_version):
            issues.append(
                f"[workspace.dependencies].{pkg_name} version ({dep_ver!r}) != {ws_version}"
            )
        elif dep_ver is None and strict_pub and not is_non_pub:
            issues.append(f"[workspace.dependencies].{pkg_name} must declare version {ws_version}")

    if (strict_all or (strict_pub and not is_non_pub)) and (
        ws_rust_version is not None and "rust-version" not in pkg_dict
    ):
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) must set rust-version ({ws_rust_version})")
    if strict_all and is_non_pub and pkg_dict.get("publish") is not False:
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) must explicitly set publish = false")

    if strict_pub and not is_non_pub:
        desc = pkg_dict.get("description")
        if not isinstance(desc, str) or not desc.strip():
            issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) must set a non-empty description")
        if ws_repository is not None:
            repo = _resolve_workspace_field(pkg_dict, "repository", ws_pkg_dict)
            if repo != ws_repository:
                issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) repository ({repo}) != {ws_repository}")
        if pkg_dict.get("publish") is False:
            issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) must not set publish = false")


def cmd_check_packaging(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.repo_root)
    issues: list[str] = []
    strict_all = bool(args.strict_metadata)
    strict_pub = bool(args.strict_publishable_metadata) or strict_all

    root_cargo_path = root / "Cargo.toml"
    if not root_cargo_path.exists():
        print("Cargo.toml is missing at repository root")
        return 0

    root_data = _load_toml(root_cargo_path)
    ws_section = root_data.get("workspace")
    ws_dict = ws_section if isinstance(ws_section, dict) else {}
    raw_ws_pkg = ws_dict.get("package")
    ws_pkg_dict = raw_ws_pkg if isinstance(raw_ws_pkg, dict) else {}
    raw_deps = ws_dict.get("dependencies")
    ws_deps = raw_deps if isinstance(raw_deps, dict) else {}

    ws_version = ws_pkg_dict.get("version") if isinstance(ws_pkg_dict.get("version"), str) else None
    if strict_pub:
        if not isinstance(ws_pkg_dict.get("rust-version"), str):
            issues.append("Cargo.toml [workspace.package].rust-version must be set")
        if not isinstance(ws_pkg_dict.get("repository"), str):
            issues.append("Cargo.toml [workspace.package].repository must be set")

    member_dirs = _workspace_member_dirs(root, root_data)
    if not member_dirs:
        issues.append("Cargo.toml [workspace].members is empty or missing")

    for member_dir in member_dirs:
        rel_dir = member_dir.relative_to(root).as_posix()
        member_toml = member_dir / "Cargo.toml"
        if not member_toml.exists():
            issues.append(f"{rel_dir}/Cargo.toml is missing for workspace member")
            continue
        member_data = _load_toml(member_toml)
        pkg_dict = member_data.get("package")
        if not isinstance(pkg_dict, dict):
            issues.append(f"{rel_dir}/Cargo.toml is missing [package] table")
            continue
        _check_member_metadata(
            rel_dir, pkg_dict, ws_pkg_dict, ws_deps, strict_all, strict_pub, issues
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
