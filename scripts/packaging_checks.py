#!/usr/bin/env python3
"""Cross-target and crates.io packaging metadata validation for egraph-rs."""

from __future__ import annotations

import json
import pathlib
import re
import tomllib
from typing import Any

NON_PUBLISHABLE_CRATES: frozenset[str] = frozenset(
    {"egraph-ffi", "egraph-python", "egraph-wasm", "egraph-cli"}
)
FACADE_CRATE_NAME = "egraph-rs"
VALID_CRATES_IO_CATEGORIES: frozenset[str] = frozenset(
    {"algorithms", "data-structures", "graphics", "mathematics", "science"}
)
KEYWORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,19}$")


def load_toml(path: pathlib.Path) -> dict[str, Any]:
    """Load a TOML manifest with typed dictionary output."""
    loaded: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    return loaded


def workspace_member_dirs(root: pathlib.Path, root_data: dict[str, Any]) -> list[pathlib.Path]:
    """Resolve [workspace].members directories relative to repo root."""
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


def resolve_workspace_field(
    pkg_dict: dict[str, Any],
    field: str,
    ws_pkg_dict: dict[str, Any],
) -> str | None:
    """Resolve a [package] field supporting `field.workspace = true`."""
    val = pkg_dict.get(field)
    if isinstance(val, dict) and val.get("workspace") is True:
        ws_val = ws_pkg_dict.get(field)
        return str(ws_val) if isinstance(ws_val, str) else None
    if isinstance(val, str):
        return val
    return None


def resolve_cargo_version(cargo_path: pathlib.Path, ws_version: str | None) -> str | None:
    """Return resolved package version string for a member Cargo.toml."""
    data = load_toml(cargo_path)
    pkg_dict = data.get("package")
    if not isinstance(pkg_dict, dict):
        return None
    ws_pkg: dict[str, Any] = {"version": ws_version} if ws_version is not None else {}
    return resolve_workspace_field(pkg_dict, "version", ws_pkg)


def _ws_str(ws_pkg_dict: dict[str, Any], key: str) -> str | None:
    val = ws_pkg_dict.get(key)
    return val if isinstance(val, str) else None


def _check_python_packaging(
    root: pathlib.Path,
    ws_version: str | None,
    ws_deps: dict[str, Any],
    issues: list[str],
) -> None:
    py_toml = root / "crates" / "python" / "Cargo.toml"
    py_proj = root / "crates" / "python" / "pyproject.toml"
    if not (py_toml.exists() and py_proj.exists()):
        return

    py_cargo_data = load_toml(py_toml)
    py_cargo_ver = resolve_cargo_version(py_toml, ws_version)
    if ws_version and py_cargo_ver != ws_version:
        issues.append(
            f"crates/python/Cargo.toml version ({py_cargo_ver}) != workspace version ({ws_version})"
        )

    pyo3_spec = py_cargo_data.get("dependencies", {}).get("pyo3")
    pyo3_features: list[str] = []
    if isinstance(pyo3_spec, dict):
        pyo3_features.extend(str(f) for f in pyo3_spec.get("features", []))
        ws_pyo3 = ws_deps.get("pyo3")
        if pyo3_spec.get("workspace") is True and isinstance(ws_pyo3, dict):
            pyo3_features.extend(str(f) for f in ws_pyo3.get("features", []))
    pyproj_data = load_toml(py_proj)
    req_py = str(pyproj_data.get("project", {}).get("requires-python", ""))
    proj_deps = [str(d) for d in pyproj_data.get("project", {}).get("dependencies", [])]

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

    wasm_cargo_ver = resolve_cargo_version(wasm_toml, ws_version)
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


def _check_publishable_crates_io_fields(
    root: pathlib.Path,
    member_dir: pathlib.Path,
    rel_dir: str,
    pkg_name: str,
    pkg_dict: dict[str, Any],
    ws_pkg_dict: dict[str, Any],
    root_license_bytes: bytes | None,
    issues: list[str],
) -> None:
    desc = pkg_dict.get("description")
    if not isinstance(desc, str) or not desc.strip():
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) must set a non-empty description")
    if pkg_dict.get("publish") is False:
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) must not set publish = false")

    ws_repository = _ws_str(ws_pkg_dict, "repository")
    repo = resolve_workspace_field(pkg_dict, "repository", ws_pkg_dict)
    if ws_repository is not None and repo != ws_repository:
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) repository ({repo}) != {ws_repository}")

    homepage = resolve_workspace_field(pkg_dict, "homepage", ws_pkg_dict)
    if not homepage or (repo and homepage != repo):
        issues.append(
            f"{rel_dir}/Cargo.toml ({pkg_name}) homepage ({homepage!r}) must match repository ({repo!r})"
        )

    expected_docs = f"https://docs.rs/{pkg_name}"
    if pkg_dict.get("documentation") != expected_docs:
        issues.append(
            f"{rel_dir}/Cargo.toml ({pkg_name}) documentation ({pkg_dict.get('documentation')!r}) "
            f"!= {expected_docs!r}"
        )

    license_val = resolve_workspace_field(pkg_dict, "license", ws_pkg_dict)
    if license_val != "MIT":
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) license ({license_val!r}) != 'MIT'")
    if "license-file" in pkg_dict:
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) must not set license-file alongside MIT")

    crate_license = member_dir / "LICENSE"
    if not crate_license.exists():
        issues.append(f"{rel_dir}/LICENSE is missing for publishable crate {pkg_name}")
    elif root_license_bytes is not None and crate_license.read_bytes() != root_license_bytes:
        issues.append(f"{rel_dir}/LICENSE differs from root LICENSE")

    readme_raw = pkg_dict.get("readme")
    if isinstance(readme_raw, dict) and readme_raw.get("workspace") is True:
        ws_readme = _ws_str(ws_pkg_dict, "readme")
        readme_path = root / ws_readme if ws_readme else None
    elif isinstance(readme_raw, str) and readme_raw.strip():
        readme_path = member_dir / readme_raw
    else:
        readme_path = None
    if readme_path is None or not readme_path.exists():
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) readme target is missing or unset")

    keywords = pkg_dict.get("keywords")
    if not isinstance(keywords, list) or not (1 <= len(keywords) <= 5):
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) keywords must contain 1..=5 entries")
    else:
        for kw in keywords:
            if not isinstance(kw, str) or not KEYWORD_RE.fullmatch(kw):
                issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) invalid keyword {kw!r}")

    categories = pkg_dict.get("categories")
    if not isinstance(categories, list) or not (1 <= len(categories) <= 5):
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) categories must contain 1..=5 entries")
    else:
        for cat in categories:
            if not isinstance(cat, str) or cat not in VALID_CRATES_IO_CATEGORIES:
                issues.append(
                    f"{rel_dir}/Cargo.toml ({pkg_name}) category {cat!r} not in allowed slugs "
                    f"{sorted(VALID_CRATES_IO_CATEGORIES)}"
                )

    meta = pkg_dict.get("metadata")
    docs_rs = meta.get("docs", {}).get("rs") if isinstance(meta, dict) and isinstance(meta.get("docs"), dict) else None
    if not isinstance(docs_rs, dict) or docs_rs.get("all-features") is not True:
        issues.append(
            f"{rel_dir}/Cargo.toml ({pkg_name}) must define [package.metadata.docs.rs] all-features = true"
        )
    else:
        rustdoc_args = docs_rs.get("rustdoc-args")
        if not isinstance(rustdoc_args, list) or "--cfg" not in rustdoc_args or "docsrs" not in rustdoc_args:
            issues.append(
                f"{rel_dir}/Cargo.toml ({pkg_name}) [package.metadata.docs.rs].rustdoc-args must include '--cfg', 'docsrs'"
            )


def _check_member_metadata(
    root: pathlib.Path,
    member_dir: pathlib.Path,
    rel_dir: str,
    pkg_dict: dict[str, Any],
    ws_pkg_dict: dict[str, Any],
    ws_deps: dict[str, Any],
    strict_all: bool,
    strict_pub: bool,
    root_license_bytes: bytes | None,
    issues: list[str],
) -> tuple[str | None, bool]:
    pkg_name = pkg_dict.get("name")
    if not isinstance(pkg_name, str) or not pkg_name:
        issues.append(f"{rel_dir}/Cargo.toml is missing package.name")
        return None, False

    ws_version = _ws_str(ws_pkg_dict, "version")
    ws_rust_version = _ws_str(ws_pkg_dict, "rust-version")
    if ws_version is not None:
        pkg_ver = resolve_workspace_field(pkg_dict, "version", ws_pkg_dict)
        if pkg_ver != ws_version:
            issues.append(f"{rel_dir}/Cargo.toml version ({pkg_ver}) != workspace ({ws_version})")

    is_non_pub = pkg_name in NON_PUBLISHABLE_CRATES
    if is_non_pub and "publish" in pkg_dict and pkg_dict["publish"] is not False:
        issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) publish={pkg_dict['publish']!r} != false")

    if ws_rust_version is not None and "rust-version" in pkg_dict:
        pkg_msrv = resolve_workspace_field(pkg_dict, "rust-version", ws_pkg_dict)
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
        _check_publishable_crates_io_fields(
            root, member_dir, rel_dir, pkg_name, pkg_dict, ws_pkg_dict, root_license_bytes, issues
        )

    return pkg_name, not is_non_pub


def collect_packaging_issues(
    root: pathlib.Path,
    strict_all: bool,
    strict_pub: bool,
) -> list[str]:
    """Validate workspace Cargo, crates.io, Python, and WASM packaging metadata."""
    issues: list[str] = []
    root_cargo_path = root / "Cargo.toml"
    if not root_cargo_path.exists():
        return ["Cargo.toml is missing at repository root"]

    root_data = load_toml(root_cargo_path)
    ws_section = root_data.get("workspace")
    ws_dict = ws_section if isinstance(ws_section, dict) else {}
    raw_ws_pkg = ws_dict.get("package")
    ws_pkg_dict = raw_ws_pkg if isinstance(raw_ws_pkg, dict) else {}
    raw_deps = ws_dict.get("dependencies")
    ws_deps = raw_deps if isinstance(raw_deps, dict) else {}

    ws_version = _ws_str(ws_pkg_dict, "version")
    root_license_path = root / "LICENSE"
    root_license_bytes = root_license_path.read_bytes() if root_license_path.exists() else None

    if strict_pub:
        for req_key in ("rust-version", "repository", "homepage", "readme", "license"):
            if not isinstance(ws_pkg_dict.get(req_key), str):
                issues.append(f"Cargo.toml [workspace.package].{req_key} must be set")
        if root_license_bytes is None:
            issues.append("Root LICENSE file is missing")
        ws_readme = _ws_str(ws_pkg_dict, "readme")
        if ws_readme and not (root / ws_readme).exists():
            issues.append(f"Cargo.toml [workspace.package].readme ({ws_readme!r}) does not exist")

    member_dirs = workspace_member_dirs(root, root_data)
    if not member_dirs:
        issues.append("Cargo.toml [workspace].members is empty or missing")

    publishable_names: list[str] = []
    for member_dir in member_dirs:
        rel_dir = member_dir.relative_to(root).as_posix()
        member_toml = member_dir / "Cargo.toml"
        if not member_toml.exists():
            issues.append(f"{rel_dir}/Cargo.toml is missing for workspace member")
            continue
        member_data = load_toml(member_toml)
        pkg_dict = member_data.get("package")
        if not isinstance(pkg_dict, dict):
            issues.append(f"{rel_dir}/Cargo.toml is missing [package] table")
            continue
        pkg_name, is_pub = _check_member_metadata(
            root,
            member_dir,
            rel_dir,
            pkg_dict,
            ws_pkg_dict,
            ws_deps,
            strict_all,
            strict_pub,
            root_license_bytes,
            issues,
        )
        if pkg_name and is_pub:
            publishable_names.append(pkg_name)

    if strict_pub:
        if FACADE_CRATE_NAME not in publishable_names:
            issues.append(f"Publishable facade crate {FACADE_CRATE_NAME!r} is missing from workspace")
        if "egraph" in publishable_names:
            issues.append("Legacy crate name 'egraph' must be renamed to 'egraph-rs'")

    _check_python_packaging(root, ws_version, ws_deps, issues)
    _check_wasm_packaging(root, ws_version, issues)
    return issues
