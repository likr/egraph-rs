#!/usr/bin/env python3
"""README link, include/exclude, asset-escape, and doc_cfg checks for publishable crates."""

from __future__ import annotations

import pathlib
import re
from typing import Any

INLINE_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\n]+)\)")
REF_LINK_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
INCLUDE_MACRO_RE = re.compile(r'\b(?:include_str|include_bytes)!\(\s*"([^"]+)"\s*\)')
ALLOWED_URL_PREFIXES = ("https://", "http://", "mailto:", "#")


def _strip_fenced_code_blocks(markdown_text: str) -> str:
    """Remove fenced code blocks (``` or ~~~) so code snippets are not parsed as links."""
    kept: list[str] = []
    fence_char: str | None = None
    fence_len = 0
    for line in markdown_text.splitlines():
        stripped = line.lstrip()
        if fence_char is None:
            if stripped.startswith(("```", "~~~")):
                ch = stripped[0]
                run = len(stripped) - len(stripped.lstrip(ch))
                if run >= 3:
                    fence_char = ch
                    fence_len = run
                    continue
            kept.append(line)
        else:
            if stripped.startswith(fence_char * fence_len) and not stripped.lstrip(fence_char).strip():
                fence_char = None
                fence_len = 0
    return "\n".join(kept)


def _extract_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")].strip()
    return target.split()[0] if target else ""


def find_relative_markdown_links(markdown_text: str) -> list[str]:
    """Return any relative Markdown link targets outside code blocks and inline code."""
    cleaned = INLINE_CODE_RE.sub("", _strip_fenced_code_blocks(markdown_text))
    relative: list[str] = []
    for raw in INLINE_LINK_RE.findall(cleaned) + REF_LINK_RE.findall(cleaned):
        target = _extract_link_target(raw)
        if target and not target.startswith(ALLOWED_URL_PREFIXES):
            relative.append(target)
    return relative


def check_crate_readme(
    member_dir: pathlib.Path,
    rel_dir: str,
    pkg_name: str,
    pkg_dict: dict[str, Any],
    facade_name: str,
    issues: list[str],
) -> None:
    """Validate dedicated README.md, facade guidance, and absolute-only links."""
    readme_raw = pkg_dict.get("readme")
    if isinstance(readme_raw, dict) and readme_raw.get("workspace") is True:
        issues.append(
            f"{rel_dir}/Cargo.toml ({pkg_name}) must set readme = 'README.md' instead of readme.workspace = true"
        )
        return
    if readme_raw != "README.md":
        issues.append(
            f"{rel_dir}/Cargo.toml ({pkg_name}) readme ({readme_raw!r}) must be 'README.md'"
        )
        return

    readme_path = member_dir / "README.md"
    if not readme_path.exists():
        issues.append(f"{rel_dir}/README.md is missing for publishable crate {pkg_name}")
        return

    text = readme_path.read_text(encoding="utf-8")
    if not text.strip():
        issues.append(f"{rel_dir}/README.md is empty for publishable crate {pkg_name}")
        return

    if pkg_name != facade_name and facade_name not in text:
        issues.append(
            f"{rel_dir}/README.md ({pkg_name}) must mention recommended facade crate '{facade_name}'"
        )

    for rel_target in find_relative_markdown_links(text):
        issues.append(
            f"{rel_dir}/README.md ({pkg_name}) contains relative link {rel_target!r}; use absolute URLs"
        )


def _has_glob_or_prefix(patterns: list[str], required: tuple[str, ...]) -> bool:
    return any(pat in required for pat in patterns)


def check_crate_include_exclude(
    member_dir: pathlib.Path,
    rel_dir: str,
    pkg_name: str,
    pkg_dict: dict[str, Any],
    issues: list[str],
) -> None:
    """Validate explicit package include/exclude lists and reject cross-crate asset escapes."""
    has_inc = "include" in pkg_dict
    has_exc = "exclude" in pkg_dict
    if not has_inc and not has_exc:
        issues.append(
            f"{rel_dir}/Cargo.toml ({pkg_name}) must define explicit 'include' or 'exclude'"
        )
    elif has_inc and has_exc:
        issues.append(
            f"{rel_dir}/Cargo.toml ({pkg_name}) must not set both 'include' and 'exclude'"
        )
    elif has_inc:
        inc = pkg_dict.get("include")
        if not isinstance(inc, list) or not inc or not all(isinstance(x, str) and x.strip() for x in inc):
            issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) 'include' must be a non-empty list[str]")
        else:
            for req in ("Cargo.toml", "LICENSE", "README.md"):
                if req not in inc:
                    issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) 'include' must contain {req!r}")
            if not _has_glob_or_prefix(inc, ("src/**/*.rs", "src/**")):
                issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) 'include' must cover 'src/**/*.rs'")
            if (member_dir / "tests").is_dir() and not _has_glob_or_prefix(inc, ("tests/**/*.rs", "tests/**")):
                issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) 'include' must cover 'tests/**/*.rs'")
            if (member_dir / "src" / "data").is_dir() and not _has_glob_or_prefix(
                inc, ("src/data/*.csv", "src/data/**", "src/**")
            ):
                issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) 'include' must cover 'src/data/*.csv'")
    else:
        exc = pkg_dict.get("exclude")
        if not isinstance(exc, list) or not exc or not all(isinstance(x, str) and x.strip() for x in exc):
            issues.append(f"{rel_dir}/Cargo.toml ({pkg_name}) 'exclude' must be a non-empty list[str]")
        else:
            for forbidden in ("Cargo.toml", "LICENSE", "README.md", "src", "src/**", "src/**/*.rs"):
                if forbidden in exc:
                    issues.append(
                        f"{rel_dir}/Cargo.toml ({pkg_name}) 'exclude' must not exclude {forbidden!r}"
                    )

    crate_root = member_dir.resolve()
    for subdir in ("src", "tests"):
        base = member_dir / subdir
        if not base.is_dir():
            continue
        for rs_file in sorted(base.rglob("*.rs")):
            text = rs_file.read_text(encoding="utf-8")
            for inc_rel in INCLUDE_MACRO_RE.findall(text):
                resolved = (rs_file.parent / inc_rel).resolve()
                if not resolved.is_relative_to(crate_root):
                    rel_rs = rs_file.relative_to(member_dir).as_posix()
                    issues.append(
                        f"{rel_dir}/{rel_rs} ({pkg_name}) escapes crate root via include_str!/include_bytes!({inc_rel!r})"
                    )


def check_crate_doc_cfg(
    member_dir: pathlib.Path,
    rel_dir: str,
    pkg_name: str,
    member_data: dict[str, Any],
    issues: list[str],
) -> None:
    """Enforce `#![cfg_attr(docsrs, feature(doc_cfg))]` and forbid `doc_auto_cfg`."""
    src_dir = member_dir / "src"
    if not src_dir.is_dir():
        return

    for rs_file in sorted(src_dir.rglob("*.rs")):
        text = rs_file.read_text(encoding="utf-8")
        if "doc_auto_cfg" in text:
            rel_rs = rs_file.relative_to(member_dir).as_posix()
            issues.append(
                f"{rel_dir}/{rel_rs} ({pkg_name}) uses removed 'doc_auto_cfg'; use 'doc_cfg'"
            )

    features = member_data.get("features")
    has_optional_features = isinstance(features, dict) and any(k != "default" for k in features)
    lib_rs = src_dir / "lib.rs"
    if has_optional_features and lib_rs.exists():
        lib_text = lib_rs.read_text(encoding="utf-8")
        if "#![cfg_attr(docsrs, feature(doc_cfg))]" not in lib_text:
            issues.append(
                f"{rel_dir}/src/lib.rs ({pkg_name}) defines optional features and must enable "
                "'#![cfg_attr(docsrs, feature(doc_cfg))]'"
            )
