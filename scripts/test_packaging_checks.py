#!/usr/bin/env python3
"""Unit tests for scripts/packaging_checks.py and scripts/readme_checks.py."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import packaging_checks as pc  # noqa: E402


def _write_minimal_workspace(
    root: pathlib.Path,
    *,
    ws_version: str = "6.0.0-beta.3",
    core_dep_pin: str = "=6.0.0-beta.3",
    facade_dep_pin: str = "=6.0.0-beta.3",
    core_readme_field: str = 'readme = "README.md"',
    core_readme_body: str = (
        "# egraph-core\n\n"
        "Use `egraph-rs` (`[docs](https://docs.rs/egraph-rs)`). "
        "Code `[ignored](rel.md)` and:\n\n```markdown\n[also-ignored](rel2.md)\n```\n"
    ),
    core_include_line: str = 'include = ["Cargo.toml", "LICENSE", "README.md", "src/**/*.rs"]',
    core_features_block: str = "",
    core_lib_rs: str = "//! Core crate.\n",
    core_extra_dep: str = "",
) -> None:
    (root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (root / "README.md").write_text("# Root\n", encoding="utf-8")
    (root / "Cargo.toml").write_text(
        f"""[workspace]
resolver = "2"
members = ["crates/core", "crates/egraph"]

[workspace.package]
version = "{ws_version}"
edition = "2024"
rust-version = "1.88"
license = "MIT"
repository = "https://github.com/likr/egraph-rs"
homepage = "https://github.com/likr/egraph-rs"
readme = "README.md"

[workspace.dependencies]
egraph-core = {{ path = "crates/core", version = "{core_dep_pin}" }}
egraph-rs = {{ path = "crates/egraph", version = "{facade_dep_pin}" }}
""",
        encoding="utf-8",
    )

    core_dir = root / "crates" / "core"
    (core_dir / "src").mkdir(parents=True)
    (core_dir / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (core_dir / "README.md").write_text(core_readme_body, encoding="utf-8")
    (core_dir / "src" / "lib.rs").write_text(core_lib_rs, encoding="utf-8")
    (core_dir / "Cargo.toml").write_text(
        f"""[package]
name = "egraph-core"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
homepage.workspace = true
{core_readme_field}
documentation = "https://docs.rs/egraph-core"
description = "Core crate"
keywords = ["graph"]
categories = ["algorithms"]
{core_include_line}

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
{core_features_block}
[dependencies]
{core_extra_dep}
""",
        encoding="utf-8",
    )

    facade_dir = root / "crates" / "egraph"
    (facade_dir / "src").mkdir(parents=True)
    (facade_dir / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (facade_dir / "README.md").write_text(
        "# egraph-rs\n\n[LICENSE](https://github.com/likr/egraph-rs/blob/master/crates/egraph/LICENSE)\n",
        encoding="utf-8",
    )
    (facade_dir / "src" / "lib.rs").write_text("//! Facade.\n", encoding="utf-8")
    (facade_dir / "Cargo.toml").write_text(
        """[package]
name = "egraph-rs"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
homepage.workspace = true
readme = "README.md"
documentation = "https://docs.rs/egraph-rs"
description = "Facade crate"
keywords = ["graph"]
categories = ["algorithms"]
include = ["Cargo.toml", "LICENSE", "README.md", "src/**/*.rs"]

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]

[dependencies]
egraph-core.workspace = true
""",
        encoding="utf-8",
    )


class PackagingChecksTests(unittest.TestCase):
    def test_valid_prerelease_workspace_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(root)
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertEqual(issues, [])

    def test_prerelease_requires_exact_equals_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(root, core_dep_pin="6.0.0-beta.3")
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertTrue(any("must be exact '=6.0.0-beta.3'" in i for i in issues), issues)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(
                root,
                ws_version="6.0.0",
                core_dep_pin="6.0.0",
                facade_dep_pin="6.0.0",
            )
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertEqual(issues, [])

    def test_member_path_only_or_unpinned_internal_dep_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(
                root,
                core_extra_dep='egraph-rs = { path = "../egraph" }',
            )
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertTrue(any("path-only internal dependency" in i for i in issues), issues)

    def test_readme_workspace_relative_links_and_facade_mention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(root, core_readme_field="readme.workspace = true")
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertTrue(any("readme.workspace = true" in i for i in issues), issues)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(
                root,
                core_readme_body="# egraph-core\n\nSee [LICENSE](LICENSE) and [guide](egraph-rs.md).\n",
            )
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertTrue(any("relative link 'LICENSE'" in i for i in issues), issues)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(root, core_readme_body="# egraph-core\n\nStandalone text.\n")
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertTrue(any("recommended facade crate 'egraph-rs'" in i for i in issues), issues)

    def test_include_exclude_and_escape_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(root, core_include_line="")
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertTrue(any("must define explicit 'include' or 'exclude'" in i for i in issues), issues)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(
                root,
                core_include_line=(
                    'include = ["Cargo.toml", "README.md", "src/**/*.rs"]\n'
                    'exclude = ["target"]'
                ),
            )
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertTrue(any("must not set both 'include' and 'exclude'" in i for i in issues), issues)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(
                root,
                core_include_line='include = ["Cargo.toml", "README.md", "src/**/*.rs"]',
            )
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertTrue(any("must contain 'LICENSE'" in i for i in issues), issues)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(
                root,
                core_lib_rs='const DATA: &str = include_str!("../../dataset/data.csv");\n',
            )
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertTrue(any("escapes crate root" in i for i in issues), issues)

    def test_doc_cfg_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(
                root,
                core_features_block='\n[features]\ndefault = []\nserde = []\n',
                core_lib_rs="//! Missing doc_cfg.\n",
            )
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertTrue(any("feature(doc_cfg)" in i for i in issues), issues)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _write_minimal_workspace(
                root,
                core_features_block='\n[features]\ndefault = []\nserde = []\n',
                core_lib_rs="#![cfg_attr(docsrs, feature(doc_auto_cfg, doc_cfg))]\n",
            )
            issues = pc.collect_packaging_issues(root, strict_all=True, strict_pub=True)
            self.assertTrue(any("removed 'doc_auto_cfg'" in i for i in issues), issues)


if __name__ == "__main__":
    unittest.main()
