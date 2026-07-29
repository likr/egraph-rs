---
name: egraph-rs-verification
description: Commands and procedures to format, lint, and run tests for Rust, WebAssembly, Python, and JS/TS in the egraph-rs project.
---

# Verification and Testing Guide for egraph-rs

This skill contains all the commands to check formatting, lint code, and run test suites within the `egraph-rs` workspace.

## Quick Reference Commands

| Language/Target | Action | Command |
| :--- | :--- | :--- |
| **Rust** | Check | `make check` |
| **Rust** | Format | `make fmt` (runs `cargo fmt --all`) |
| **Rust** | Lint | `make lint` (runs `cargo clippy ...`) |
| **Rust** | All Tests | `make test` (runs `cargo test --workspace`) |
| **Rust** | Crate Tests | `make test-crate CRATE=<crate-name>` |
| **Rust** | Specific Test | `cargo test -p <crate-name> <test-name>` |
| **Rust** | Ignored / Accuracy Test | `cargo test -p <crate-name> --test <test-name> -- --ignored --nocapture` |
| **WASM** | All Tests | `wasm-pack test --node crates/wasm` |
| **WASM** | Specific Test | `wasm-pack test --node crates/wasm --test <test-name>` |
| **WASM** | Build | `npm run wasm-build` |
| **Python** | Build | `make python-build` (compiles bindings via Maturin) |
| **Python** | All Tests | `make python-test` |
| **Python** | Module Tests | `make python-test-module MODULE=<module>` |
| **Python** | Specific Case | Run from `crates/python`: `python -m unittest tests.test_<module>.TestClass.test_method` |
| **Python** | Build Docs | `make python-docs` |
| **Python** | Doctests | `make python-doctest` |
| **JS/TS** | Format | `npx prettier --write .` |
| **Examples** | Run | `npm start` |

## Mandatory Steps Before Completion

Whenever a task is ready to be completed:
1. Run `cargo fmt --all` to format the Rust codebase.
2. Run `cargo clippy --workspace --all-targets --all-features -- -D warnings` and ensure 0 warnings or errors.
3. Run `cargo test --workspace` to ensure all workspace tests pass.
4. Run relevant binding tests (WASM, Python) if you modified the bindings, drawing, or layout crates.

## Git Command Precautions
- **Always use `--no-pager`**: When querying Git history or diffs in CLI (e.g. `git diff`, `git log`, `git show`), prefix with `git --no-pager` to prevent terminal lockups or interactive page-hangs.
  - Correct: `git --no-pager diff`
  - Correct: `git --no-pager log`

## System Dependencies & Environment Variables

- **`RUST_FONTCONFIG_DLOPEN=1`**: When checking, building, or testing crates in the workspace, you may encounter compile-time failures from `yeslogic-fontconfig-sys` if the system lacks `fontconfig` libraries. Setting `RUST_FONTCONFIG_DLOPEN=1` bypasses the compile-time link checks by dynamically loading the library at runtime.
  - Example: `RUST_FONTCONFIG_DLOPEN=1 cargo test --workspace`

