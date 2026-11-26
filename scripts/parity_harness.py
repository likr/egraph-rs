#!/usr/bin/env python3
"""Cross-target golden fixture & adversarial contract harness for egraph.

Supports `CsrGraph` / `Layout2d` Python bindings and optional Node.js WASM
cross-checking via `--wasm-pkg`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_SCRIPTS_DIR = str(pathlib.Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from parity import run_all_suites


def main() -> int:
    parser = argparse.ArgumentParser(description="egraph cross-target parity harness")
    parser.add_argument("--wasm-pkg", type=pathlib.Path, default=None, help="Path to built WASM pkg/")
    parser.add_argument("--wasm-only", action="store_true", help="Skip Python checks and run only WASM Node checks")
    parser.add_argument("--json", type=pathlib.Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    results = run_all_suites(wasm_pkg=args.wasm_pkg, wasm_only=args.wasm_only)

    print(f"{'CASE':<40} {'STATUS':<24} {'ELAPSED(s)':>10}")
    print("-" * 78)
    for row in results:
        print(f"{row['case']:<40} {row['status']:<24} {row['elapsed_sec']:>10.4f}")

    if args.json is not None:
        args.json.write_text(json.dumps({"results": results}, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
