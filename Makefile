export RUST_FONTCONFIG_DLOPEN=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ifneq ($(shell command -v sccache 2>/dev/null),)
export RUSTC_WRAPPER ?= sccache
endif

VENV_DIR := $(CURDIR)/.venv
VENV_PY := $(VENV_DIR)/bin/python
VENV_MATURIN := $(VENV_DIR)/bin/maturin

.PHONY: help fmt lint check test test-crate \
	venv-setup verify-fast verify-all invariants invariants-strict parity sccache-stats \
	wasm-check wasm-build wasm-test wasm-size \
	worktree-create worktree-list worktree-clean \
	python-build python-test python-test-module python-docs python-doctest python-clean \
	all clean

# Default target
help:
	@echo "egraph-rs Makefile - Task Runner (Single-Thread Pure-Rust / Native+Python+WASM)"
	@echo ""
	@echo "Harness & Parallel Worktree Tasks:"
	@echo "  make venv-setup                     - Provision worktree-local Python .venv via uv"
	@echo "  make verify-fast [CRATE=<name>]     - Tier 0: fmt + clippy/nextest on changed or CRATE + thread scan"
	@echo "  make verify-all                     - Full Rust + Python + WASM (+ bundle/size) + invariants + parity verification"
	@echo "  make invariants                     - Check single-thread, wasm32, RNG, size & metadata (report)"
	@echo "  make invariants-strict              - Check invariants strictly (fail on any v6 violation)"
	@echo "  make parity                         - Run cross-target parity & adversarial harness"
	@echo "  make worktree-create NAME=<slug>    - Create isolated .worktrees/<slug> + wt/<slug> + .venv"
	@echo "  make worktree-list                  - List all git worktrees, dirty state & .venv status"
	@echo "  make worktree-clean NAME=<slug>     - Safely remove .worktrees/<slug> & wt/<slug> branch"
	@echo "  make sccache-stats                  - Show sccache compilation cache hit statistics"
	@echo ""
	@echo "Rust Tasks:"
	@echo "  make fmt                            - Format Rust code with rustfmt"
	@echo "  make lint                           - Run clippy linter with strict warnings"
	@echo "  make check                          - Run cargo check on workspace"
	@echo "  make test                           - Run all Rust tests in workspace"
	@echo "  make test-crate CRATE=<name>        - Run tests for specific crate"
	@echo ""
	@echo "WASM Tasks:"
	@echo "  make wasm-check                     - Check egraph-wasm for wasm32-unknown-unknown"
	@echo "  make wasm-build                     - Build release web WASM bundle + wasm-opt -Oz"
	@echo "  make wasm-test                      - Run wasm-pack test --node crates/wasm"
	@echo "  make wasm-size                      - Report .wasm raw & gzipped size vs 600 KiB budget"
	@echo ""
	@echo "Python Tasks (uses worktree-local $(VENV_DIR)):"
	@echo "  make python-build                   - Build release Python bindings into local .venv"
	@echo "  make python-test                    - Run all Python tests in local .venv"
	@echo "  make python-test-module MODULE=<m>  - Run specific Python test module"
	@echo "  make python-docs                    - Build Python documentation (HTML)"
	@echo "  make python-doctest                 - Run Python doctests"
	@echo "  make python-clean                   - Clean Python build artifacts"
	@echo ""
	@echo "Combined Tasks:"
	@echo "  make all                            - Format, lint, and test everything"
	@echo "  make clean                          - Clean all build artifacts"

# Harness & Worktree Tasks
venv-setup:
	./scripts/setup_venv.sh

verify-fast:
ifdef CRATE
	./scripts/verify.sh --fast --crate $(CRATE)
else
	./scripts/verify.sh --fast
endif

verify-all:
	./scripts/verify.sh --all --build-wasm

invariants:
	./scripts/check_invariants.sh --mode report

invariants-strict:
	./scripts/check_invariants.sh --mode strict --check-wasm-size

parity:
	./scripts/verify.sh --parity

sccache-stats:
	sccache --show-stats

worktree-create:
	@if [ -z "$(NAME)" ]; then echo "Error: NAME=<slug> required."; exit 2; fi
	./scripts/worktree.sh create "$(NAME)"

worktree-list:
	./scripts/worktree.sh list

worktree-clean:
	@if [ -z "$(NAME)" ]; then echo "Error: NAME=<slug> required."; exit 2; fi
	./scripts/worktree.sh clean "$(NAME)" --delete-branch

# Rust Tasks
fmt:
	@echo "Formatting Rust code..."
	cargo fmt --all

lint:
	@echo "Running clippy linter..."
	cargo clippy --workspace --all-targets --all-features -- -D warnings

check:
	@echo "Checking Rust code..."
	cargo check --workspace

test:
	@echo "Running all Rust tests..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		cargo nextest run --workspace; \
	else \
		cargo test --workspace; \
	fi

test-crate:
	@if [ -z "$(CRATE)" ]; then \
		echo "Error: CRATE parameter required. Usage: make test-crate CRATE=<crate-name>"; \
		exit 1; \
	fi
	@echo "Running tests for crate: $(CRATE)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		cargo nextest run -p $(CRATE); \
	else \
		cargo test -p $(CRATE); \
	fi

# WASM Tasks
wasm-check:
	cargo check -p egraph-wasm --target wasm32-unknown-unknown

wasm-build:
	wasm-pack build --target web --release -d pkg crates/wasm
	@if command -v wasm-opt >/dev/null 2>&1 && [ -f crates/wasm/pkg/egraph_wasm_bg.wasm ]; then \
		wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
			crates/wasm/pkg/egraph_wasm_bg.wasm -o crates/wasm/pkg/egraph_wasm_bg.wasm; \
	fi

wasm-test:
	wasm-pack test --node crates/wasm

wasm-size:
	./scripts/check_invariants.sh --mode report --check-wasm-size

# Python Tasks (isolated per-worktree .venv)
python-build: venv-setup
	@echo "Building Python bindings into $(VENV_DIR)..."
	cd crates/python && VIRTUAL_ENV="$(VENV_DIR)" "$(VENV_MATURIN)" develop --release

python-test: venv-setup
	@echo "Running all Python tests..."
	cd crates/python && VIRTUAL_ENV="$(VENV_DIR)" "$(VENV_PY)" -m unittest discover -s tests

python-test-module: venv-setup
	@if [ -z "$(MODULE)" ]; then \
		echo "Error: MODULE parameter required. Usage: make python-test-module MODULE=<module-name>"; \
		exit 1; \
	fi
	@echo "Running Python test module: $(MODULE)..."
	cd crates/python && VIRTUAL_ENV="$(VENV_DIR)" "$(VENV_PY)" -m unittest tests.$(MODULE)

python-docs:
	@echo "Building Python documentation..."
	$(MAKE) -C crates/python/docs html

python-doctest:
	@echo "Running Python doctests..."
	$(MAKE) -C crates/python/docs doctest

python-clean:
	@echo "Cleaning Python build artifacts..."
	cd crates/python && rm -rf build/ dist/ *.egg-info target/
	$(MAKE) -C crates/python/docs clean

# Combined Tasks
all: fmt verify-all
	@echo "All tasks completed successfully!"

clean: python-clean
	@echo "Cleaning Rust build artifacts..."
	cargo clean
	@echo "All build artifacts cleaned!"
