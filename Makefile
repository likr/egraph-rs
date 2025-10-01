.PHONY: help fmt lint check test test-crate python-build python-test python-test-module python-docs python-doctest python-clean all clean

# Default target
help:
	@echo "egraph-rs Makefile - Task Runner"
	@echo ""
	@echo "Rust Tasks:"
	@echo "  make fmt                    - Format Rust code with rustfmt"
	@echo "  make lint                   - Run clippy linter with strict warnings"
	@echo "  make check                  - Run cargo check on workspace"
	@echo "  make test                   - Run all Rust tests in workspace"
	@echo "  make test-crate CRATE=<name> - Run tests for specific crate"
	@echo ""
	@echo "Python Tasks:"
	@echo "  make python-build           - Build Python bindings with maturin"
	@echo "  make python-test            - Run all Python tests"
	@echo "  make python-test-module MODULE=<name> - Run specific Python test module"
	@echo "  make python-docs            - Build Python documentation (HTML)"
	@echo "  make python-doctest         - Run Python doctests"
	@echo "  make python-clean           - Clean Python build artifacts"
	@echo ""
	@echo "Combined Tasks:"
	@echo "  make all                    - Format, lint, and test everything"
	@echo "  make clean                  - Clean all build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make test-crate CRATE=petgraph-layout-mds"
	@echo "  make python-test-module MODULE=test_sgd"

# Rust: Format code
fmt:
	@echo "Formatting Rust code..."
	cargo fmt --all

# Rust: Lint code
lint:
	@echo "Running clippy linter..."
	cargo clippy --workspace --all-targets --all-features -- -D warnings

# Rust: Check code
check:
	@echo "Checking Rust code..."
	cargo check --workspace

# Rust: Run all tests
test:
	@echo "Running all Rust tests..."
	cargo test --workspace

# Rust: Run tests for specific crate
test-crate:
	@if [ -z "$(CRATE)" ]; then \
		echo "Error: CRATE parameter required. Usage: make test-crate CRATE=<crate-name>"; \
		exit 1; \
	fi
	@echo "Running tests for crate: $(CRATE)..."
	cargo test -p $(CRATE)

# Python: Build bindings
python-build:
	@echo "Building Python bindings..."
	cd crates/python && maturin develop --release

# Python: Run all tests
python-test:
	@echo "Running all Python tests..."
	cd crates/python && python -m unittest discover tests

# Python: Run specific test module
python-test-module:
	@if [ -z "$(MODULE)" ]; then \
		echo "Error: MODULE parameter required. Usage: make python-test-module MODULE=<module-name>"; \
		exit 1; \
	fi
	@echo "Running Python test module: $(MODULE)..."
	cd crates/python && python -m unittest tests.$(MODULE)

# Python: Build documentation
python-docs:
	@echo "Building Python documentation..."
	$(MAKE) -C crates/python/docs html

# Python: Run doctests
python-doctest:
	@echo "Running Python doctests..."
	$(MAKE) -C crates/python/docs doctest

# Python: Clean build artifacts
python-clean:
	@echo "Cleaning Python build artifacts..."
	cd crates/python && rm -rf build/ dist/ *.egg-info target/
	$(MAKE) -C crates/python/docs clean

# Combined: Format, lint, and test everything
all: fmt lint test python-test
	@echo "All tasks completed successfully!"

# Clean all build artifacts
clean: python-clean
	@echo "Cleaning Rust build artifacts..."
	cargo clean
	@echo "All build artifacts cleaned!"
