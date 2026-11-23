## Summary
<!-- Describe what this PR changes and why. -->

## Related Issue
<!-- e.g., Closes #123 -->

## Change Type
- [ ] `feat`: New feature or layout/metric/binding API
- [ ] `fix`: Bug fix or numerical/edge-case correction
- [ ] `perf`: Single-threaded algorithmic or memory optimization
- [ ] `refactor`: Internal code structure or cleanup
- [ ] `docs`: Documentation, README, CHANGELOG, or examples
- [ ] `test`: Test coverage, parity harness, or smoke checks
- [ ] `chore`: CI/workflows, packaging, dependencies, or `.agents/` updates
- [ ] `BREAKING CHANGE`: Public API or behavior breaking change (explain migration below)

## Affected Targets
- [ ] Native Rust (`egraph` / core crates)
- [ ] Python (`crates/python`, `_core.pyi`, `egraph.interop`)
- [ ] WebAssembly / TypeScript (`crates/wasm`, `pkg/`, `js/examples`)
- [ ] CLI (`crates/cli`)
- [ ] CI / Release / Docs

## Verification Commands Run
<!-- List exact single-threaded commands and outcomes (e.g., `./scripts/verify.sh --all --build-wasm --docs`). -->
- `EGRAPH_PARALLEL_WORKERS=1 ./scripts/verify.sh ...`

## Mandatory Checklist
- [ ] Strictly single-threaded pure CPU Rust: zero `rayon`, `std::thread`, Web Workers, `SharedArrayBuffer`, `wgpu`, or C/Fortran FFI (`ndarray` uses `default-features = false`).
- [ ] Deterministic explicit `u64` RNG (`DeterministicRng::from_seed_u64`), zero `thread_rng()` / `from_entropy()`, and zero non-test `.unwrap()` / `expect()` / `panic!` / `unreachable!` / `unsafe`.
- [ ] Native / Python / WASM API parity and `--no-default-features` WASM builds checked; `egraph_wasm_bg.wasm` gzip remains `<= 600 KiB`.
- [ ] Python bindings checked with `mypy --strict` and `stubtest egraph` when `.py` / `.pyi` / PyO3 exports changed.
- [ ] All code, comments, documentation, commit messages, and PR/Issue prose are written in English (`jp_cjk=0`).
