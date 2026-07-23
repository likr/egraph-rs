# egraph-rs Maintenance Plan

## 1. Codebase Overview & Status
- **Repository**: `https://github.com/likr/egraph-rs`
- **Workspace Structure**: Cargo workspace with 16 member crates covering graph algorithms, layouts (SGD, Kamada-Kawai, MDS, TsNet, Omega, etc.), linear algebra, datasets, CLI, WebAssembly, and Python bindings (`egraph-python`).
- **Recent Upgrades**: 
  - Petgraph upgraded to `0.8`.
  - PyO3 and NumPy upgraded to `0.28`.

## 2. Identified Build & Test Issues
When running `cargo test --workspace`, compilation fails due to a dependency version conflict involving `ndarray`:
- **NumPy 0.28 & PyO3 0.28** (in `egraph-python`) require **`ndarray 0.17`**.
- **Linfa 0.8 / Linfa-Clustering / Linfa-NN** (used in `petgraph-clustering` and `petgraph-quality-metrics`) require **`ndarray 0.16`**.
- Rust does not permit multiple incompatible major versions of `ndarray` for shared types (like `Array2`, `Dataset`, `BallTree`), resulting in E0308 type mismatch errors and unsatisfied trait bounds.

## 3. Proposed Resolution Options
1. **Option A: Align on `ndarray 0.16` (Downgrade PyO3/NumPy to 0.26)**
   - Revert PyO3 and NumPy dependencies back to `0.26` to match `ndarray 0.16` expected by `linfa 0.8`.
   - Revert PyO3 0.28 syntax updates (`.cast()` -> `.downcast()`, removing `py` parameters, etc.).
   - *Pros*: Restores compatibility with `linfa` without rewriting ML algorithms.
   - *Cons*: Reverts recent PyO3/NumPy modernization.

2. **Option B: Remove or Replace `linfa` Dependency**
   - Implement lightweight k-means and nearest neighbor / ball tree functionality natively or using alternative crates compatible with `ndarray 0.17`.
   - *Pros*: Fully embraces modern NumPy 0.28 / PyO3 0.28 / Ndarray 0.17 stack across the entire workspace.
   - *Cons*: Requires rewriting spectral clustering and neighborhood preservation logic.

3. **Option C: Upgrade or Fork Linfa / Use Feature Flags**
   - Check if `linfa` can be updated or if a fork supporting `ndarray 0.17` is available.

## 4. Recommended Action Plan
1. Document the dependency conflict (done in this maintenance plan).
2. Discuss with maintainers whether to downgrade PyO3/NumPy to 0.26 (Option A) or migrate away from `linfa` to support NumPy 0.28 (Option B).
3. Execute the chosen resolution strategy and ensure `cargo test --workspace` and `make all` pass cleanly.

