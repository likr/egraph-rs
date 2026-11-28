//! WebAssembly bindings for the single-threaded Algebraic Multigrid Preconditioned Conjugate
//! Gradient (`AmgPcgSolver`) linear solver.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, invalid_input_err, parse_non_negative_usize};
use crate::js_options::{parse_amg_laplacian_kind, parse_amg_pcg_config, solver_status_str};
use crate::linalg_matrix::JsCsrMatrix;
use egraph::core::CoreError;
use egraph::linalg::{
    AmgPcgBatchResult, AmgPcgSolveResult, AmgPcgSolver, AmgPcgWorkspace, CsrMatrix, LaplacianKind,
};
use wasm_bindgen::prelude::*;

/// Single-RHS solution and diagnostics returned by `AmgPcgSolver.solve()`.
#[wasm_bindgen(js_name = AmgPcgSolveResult)]
#[derive(Debug, Clone)]
pub struct JsAmgPcgSolveResult {
    x_vec: Vec<f64>,
    res: AmgPcgSolveResult,
}

#[wasm_bindgen(js_class = AmgPcgSolveResult)]
impl JsAmgPcgSolveResult {
    /// Returns a copy of the solution vector (`Float64Array`).
    pub fn x(&self) -> Vec<f64> {
        self.x_vec.clone()
    }

    /// Number of PCG iterations executed.
    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize {
        self.res.iterations
    }

    /// True Euclidean residual norm `||b - A x||_2`.
    #[wasm_bindgen(getter, js_name = residualNorm)]
    pub fn residual_norm(&self) -> f64 {
        self.res.residual_norm
    }

    /// Relative residual norm `||b - A x||_2 / ||b||_2`.
    #[wasm_bindgen(getter, js_name = relativeResidual)]
    pub fn relative_residual(&self) -> f64 {
        self.res.relative_residual
    }

    /// `true` if the stopping tolerance was met.
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.res.converged
    }

    /// Final solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.res.status).to_owned()
    }
}

/// Batched-RHS solution and diagnostics returned by `AmgPcgSolver.solveBatch()`.
#[wasm_bindgen(js_name = AmgPcgBatchResult)]
#[derive(Debug, Clone)]
pub struct JsAmgPcgBatchResult {
    num_nodes: usize,
    ncols: usize,
    x_flat: Vec<f64>,
    res: AmgPcgBatchResult,
}

#[wasm_bindgen(js_class = AmgPcgBatchResult)]
impl JsAmgPcgBatchResult {
    /// Returns a flat row-major `(numNodes, ncols)` `Float64Array` copy of all solution columns.
    pub fn x(&self) -> Vec<f64> {
        self.x_flat.clone()
    }

    /// Number of rows (`numNodes`).
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Number of RHS columns (`ncols`).
    #[wasm_bindgen(getter)]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Sum of PCG iterations across all columns.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.res.total_iterations
    }

    /// Maximum PCG iterations executed across all columns.
    #[wasm_bindgen(getter, js_name = maxIterationsUsed)]
    pub fn max_iterations_used(&self) -> usize {
        self.res.max_iterations_used
    }

    /// Minimum PCG iterations executed across all columns.
    #[wasm_bindgen(getter, js_name = minIterationsUsed)]
    pub fn min_iterations_used(&self) -> usize {
        self.res.min_iterations_used
    }

    /// Maximum true residual norm across all columns.
    #[wasm_bindgen(getter, js_name = maxResidualNorm)]
    pub fn max_residual_norm(&self) -> f64 {
        self.res.max_residual_norm
    }

    /// Maximum relative residual norm across all columns.
    #[wasm_bindgen(getter, js_name = maxRelativeResidual)]
    pub fn max_relative_residual(&self) -> f64 {
        self.res.max_relative_residual
    }

    /// `true` if every column converged within tolerance.
    #[wasm_bindgen(getter, js_name = allConverged)]
    pub fn all_converged(&self) -> bool {
        self.res.all_converged
    }

    /// Returns per-column iteration counts (`Uint32Array`).
    #[wasm_bindgen(js_name = iterationsPerCol)]
    pub fn iterations_per_col(&self) -> Vec<u32> {
        self.res
            .iterations_per_col
            .iter()
            .map(|&v| v as u32)
            .collect()
    }

    /// Returns per-column true residual norms (`Float64Array`).
    #[wasm_bindgen(js_name = residualNorms)]
    pub fn residual_norms(&self) -> Vec<f64> {
        self.res.residual_norms.clone()
    }

    /// Returns per-column relative residual norms (`Float64Array`).
    #[wasm_bindgen(js_name = relativeResiduals)]
    pub fn relative_residuals(&self) -> Vec<f64> {
        self.res.relative_residuals.clone()
    }

    /// Returns per-column convergence flags (`Uint8Array` with `1` for converged, `0` otherwise).
    #[wasm_bindgen(js_name = convergedPerCol)]
    pub fn converged_per_col(&self) -> Vec<u8> {
        self.res
            .converged_per_col
            .iter()
            .map(|&v| u8::from(v))
            .collect()
    }
}

/// Hardened single-threaded Algebraic Multigrid Preconditioned Conjugate Gradient (`AmgPcgSolver`)
/// supporting symmetric CSR matrices, regularized/shifted graph Laplacians, batched RHS, and
/// cooperative `step()` execution.
#[wasm_bindgen(js_name = AmgPcgSolver)]
#[derive(Debug, Clone)]
pub struct JsAmgPcgSolver {
    inner: AmgPcgSolver,
    ws: AmgPcgWorkspace,
}

#[wasm_bindgen(js_class = AmgPcgSolver)]
impl JsAmgPcgSolver {
    /// Constructs an `AmgPcgSolver` directly from raw symmetric CSR buffers.
    #[wasm_bindgen(js_name = fromCsr)]
    pub fn from_csr(
        num_nodes: f64,
        indptr: &[u32],
        indices: &[u32],
        values: &[f64],
        options: &JsValue,
    ) -> Result<JsAmgPcgSolver, JsValue> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (cfg, ns) = parse_amg_pcg_config(options)?;
        let mat = CsrMatrix::from_csr(n, indptr.to_vec(), indices.to_vec(), values.to_vec())
            .map_err(core_err_to_js)?;
        let inner =
            AmgPcgSolver::with_near_nullspace(&mat, ns.as_deref(), cfg).map_err(core_err_to_js)?;
        let ws = inner.new_workspace();
        Ok(Self { inner, ws })
    }

    /// Constructs an `AmgPcgSolver` directly from an existing `CsrMatrix`.
    #[wasm_bindgen(js_name = fromCsrMatrix)]
    pub fn from_csr_matrix(
        matrix: &JsCsrMatrix,
        options: &JsValue,
    ) -> Result<JsAmgPcgSolver, JsValue> {
        let (cfg, ns) = parse_amg_pcg_config(options)?;
        let inner = AmgPcgSolver::with_near_nullspace(matrix.inner(), ns.as_deref(), cfg)
            .map_err(core_err_to_js)?;
        let ws = inner.new_workspace();
        Ok(Self { inner, ws })
    }

    /// Constructs an `AmgPcgSolver` for `I + gamma * L_sym` with exact `D^{1/2} 1` near-nullspace.
    #[wasm_bindgen(js_name = fromGraphRegularizedLaplacian)]
    pub fn from_graph_regularized_laplacian(
        graph: &JsCsrGraph,
        gamma: Option<f64>,
        options: &JsValue,
    ) -> Result<JsAmgPcgSolver, JsValue> {
        let (cfg, _) = parse_amg_pcg_config(options)?;
        let inner =
            AmgPcgSolver::from_regularized_sym_laplacian(graph.inner(), gamma.unwrap_or(1.0), cfg)
                .map_err(core_err_to_js)?;
        let ws = inner.new_workspace();
        Ok(Self { inner, ws })
    }

    /// Constructs an `AmgPcgSolver` for a shifted graph Laplacian (`"combinatorial"` or `"symmetricNormalized"`).
    #[wasm_bindgen(js_name = fromGraphLaplacian)]
    pub fn from_graph_laplacian(
        graph: &JsCsrGraph,
        kind: &JsValue,
        shift: Option<f64>,
        options: &JsValue,
    ) -> Result<JsAmgPcgSolver, JsValue> {
        let lap_kind = if kind.is_undefined() || kind.is_null() {
            LaplacianKind::Combinatorial
        } else if let Some(k) = kind.as_string() {
            parse_amg_laplacian_kind(&k)?
        } else {
            return Err(invalid_input_err(
                "AmgPcgSolver.fromGraphLaplacian: kind must be 'combinatorial' or 'symmetricNormalized' (pass options as the 4th argument)",
            ));
        };
        let (cfg, _) = parse_amg_pcg_config(options)?;
        let inner =
            AmgPcgSolver::from_laplacian(graph.inner(), lap_kind, shift.unwrap_or(0.0), cfg)
                .map_err(core_err_to_js)?;
        let ws = inner.new_workspace();
        Ok(Self { inner, ws })
    }

    /// Solves `A x = b` synchronously with optional initial guess `x0`.
    pub fn solve(
        &mut self,
        b: &[f64],
        x0: Option<Vec<f64>>,
    ) -> Result<JsAmgPcgSolveResult, JsValue> {
        self.inner
            .init_solve(b, x0.as_deref(), &mut self.ws)
            .map_err(core_err_to_js)?;
        self.inner.run(&mut self.ws).map_err(core_err_to_js)?;
        Ok(JsAmgPcgSolveResult {
            x_vec: self.ws.x().to_vec(),
            res: self.ws.result(),
        })
    }

    /// Solves `A X = B` for flat row-major `(numNodes, ncols)` RHS matrix `bFlat`.
    #[wasm_bindgen(js_name = solveBatch)]
    pub fn solve_batch(
        &mut self,
        b_flat: &[f64],
        ncols: f64,
        x0_flat: Option<Vec<f64>>,
    ) -> Result<JsAmgPcgBatchResult, JsValue> {
        let nc = parse_non_negative_usize(ncols, "ncols")?;
        let n = self.inner.num_nodes();
        let expected = n
            .checked_mul(nc)
            .ok_or_else(|| invalid_input_err("AmgPcgSolver.solveBatch size overflow"))?;
        let mut x_out = match x0_flat {
            Some(v) => {
                if v.len() != expected {
                    return Err(core_err_to_js(CoreError::dimension_mismatch(
                        expected,
                        v.len(),
                        "AmgPcgSolver.solveBatch::x0Flat",
                    )));
                }
                v
            }
            None => vec![0.0_f64; expected],
        };
        let res = self
            .inner
            .solve_batch(b_flat, &mut x_out, nc, &mut self.ws)
            .map_err(core_err_to_js)?;
        Ok(JsAmgPcgBatchResult {
            num_nodes: n,
            ncols: nc,
            x_flat: x_out,
            res,
        })
    }

    /// Initializes a cooperative PCG solve for `b` and optional initial guess `x0`.
    #[wasm_bindgen(js_name = startSolve)]
    pub fn start_solve(&mut self, b: &[f64], x0: Option<Vec<f64>>) -> Result<String, JsValue> {
        let status = self
            .inner
            .init_solve(b, x0.as_deref(), &mut self.ws)
            .map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Executes a single cooperative PCG iteration and returns the updated status string.
    pub fn step(&mut self) -> Result<String, JsValue> {
        let status = self.inner.step(&mut self.ws).map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Executes up to `steps` cooperative PCG iterations and returns the updated status string.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, steps: f64) -> Result<String, JsValue> {
        let n_steps = parse_non_negative_usize(steps, "steps")?;
        let status = self
            .inner
            .run_steps(&mut self.ws, n_steps)
            .map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Runs the active cooperative solve to completion and returns the final status string.
    pub fn run(&mut self) -> Result<String, JsValue> {
        let status = self.inner.run(&mut self.ws).map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Returns a copy of the current solution vector (`Float64Array`).
    pub fn solution(&self) -> Vec<f64> {
        self.ws.x().to_vec()
    }

    /// Returns the matrix dimension `numNodes`.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the number of AMG hierarchy levels.
    #[wasm_bindgen(getter, js_name = numLevels)]
    pub fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }

    /// Returns the coarsest-level matrix dimension.
    #[wasm_bindgen(getter, js_name = coarsestSize)]
    pub fn coarsest_size(&self) -> usize {
        self.inner.coarsest_size()
    }

    /// Returns the AMG operator complexity.
    #[wasm_bindgen(getter, js_name = operatorComplexity)]
    pub fn operator_complexity(&self) -> f64 {
        self.inner.operator_complexity()
    }

    /// Returns the maximum CSR nonzero count across all AMG hierarchy levels.
    #[wasm_bindgen(getter, js_name = maxLevelNnz)]
    pub fn max_level_nnz(&self) -> usize {
        self.inner.max_level_nnz()
    }

    /// Returns the current solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.ws.status()).to_owned()
    }

    /// Returns the number of PCG iterations executed in the active solve.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.ws.iteration()
    }

    /// Returns the maximum scheduled iterations for the active solve.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.ws.total_iterations()
    }

    /// Returns `true` when the active solve has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.ws.is_finished()
    }

    /// Returns the most recent Euclidean residual norm.
    #[wasm_bindgen(getter, js_name = residualNorm)]
    pub fn residual_norm(&self) -> f64 {
        self.ws.residual_norm()
    }

    /// Returns the most recent relative Euclidean residual norm.
    #[wasm_bindgen(getter, js_name = relativeResidual)]
    pub fn relative_residual(&self) -> f64 {
        self.ws.relative_residual()
    }

    /// Returns `true` if the active solve converged within tolerance.
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.ws.converged()
    }
}
