use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar};
use egraph_linalg::csr_matrix::{
    CsrMatrix, LaplacianKind, combinatorial_laplacian, symmetric_normalized_laplacian,
};

use super::config::AmgPcgConfig;
use super::hierarchy::{AmgLevel, GalerkinScratch, build_double_pairwise_level};
use super::smoother::{CoarsestSolver, build_coarsest_solver, compute_safe_inv_diag};
use super::workspace::AmgPcgWorkspace;

/// Hardened, strictly single-threaded Algebraic Multigrid Preconditioned Conjugate Gradient solver.
#[derive(Debug, Clone)]
pub struct AmgPcgSolver {
    pub(super) num_nodes: usize,
    pub(super) config: AmgPcgConfig,
    pub(super) levels: Vec<AmgLevel>,
    pub(super) coarsest_a: CsrMatrix,
    pub(super) coarsest_solver: CoarsestSolver,
    pub(super) zero_row_mask: Vec<bool>,
    pub(super) inv_diag0: Vec<f64>,
    pub(super) max_diag0: f64,
    input_nnz: usize,
}

impl AmgPcgSolver {
    /// Builds an [`AmgPcgSolver`] for a symmetric positive (semi-)definite [`CsrMatrix`]
    /// using uniform $\mathbf{1}$ near-nullspace initialization.
    pub fn new(matrix: &CsrMatrix, config: AmgPcgConfig) -> CoreResult<Self> {
        Self::with_near_nullspace(matrix, None, config)
    }

    /// Builds an [`AmgPcgSolver`] for `matrix` with an optional custom `near_nullspace` vector.
    pub fn with_near_nullspace(
        matrix: &CsrMatrix,
        near_nullspace: Option<&[f64]>,
        config: AmgPcgConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = matrix.num_nodes();
        if let Some(ns) = near_nullspace {
            if ns.len() != n {
                return Err(CoreError::dimension_mismatch(
                    n,
                    ns.len(),
                    "AmgPcgSolver::near_nullspace",
                ));
            }
            if ns.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::non_finite("AmgPcgSolver::near_nullspace"));
            }
        }

        if !matrix.is_symmetric(config.sym_tol) {
            return Err(CoreError::invalid_input(
                "AmgPcgSolver requires a symmetric matrix within config.sym_tol",
            ));
        }

        let a0 = if matrix.is_symmetric(0.0) {
            matrix.clone()
        } else {
            matrix.symmetrize()?
        };

        if n == 0 {
            return Ok(Self {
                num_nodes: 0,
                config,
                levels: Vec::new(),
                coarsest_a: a0,
                coarsest_solver: CoarsestSolver::DensePinverse {
                    n: 0,
                    pinv: Vec::new(),
                },
                zero_row_mask: Vec::new(),
                inv_diag0: Vec::new(),
                max_diag0: 0.0,
                input_nnz: 0,
            });
        }

        let diag0 = a0.diagonal();
        let mut max_diag0 = 0.0_f64;
        for &d in diag0 {
            if !d.is_finite() {
                return Err(CoreError::non_finite("AmgPcgSolver::diagonal"));
            }
            if d > max_diag0 {
                max_diag0 = d;
            }
        }
        let neg_thresh = if max_diag0 > 0.0 {
            -config.diag_eps * max_diag0
        } else {
            0.0
        };

        let mut zero_row_mask = vec![false; n];
        for (row, &d) in diag0.iter().enumerate() {
            if d < neg_thresh {
                return Err(CoreError::invalid_input(format!(
                    "negative diagonal entry at row {row}: {d}"
                )));
            }
            if d <= 0.0 {
                let start = a0.indptr()[row] as usize;
                let end = a0.indptr()[row + 1] as usize;
                let row_u32 = row as u32;
                let mut has_nonzero_off = false;
                for pos in start..end {
                    if a0.indices()[pos] != row_u32 && a0.values()[pos] != 0.0 {
                        has_nonzero_off = true;
                        break;
                    }
                }
                if has_nonzero_off {
                    return Err(CoreError::invalid_input(format!(
                        "zero diagonal entry on non-zero row {row}"
                    )));
                }
                zero_row_mask[row] = true;
            }
        }

        let inv_diag0 = compute_safe_inv_diag(diag0, max_diag0, Some(&zero_row_mask));
        let input_nnz = a0.nnz();

        let mut current_a = a0;
        let mut current_ns: Vec<f64> = match near_nullspace {
            Some(ns) => ns.to_vec(),
            None => vec![1.0_f64; n],
        };

        let mut levels = Vec::new();
        let mut rng = DeterministicRng::from_seed_u64(config.seed);
        let mut gal_scratch = GalerkinScratch::default();

        for _level_idx in 0..config.max_levels {
            let nc_curr = current_a.num_nodes();
            if nc_curr <= config.max_coarse_size {
                break;
            }

            let Some(built) = build_double_pairwise_level(
                &current_a,
                &current_ns,
                &config,
                &mut rng,
                &mut gal_scratch,
            )?
            else {
                break;
            };

            let curr_inv_diag = if levels.is_empty() {
                inv_diag0.clone()
            } else {
                let d = current_a.diagonal();
                let m = d.iter().copied().fold(0.0_f64, f64::max);
                compute_safe_inv_diag(d, m, None)
            };

            levels.push(AmgLevel {
                a: current_a,
                inv_diag: curr_inv_diag,
                agg_id: built.agg_id,
                p_val: built.p_val,
            });
            current_a = built.next_a;
            current_ns = built.next_ns;
        }

        let coarsest_solver = build_coarsest_solver(&current_a, &config)?;

        Ok(Self {
            num_nodes: n,
            config,
            levels,
            coarsest_a: current_a,
            coarsest_solver,
            zero_row_mask,
            inv_diag0,
            max_diag0,
            input_nnz,
        })
    }

    /// Builds an [`AmgPcgSolver`] for the regularized symmetric normalized Laplacian
    /// $A = I + \gamma L_{\mathrm{sym}}$ ($\gamma \ge 0$) with exact $D^{1/2}\mathbf{1}$ near-nullspace.
    pub fn from_regularized_sym_laplacian<W: Scalar>(
        graph: &CsrGraph<W>,
        gamma: f64,
        config: AmgPcgConfig,
    ) -> CoreResult<Self> {
        if !gamma.is_finite() {
            return Err(CoreError::non_finite(
                "AmgPcgSolver::from_regularized_sym_laplacian::gamma",
            ));
        }
        if gamma < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "gamma must be >= 0.0, got {gamma}"
            )));
        }
        let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(graph);
        let a = l_sym.scale_and_shift(gamma, 1.0)?;
        let near_nullspace: Vec<f64> = inv_sqrt_deg
            .into_iter()
            .map(|inv| if inv > 0.0 { 1.0 / inv } else { 1.0 })
            .collect();
        Self::with_near_nullspace(&a, Some(&near_nullspace), config)
    }

    /// Builds an [`AmgPcgSolver`] for the shifted graph Laplacian $L + \mathrm{shift}\cdot I$
    /// (`shift >= 0.0`) with matching near-nullspace ($D^{1/2}\mathbf{1}$ for `SymmetricNormalized`,
    /// $\mathbf{1}$ for `Combinatorial`).
    pub fn from_laplacian<W: Scalar>(
        graph: &CsrGraph<W>,
        kind: LaplacianKind,
        shift: f64,
        config: AmgPcgConfig,
    ) -> CoreResult<Self> {
        if !shift.is_finite() {
            return Err(CoreError::non_finite("AmgPcgSolver::from_laplacian::shift"));
        }
        if shift < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "shift must be >= 0.0, got {shift}"
            )));
        }
        match kind {
            LaplacianKind::Combinatorial => {
                let l = combinatorial_laplacian(graph);
                let a = if shift == 0.0 {
                    l
                } else {
                    l.scale_and_shift(1.0, shift)?
                };
                Self::with_near_nullspace(&a, None, config)
            }
            LaplacianKind::SymmetricNormalized => {
                let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(graph);
                let a = if shift == 0.0 {
                    l_sym
                } else {
                    l_sym.scale_and_shift(1.0, shift)?
                };
                let near_nullspace: Vec<f64> = inv_sqrt_deg
                    .into_iter()
                    .map(|inv| if inv > 0.0 { 1.0 / inv } else { 1.0 })
                    .collect();
                Self::with_near_nullspace(&a, Some(&near_nullspace), config)
            }
            LaplacianKind::RandomWalk => Err(CoreError::unsupported(
                "AmgPcgSolver does not support asymmetric RandomWalk Laplacian; use Combinatorial or SymmetricNormalized",
            )),
        }
    }

    /// Returns the dimension $n$ of the linear system.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of levels in the AMG hierarchy (including the coarsest level).
    #[inline]
    pub fn num_levels(&self) -> usize {
        if self.num_nodes == 0 {
            0
        } else {
            self.levels.len() + 1
        }
    }

    /// Returns the dimension of the coarsest AMG level.
    #[inline]
    pub fn coarsest_size(&self) -> usize {
        self.coarsest_a.num_nodes()
    }

    /// Returns the AMG operator complexity $\sum_{l=0}^L \mathrm{nnz}(A_l) / \mathrm{nnz}(A_0)$.
    pub fn operator_complexity(&self) -> f64 {
        if self.input_nnz == 0 {
            return 1.0;
        }
        let mut total_nnz = 0usize;
        for lvl in &self.levels {
            total_nnz += lvl.a.nnz();
        }
        total_nnz += self.coarsest_a.nnz();
        (total_nnz as f64) / (self.input_nnz as f64)
    }

    /// Returns the maximum CSR nonzero count across all levels in the AMG hierarchy.
    pub fn max_level_nnz(&self) -> usize {
        let mut max_nnz = self.coarsest_a.nnz();
        for lvl in &self.levels {
            max_nnz = max_nnz.max(lvl.a.nnz());
        }
        max_nnz
    }

    /// Returns the active [`AmgPcgConfig`].
    #[inline]
    pub fn config(&self) -> AmgPcgConfig {
        self.config
    }

    /// Allocates a workspace pre-sized for this solver.
    pub fn new_workspace(&self) -> AmgPcgWorkspace {
        AmgPcgWorkspace::for_solver(self)
    }

    #[inline]
    pub(super) fn finest_matrix(&self) -> &CsrMatrix {
        if let Some(lvl0) = self.levels.first() {
            &lvl0.a
        } else {
            &self.coarsest_a
        }
    }
}
