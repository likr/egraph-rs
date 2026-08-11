pub mod aggregation;
pub mod hierarchy;
pub mod matrix;
pub mod prolongator;
pub mod smoother;

use crate::SparseSymmetricMatrix;
use ndarray::Array1;
use petgraph_drawing::DrawingValue;

use hierarchy::{build_hierarchy, Hierarchy};
use matrix::CsrMatrix;
use smoother::jacobi_smooth;

/// Algebraic Multigrid (AMG) Preconditioner and Solver
#[derive(Debug, Clone)]
pub struct AmgSolver<S> {
    pub max_iterations: usize,
    pub tolerance: S,
    pub theta: S,
    pub max_levels: usize,
    pub max_coarse_size: usize,
    pub pre_sweeps: usize,
    pub post_sweeps: usize,
    pub omega: S,
}

impl<S> Default for AmgSolver<S>
where
    S: num_traits::Float
        + num_traits::Zero
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Default
        + std::cmp::PartialOrd
        + DrawingValue,
{
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: S::from_f64(1e-6).unwrap(),
            theta: S::from_f64(0.25).unwrap(),
            max_levels: 10,
            max_coarse_size: 50,
            pre_sweeps: 1,
            post_sweeps: 1,
            omega: S::from_f64(0.6666666666666666).unwrap(), // 2/3
        }
    }
}

impl<S> AmgSolver<S>
where
    S: Copy
        + num_traits::Float
        + num_traits::Zero
        + std::ops::AddAssign
        + std::ops::SubAssign
        + Default
        + std::cmp::PartialOrd
        + std::fmt::Debug
        + DrawingValue,
{
    fn v_cycle(
        &self,
        hierarchy: &Hierarchy<S>,
        level_idx: usize,
        b: &Array1<S>,
        x: &mut Array1<S>,
    ) {
        if level_idx == hierarchy.levels.len() {
            // Exact solve at coarsest level (or just many Jacobi sweeps for now)
            jacobi_smooth(&hierarchy.coarsest_a, b, x, self.omega, 20);
            return;
        }

        let level = &hierarchy.levels[level_idx];

        // Pre-smooth
        jacobi_smooth(&level.a, b, x, self.omega, self.pre_sweeps);

        // Compute residual
        let mut r = Array1::zeros(level.a.rows);
        level.a.multiply_into(x, &mut r);
        for i in 0..r.len() {
            r[i] = b[i] - r[i];
        }

        // Restrict residual
        let r_coarse = level.r.multiply(&r);

        // Coarse grid correction
        let mut e_coarse = Array1::zeros(level.r.rows);
        self.v_cycle(hierarchy, level_idx + 1, &r_coarse, &mut e_coarse);

        // Prolongate and add
        let e_fine = level.p.multiply(&e_coarse);
        for i in 0..x.len() {
            x[i] += e_fine[i];
        }

        // Post-smooth
        jacobi_smooth(&level.a, b, x, self.omega, self.post_sweeps);
    }

    /// Solves the system using AMG as a standalone solver
    pub fn solve(
        &self,
        matrix: &SparseSymmetricMatrix<S>,
        b: &Array1<S>,
        x: &mut Array1<S>,
    ) -> usize {
        let a_csr = CsrMatrix::from_symmetric(matrix);
        let hierarchy = build_hierarchy(
            a_csr.clone(),
            self.theta,
            self.max_levels,
            self.max_coarse_size,
        );

        let mut r = Array1::zeros(a_csr.rows);
        a_csr.multiply_into(x, &mut r);
        for i in 0..r.len() {
            r[i] = b[i] - r[i];
        }

        let mut b_norm = S::zero();
        for &val in b.iter() {
            b_norm += val * val;
        }
        b_norm = b_norm.sqrt();
        if b_norm == S::zero() {
            b_norm = S::one();
        }

        for iter in 0..self.max_iterations {
            let mut r_norm = S::zero();
            for &val in r.iter() {
                r_norm += val * val;
            }
            r_norm = r_norm.sqrt();

            if r_norm / b_norm < self.tolerance {
                return iter;
            }

            self.v_cycle(&hierarchy, 0, b, x);

            a_csr.multiply_into(x, &mut r);
            for i in 0..r.len() {
                r[i] = b[i] - r[i];
            }
        }

        self.max_iterations
    }

    /// Solves the system `A * x = b` by applying a single V-cycle (used for preconditioning)
    pub fn apply_preconditioner(&self, hierarchy: &Hierarchy<S>, r: &Array1<S>, z: &mut Array1<S>) {
        z.fill(S::zero());
        self.v_cycle(hierarchy, 0, r, z);
    }
}
