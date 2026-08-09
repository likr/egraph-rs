use ndarray::Array1;
use petgraph_distance::SparseSymmetricMatrix;
use petgraph_drawing::DrawingValue;
use std::collections::HashMap;

/// Trait for building a solver, often caching preconditioners.
pub trait LinearSolverBuilder<S> {
    type Solver: LinearSolver<S>;

    /// Builds the linear solver, consuming the matrix.
    fn build(&self, matrix: SparseSymmetricMatrix<S>) -> Self::Solver;
}

/// Trait for solving linear systems `A * x = b`.
pub trait LinearSolver<S> {
    /// Returns a reference to the internal matrix.
    fn matrix(&self) -> &SparseSymmetricMatrix<S>;

    /// Solves the system `matrix * x = b`.
    /// Returns the number of iterations taken.
    fn solve(&self, b: &Array1<S>, x: &mut Array1<S>) -> usize;
}

/// Standard Conjugate Gradient without preconditioning.
#[derive(Debug, Clone)]
pub struct CgSolver<S> {
    pub max_iterations: usize,
    pub tolerance: S,
}

pub struct CgSolverInstance<S> {
    matrix: SparseSymmetricMatrix<S>,
    max_iterations: usize,
    tolerance: S,
}

impl<S> LinearSolverBuilder<S> for CgSolver<S>
where
    S: DrawingValue + Default,
{
    type Solver = CgSolverInstance<S>;

    fn build(&self, matrix: SparseSymmetricMatrix<S>) -> Self::Solver {
        CgSolverInstance {
            matrix,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
        }
    }
}

impl<S> LinearSolver<S> for CgSolverInstance<S>
where
    S: DrawingValue + Default,
{
    fn matrix(&self) -> &SparseSymmetricMatrix<S> {
        &self.matrix
    }

    fn solve(&self, b: &Array1<S>, x: &mut Array1<S>) -> usize {
        let n = self.matrix.dim();
        let mut r = Array1::zeros(n);
        let mut p = Array1::zeros(n);
        let mut q = Array1::zeros(n);

        self.matrix.multiply_into(x, &mut r);
        for i in 0..n {
            r[i] = b[i] - r[i];
            p[i] = r[i];
        }

        let mut rsold = r.dot(&r);
        if rsold < self.tolerance * self.tolerance {
            return 0;
        }

        for iter in 0..self.max_iterations {
            self.matrix.multiply_into(&p, &mut q);
            let alpha = rsold / p.dot(&q);

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * q[i];
            }

            let rsnew = r.dot(&r);
            if rsnew < self.tolerance * self.tolerance {
                return iter + 1;
            }

            let beta = rsnew / rsold;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rsold = rsnew;
        }
        self.max_iterations
    }
}

/// Conjugate Gradient with Jacobi (diagonal) preconditioning.
#[derive(Debug, Clone)]
pub struct JacobiCgSolver<S> {
    pub max_iterations: usize,
    pub tolerance: S,
}

pub struct JacobiCgSolverInstance<S> {
    matrix: SparseSymmetricMatrix<S>,
    max_iterations: usize,
    tolerance: S,
}

impl<S> LinearSolverBuilder<S> for JacobiCgSolver<S>
where
    S: DrawingValue + Default,
{
    type Solver = JacobiCgSolverInstance<S>;

    fn build(&self, matrix: SparseSymmetricMatrix<S>) -> Self::Solver {
        JacobiCgSolverInstance {
            matrix,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
        }
    }
}

impl<S> LinearSolver<S> for JacobiCgSolverInstance<S>
where
    S: DrawingValue + Default,
{
    fn matrix(&self) -> &SparseSymmetricMatrix<S> {
        &self.matrix
    }

    fn solve(&self, b: &Array1<S>, x: &mut Array1<S>) -> usize {
        let n = self.matrix.dim();
        let mut r = Array1::zeros(n);
        let mut z = Array1::zeros(n);
        let mut q = Array1::zeros(n);

        self.matrix.multiply_into(x, &mut r);
        for i in 0..n {
            r[i] = b[i] - r[i];
            let diag = self.matrix.diagonal()[i];
            z[i] = if diag > S::zero() { r[i] / diag } else { r[i] };
        }
        let mut p = z.clone();
        let mut rsold = r.dot(&z);

        if rsold < self.tolerance * self.tolerance {
            return 0;
        }

        for iter in 0..self.max_iterations {
            self.matrix.multiply_into(&p, &mut q);
            let alpha = rsold / p.dot(&q);

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * q[i];
            }

            for i in 0..n {
                let diag = self.matrix.diagonal()[i];
                z[i] = if diag > S::zero() { r[i] / diag } else { r[i] };
            }

            let rsnew = r.dot(&z);
            if rsnew < self.tolerance * self.tolerance {
                return iter + 1;
            }

            let beta = rsnew / rsold;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rsold = rsnew;
        }
        self.max_iterations
    }
}

/// IC(0) Incomplete Cholesky preconditioner
#[derive(Debug, Clone)]
pub struct IncompleteCholeskyPreconditioner<S> {
    n: usize,
    diagonal: Vec<S>,
    row_entries: Vec<Vec<(usize, S)>>,
    col_entries: Vec<Vec<(usize, S)>>,
}

impl<S> IncompleteCholeskyPreconditioner<S>
where
    S: DrawingValue + Default,
{
    pub fn from_matrix(matrix: &SparseSymmetricMatrix<S>) -> Self {
        let n = matrix.dim();
        let mut adjacency: Vec<HashMap<usize, S>> = vec![HashMap::new(); n];
        for &(i, j, val) in matrix.edges() {
            adjacency[i].insert(j, val);
            adjacency[j].insert(i, val);
        }

        let mut diagonal = vec![S::zero(); n];
        let mut row_entries: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];
        let mut col_entries: Vec<Vec<(usize, S)>> = vec![Vec::new(); n];

        for i in 0..n {
            for (&j, &val) in &adjacency[i] {
                if j < i {
                    row_entries[i].push((j, val));
                    col_entries[j].push((i, val));
                }
            }
        }

        for i in 0..n {
            row_entries[i].sort_by_key(|&(col, _)| col);
            col_entries[i].sort_by_key(|&(row, _)| row);
        }

        for i in 0..n {
            let mut sum = S::zero();
            for &(_, l_ik) in &row_entries[i] {
                sum += l_ik * l_ik;
            }

            let aii = matrix.diagonal()[i];
            diagonal[i] = (aii - sum).max(S::zero()).sqrt();

            if diagonal[i] <= S::zero() {
                diagonal[i] = S::from_f32(1e-6).unwrap();
            }

            for (&j, &val) in &adjacency[i] {
                let entry_pos = if j > i {
                    row_entries[j].iter().position(|&(col, _)| col == i)
                } else {
                    None
                };
                if let Some(entry_pos) = entry_pos {
                    let mut sum = S::zero();
                    let mut i_ptr = 0;
                    let mut j_ptr = 0;

                    while i_ptr < row_entries[i].len() && j_ptr < row_entries[j].len() {
                        let (i_col, i_val) = row_entries[i][i_ptr];
                        let (j_col, j_val) = row_entries[j][j_ptr];

                        if i_col == j_col && i_col < i {
                            sum += i_val * j_val;
                            i_ptr += 1;
                            j_ptr += 1;
                        } else if i_col < j_col {
                            i_ptr += 1;
                        } else {
                            j_ptr += 1;
                        }
                    }

                    let a_ji = val;
                    let new_value = (a_ji - sum) / diagonal[i];

                    row_entries[j][entry_pos].1 = new_value;
                    if let Some(col_pos) = col_entries[i].iter().position(|&(row, _)| row == j) {
                        col_entries[i][col_pos].1 = new_value;
                    }
                }
            }
        }

        Self {
            n,
            diagonal,
            row_entries,
            col_entries,
        }
    }

    pub fn apply(&self, r: &Array1<S>, z: &mut Array1<S>) {
        let mut y = Array1::zeros(self.n);
        for i in 0..self.n {
            let mut sum = S::zero();
            for &(j, l_ij) in &self.row_entries[i] {
                sum += l_ij * y[j];
            }
            y[i] = (r[i] - sum) / self.diagonal[i];
        }

        z.fill(S::zero());
        for i in (0..self.n).rev() {
            let mut sum = S::zero();
            for &(j, l_ji) in &self.col_entries[i] {
                sum += l_ji * z[j];
            }
            z[i] = (y[i] - sum) / self.diagonal[i];
        }
    }
}

/// Conjugate Gradient with IC(0) preconditioning.
#[derive(Debug, Clone)]
pub struct Ic0CgSolver<S> {
    pub max_iterations: usize,
    pub tolerance: S,
}

pub struct Ic0CgSolverInstance<S> {
    matrix: SparseSymmetricMatrix<S>,
    preconditioner: IncompleteCholeskyPreconditioner<S>,
    max_iterations: usize,
    tolerance: S,
}

impl<S> LinearSolverBuilder<S> for Ic0CgSolver<S>
where
    S: DrawingValue + Default,
{
    type Solver = Ic0CgSolverInstance<S>;

    fn build(&self, matrix: SparseSymmetricMatrix<S>) -> Self::Solver {
        let preconditioner = IncompleteCholeskyPreconditioner::from_matrix(&matrix);
        Ic0CgSolverInstance {
            matrix,
            preconditioner,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
        }
    }
}

impl<S> LinearSolver<S> for Ic0CgSolverInstance<S>
where
    S: DrawingValue + Default,
{
    fn matrix(&self) -> &SparseSymmetricMatrix<S> {
        &self.matrix
    }

    fn solve(&self, b: &Array1<S>, x: &mut Array1<S>) -> usize {
        let n = self.matrix.dim();
        let mut r = Array1::zeros(n);
        let mut z = Array1::zeros(n);
        let mut q = Array1::zeros(n);

        self.matrix.multiply_into(x, &mut r);
        for i in 0..n {
            r[i] = b[i] - r[i];
        }

        self.preconditioner.apply(&r, &mut z);
        let mut p = z.clone();
        let mut rsold = r.dot(&z);

        if rsold < self.tolerance * self.tolerance {
            return 0;
        }

        for iter in 0..self.max_iterations {
            self.matrix.multiply_into(&p, &mut q);
            let alpha = rsold / p.dot(&q);

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * q[i];
            }

            self.preconditioner.apply(&r, &mut z);

            let rsnew = r.dot(&z);
            if rsnew < self.tolerance * self.tolerance {
                return iter + 1;
            }

            let beta = rsnew / rsold;
            for i in 0..n {
                p[i] = beta * p[i] + z[i];
            }
            rsold = rsnew;
        }
        self.max_iterations
    }
}

/// Basic Algebraic Multigrid (AMG) Preconditioner (Smoothed Aggregation)
#[derive(Debug, Clone)]
pub struct AmgCgSolver<S> {
    pub max_iterations: usize,
    pub tolerance: S,
}

pub struct AmgCgSolverInstance<S> {
    matrix: SparseSymmetricMatrix<S>,
    amg: crate::amg::AmgSolver<S>,
    hierarchy: crate::amg::hierarchy::Hierarchy<S>,
    max_iterations: usize,
    tolerance: S,
}

impl<S> LinearSolverBuilder<S> for AmgCgSolver<S>
where
    S: DrawingValue + Default,
{
    type Solver = AmgCgSolverInstance<S>;

    fn build(&self, matrix: SparseSymmetricMatrix<S>) -> Self::Solver {
        let amg = crate::amg::AmgSolver::<S>::default();
        let a_csr = crate::amg::matrix::CsrMatrix::from_symmetric(&matrix);
        let hierarchy = crate::amg::hierarchy::build_hierarchy(
            a_csr,
            amg.theta,
            amg.max_levels,
            amg.max_coarse_size,
        );

        AmgCgSolverInstance {
            matrix,
            amg,
            hierarchy,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
        }
    }
}

impl<S> LinearSolver<S> for AmgCgSolverInstance<S>
where
    S: DrawingValue + Default,
{
    fn matrix(&self) -> &SparseSymmetricMatrix<S> {
        &self.matrix
    }

    fn solve(&self, b: &Array1<S>, x: &mut Array1<S>) -> usize {
        let n = self.matrix.dim();
        let mut r = Array1::zeros(n);
        let mut z = Array1::zeros(n);
        let mut q = Array1::zeros(n);

        self.matrix.multiply_into(x, &mut r);
        for i in 0..n {
            r[i] = b[i] - r[i];
        }

        self.amg.apply_preconditioner(&self.hierarchy, &r, &mut z);
        let mut p = z.clone();
        let mut rsold = r.dot(&z);

        if rsold < self.tolerance * self.tolerance {
            return 0;
        }

        for iter in 0..self.max_iterations {
            self.matrix.multiply_into(&p, &mut q);
            let alpha = rsold / p.dot(&q);

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * q[i];
            }

            self.amg.apply_preconditioner(&self.hierarchy, &r, &mut z);

            let rsnew = r.dot(&z);
            if rsnew < self.tolerance * self.tolerance {
                return iter + 1;
            }

            let beta = rsnew / rsold;
            for i in 0..n {
                p[i] = beta * p[i] + z[i];
            }
            rsold = rsnew;
        }
        self.max_iterations
    }
}
