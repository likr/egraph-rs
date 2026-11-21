use crate::solver::chebyshev::compute_hutchinson_diagonal_from_residual;
use crate::*;
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use rand::Rng;

pub(crate) fn generate_rademacher_vectors<S, R>(n: usize, m: usize, rng: &mut R) -> Array2<S>
where
    S: Float,
    R: Rng,
{
    Array2::from_shape_fn((n, m), |_| {
        if rng.gen::<bool>() {
            S::one()
        } else {
            -S::one()
        }
    })
}

pub(crate) fn resolve_lambda_max<S, R>(
    laplacian: &SparseSymmetricMatrix<S>,
    lambda_max: Option<S>,
    rng: &mut R,
) -> Result<S, String>
where
    S: Float + std::iter::Sum + std::ops::AddAssign + Default + ScalarOperand,
    R: Rng,
{
    match lambda_max {
        Some(l) => {
            if !l.is_finite() || l <= S::zero() {
                return Err("lambda_max must be finite and positive".to_string());
            }
            Ok(l)
        }
        None => {
            if laplacian.dim() == 0 {
                Ok(S::one())
            } else {
                Ok(estimate_lambda_max(
                    laplacian,
                    rng,
                    100,
                    S::from(1e-4).unwrap(),
                ))
            }
        }
    }
}

pub(crate) fn build_pivot_vectors_with<S, F>(
    n: usize,
    pivots: &[usize],
    mut eval_vec: F,
) -> Result<Vec<Array1<S>>, String>
where
    S: Float,
    F: FnMut(&Array1<S>) -> Array1<S>,
{
    let mut pivot_vectors = Vec::with_capacity(pivots.len());
    for &p in pivots {
        if p >= n {
            return Err("Pivot index out of bounds".to_string());
        }
        let mut e = Array1::zeros(n);
        e[p] = S::one();
        pivot_vectors.push(eval_vec(&e));
    }
    Ok(pivot_vectors)
}

pub(crate) fn compute_hutchinson_diagonal_with<S, R, F>(
    laplacian: &SparseSymmetricMatrix<S>,
    num_hutchinson: Option<usize>,
    g_hat_0: S,
    rng: &mut R,
    eval_baseline: F,
) -> Option<Array1<S>>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive,
    R: Rng,
    F: FnOnce(&Array2<S>, &Array1<S>) -> (Array2<S>, Array2<S>, Vec<S>),
{
    let n = laplacian.dim();
    match num_hutchinson {
        Some(m) if m > 0 && n > 0 => {
            let u1 = laplacian.stationary_vector();
            let rademacher = generate_rademacher_vectors(n, m, rng);
            let (_, _, residual_diag_sum) = eval_baseline(&rademacher, &u1);
            Some(compute_hutchinson_diagonal_from_residual(
                &residual_diag_sum,
                &u1,
                g_hat_0,
                m,
            ))
        }
        Some(m) if m > 0 && n == 0 => Some(Array1::zeros(0)),
        _ => None,
    }
}

/// Builder for PivotedDiffusionKernel
pub struct PivotedDiffusionKernelBuilder<'a, S> {
    pub laplacian: &'a SparseSymmetricMatrix<S>,
    pub t: S,
    pub degree: usize,
    pub pivots: Vec<usize>,
    pub lambda_max: Option<S>,
    pub num_hutchinson: Option<usize>,
}

impl<'a, S> PivotedDiffusionKernelBuilder<'a, S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive,
{
    pub fn new(
        laplacian: &'a SparseSymmetricMatrix<S>,
        t: S,
        degree: usize,
        pivots: Vec<usize>,
    ) -> Self {
        Self {
            laplacian,
            t,
            degree,
            pivots,
            lambda_max: None,
            num_hutchinson: None,
        }
    }

    pub fn lambda_max(mut self, lambda_max: S) -> Self {
        self.lambda_max = Some(lambda_max);
        self
    }

    pub fn num_hutchinson(mut self, num_hutchinson: Option<usize>) -> Self {
        self.num_hutchinson = num_hutchinson;
        self
    }

    pub fn build<R: Rng>(self, rng: &mut R) -> Result<PivotedDiffusionKernel<S>, String> {
        if !self.t.is_finite() || self.t < S::zero() {
            return Err("t must be finite and >= 0".to_string());
        }
        let n = self.laplacian.dim();
        for &p in &self.pivots {
            if p >= n {
                return Err("Pivot index out of bounds".to_string());
            }
        }

        let lambda_max = resolve_lambda_max(self.laplacian, self.lambda_max, rng)?;

        let pivot_vectors = build_pivot_vectors_with(n, &self.pivots, |e| {
            chebyshev_approximation_vec(self.laplacian, self.t, self.degree, lambda_max, e)
        })?;

        let diagonal = compute_hutchinson_diagonal_with(
            self.laplacian,
            self.num_hutchinson,
            S::one(),
            rng,
            |rademacher, u1| {
                chebyshev_approximation_with_baseline(
                    self.laplacian,
                    self.t,
                    self.degree,
                    lambda_max,
                    rademacher,
                    u1,
                )
            },
        );

        Ok(PivotedDiffusionKernel {
            pivots: self.pivots,
            vectors: pivot_vectors,
            diagonal,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_pivoted_diffusion_kernel_builder_rejects_invalid_t() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        g.add_edge(n0, n1, ());
        let lap = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0_f64);
        let mut rng = StdRng::seed_from_u64(1);
        for bad_t in [-1.0, f64::NAN, f64::INFINITY] {
            let err = PivotedDiffusionKernelBuilder::new(&lap, bad_t, 10, vec![0])
                .build(&mut rng)
                .unwrap_err();
            assert!(err.contains("t"));
        }
        assert!(PivotedDiffusionKernelBuilder::new(&lap, 0.0, 10, vec![0])
            .build(&mut rng)
            .is_ok());
    }
}
