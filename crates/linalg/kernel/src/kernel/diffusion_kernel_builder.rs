use crate::*;
use ndarray::ScalarOperand;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use petgraph::{
    graph::IndexType,
    prelude::NodeIndex,
    visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::Rng;
use std::hash::Hash;

/// Builder for DiffusionKernel
pub struct DiffusionKernelBuilder<'a, S> {
    pub laplacian: &'a SparseSymmetricMatrix<S>,
    pub t: S,
    pub degree: usize,
    pub lambda_max: Option<S>,
}

impl<'a, S> DiffusionKernelBuilder<'a, S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive,
{
    pub fn new(laplacian: &'a SparseSymmetricMatrix<S>, t: S, degree: usize) -> Self {
        Self {
            laplacian,
            t,
            degree,
            lambda_max: None,
        }
    }

    pub fn lambda_max(mut self, lambda_max: S) -> Self {
        self.lambda_max = Some(lambda_max);
        self
    }

    pub fn build<R: Rng>(self, rng: &mut R) -> Result<DiffusionKernel<S>, String> {
        if !self.t.is_finite() || self.t < S::zero() {
            return Err("t must be finite and >= 0".to_string());
        }
        let lambda_max = match self.lambda_max {
            Some(l) => {
                if !l.is_finite() || l <= S::zero() {
                    return Err("lambda_max must be finite and positive".to_string());
                }
                l
            }
            None => {
                if self.laplacian.dim() == 0 {
                    S::one()
                } else {
                    estimate_lambda_max(self.laplacian, rng, 100, S::from_f64(1e-4).unwrap())
                }
            }
        };

        let n = self.laplacian.dim();
        let mut eye = Array2::zeros((n, n));
        for i in 0..n {
            eye[[i, i]] = S::one();
        }

        let matrix = chebyshev_approximation(self.laplacian, self.t, self.degree, lambda_max, &eye);

        Ok(DiffusionKernel { n, matrix })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_diffusion_kernel_builder_rejects_invalid_t() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        g.add_edge(n0, n1, ());
        let lap = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0_f64);
        let mut rng = StdRng::seed_from_u64(1);
        for bad_t in [-1.0, f64::NAN, f64::INFINITY] {
            let err = DiffusionKernelBuilder::new(&lap, bad_t, 10)
                .build(&mut rng)
                .unwrap_err();
            assert!(err.contains("t"));
        }
        assert!(DiffusionKernelBuilder::new(&lap, 0.0, 10)
            .build(&mut rng)
            .is_ok());
    }
}
