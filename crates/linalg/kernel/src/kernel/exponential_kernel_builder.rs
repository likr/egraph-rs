use crate::*;
use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Builder for ExponentialKernel
#[derive(Clone, Debug)]
pub struct ExponentialKernelBuilder<S> {
    pub gamma: S,
}

impl<S: Float> ExponentialKernelBuilder<S> {
    pub fn new(gamma: S) -> Self {
        Self { gamma }
    }

    pub fn gamma(mut self, gamma: S) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn build<K>(self, kernel: K) -> Result<ExponentialKernel<K, S>, String>
    where
        S: DrawingValue,
    {
        if !self.gamma.is_finite() || self.gamma < S::zero() {
            return Err("gamma must be finite and non-negative".to_string());
        }
        Ok(ExponentialKernel::new(kernel, self.gamma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_exponential_kernel_builder_validates_gamma() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        g.add_edge(n0, n1, ());
        let lap = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0_f64);
        let mut rng = StdRng::seed_from_u64(1);
        let dk = DiffusionKernelBuilder::new(&lap, 1.0, 10)
            .build(&mut rng)
            .unwrap();

        for bad_gamma in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = ExponentialKernelBuilder::new(bad_gamma)
                .build(dk.clone())
                .unwrap_err();
            assert!(err.contains("gamma"));
        }
        assert!(ExponentialKernelBuilder::new(0.0).build(dk.clone()).is_ok());
        assert!(ExponentialKernelBuilder::new(0.5).build(dk).is_ok());
    }
}
