use crate::*;
use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Builder for TKernel
#[derive(Clone, Debug)]
pub struct TKernelBuilder<S> {
    pub alpha: S,
}

impl<S: Float> TKernelBuilder<S> {
    pub fn new(alpha: S) -> Self {
        Self { alpha }
    }

    pub fn alpha(mut self, alpha: S) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn build<K>(self, kernel: K) -> Result<TKernel<K, S>, String>
    where
        S: DrawingValue,
    {
        if !self.alpha.is_finite() || self.alpha <= S::zero() {
            return Err("alpha must be finite and positive".to_string());
        }
        Ok(TKernel::new(kernel, self.alpha))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_t_kernel_builder_validates_alpha() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        g.add_edge(n0, n1, ());
        let lap = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0_f64);
        let mut rng = StdRng::seed_from_u64(1);
        let dk = DiffusionKernelBuilder::new(&lap, 1.0, 10)
            .build(&mut rng)
            .unwrap();

        for bad_alpha in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = TKernelBuilder::new(bad_alpha)
                .build(dk.clone())
                .unwrap_err();
            assert!(err.contains("alpha"));
        }
        assert!(TKernelBuilder::new(1.0).build(dk).is_ok());
    }
}
