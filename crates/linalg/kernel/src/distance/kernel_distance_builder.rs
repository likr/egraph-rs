use crate::*;
use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Builder for KernelDistance
#[derive(Clone, Debug)]
pub struct KernelDistanceBuilder<S> {
    pub min_dist: S,
}

impl<S: Float> KernelDistanceBuilder<S> {
    pub fn new() -> Self {
        Self {
            min_dist: S::zero(),
        }
    }

    pub fn min_dist(mut self, min_dist: S) -> Self {
        self.min_dist = min_dist;
        self
    }

    pub fn build<K>(self, kernel: K) -> Result<KernelDistance<K, S>, String>
    where
        S: DrawingValue,
    {
        if !self.min_dist.is_finite() || self.min_dist < S::zero() {
            return Err("min_dist must be finite and non-negative".to_string());
        }
        Ok(KernelDistance::new(kernel).min_dist(self.min_dist))
    }
}

impl<S: Float> Default for KernelDistanceBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_kernel_distance_builder_validates_min_dist() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        g.add_edge(n0, n1, ());
        let lap = SparseSymmetricMatrix::symmetric_normalized_laplacian(&g, |_| 1.0_f64);
        let mut rng = StdRng::seed_from_u64(1);
        let dk = DiffusionKernelBuilder::new(&lap, 1.0, 10)
            .build(&mut rng)
            .unwrap();

        for bad_min_dist in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = KernelDistanceBuilder::new()
                .min_dist(bad_min_dist)
                .build(dk.clone())
                .unwrap_err();
            assert!(err.contains("min_dist"));
        }
        assert!(KernelDistanceBuilder::new()
            .min_dist(0.0)
            .build(dk.clone())
            .is_ok());
        assert!(KernelDistanceBuilder::new()
            .min_dist(1e-3)
            .build(dk)
            .is_ok());
    }
}
