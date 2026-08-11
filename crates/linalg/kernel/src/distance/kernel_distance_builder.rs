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
        Ok(KernelDistance::new(kernel).min_dist(self.min_dist))
    }
}

impl<S: Float> Default for KernelDistanceBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}
