use crate::*;
use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Builder for GaussianKernel
#[derive(Clone, Debug)]
pub struct GaussianKernelBuilder<S> {
    pub gamma: S,
}

impl<S: Float> GaussianKernelBuilder<S> {
    pub fn new(gamma: S) -> Self {
        Self { gamma }
    }

    pub fn gamma(mut self, gamma: S) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn build<K>(self, kernel: K) -> Result<GaussianKernel<K, S>, String>
    where
        S: DrawingValue,
    {
        Ok(GaussianKernel::new(kernel, self.gamma))
    }
}
