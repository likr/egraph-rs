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
        Ok(ExponentialKernel::new(kernel, self.gamma))
    }
}
