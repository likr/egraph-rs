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
        Ok(TKernel::new(kernel, self.alpha))
    }
}
