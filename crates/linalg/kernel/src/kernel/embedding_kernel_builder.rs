use crate::*;
use ndarray::Array2;
use petgraph::visit::IntoNodeIdentifiers;
use petgraph_drawing::DrawingValue;
use std::hash::Hash;

/// Builder for EmbeddingKernel
#[derive(Clone, Debug, Default)]
pub struct EmbeddingKernelBuilder;

impl EmbeddingKernelBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn build<G, N, S>(
        self,
        graph: G,
        embedding: Array2<S>,
    ) -> Result<EmbeddingKernel<N, S>, String>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Eq + Hash + Copy,
        S: DrawingValue,
    {
        Ok(EmbeddingKernel::new(graph, embedding))
    }
}
