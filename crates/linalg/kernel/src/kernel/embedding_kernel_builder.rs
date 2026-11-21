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
        EmbeddingKernel::try_new(graph, embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use petgraph::graph::{NodeIndex, UnGraph};

    #[test]
    fn test_embedding_kernel_builder_rejects_row_mismatch() {
        let mut g10 = UnGraph::<(), ()>::new_undirected();
        for _ in 0..10 {
            g10.add_node(());
        }
        let err = EmbeddingKernelBuilder::new()
            .build::<_, NodeIndex, f64>(&g10, Array2::<f64>::zeros((2, 2)))
            .unwrap_err();
        assert!(err.contains("must match graph node count"));
    }
}
