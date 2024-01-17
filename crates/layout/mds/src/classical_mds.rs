use crate::{double_centering::double_centering, eigendecomposition::eigendecomposition};
use ndarray::prelude::*;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
use petgraph_algorithm_shortest_path::{all_sources_dijkstra, DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Drawing2D, DrawingD, DrawingIndex};

pub struct ClassicalMds {
    pub eps: f32,
    b: Array2<f32>,
}

impl ClassicalMds {
    pub fn new<G, F>(graph: G, length: F) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> f32,
    {
        let distance_matrix = all_sources_dijkstra(graph, length);
        Self::new_with_distance_matrix(distance_matrix)
    }

    pub fn new_with_distance_matrix<N>(distance_matrix: FullDistanceMatrix<N, f32>) -> Self
    where
        N: DrawingIndex,
    {
        let (n, m) = distance_matrix.shape();
        let mut delta = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                delta[[i, j]] = distance_matrix.get_by_index(i, j).powi(2);
            }
        }
        let b = double_centering(&delta);
        Self { eps: 1e-3, b }
    }

    pub fn run_2d<G>(&self, graph: G) -> Drawing2D<G::NodeId, f32>
    where
        G: IntoEdges + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Ord,
    {
        let (e, v) = eigendecomposition(&self.b, 2, self.eps);
        let xy = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
        let mut drawing = Drawing2D::new(graph);
        for (i, u) in graph.node_identifiers().enumerate() {
            drawing.position_mut(u).map(|p| {
                p.0 = xy[[i, 0]];
                p.1 = xy[[i, 1]];
            });
        }
        drawing
    }

    pub fn run<G>(&self, graph: G, d: usize) -> DrawingD<G::NodeId, f32>
    where
        G: IntoEdges + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Ord,
    {
        let (e, v) = eigendecomposition(&self.b, d, self.eps);
        let x = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
        let mut drawing = DrawingD::new(graph);
        drawing.set_dimension(d);
        for (i, u) in graph.node_identifiers().enumerate() {
            drawing.position_mut(u).map(|p| {
                for j in 0..d {
                    p.0[j] = x[[i, j]];
                }
            });
        }
        drawing
    }
}
