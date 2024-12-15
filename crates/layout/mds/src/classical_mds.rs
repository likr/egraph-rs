use crate::{double_centering::double_centering, eigendecomposition::eigendecomposition};
use ndarray::prelude::*;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
use petgraph_algorithm_shortest_path::{all_sources_dijkstra, DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Drawing, DrawingEuclidean, DrawingEuclidean2d, DrawingIndex};

pub struct ClassicalMds<N> {
    pub eps: f32,
    indices: Vec<N>,
    b: Array2<f32>,
}

impl<N> ClassicalMds<N>
where
    N: DrawingIndex,
{
    pub fn new<G, F>(graph: G, length: F) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Copy + Ord + Into<N>,
        F: FnMut(G::EdgeRef) -> f32,
        N: Copy,
    {
        let distance_matrix = all_sources_dijkstra(graph, length);
        Self::new_with_distance_matrix(&distance_matrix)
    }

    pub fn new_with_distance_matrix<N2>(distance_matrix: &FullDistanceMatrix<N2, f32>) -> Self
    where
        N2: DrawingIndex + Copy + Into<N>,
    {
        let (n, m) = distance_matrix.shape();
        let mut delta = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                delta[[i, j]] = distance_matrix.get_by_index(i, j).powi(2);
            }
        }
        let b = double_centering(&delta);
        Self {
            eps: 1e-3,
            indices: distance_matrix
                .row_indices()
                .map(|u| u.into())
                .collect::<Vec<_>>(),
            b,
        }
    }

    pub fn run_2d(&self) -> DrawingEuclidean2d<N, f32>
    where
        N: Copy,
    {
        let (e, v) = eigendecomposition(&self.b, 2, self.eps);
        let xy = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
        let mut drawing = DrawingEuclidean2d::from_node_indices(&self.indices);
        for (i, &u) in self.indices.iter().enumerate() {
            drawing.position_mut(u).map(|p| {
                p.0 = xy[[i, 0]];
                p.1 = xy[[i, 1]];
            });
        }
        drawing
    }

    pub fn run(&self, d: usize) -> DrawingEuclidean<N, f32>
    where
        N: Copy,
    {
        let (e, v) = eigendecomposition(&self.b, d, self.eps);
        let x = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
        let mut drawing = DrawingEuclidean::from_node_indices(&self.indices, d);
        for (i, &u) in self.indices.iter().enumerate() {
            drawing.position_mut(u).map(|p| {
                for j in 0..d {
                    p.0[j] = x[[i, j]];
                }
            });
        }
        drawing
    }
}
