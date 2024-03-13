use crate::{double_centering::double_centering, eigendecomposition::eigendecomposition};
use ndarray::prelude::*;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
use petgraph_algorithm_shortest_path::{multi_source_dijkstra, DistanceMatrix};
use petgraph_drawing::{Drawing, DrawingEuclidean, DrawingEuclidean2d, DrawingIndex};

pub struct PivotMds<N> {
    pub eps: f32,
    indices: Vec<N>,
    c: Array2<f32>,
}

impl<N> PivotMds<N>
where
    N: DrawingIndex,
{
    pub fn new<G, F>(graph: G, length: F, sources: &[G::NodeId]) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Ord + Into<N>,
        F: FnMut(G::EdgeRef) -> f32,
    {
        let distance_matrix = multi_source_dijkstra(graph, length, &sources);
        Self::new_with_distance_matrix(distance_matrix)
    }

    pub fn new_with_distance_matrix<N2, D>(distance_matrix: D) -> Self
    where
        N2: DrawingIndex + Copy + Into<N>,
        D: DistanceMatrix<N2, f32>,
    {
        let (n, m) = distance_matrix.shape();
        let mut delta = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                delta[[i, j]] = distance_matrix.get_by_index(i, j).powi(2);
            }
        }
        let c = double_centering(&delta);
        Self {
            eps: 1e-3,
            indices: distance_matrix
                .row_indices()
                .map(|u| u.into())
                .collect::<Vec<_>>(),
            c,
        }
    }

    pub fn run_2d(&self) -> DrawingEuclidean2d<N, f32>
    where
        N: Copy,
    {
        let ct_c = self.c.t().dot(&self.c);
        let (e, v) = eigendecomposition(&ct_c, 2, self.eps);
        let xy = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
        let xy = self.c.dot(&xy);
        let mut drawing = DrawingEuclidean2d::from_node_indices(&self.indices);
        for (i, &u) in self.indices.iter().enumerate() {
            drawing.position_mut(u).map(|p| {
                p.0 = xy[[i, 0]];
                p.1 = xy[[i, 1]]
            });
        }
        drawing
    }

    pub fn run(&self, d: usize) -> DrawingEuclidean<N, f32>
    where
        N: Copy,
    {
        let ct_c = self.c.t().dot(&self.c);
        let (e, v) = eigendecomposition(&ct_c, 2, self.eps);
        let x = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
        let x = self.c.dot(&x);
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
