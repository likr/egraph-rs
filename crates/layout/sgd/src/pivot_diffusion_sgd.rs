use crate::sparse_sgd::proportional_sampling;
use crate::Sgd;
use crate::SparseSgd;
use ndarray::{Array1, ScalarOperand};
use num_traits::Float;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeIndexable};
use petgraph_distance::SparseSymmetricMatrix;
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_diffusion_kernel::{DiffusionKernel, PivotDiffusionDistanceMatrix};
use rand::Rng;

/// Builder for creating Pivot-based Diffusion SGD layout instances.
///
/// Uses exact single-source heat diffusion from incrementally selected pivot nodes,
/// with Hutchinson trace estimation for diagonal elements.
///
/// Distances are defined as:
/// $$D_{ij} = \sqrt{-4t \log \left( \frac{K_{ij}}{\sqrt{K_{ii} K_{jj}}} \right)}$$
///
/// Pivot selection uses max-min random sampling on these distances, computing only
/// h heat diffusion vectors (one per selected pivot) for O(h·degree·|E|) total cost.
pub struct PivotDiffusionSgd<S> {
    /// Heat diffusion time parameter t
    t: S,
    /// Degree of Chebyshev polynomial approximation
    degree: usize,
    /// Number of Rademacher vectors for Hutchinson diagonal trace estimation
    num_vectors: usize,
    /// Number of pivot nodes to select
    h: usize,
}

impl<S> Default for PivotDiffusionSgd<S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive
        + DrawingValue,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S> PivotDiffusionSgd<S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive
        + DrawingValue,
{
    pub fn new() -> Self {
        Self {
            t: S::from_f64(1.0).unwrap(),
            degree: 30,
            num_vectors: 32,
            h: 50,
        }
    }

    /// Sets the diffusion time parameter t.
    pub fn t(&mut self, t: S) -> &mut Self {
        self.t = t;
        self
    }

    /// Sets the degree of Chebyshev polynomial approximation.
    pub fn degree(&mut self, degree: usize) -> &mut Self {
        self.degree = degree;
        self
    }

    /// Sets the number of vectors used for Hutchinson diagonal trace estimation.
    pub fn num_vectors(&mut self, num_vectors: usize) -> &mut Self {
        self.num_vectors = num_vectors;
        self
    }

    /// Sets the number of pivot nodes.
    pub fn h(&mut self, h: usize) -> &mut Self {
        self.h = h;
        self
    }

    /// Selects pivot nodes using incremental max-min random sampling on heat diffusion distances.
    ///
    /// Each new pivot's distance vector is computed on demand via exact single-source
    /// heat diffusion, keeping total cost at O(h·degree·|E|) rather than O(n·degree·|E|).
    pub fn choose_pivot<G, R>(
        &self,
        graph: G,
        laplacian: &SparseSymmetricMatrix<S>,
        rng: &mut R,
    ) -> (Vec<G::NodeId>, PivotDiffusionDistanceMatrix<G::NodeId, S>)
    where
        G: IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex,
        R: Rng,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        let n = nodes.len();
        let h = self.h.min(n);
        let kernel = DiffusionKernel::new(laplacian, self.t, self.degree, self.num_vectors, rng);

        let mut pivot_indices = Vec::with_capacity(h);
        let mut pivot_distances = Vec::with_capacity(h);

        // Select first pivot randomly and compute its distance vector
        let p0 = rng.gen_range(0..n);
        let d0 = PivotDiffusionDistanceMatrix::pivot_distance_vector(
            laplacian,
            &kernel,
            self.t,
            self.degree,
            p0,
        );
        let mut min_d = Array1::from_vec(d0.clone());
        pivot_indices.push(p0);
        pivot_distances.push(d0);

        // Incrementally select remaining pivots
        for _ in 1..h {
            let pk = proportional_sampling(&min_d, rng);
            let dk = PivotDiffusionDistanceMatrix::pivot_distance_vector(
                laplacian,
                &kernel,
                self.t,
                self.degree,
                pk,
            );
            for (j, &d) in dk.iter().enumerate() {
                min_d[j] = min_d[j].min(d);
            }
            pivot_indices.push(pk);
            pivot_distances.push(dk);
        }

        let pivot_nodes = pivot_indices.iter().map(|&i| nodes[i]).collect();
        let dm = PivotDiffusionDistanceMatrix::from_precomputed(
            graph,
            &pivot_indices,
            pivot_distances,
            S::zero(),
        );
        (pivot_nodes, dm)
    }

    /// Builds an SGD instance with heat diffusion distances and max-min random pivot sampling.
    pub fn build<G, F, R>(
        &self,
        graph: G,
        laplacian: &SparseSymmetricMatrix<S>,
        length: F,
        rng: &mut R,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let (pivots, distance_matrix) = self.choose_pivot(graph, laplacian, rng);
        SparseSgd::new()
            .h(pivots.len())
            .build_with_pivot_and_distance_matrix(graph, length, &pivots, &distance_matrix)
    }

    /// Builds an SGD instance with pre-selected pivot nodes.
    pub fn build_with_pivot<G, F, R>(
        &self,
        graph: G,
        laplacian: &SparseSymmetricMatrix<S>,
        length: F,
        pivots: &[G::NodeId],
        rng: &mut R,
    ) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let kernel = DiffusionKernel::new(laplacian, self.t, self.degree, self.num_vectors, rng);
        let pivot_indices: Vec<usize> = pivots.iter().map(|&p| graph.to_index(p)).collect();
        let distance_matrix = PivotDiffusionDistanceMatrix::new(
            graph,
            laplacian,
            &kernel,
            self.t,
            self.degree,
            &pivot_indices,
            S::zero(),
        );

        SparseSgd::new()
            .h(pivots.len())
            .build_with_pivot_and_distance_matrix(graph, length, pivots, &distance_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use petgraph_distance::{Laplacian, StandardLaplacian};
    use rand::SeedableRng;

    #[test]
    fn test_pivot_diffusion_sgd_basic() {
        let n = 5;
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
        for i in 0..(n - 1) {
            graph.add_edge(nodes[i], nodes[i + 1], ());
        }

        let laplacian = StandardLaplacian.build(&graph, &mut |_| 1.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let sgd = PivotDiffusionSgd::new()
            .h(2)
            .t(0.5)
            .degree(20)
            .num_vectors(16)
            .build(&graph, &laplacian, |_| 1.0, &mut rng);

        let node_pairs = sgd.node_pairs();
        assert!(!node_pairs.is_empty());
        for &(_, _, dij, dji, wij, wji) in node_pairs {
            assert!(dij > 0.0);
            assert!(dji > 0.0);
            assert!(wij >= 0.0);
            assert!(wji > 0.0);
        }
    }
}
