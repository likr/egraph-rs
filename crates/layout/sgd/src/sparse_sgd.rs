use crate::Sgd;
use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_algorithm_shortest_path::{
    dijkstra_with_distance_matrix, multi_source_dijkstra, DistanceMatrix, SubDistanceMatrix,
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::prelude::*;
use std::collections::{HashMap, HashSet};

/// Sparse Stochastic Gradient Descent (SGD) implementation for graph layout.
///
/// This implementation uses a sparse approximation technique that selects a subset of
/// pivot nodes and computes shortest-path distances only from these pivot nodes to all
/// other nodes. This approach significantly reduces the computational complexity for
/// large graphs compared to the full SGD algorithm, while still producing good layouts.
///
/// The algorithm works by:
/// 1. Selecting a set of pivot nodes
/// 2. Computing shortest paths from these pivots to all other nodes
/// 3. Assigning each non-pivot node to its closest pivot
/// 4. Using these pivot-based distances to drive the layout optimization
pub struct SparseSgd<S> {
    /// List of node pairs to be considered during layout optimization.
    /// Each tuple contains (i, j, distance_ij, distance_ji, weight_ij, weight_ji)
    node_pairs: Vec<(usize, usize, S, S, S, S)>,
}

impl<S> SparseSgd<S> {
    /// Creates a new SparseSgd instance from a graph.
    ///
    /// This constructor selects pivot nodes randomly and computes the shortest path
    /// distances to set up the sparse SGD algorithm.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `h` - The number of pivot nodes to use (controls sparsity)
    ///
    /// # Returns
    /// A new SparseSgd instance configured with appropriate node pairs
    pub fn new<G, F>(graph: G, length: F, h: usize) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let mut rng = rand::thread_rng();
        SparseSgd::new_with_rng(graph, length, h, &mut rng)
    }

    /// Creates a new SparseSgd instance with a provided random number generator.
    ///
    /// This variant of the constructor allows using a specific random number generator
    /// for deterministic behavior in testing scenarios.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `h` - The number of pivot nodes to use (controls sparsity)
    /// * `rng` - The random number generator to use for selecting pivots
    ///
    /// # Returns
    /// A new SparseSgd instance configured with appropriate node pairs
    pub fn new_with_rng<G, F, R>(graph: G, length: F, h: usize, rng: &mut R) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        S: DrawingValue,
    {
        let mut length = length;
        let n = graph.node_count();
        let h = h.min(n);
        let (pivot, d) = Self::choose_pivot(graph, &mut length, h, rng);
        Self::new_with_pivot_and_distance_matrix(graph, length, &pivot, &d)
    }

    /// Creates a new SparseSgd instance with pre-selected pivot nodes.
    ///
    /// This constructor allows using a specific set of pivot nodes instead of
    /// selecting them randomly, which can be useful when you have domain knowledge
    /// about which nodes might make good pivots.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `pivot` - A slice of node IDs to use as pivot nodes
    ///
    /// # Returns
    /// A new SparseSgd instance configured with the specified pivot nodes
    pub fn new_with_pivot<G, F>(graph: G, mut length: F, pivot: &[G::NodeId]) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let d = multi_source_dijkstra(graph, &mut length, pivot);
        Self::new_with_pivot_and_distance_matrix(graph, &mut length, pivot, &d)
    }

    /// Creates a new SparseSgd instance with pre-selected pivot nodes and a pre-computed distance matrix.
    ///
    /// This is the most low-level constructor, useful when you have already computed
    /// the distance matrix or want to provide a custom distance matrix.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `pivot` - A slice of node IDs to use as pivot nodes
    /// * `distance_matrix` - A pre-computed distance matrix from pivots to all nodes
    ///
    /// # Returns
    /// A new SparseSgd instance configured with the specified pivot nodes and distances
    pub fn new_with_pivot_and_distance_matrix<G, F, D>(
        graph: G,
        mut length: F,
        pivot: &[G::NodeId],
        distance_matrix: &D,
    ) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        D: DistanceMatrix<G::NodeId, S>,
        S: DrawingValue,
    {
        let indices = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect::<HashMap<_, _>>();
        let n = indices.len();
        let h = pivot.len();
        let mut node_pairs = vec![];
        let mut edges = HashSet::new();
        for edge in graph.edge_references() {
            let i = indices[&edge.source()];
            let j = indices[&edge.target()];
            let dij = length(edge);
            let wij = S::one() / (dij * dij);
            node_pairs.push((i, j, dij, dij, wij, wij));
            edges.insert((i, j));
            edges.insert((j, i));
        }

        let r = (0..n)
            .map(|j| {
                (0..h)
                    .min_by_key(|&i| OrderedFloat(distance_matrix.get_by_index(i, j)))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut r_nodes = vec![vec![]; h];
        for j in 0..n {
            r_nodes[r[j]].push(j);
        }

        for (k, &u) in pivot.iter().enumerate() {
            let p = indices[&u];
            for i in 0..n {
                if edges.contains(&(p, i)) || p == i {
                    continue;
                }
                let dpi = distance_matrix.get_by_index(k, i);
                let wpi = S::one() / (dpi * dpi);
                let spi = S::from_usize(
                    r_nodes[k]
                        .iter()
                        .filter(|&&j| {
                            S::from_usize(2).unwrap() * distance_matrix.get_by_index(k, j) <= dpi
                        })
                        .count(),
                )
                .unwrap();
                node_pairs.push((p, i, dpi, dpi, spi * wpi, S::zero()));
            }
        }
        SparseSgd { node_pairs }
    }

    /// Selects pivot nodes using a max-min randomized algorithm.
    ///
    /// This method implements a maximal-minimal distance pivot selection strategy.
    /// It first selects a random node as the first pivot, then iteratively selects
    /// nodes that maximize the minimum distance to all previously selected pivots.
    ///
    /// # Parameters
    /// * `graph` - The input graph to select pivots from
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `h` - The number of pivot nodes to select
    /// * `rng` - The random number generator to use
    ///
    /// # Returns
    /// A tuple containing:
    /// - A vector of selected pivot node IDs
    /// - A SubDistanceMatrix containing distances from pivots to all nodes
    pub fn choose_pivot<G, F, R>(
        graph: G,
        length: F,
        h: usize,
        rng: &mut R,
    ) -> (Vec<G::NodeId>, SubDistanceMatrix<G::NodeId, S>)
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        S: DrawingValue,
    {
        max_min_random_sp(graph, length, h, rng)
    }
}

/// Implementation of the Sgd trait for SparseSgd
///
/// This provides the core SGD functionality for the sparse graph layout algorithm,
/// allowing it to work with the common SGD framework.
impl<S> Sgd<S> for SparseSgd<S> {
    /// Returns a reference to the node pairs data structure.
    ///
    /// For the sparse implementation, this contains:
    /// - All direct edge connections (from the original graph)
    /// - Additional pivot-to-node connections
    fn node_pairs(&self) -> &Vec<(usize, usize, S, S, S, S)> {
        &self.node_pairs
    }

    /// Returns a mutable reference to the node pairs data structure.
    ///
    /// This allows modifying the node pairs during execution if needed.
    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, S, S, S, S)> {
        &mut self.node_pairs
    }
}

/// Implementation of the maximal-minimal distance pivot selection strategy.
///
/// This function selects pivot nodes that are maximally distant from each other,
/// which helps ensure good coverage of the graph for sparse layouts.
///
/// The strategy works as follows:
/// 1. Select a random node as the first pivot
/// 2. Compute shortest paths from this pivot to all other nodes
/// 3. Select the next pivot as the node that maximizes its minimum distance to all existing pivots
/// 4. Repeat steps 2-3 until the desired number of pivots is selected
///
/// # Parameters
/// * `graph` - The input graph to select pivots from
/// * `length` - A function that maps edges to their lengths/weights
/// * `h` - The number of pivot nodes to select
/// * `rng` - The random number generator to use
///
/// # Returns
/// A tuple containing:
/// - A vector of selected pivot node IDs
/// - A SubDistanceMatrix containing distances from pivots to all nodes
fn max_min_random_sp<G, F, R, S>(
    graph: G,
    length: F,
    h: usize,
    rng: &mut R,
) -> (Vec<G::NodeId>, SubDistanceMatrix<G::NodeId, S>)
where
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
    G::NodeId: DrawingIndex + Ord,
    F: FnMut(G::EdgeRef) -> S,
    R: Rng,
    S: DrawingValue,
{
    let indices = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<_, _>>();
    let nodes = graph.node_identifiers().collect::<Vec<_>>();
    let mut length = length;
    let n = indices.len();
    let mut pivot = vec![];
    pivot.push(nodes[rng.gen_range(0..n)]);
    let mut distance_matrix = SubDistanceMatrix::empty(graph);
    distance_matrix.push(pivot[0]);
    dijkstra_with_distance_matrix(graph, &mut length, pivot[0], &mut distance_matrix);
    let mut min_d = Array1::from_elem(n, S::infinity());
    for k in 1..h {
        for j in 0..n {
            min_d[j] = min_d[j].min(distance_matrix.get_by_index(k - 1, j));
        }
        pivot.push(nodes[proportional_sampling(&min_d, rng)]);
        distance_matrix.push(pivot[k]);
        dijkstra_with_distance_matrix(graph, &mut length, pivot[k], &mut distance_matrix);
    }
    (pivot, distance_matrix)
}

/// Selects a node index with probability proportional to its distance value.
///
/// This function is used in the pivot selection process to randomly choose the
/// next pivot with probability proportional to its minimum distance from all
/// previously selected pivots. This ensures that nodes that are farther from
/// existing pivots have a higher chance of being selected as the next pivot.
///
/// # Parameters
/// * `values` - An array of distance values for each node
/// * `rng` - The random number generator to use
///
/// # Returns
/// The index of the selected node
///
/// # Panics
/// This function will panic if all values are zero or if an unexpected state is reached.
fn proportional_sampling<R, S>(values: &Array1<S>, rng: &mut R) -> usize
where
    R: Rng,
    S: DrawingValue,
{
    let n = values.len();
    let mut s = 0.;
    for i in 0..n {
        s += values[i].to_f32().unwrap();
    }
    if s == 0. {
        panic!("could not choice pivot");
    }
    let x = rng.gen_range(0.0..s);
    s = 0.;
    for i in 0..n {
        s += values[i].to_f32().unwrap();
        if x < s {
            return i;
        }
    }
    panic!("unreachable");
}
