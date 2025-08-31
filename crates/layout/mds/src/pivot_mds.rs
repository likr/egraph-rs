use crate::{double_centering::double_centering, eigendecomposition::eigendecomposition};
use ndarray::prelude::*;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
use petgraph_algorithm_shortest_path::{multi_source_dijkstra, DistanceMatrix};
use petgraph_drawing::{Drawing, DrawingEuclidean, DrawingEuclidean2d, DrawingIndex, DrawingValue};

/// Pivot-based Multidimensional Scaling (Pivot MDS) implementation.
///
/// Pivot MDS is a more efficient version of Classical MDS that reduces computational
/// complexity by using only a subset of nodes (pivots) for distance calculations.
/// This makes it suitable for larger graphs where Classical MDS would be too expensive.
///
/// The algorithm works by:
/// 1. Selecting a subset of nodes as pivots
/// 2. Computing distances from all nodes to these pivots
/// 3. Applying MDS on this reduced distance matrix
/// 4. Using a linear transformation to map all nodes into the resulting space
///
/// This approach reduces the complexity from O(n³) to approximately O(n·k²), where
/// k is the number of pivot nodes, making it much more scalable.
///
/// # Examples
///
/// ```
/// use petgraph::prelude::*;
/// use petgraph_layout_mds::PivotMds;
///
/// // Create a simple graph
/// let mut graph = Graph::new_undirected();
/// let n1 = graph.add_node(());
/// let n2 = graph.add_node(());
/// let n3 = graph.add_node(());
/// let n4 = graph.add_node(());
/// graph.add_edge(n1, n2, ());
/// graph.add_edge(n2, n3, ());
/// graph.add_edge(n3, n4, ());
///
/// // Use n1 and n3 as pivot nodes
/// let pivot_nodes = vec![n1, n3];
/// let mds = PivotMds::<NodeIndex>::new(&graph, |_| 1.0, &pivot_nodes);
///
/// // Run Pivot MDS to get a 2D layout
/// let drawing = mds.run_2d();
/// ```
pub struct PivotMds<N, S> {
    /// Convergence threshold for eigendecomposition.
    ///
    /// Lower values will result in more accurate layouts but may require more iterations.
    pub eps: S,

    /// Node indices of the graph.
    indices: Vec<N>,

    /// Double-centered distance matrix.
    c: Array2<S>,
}

impl<N, S> PivotMds<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    /// Creates a new Pivot MDS instance from a graph, edge length function, and pivot nodes.
    ///
    /// This method computes distances from all nodes to the specified pivot nodes,
    /// which are then used to initialize the MDS computation. The choice of pivot nodes
    /// can significantly affect the quality of the layout, so it's often beneficial
    /// to select nodes that are well-distributed throughout the graph.
    ///
    /// # Parameters
    ///
    /// * `graph`: A graph structure that can be traversed for edges and nodes
    /// * `length`: A function that calculates the length of each edge in the graph
    /// * `sources`: A slice of node IDs to use as pivot nodes
    ///
    /// # Type Parameters
    ///
    /// * `G`: Graph type that implements the required graph traversal traits
    /// * `F`: Function type for calculating edge lengths
    ///
    /// # Returns
    ///
    /// A new `PivotMds` instance ready to compute a layout
    pub fn new<G, F>(graph: G, length: F, sources: &[G::NodeId]) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Ord + Into<N>,
        F: FnMut(G::EdgeRef) -> S,
    {
        let distance_matrix = multi_source_dijkstra(graph, length, sources);
        Self::new_with_distance_matrix(&distance_matrix)
    }

    /// Creates a new Pivot MDS instance from a pre-computed distance matrix.
    ///
    /// This method is useful when distances from nodes to pivots have already been computed
    /// or when using a custom distance calculation method.
    ///
    /// # Parameters
    ///
    /// * `distance_matrix`: A matrix containing distances from all nodes to pivot nodes
    ///
    /// # Type Parameters
    ///
    /// * `N2`: Node index type of the distance matrix
    /// * `D`: Distance matrix type
    ///
    /// # Returns
    ///
    /// A new `PivotMds` instance ready to compute a layout
    pub fn new_with_distance_matrix<N2, D>(distance_matrix: &D) -> Self
    where
        N2: DrawingIndex + Copy + Into<N>,
        D: DistanceMatrix<N2, S>,
    {
        let (n, m) = distance_matrix.shape();
        let mut delta = Array2::zeros((m, n));
        for i in 0..n {
            for j in 0..m {
                delta[[j, i]] = distance_matrix.get_by_index(i, j).powi(2);
            }
        }
        let c = double_centering(&delta);
        Self {
            eps: (1e-3).into(),
            indices: distance_matrix
                .col_indices()
                .map(|u| u.into())
                .collect::<Vec<_>>(),
            c,
        }
    }

    /// Computes a 2D layout of the graph using Pivot MDS.
    ///
    /// This is a convenience method for the common case of creating a 2D visualization.
    /// It uses the top 2 eigenvectors to position nodes in a 2D space.
    ///
    /// The algorithm:
    /// 1. Computes the inner product matrix C^T·C
    /// 2. Performs eigendecomposition on this matrix
    /// 3. Maps the nodes to the resulting space using the centered distance matrix
    ///
    /// # Returns
    ///
    /// A `DrawingEuclidean2d` instance containing the 2D coordinates of all nodes.
    pub fn run_2d(&self) -> DrawingEuclidean2d<N, S>
    where
        N: Copy,
        S: Default,
    {
        let ct_c = self.c.t().dot(&self.c);
        let (mut e, v) = eigendecomposition(&ct_c, 2, self.eps);

        // Filter out negative or very small eigenvalues to avoid NaN values
        let epsilon = (1e-10).into();
        for i in 0..e.len() {
            if e[i] < epsilon {
                e[i] = S::zero();
            }
        }

        let xy = v.dot(&Array2::from_diag(&e.mapv(|v| v.max(S::zero()).sqrt())));
        let xy = self.c.dot(&xy);
        let mut drawing = DrawingEuclidean2d::from_node_indices(&self.indices);
        for (i, &u) in self.indices.iter().enumerate() {
            if let Some(p) = drawing.position_mut(u) {
                p.0 = xy[[i, 0]];
                p.1 = xy[[i, 1]];
            }
        }
        drawing
    }

    /// Computes a d-dimensional layout of the graph using Pivot MDS.
    ///
    /// This method allows creating layouts in arbitrary dimensions by using
    /// the top d eigenvectors to position nodes in a d-dimensional space.
    ///
    /// # Parameters
    ///
    /// * `d`: The number of dimensions for the resulting layout
    ///
    /// # Returns
    ///
    /// A `DrawingEuclidean` instance containing the d-dimensional coordinates of all nodes.
    pub fn run(&self, d: usize) -> DrawingEuclidean<N, S>
    where
        N: Copy,
        S: Default,
    {
        let ct_c = self.c.t().dot(&self.c);
        let (mut e, v) = eigendecomposition(&ct_c, d, self.eps);

        // Filter out negative or very small eigenvalues to avoid NaN values
        let epsilon = (1e-10).into();
        for i in 0..e.len() {
            if e[i] < epsilon {
                e[i] = S::zero();
            }
        }

        let x = v.dot(&Array2::from_diag(&e.mapv(|v| v.max(S::zero()).sqrt())));
        let x = self.c.dot(&x);
        let mut drawing = DrawingEuclidean::from_node_indices(&self.indices, d);
        for (i, &u) in self.indices.iter().enumerate() {
            if let Some(p) = drawing.position_mut(u) {
                for j in 0..d {
                    p.0[j] = x[[i, j]];
                }
            }
        }
        drawing
    }
}
