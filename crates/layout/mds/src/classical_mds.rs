use crate::{double_centering::double_centering, eigendecomposition::eigendecomposition};
use ndarray::prelude::*;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
use petgraph_algorithm_shortest_path::{all_sources_dijkstra, DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Drawing, DrawingEuclidean, DrawingEuclidean2d, DrawingIndex};

/// Classical Multidimensional Scaling (CMDS) implementation.
///
/// This struct implements the standard MDS algorithm that works by:
/// 1. Computing a full distance matrix between all pairs of nodes
/// 2. Applying double centering to convert distances to dot products
/// 3. Computing eigenvectors of the resulting matrix
/// 4. Using the top eigenvectors to place nodes in a lower-dimensional space
///
/// While this approach provides optimal results, it requires O(n²) memory and
/// has O(n³) time complexity, making it suitable only for smaller graphs.
///
/// # Examples
///
/// ```
/// use petgraph::prelude::*;
/// use petgraph_layout_mds::ClassicalMds;
///
/// // Create a simple graph
/// let mut graph = Graph::new_undirected();
/// let n1 = graph.add_node(());
/// let n2 = graph.add_node(());
/// let n3 = graph.add_node(());
/// graph.add_edge(n1, n2, ());
/// graph.add_edge(n2, n3, ());
///
/// // Create a ClassicalMds instance
/// let mds = ClassicalMds::<NodeIndex>::new(&graph, |_| 1.0);
///
/// // Run MDS to get a 2D layout
/// let drawing = mds.run_2d();
/// ```
pub struct ClassicalMds<N> {
    /// Convergence threshold for eigendecomposition.
    ///
    /// Lower values will result in more accurate layouts but may require more iterations.
    pub eps: f32,

    /// Node indices of the graph.
    indices: Vec<N>,

    /// Double-centered distance matrix.
    b: Array2<f32>,
}

impl<N> ClassicalMds<N>
where
    N: DrawingIndex,
{
    /// Creates a new Classical MDS instance from a graph and an edge length function.
    ///
    /// This method computes the all-pairs shortest paths in the graph using Dijkstra's algorithm
    /// to create a distance matrix, which is then used to initialize the MDS computation.
    ///
    /// # Parameters
    ///
    /// * `graph`: A graph structure that can be traversed for edges and nodes
    /// * `length`: A function that calculates the length of each edge in the graph
    ///
    /// # Type Parameters
    ///
    /// * `G`: Graph type that implements the required graph traversal traits
    /// * `F`: Function type for calculating edge lengths
    ///
    /// # Returns
    ///
    /// A new `ClassicalMds` instance ready to compute a layout
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

    /// Creates a new Classical MDS instance from a pre-computed distance matrix.
    ///
    /// This method is useful when the distance matrix has already been computed
    /// or when using a custom distance calculation method.
    ///
    /// # Parameters
    ///
    /// * `distance_matrix`: A matrix containing distances between all pairs of nodes
    ///
    /// # Type Parameters
    ///
    /// * `N2`: Node index type of the distance matrix
    ///
    /// # Returns
    ///
    /// A new `ClassicalMds` instance ready to compute a layout
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

    /// Computes a 2D layout of the graph using Classical MDS.
    ///
    /// This is a convenience method for the common case of creating a 2D visualization.
    /// It uses the top 2 eigenvectors of the centered distance matrix to place nodes.
    ///
    /// # Returns
    ///
    /// A `DrawingEuclidean2d` instance containing the 2D coordinates of all nodes.
    pub fn run_2d(&self) -> DrawingEuclidean2d<N, f32>
    where
        N: Copy,
    {
        let (e, v) = eigendecomposition(&self.b, 2, self.eps);
        let xy = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
        let mut drawing = DrawingEuclidean2d::from_node_indices(&self.indices);
        for (i, &u) in self.indices.iter().enumerate() {
            if let Some(p) = drawing.position_mut(u) {
                p.0 = xy[[i, 0]];
                p.1 = xy[[i, 1]];
            }
        }
        drawing
    }

    /// Computes a d-dimensional layout of the graph using Classical MDS.
    ///
    /// This method allows creating layouts in arbitrary dimensions by using
    /// the top d eigenvectors of the centered distance matrix.
    ///
    /// # Parameters
    ///
    /// * `d`: The number of dimensions for the resulting layout
    ///
    /// # Returns
    ///
    /// A `DrawingEuclidean` instance containing the d-dimensional coordinates of all nodes.
    pub fn run(&self, d: usize) -> DrawingEuclidean<N, f32>
    where
        N: Copy,
    {
        let (e, v) = eigendecomposition(&self.b, d, self.eps);
        let x = v.dot(&Array2::from_diag(&e.mapv(|v| v.sqrt())));
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
