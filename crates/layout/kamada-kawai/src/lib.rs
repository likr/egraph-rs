//! # Kamada-Kawai Graph Layout Algorithm
//!
//! This crate provides an implementation of the Kamada-Kawai graph layout algorithm,
//! a force-directed layout method for drawing graphs in 2D space.
//!
//! The algorithm works by modeling the graph as a system of springs, where:
//! - Each pair of nodes is connected by a spring
//! - The ideal length of each spring is proportional to the shortest path distance between nodes
//! - The algorithm iteratively moves nodes to minimize the energy of the spring system
//!
//! ## References
//!
//! Kamada, T., & Kawai, S. (1989). An algorithm for drawing general undirected graphs.
//! Information Processing Letters, 31(1), 7-15.
//!
//! ## Example
//!
//! ```
//! use petgraph::prelude::*;
//! use petgraph_layout_kamada_kawai::KamadaKawai;
//! use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex};
//!
//! // Create an undirected graph
//! let mut graph = Graph::new_undirected();
//! let n1 = graph.add_node(());
//! let n2 = graph.add_node(());
//! let n3 = graph.add_node(());
//! graph.add_edge(n1, n2, ());
//! graph.add_edge(n2, n3, ());
//!
//! // Create initial placement
//! let mut drawing = DrawingEuclidean2d::<NodeIndex, f32>::initial_placement(&graph);
//!
//! // Create and run the Kamada-Kawai layout algorithm
//! let kamada_kawai = KamadaKawai::new(&graph, |_| 1.0);
//! kamada_kawai.run(&mut drawing);
//! ```
//!
use ndarray::prelude::*;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount};
use petgraph_algorithm_shortest_path::{all_sources_dijkstra, DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};

fn norm<S>(x: S, y: S) -> S
where
    S: DrawingValue,
{
    x.hypot(y).max(S::one())
}

/// Implementation of the Kamada-Kawai graph layout algorithm.
///
/// This struct stores the spring constants and ideal distances between nodes,
/// calculated based on the graph's topology, as well as the convergence threshold.
pub struct KamadaKawai<S> {
    /// Spring constants for each pair of nodes
    k: Array2<S>,
    /// Ideal distances between nodes
    l: Array2<S>,
    /// Convergence threshold
    pub eps: S,
}

impl<S> KamadaKawai<S> {
    /// Creates a new Kamada-Kawai layout algorithm instance from a graph.
    ///
    /// This constructor calculates the shortest path distances between all pairs of nodes
    /// in the graph, using the provided edge length function.
    ///
    /// # Arguments
    ///
    /// * `graph` - The input graph
    /// * `length` - A function that returns the length of each edge
    ///
    /// # Returns
    ///
    /// A new `KamadaKawai` instance
    pub fn new<G, F>(graph: G, length: F) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let l = all_sources_dijkstra(graph, length);
        KamadaKawai::new_with_distance_matrix(&l)
    }

    /// Creates a new Kamada-Kawai layout algorithm instance from a pre-computed distance matrix.
    ///
    /// This constructor uses an existing distance matrix rather than calculating
    /// it from the graph. This can be useful when the distances have already been
    /// computed or when using a custom distance metric.
    ///
    /// # Arguments
    ///
    /// * `d` - A full distance matrix containing shortest path distances between all pairs of nodes
    ///
    /// # Returns
    ///
    /// A new `KamadaKawai` instance
    pub fn new_with_distance_matrix<N>(d: &FullDistanceMatrix<N, S>) -> Self
    where
        N: DrawingIndex,
        S: DrawingValue,
    {
        let eps = S::from_f32(1e-1).unwrap();
        let n = d.shape().0;

        let mut l = Array2::zeros((n, n));
        let mut k = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                l[[i, j]] = d.get_by_index(i, j);
                k[[i, j]] = S::one() / (l[[i, j]] * l[[i, j]]);
            }
        }
        KamadaKawai { k, l, eps }
    }

    /// Selects the node with the maximum energy gradient to move next.
    ///
    /// This method calculates the energy gradient for each node and returns
    /// the index of the node with the maximum gradient magnitude, if it exceeds
    /// the convergence threshold. If all nodes have gradients below the threshold,
    /// it returns None, indicating the layout has converged.
    ///
    /// # Arguments
    ///
    /// * `drawing` - The current node positions
    ///
    /// # Returns
    ///
    /// The index of the selected node, or None if the layout has converged
    pub fn select_node<N>(&self, drawing: &DrawingEuclidean2d<N, S>) -> Option<usize>
    where
        N: DrawingIndex,
        S: DrawingValue,
    {
        let n = drawing.len();
        let KamadaKawai { k, l, eps, .. } = self;
        let mut delta2_max = S::zero();
        let mut m_target = 0;
        for m in 0..n {
            let xm = drawing.raw_entry(m).0;
            let ym = drawing.raw_entry(m).1;
            let mut dedx = S::zero();
            let mut dedy = S::zero();
            for i in 0..n {
                if i != m {
                    let xi = drawing.raw_entry(i).0;
                    let yi = drawing.raw_entry(i).1;
                    let dx = xm - xi;
                    let dy = ym - yi;
                    let d = norm(dx, dy);
                    dedx += k[[m, i]] * (S::one() - l[[m, i]] / d) * dx;
                    dedy += k[[m, i]] * (S::one() - l[[m, i]] / d) * dy;
                }
            }
            let delta2 = dedx * dedx + dedy * dedy;
            if delta2 > delta2_max {
                delta2_max = delta2;
                m_target = m;
            }
        }

        if delta2_max < *eps * *eps {
            None
        } else {
            Some(m_target)
        }
    }

    /// Moves a single node to reduce its energy.
    ///
    /// This method calculates the optimal position for the specified node
    /// using a second-order approximation of the energy function, and updates
    /// its position in the drawing.
    ///
    /// # Arguments
    ///
    /// * `m` - The index of the node to move
    /// * `drawing` - The current node positions, which will be updated
    pub fn apply_to_node<N>(&self, m: usize, drawing: &mut DrawingEuclidean2d<N, S>)
    where
        N: DrawingIndex,
        S: DrawingValue,
    {
        let n = drawing.len();
        let KamadaKawai { k, l, .. } = self;
        let xm = drawing.raw_entry(m).0;
        let ym = drawing.raw_entry(m).1;
        let mut hxx = S::zero();
        let mut hyy = S::zero();
        let mut hxy = S::zero();
        let mut dedx = S::zero();
        let mut dedy = S::zero();
        for i in 0..n {
            if i != m {
                let xi = drawing.raw_entry(i).0;
                let yi = drawing.raw_entry(i).1;
                let dx = xm - xi;
                let dy = ym - yi;
                let d = norm(dx, dy);
                let d3 = d * d * d;
                hxx += k[[m, i]] * (S::one() - l[[m, i]] * dy * dy / d3);
                hyy += k[[m, i]] * (S::one() - l[[m, i]] * dx * dx / d3);
                hxy += k[[m, i]] * l[[m, i]] * dx * dy / d3;
                dedx += k[[m, i]] * (S::one() - l[[m, i]] / d) * dx;
                dedy += k[[m, i]] * (S::one() - l[[m, i]] / d) * dy;
            }
        }
        let det = hxx * hyy - hxy * hxy;
        let delta_x = (hyy * dedx - hxy * dedy) / det;
        let delta_y = (hxx * dedy - hxy * dedx) / det;
        drawing.raw_entry_mut(m).0 -= delta_x;
        drawing.raw_entry_mut(m).1 -= delta_y;
    }

    /// Runs the Kamada-Kawai algorithm until convergence.
    ///
    /// This method repeatedly selects the node with the maximum energy gradient
    /// and moves it to reduce the energy, until the layout converges.
    ///
    /// # Arguments
    ///
    /// * `drawing` - The initial node positions, which will be updated to the final layout
    pub fn run<N>(&self, drawing: &mut DrawingEuclidean2d<N, S>)
    where
        N: DrawingIndex,
        S: DrawingValue,
    {
        while let Some(m) = self.select_node(drawing) {
            self.apply_to_node(m, drawing);
        }
    }
}

#[test]
fn test_kamada_kawai() {
    use petgraph::Graph;

    let n = 10;
    let mut graph = Graph::new_undirected();
    let nodes = (0..n).map(|_| graph.add_node(())).collect::<Vec<_>>();
    for i in 0..n {
        for j in 0..i {
            graph.add_edge(nodes[j], nodes[i], ());
        }
    }

    let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }

    let kamada_kawai = KamadaKawai::new(&graph, &mut |_| 1.);
    kamada_kawai.run(&mut coordinates);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }
}
