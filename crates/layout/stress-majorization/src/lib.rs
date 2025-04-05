//! Stress Majorization implementation for graph layout.
//!
//! This crate provides an implementation of the Stress Majorization algorithm,
//! a force-directed graph layout technique that minimizes a stress function.
//! The stress function measures the difference between the Euclidean distances
//! in the layout and the desired or theoretical distances (typically shortest path
//! distances in the graph).
//!
//! # Algorithm
//!
//! Stress Majorization works by iteratively solving a sequence of quadratic problems
//! that approximate the stress function. Each iteration improves the layout by
//! moving nodes to positions that reduce the overall stress.
//!
//! The algorithm implemented here is based on:
//!
//! Gansner, E. R., Koren, Y., & North, S. (2004). Graph drawing by stress
//! majorization. In International Symposium on Graph Drawing (pp. 239-250).
//!
//! # Usage
//!
//! ```
//! use petgraph::prelude::*;
//! use petgraph_drawing::DrawingEuclidean2d;
//! use petgraph_layout_stress_majorization::StressMajorization;
//!
//! // Create a graph
//! let mut graph = Graph::new_undirected();
//! let n1 = graph.add_node(());
//! let n2 = graph.add_node(());
//! let n3 = graph.add_node(());
//! graph.add_edge(n1, n2, ());
//! graph.add_edge(n2, n3, ());
//!
//! // Create initial placement
//! let mut drawing = DrawingEuclidean2d::initial_placement(&graph);
//!
//! // Run stress majorization
//! let mut sm = StressMajorization::new(&graph, &drawing, |_| 1.0);
//! sm.run(&mut drawing);
//! ```

use ndarray::prelude::*;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount};
use petgraph_algorithm_shortest_path::{all_sources_dijkstra, DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex};

/// Computes the optimal step length (alpha) in the conjugate gradient method.
///
/// The line search finds the value of alpha that minimizes the function value
/// when moving in the direction d.
///
/// # Arguments
///
/// * `a` - The coefficient matrix
/// * `dx` - The gradient vector
/// * `d` - The search direction
///
/// # Returns
///
/// The optimal step length alpha
fn line_search(a: &Array2<f32>, dx: &Array1<f32>, d: &Array1<f32>) -> f32 {
    let n = dx.len();
    let mut alpha = -d.dot(dx);
    let mut s = 0.;
    for i in 0..n {
        for j in 0..n {
            s += d[i] * d[j] * a[[i, j]];
        }
    }
    alpha /= s;
    alpha
}

/// Computes the gradient (delta_f) of the quadratic function f(x) = (1/2)x^T A x - b^T x.
///
/// # Arguments
///
/// * `a` - The coefficient matrix A
/// * `b` - The vector b
/// * `x` - The current position x
/// * `dx` - Output parameter where the gradient will be stored
fn delta_f(a: &Array2<f32>, b: &Array1<f32>, x: &Array1<f32>, dx: &mut Array1<f32>) {
    let n = b.len();
    for i in 0..n {
        dx[i] = 0.;
        for j in 0..n {
            dx[i] += a[[i, j]] * x[j];
        }
        dx[i] -= b[i];
    }
}

/// Solves a system of linear equations Ax = b using the conjugate gradient method.
///
/// The conjugate gradient method is an iterative algorithm for solving systems
/// of the form Ax = b where A is a symmetric positive-definite matrix.
///
/// # Arguments
///
/// * `a` - The coefficient matrix A
/// * `b` - The right-hand side vector b
/// * `x` - The initial guess for x, which will be updated with the solution
/// * `epsilon` - The convergence threshold (algorithm stops when the residual norm is less than this value)
pub fn conjugate_gradient(a: &Array2<f32>, b: &Array1<f32>, x: &mut Array1<f32>, epsilon: f32) {
    let n = b.len();
    let mut dx = Array1::zeros(n);
    let mut d = Array1::zeros(n);
    delta_f(a, b, x, &mut dx);
    for i in 0..n {
        d[i] = -dx[i];
    }
    let mut dx_norm0 = dx.dot(&dx);
    for _ in 0..n {
        let alpha = line_search(a, &dx, &d);
        for i in 0..n {
            x[i] += alpha * d[i];
        }
        delta_f(a, b, x, &mut dx);
        let dx_norm = dx.dot(&dx);
        if dx_norm < epsilon {
            break;
        }
        let beta = dx_norm / dx_norm0;
        dx_norm0 = dx_norm;
        for i in 0..n {
            d[i] = beta * d[i] - dx[i];
        }
    }
}

/// Computes the stress value for the current layout.
///
/// The stress is defined as the weighted sum of squared differences between
/// the Euclidean distances in the layout and the desired distances.
///
/// # Arguments
///
/// * `x` - The x-coordinates of nodes
/// * `y` - The y-coordinates of nodes
/// * `w` - The weight matrix
/// * `d` - The desired distance matrix
///
/// # Returns
///
/// The stress value
fn stress(x: &Array1<f32>, y: &Array1<f32>, w: &Array2<f32>, d: &Array2<f32>) -> f32 {
    let n = x.len() + 1;
    let mut s = 0.;
    for j in 1..n - 1 {
        for i in 0..j {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let norm = (dx * dx + dy * dy).sqrt();
            let dij = d[[i, j]];
            let wij = w[[i, j]];
            let e = norm - dij;
            s += wij * e * e;
        }
    }
    for i in 0..n - 1 {
        let j = n - 1;
        let dx = x[i];
        let dy = y[i];
        let norm = (dx * dx + dy * dy).sqrt();
        let dij = d[[i, j]];
        let wij = w[[i, j]];
        let e = norm - dij;
        s += wij * e * e;
    }
    s
}

/// An implementation of the Stress Majorization algorithm for graph layout.
///
/// Stress Majorization is a force-directed technique that iteratively minimizes
/// a stress function by solving a series of simpler quadratic problems.
/// This implementation supports 2D layouts with weighted edges.
pub struct StressMajorization {
    d: Array2<f32>,
    w: Array2<f32>,
    l_w: Array2<f32>,
    l_z: Array2<f32>,
    b: Array1<f32>,
    stress: f32,
    x_x: Array1<f32>,
    x_y: Array1<f32>,
    epsilon: f32,
}

impl StressMajorization {
    /// Creates a new `StressMajorization` instance from a graph and an initial drawing.
    ///
    /// # Arguments
    ///
    /// * `graph` - The input graph
    /// * `drawing` - The initial node positions
    /// * `length` - A function that returns the desired length for each edge
    ///
    /// # Returns
    ///
    /// A new `StressMajorization` instance
    pub fn new<G, F>(
        graph: G,
        drawing: &DrawingEuclidean2d<G::NodeId, f32>,
        length: F,
    ) -> StressMajorization
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> f32,
    {
        let d = all_sources_dijkstra(graph, length);
        StressMajorization::new_with_distance_matrix(drawing, &d)
    }

    /// Creates a new `StressMajorization` instance using a pre-computed distance matrix.
    ///
    /// This is useful when you already have shortest path distances available,
    /// or want to use custom distances instead of computing them from the graph.
    ///
    /// # Arguments
    ///
    /// * `drawing` - The initial node positions
    /// * `distance_matrix` - The pre-computed distance matrix
    ///
    /// # Returns
    ///
    /// A new `StressMajorization` instance
    pub fn new_with_distance_matrix<N>(
        drawing: &DrawingEuclidean2d<N, f32>,
        distance_matrix: &FullDistanceMatrix<N, f32>,
    ) -> StressMajorization
    where
        N: DrawingIndex,
    {
        let n = drawing.len();
        let mut d = Array2::zeros((n, n));
        let w = Array2::zeros((n, n));
        let l_w = Array2::zeros((n - 1, n - 1));
        let mut x_x = Array1::zeros(n - 1);
        let mut x_y = Array1::zeros(n - 1);
        for i in 0..n - 1 {
            x_x[i] = drawing.raw_entry(i).0 - drawing.raw_entry(n - 1).0;
            x_y[i] = drawing.raw_entry(i).1 - drawing.raw_entry(n - 1).1;
        }
        for i in 0..n {
            for j in 0..n {
                d[[i, j]] = distance_matrix.get_by_index(i, j);
            }
        }

        let epsilon = 1e-4;
        let l_z = Array2::zeros((n - 1, n - 1));
        let b = Array1::zeros(n - 1);
        let mut sm = StressMajorization {
            b,
            d,
            l_w,
            l_z,
            w,
            x_x,
            x_y,
            stress: f32::INFINITY,
            epsilon,
        };
        sm.update_weight(|_, _, dij, _| 1. / (dij * dij));
        sm
    }

    /// Performs a single iteration of the stress majorization algorithm.
    ///
    /// This function updates the node positions in the drawing to reduce the stress.
    ///
    /// # Arguments
    ///
    /// * `drawing` - The current node positions, which will be updated
    ///
    /// # Returns
    ///
    /// The relative change in stress (as a fraction of the previous stress value)
    pub fn apply<N>(&mut self, drawing: &mut DrawingEuclidean2d<N, f32>) -> f32
    where
        N: DrawingIndex,
    {
        let n = drawing.len();
        let StressMajorization {
            b, d, l_w, l_z, w, ..
        } = self;
        for i in 0..n {
            drawing.raw_entry_mut(i).0 -= drawing.raw_entry(n - 1).0;
            drawing.raw_entry_mut(i).1 -= drawing.raw_entry(n - 1).1;
        }
        for i in 1..n - 1 {
            for j in 0..i {
                let dx = drawing.raw_entry(i).0 - drawing.raw_entry(j).0;
                let dy = drawing.raw_entry(i).1 - drawing.raw_entry(j).1;
                let norm = (dx * dx + dy * dy).sqrt();
                let lij = if norm < 1e-4 {
                    0.
                } else {
                    -w[[i, j]] * d[[i, j]] / norm
                };
                l_z[[i, j]] = lij;
                l_z[[j, i]] = lij;
            }
        }
        for i in 0..n - 1 {
            let mut s = 0.;
            for j in 0..n - 1 {
                if i != j {
                    s -= l_z[[i, j]];
                }
            }
            let j = n - 1;
            let dx = drawing.raw_entry(i).0;
            let dy = drawing.raw_entry(i).1;
            let norm = (dx * dx + dy * dy).sqrt();
            s -= if norm < 1e-4 {
                0.
            } else {
                -w[[i, j]] * d[[i, j]] / norm
            };
            l_z[[i, i]] = s;
        }

        for i in 0..n - 1 {
            self.x_x[i] = drawing.raw_entry(i).0;
            let mut s = 0.;
            for j in 0..n - 1 {
                s += l_z[[i, j]] * drawing.raw_entry(j).0;
            }
            b[i] = s;
        }
        conjugate_gradient(l_w, b, &mut self.x_x, self.epsilon);

        for i in 0..n - 1 {
            self.x_y[i] = drawing.raw_entry(i).1;
            let mut s = 0.;
            for j in 0..n - 1 {
                s += l_z[[i, j]] * drawing.raw_entry(j).1;
            }
            b[i] = s;
        }
        conjugate_gradient(l_w, b, &mut self.x_y, self.epsilon);

        let stress = stress(&self.x_x, &self.x_y, w, d);
        let diff = (self.stress - stress) / self.stress;
        self.stress = stress;
        for i in 0..n - 1 {
            drawing.raw_entry_mut(i).0 = self.x_x[i];
            drawing.raw_entry_mut(i).1 = self.x_y[i];
        }
        diff
    }

    /// Runs the stress majorization algorithm until convergence.
    ///
    /// This function repeatedly applies the algorithm until the relative
    /// change in stress falls below the threshold specified by `epsilon`.
    ///
    /// # Arguments
    ///
    /// * `coordinates` - The current node positions, which will be updated
    pub fn run<N>(&mut self, coordinates: &mut DrawingEuclidean2d<N, f32>)
    where
        N: DrawingIndex,
    {
        loop {
            if self.apply(coordinates) < self.epsilon {
                break;
            }
        }
    }

    /// Updates the weights used in the stress calculation.
    ///
    /// This allows customizing how different pairs of nodes contribute to the overall
    /// stress. By default, weights are set to 1/(d_ij^2) where d_ij is the desired distance.
    ///
    /// # Arguments
    ///
    /// * `weight` - A function that takes (i, j, d_ij, current_w_ij) and returns the new weight
    pub fn update_weight<F>(&mut self, mut weight: F)
    where
        F: FnMut(usize, usize, f32, f32) -> f32,
    {
        let n = self.x_x.len() + 1;

        for j in 1..n {
            for i in 0..j {
                let wij = weight(i, j, self.d[[i, j]], self.w[[i, j]]);
                self.w[[i, j]] = wij;
                self.w[[j, i]] = wij;
            }
        }

        for i in 0..n - 1 {
            self.l_w[[i, i]] = 0.;
        }
        for j in 1..n - 1 {
            for i in 0..j {
                let wij = self.w[[i, j]];
                self.l_w[[i, j]] = -wij;
                self.l_w[[j, i]] = -wij;
                self.l_w[[i, i]] += wij;
                self.l_w[[j, j]] += wij;
            }
        }
        for i in 0..n - 1 {
            let j = n - 1;
            self.l_w[[i, i]] += self.w[[i, j]];
        }
        self.stress = stress(&self.x_x, &self.x_y, &self.w, &self.d);
    }
}

#[test]
fn test_conjugate_gradient() {
    let a = arr2(&[[3., 1.], [1., 2.]]);
    let b = arr1(&[6., 7.]);
    let mut x = arr1(&[2., 1.]);
    let epsilon = 1e-4;
    conjugate_gradient(&a, &b, &mut x, epsilon);
    let x_exact = [1., 3.];
    let mut d = 0.;
    for i in 0..x.len() {
        let dx = x[i] - x_exact[i];
        d += dx * dx;
    }
    assert!(d < epsilon);
}

#[test]
fn test_stress_majorization() {
    use petgraph::Graph;

    let n = 10;
    let mut graph = Graph::new_undirected();
    let nodes = (0..n).map(|_| graph.add_node(())).collect::<Vec<_>>();
    for j in 1..n {
        for i in 0..j {
            graph.add_edge(nodes[i], nodes[j], ());
        }
    }
    let mut coordinates = DrawingEuclidean2d::initial_placement(&graph);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }

    let mut stress_majorization = StressMajorization::new(&graph, &coordinates, &mut |_| 1.);
    stress_majorization.run(&mut coordinates);

    for &u in &nodes {
        println!("{:?}", coordinates.position(u));
    }
}
