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
use petgraph_drawing::{
    Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue, MetricEuclidean2d,
};

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
fn line_search<S: DrawingValue>(a: &Array2<S>, dx: &Array1<S>, d: &Array1<S>) -> S {
    let n = dx.len();
    let mut alpha = -d.dot(dx);
    let mut s = S::zero();
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
fn delta_f<S: DrawingValue>(a: &Array2<S>, b: &Array1<S>, x: &Array1<S>, dx: &mut Array1<S>) {
    let n = b.len();
    for i in 0..n {
        dx[i] = S::zero();
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
pub fn conjugate_gradient<S: DrawingValue>(
    a: &Array2<S>,
    b: &Array1<S>,
    x: &mut Array1<S>,
    epsilon: S,
) {
    let n = b.len();
    if n == 0 {
        return;
    }
    let mut dx = Array1::zeros(n);
    let mut d = Array1::zeros(n);
    delta_f(a, b, x, &mut dx);
    for i in 0..n {
        d[i] = -dx[i];
    }
    let mut dx_norm0 = dx.dot(&dx);
    // Exit early before line_search if the residual is already zero/converged (avoids 0/0 division on singular/exact inputs).
    if !dx_norm0.is_finite() || dx_norm0 < epsilon || dx_norm0 == S::zero() {
        return;
    }
    for _ in 0..n {
        let alpha = line_search(a, &dx, &d);
        if !alpha.is_finite() || (0..n).any(|i| !(x[i] + alpha * d[i]).is_finite()) {
            break;
        }
        for i in 0..n {
            x[i] += alpha * d[i];
        }
        delta_f(a, b, x, &mut dx);
        let dx_norm = dx.dot(&dx);
        if !dx_norm.is_finite() || dx_norm < epsilon || dx_norm0 == S::zero() {
            break;
        }
        let beta = dx_norm / dx_norm0;
        if !beta.is_finite() {
            break;
        }
        dx_norm0 = dx_norm;
        for i in 0..n {
            d[i] = beta * d[i] - dx[i];
        }
    }
}

/// Validates that all coordinates in the drawing are finite.
fn validate_finite_drawing<N: DrawingIndex, S: DrawingValue>(
    drawing: &DrawingEuclidean2d<N, S>,
) -> Result<(), String> {
    for i in 0..drawing.len() {
        let entry = drawing.raw_entry(i);
        if !entry.0.is_finite() || !entry.1.is_finite() {
            return Err(format!(
                "StressMajorization: drawing coordinate at node index {} must be finite",
                i
            ));
        }
    }
    Ok(())
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
fn stress<S: DrawingValue>(x: &Array1<S>, y: &Array1<S>, w: &Array2<S>, d: &Array2<S>) -> S {
    let n = w.nrows();
    if n <= 1 {
        return S::zero();
    }
    let mut s = S::zero();
    for j in 1..n - 1 {
        for i in 0..j {
            let dij = d[[i, j]];
            let wij = w[[i, j]];
            if !dij.is_finite() || !wij.is_finite() || wij == S::zero() {
                continue;
            }
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let norm = (dx * dx + dy * dy).sqrt();
            let e = norm - dij;
            s += wij * e * e;
        }
    }
    for i in 0..n - 1 {
        let j = n - 1;
        let dij = d[[i, j]];
        let wij = w[[i, j]];
        if !dij.is_finite() || !wij.is_finite() || wij == S::zero() {
            continue;
        }
        let dx = x[i];
        let dy = y[i];
        let norm = (dx * dx + dy * dy).sqrt();
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
pub struct StressMajorization<S> {
    d: Array2<S>,
    w: Array2<S>,
    l_w: Array2<S>,
    l_z: Array2<S>,
    b: Array1<S>,
    stress: S,
    x_x: Array1<S>,
    x_y: Array1<S>,
    pub epsilon: S,
    pub max_iterations: usize,
}

impl<S> StressMajorization<S>
where
    S: DrawingValue,
{
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
    pub fn new<G, F>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, S>, length: F) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
    {
        Self::try_new(graph, drawing, length).unwrap_or_else(|e| panic!("{}", e))
    }

    /// Fallible variant of [`StressMajorization::new`] validating node count and finite coordinates.
    pub fn try_new<G, F>(
        graph: G,
        drawing: &DrawingEuclidean2d<G::NodeId, S>,
        length: F,
    ) -> Result<Self, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
    {
        let graph_n = graph.node_count();
        let drawing_n = drawing.len();
        if graph_n != drawing_n {
            return Err(format!(
                "StressMajorization::try_new: graph node count ({}) does not match drawing length ({})",
                graph_n, drawing_n
            ));
        }
        validate_finite_drawing(drawing)?;
        let d = all_sources_dijkstra(graph, length);
        Self::try_new_with_distance_matrix(drawing, &d)
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
        drawing: &DrawingEuclidean2d<N, S>,
        distance_matrix: &FullDistanceMatrix<N, S>,
    ) -> Self
    where
        N: DrawingIndex,
    {
        Self::try_new_with_distance_matrix(drawing, distance_matrix)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Fallible variant of [`StressMajorization::new_with_distance_matrix`] validating shapes,
    /// finite drawing coordinates, and handling `n <= 1` without `0usize - 1` underflow.
    pub fn try_new_with_distance_matrix<N>(
        drawing: &DrawingEuclidean2d<N, S>,
        distance_matrix: &FullDistanceMatrix<N, S>,
    ) -> Result<Self, String>
    where
        N: DrawingIndex,
    {
        let n = drawing.len();
        let (rows, cols) = distance_matrix.shape();
        if rows != cols {
            return Err(format!(
                "StressMajorization::try_new_with_distance_matrix: expected square distance matrix, got shape ({}, {})",
                rows, cols
            ));
        }
        if rows != n {
            return Err(format!(
                "StressMajorization::try_new_with_distance_matrix: drawing length ({}) does not match distance matrix size ({})",
                n, rows
            ));
        }
        validate_finite_drawing(drawing)?;

        let epsilon = S::from_f64(1e-4).unwrap();
        let max_iterations = 100; // Default value
        if n == 0 {
            return Ok(StressMajorization {
                b: Array1::zeros(0),
                d: Array2::zeros((0, 0)),
                l_w: Array2::zeros((0, 0)),
                l_z: Array2::zeros((0, 0)),
                w: Array2::zeros((0, 0)),
                x_x: Array1::zeros(0),
                x_y: Array1::zeros(0),
                stress: S::zero(),
                epsilon,
                max_iterations,
            });
        }

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
            stress: S::infinity(),
            epsilon,
            max_iterations,
        };
        sm.update_weight(|_, _, dij, _| {
            if dij.is_finite() && dij > S::zero() {
                let w = S::one() / (dij * dij);
                if w.is_finite() {
                    w
                } else {
                    S::zero()
                }
            } else {
                S::zero()
            }
        });
        Ok(sm)
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
    pub fn apply<N>(&mut self, drawing: &mut DrawingEuclidean2d<N, S>) -> S
    where
        N: DrawingIndex,
    {
        self.try_apply(drawing).unwrap_or_else(|e| panic!("{}", e))
    }

    /// Fallible variant of [`StressMajorization::apply`] validating drawing length and finite coordinates.
    pub fn try_apply<N>(&mut self, drawing: &mut DrawingEuclidean2d<N, S>) -> Result<S, String>
    where
        N: DrawingIndex,
    {
        let n = self.n();
        if drawing.len() != n {
            return Err(format!(
                "StressMajorization::try_apply: drawing length ({}) does not match instance node count ({})",
                drawing.len(),
                n
            ));
        }
        validate_finite_drawing(drawing)?;
        if n <= 1 {
            return Ok(S::zero());
        }

        let StressMajorization {
            b, d, l_w, l_z, w, ..
        } = self;
        let MetricEuclidean2d(ref_x, ref_y) = *drawing.raw_entry(n - 1);
        for i in 0..n {
            drawing.raw_entry_mut(i).0 -= ref_x;
            drawing.raw_entry_mut(i).1 -= ref_y;
        }
        let min_norm = S::from_f64(1e-4).unwrap();
        for i in 1..n - 1 {
            for j in 0..i {
                let wij = w[[i, j]];
                let dij = d[[i, j]];
                let lij = if !wij.is_finite() || !dij.is_finite() || wij == S::zero() {
                    S::zero()
                } else {
                    let dx = drawing.raw_entry(i).0 - drawing.raw_entry(j).0;
                    let dy = drawing.raw_entry(i).1 - drawing.raw_entry(j).1;
                    let norm = (dx * dx + dy * dy).sqrt();
                    if norm < min_norm {
                        S::zero()
                    } else {
                        -wij * dij / norm
                    }
                };
                l_z[[i, j]] = lij;
                l_z[[j, i]] = lij;
            }
        }
        for i in 0..n - 1 {
            let mut s = S::zero();
            for j in 0..n - 1 {
                if i != j {
                    s -= l_z[[i, j]];
                }
            }
            let j = n - 1;
            let wij = w[[i, j]];
            let dij = d[[i, j]];
            if wij.is_finite() && dij.is_finite() && wij != S::zero() {
                let dx = drawing.raw_entry(i).0;
                let dy = drawing.raw_entry(i).1;
                let norm = (dx * dx + dy * dy).sqrt();
                if norm >= min_norm {
                    s -= -wij * dij / norm;
                }
            }
            l_z[[i, i]] = s;
        }

        for i in 0..n - 1 {
            self.x_x[i] = drawing.raw_entry(i).0;
            let mut s = S::zero();
            for j in 0..n - 1 {
                s += l_z[[i, j]] * drawing.raw_entry(j).0;
            }
            b[i] = s;
        }
        conjugate_gradient(l_w, b, &mut self.x_x, self.epsilon);

        for i in 0..n - 1 {
            self.x_y[i] = drawing.raw_entry(i).1;
            let mut s = S::zero();
            for j in 0..n - 1 {
                s += l_z[[i, j]] * drawing.raw_entry(j).1;
            }
            b[i] = s;
        }
        conjugate_gradient(l_w, b, &mut self.x_y, self.epsilon);

        let new_stress = stress(&self.x_x, &self.x_y, w, d);
        let diff = if self.stress.is_finite() && self.stress > S::zero() && new_stress.is_finite() {
            (self.stress - new_stress) / self.stress
        } else {
            S::zero()
        };
        self.stress = new_stress;
        for i in 0..n - 1 {
            drawing.raw_entry_mut(i).0 = self.x_x[i];
            drawing.raw_entry_mut(i).1 = self.x_y[i];
        }
        Ok(diff)
    }

    /// Runs the stress majorization algorithm until convergence.
    ///
    /// This function repeatedly applies the algorithm until the relative
    /// change in stress falls below the threshold specified by `epsilon`
    /// or the maximum number of iterations is reached.
    ///
    /// # Arguments
    ///
    /// * `coordinates` - The current node positions, which will be updated
    pub fn run<N>(&mut self, coordinates: &mut DrawingEuclidean2d<N, S>)
    where
        N: DrawingIndex,
    {
        self.try_run(coordinates)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Fallible variant of [`StressMajorization::run`] validating drawing length and finite coordinates.
    pub fn try_run<N>(&mut self, coordinates: &mut DrawingEuclidean2d<N, S>) -> Result<(), String>
    where
        N: DrawingIndex,
    {
        let n = self.n();
        if coordinates.len() != n {
            return Err(format!(
                "StressMajorization::try_run: drawing length ({}) does not match instance node count ({})",
                coordinates.len(),
                n
            ));
        }
        validate_finite_drawing(coordinates)?;
        if n <= 1 {
            return Ok(());
        }
        for _ in 0..self.max_iterations {
            if self.try_apply(coordinates)? < self.epsilon {
                break;
            }
        }
        Ok(())
    }

    /// Returns the number of nodes in this stress majorization instance.
    pub fn n(&self) -> usize {
        self.d.nrows()
    }

    fn rebuild_laplacian_and_stress(&mut self) {
        let n = self.n();
        if n <= 1 {
            self.stress = S::zero();
            return;
        }
        for i in 0..n - 1 {
            self.l_w[[i, i]] = S::zero();
        }
        for j in 1..n - 1 {
            for i in 0..j {
                let raw = self.w[[i, j]];
                let wij = if raw.is_finite() { raw } else { S::zero() };
                self.l_w[[i, j]] = -wij;
                self.l_w[[j, i]] = -wij;
                self.l_w[[i, i]] += wij;
                self.l_w[[j, j]] += wij;
            }
        }
        for i in 0..n - 1 {
            let j = n - 1;
            let raw = self.w[[i, j]];
            let wij = if raw.is_finite() { raw } else { S::zero() };
            self.l_w[[i, i]] += wij;
        }
        self.stress = stress(&self.x_x, &self.x_y, &self.w, &self.d);
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
        F: FnMut(usize, usize, S, S) -> S,
    {
        let _ = self.try_update_weight::<std::convert::Infallible, _>(|i, j, dij, wij| {
            Ok(weight(i, j, dij, wij))
        });
    }

    /// Fallible variant of [`StressMajorization::update_weight`] propagating callback errors
    /// without partial updates.
    pub fn try_update_weight<E, F>(&mut self, mut weight: F) -> Result<(), E>
    where
        F: FnMut(usize, usize, S, S) -> Result<S, E>,
    {
        let n = self.n();
        if n <= 1 {
            self.rebuild_laplacian_and_stress();
            return Ok(());
        }

        let mut updates = Vec::with_capacity(n * (n - 1) / 2);
        for j in 1..n {
            for i in 0..j {
                let wij = weight(i, j, self.d[[i, j]], self.w[[i, j]])?;
                updates.push((i, j, wij));
            }
        }
        for (i, j, wij) in updates {
            self.w[[i, j]] = wij;
            self.w[[j, i]] = wij;
        }

        self.rebuild_laplacian_and_stress();
        Ok(())
    }

    /// Fallible variant of [`StressMajorization::update_weight_from_array`] validating shape
    /// and finite non-negative weights without partial updates.
    pub fn try_update_weight_from_array(
        &mut self,
        w: ndarray::ArrayView2<S>,
    ) -> Result<(), String> {
        let n = self.n();
        if w.dim() != (n, n) {
            return Err(format!(
                "StressMajorization::update_weight_from_array: expected shape ({}, {}), got {:?}",
                n,
                n,
                w.dim()
            ));
        }
        for j in 1..n {
            for i in 0..j {
                let wij = w[[i, j]];
                if !wij.is_finite() || wij < S::zero() {
                    return Err(format!(
                        "StressMajorization::update_weight_from_array: weight at ({}, {}) must be finite and non-negative",
                        i, j
                    ));
                }
            }
        }
        for j in 1..n {
            for i in 0..j {
                let wij = w[[i, j]];
                self.w[[i, j]] = wij;
                self.w[[j, i]] = wij;
            }
        }
        self.rebuild_laplacian_and_stress();
        Ok(())
    }

    /// Updates the weights from an `(n, n)` array and rebuilds the reduced Laplacian `l_w`.
    ///
    /// # Panics
    ///
    /// Panics if `w.dim() != (n, n)` or any off-diagonal weight is non-finite or negative.
    pub fn update_weight_from_array(&mut self, w: ndarray::ArrayView2<S>) {
        self.try_update_weight_from_array(w)
            .unwrap_or_else(|e| panic!("{}", e));
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

#[test]
fn test_stress_majorization_parameters() {
    use petgraph::Graph;

    // Create a simple graph
    let mut graph = Graph::new_undirected();
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());
    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());

    let coordinates = DrawingEuclidean2d::initial_placement(&graph);

    // Default parameters
    let mut stress_majorization = StressMajorization::new(&graph, &coordinates, &mut |_| 1.0);

    // Check default values
    assert_eq!(stress_majorization.epsilon, 1e-4);
    assert_eq!(stress_majorization.max_iterations, 100);

    // Update parameters
    stress_majorization.epsilon = 1e-6;
    stress_majorization.max_iterations = 200;

    // Check updated values
    assert_eq!(stress_majorization.epsilon, 1e-6);
    assert_eq!(stress_majorization.max_iterations, 200);
}

#[test]
fn test_stress_majorization_update_weight_from_array() {
    use petgraph::Graph;

    let mut graph = Graph::new_undirected();
    let n0 = graph.add_node(());
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());
    graph.add_edge(n0, n1, ());
    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n0, ());

    let mut coords_fn = DrawingEuclidean2d::initial_placement(&graph);
    let mut coords_arr = coords_fn.clone();

    let mut sm_fn = StressMajorization::new(&graph, &coords_fn, |_| 1.0f64);
    let mut sm_arr = StressMajorization::new(&graph, &coords_arr, |_| 1.0f64);

    let weights = arr2(&[
        [0.0, 2.0, 0.5, 1.5],
        [2.0, 0.0, 3.0, 0.25],
        [0.5, 3.0, 0.0, 4.0],
        [1.5, 0.25, 4.0, 0.0],
    ]);

    sm_fn.update_weight(|i, j, _, _| weights[[i, j]]);
    sm_arr.update_weight_from_array(weights.view());

    assert_eq!(sm_fn.w, sm_arr.w);
    assert_eq!(sm_fn.l_w, sm_arr.l_w);
    assert!((sm_fn.stress - sm_arr.stress).abs() < 1e-12);

    let diff_fn = sm_fn.apply(&mut coords_fn);
    let diff_arr = sm_arr.apply(&mut coords_arr);
    assert!((diff_fn - diff_arr).abs() < 1e-12);

    for u in [n0, n1, n2, n3] {
        assert!((coords_fn.x(u).unwrap() - coords_arr.x(u).unwrap()).abs() < 1e-12);
        assert!((coords_fn.y(u).unwrap() - coords_arr.y(u).unwrap()).abs() < 1e-12);
    }
}

#[test]
fn test_stress_majorization_empty_and_singleton_graphs() {
    use petgraph::graph::UnGraph;

    // n == 0: no 0usize - 1 underflow, n() == 0, apply/run/update_weight are clean no-ops.
    let g0: UnGraph<(), ()> = UnGraph::new_undirected();
    let mut d0: DrawingEuclidean2d<_, f64> = DrawingEuclidean2d::initial_placement(&g0);
    let dm0 = FullDistanceMatrix::from_array(&g0, Array2::zeros((0, 0)));
    let mut sm0 = StressMajorization::try_new(&g0, &d0, |_| 1.0).unwrap();
    assert_eq!(sm0.n(), 0);
    assert_eq!(sm0.try_apply(&mut d0).unwrap(), 0.0);
    sm0.try_run(&mut d0).unwrap();
    sm0.try_update_weight::<&str, _>(|_, _, _, w| Ok(w))
        .unwrap();
    let mut sm0_dm = StressMajorization::try_new_with_distance_matrix(&d0, &dm0).unwrap();
    assert_eq!(sm0_dm.n(), 0);
    sm0_dm.try_run(&mut d0).unwrap();

    // n == 1: singleton graph works cleanly with n() == 1.
    let mut g1: UnGraph<(), ()> = UnGraph::new_undirected();
    let u0 = g1.add_node(());
    let mut d1: DrawingEuclidean2d<_, f64> = DrawingEuclidean2d::initial_placement(&g1);
    let orig_pos = (d1.x(u0).unwrap(), d1.y(u0).unwrap());
    let dm1 = FullDistanceMatrix::from_array(&g1, Array2::zeros((1, 1)));
    let mut sm1 = StressMajorization::try_new(&g1, &d1, |_| 1.0).unwrap();
    assert_eq!(sm1.n(), 1);
    assert_eq!(sm1.try_apply(&mut d1).unwrap(), 0.0);
    sm1.try_run(&mut d1).unwrap();
    assert_eq!((d1.x(u0).unwrap(), d1.y(u0).unwrap()), orig_pos);
    let mut sm1_dm = StressMajorization::try_new_with_distance_matrix(&d1, &dm1).unwrap();
    assert_eq!(sm1_dm.n(), 1);
    sm1_dm.try_run(&mut d1).unwrap();
}

#[test]
fn test_stress_majorization_fallible_validation_and_atomic_update() {
    use petgraph::graph::UnGraph;

    let mut g2: UnGraph<(), ()> = UnGraph::new_undirected();
    let a = g2.add_node(());
    let b = g2.add_node(());
    g2.add_edge(a, b, ());
    let mut d2: DrawingEuclidean2d<_, f64> = DrawingEuclidean2d::initial_placement(&g2);
    let dm2 = FullDistanceMatrix::from_array(&g2, arr2(&[[0.0, 1.0], [1.0, 0.0]]));

    let mut g3: UnGraph<(), ()> = UnGraph::new_undirected();
    let n0 = g3.add_node(());
    let n1 = g3.add_node(());
    let n2 = g3.add_node(());
    g3.add_edge(n0, n1, ());
    g3.add_edge(n1, n2, ());
    let mut d3: DrawingEuclidean2d<_, f64> = DrawingEuclidean2d::initial_placement(&g3);

    // Mismatched graph/drawing/matrix sizes return Err without panicking.
    assert!(StressMajorization::try_new(&g3, &d2, |_| 1.0).is_err());
    assert!(StressMajorization::try_new_with_distance_matrix(&d3, &dm2).is_err());

    let mut sm3 = StressMajorization::try_new(&g3, &d3, |_| 1.0).unwrap();
    assert!(sm3.try_apply(&mut d2).is_err());
    assert!(sm3.try_run(&mut d2).is_err());

    // Non-finite drawing coordinates are rejected.
    let mut d3_nan = d3.clone();
    d3_nan.raw_entry_mut(0).0 = f64::NAN;
    assert!(StressMajorization::try_new(&g3, &d3_nan, |_| 1.0).is_err());
    assert!(sm3.try_apply(&mut d3_nan).is_err());
    assert!(sm3.try_run(&mut d3_nan).is_err());

    // try_update_weight leaves weights unchanged when callback errors partway through.
    let before_w = sm3.w.clone();
    let mut calls = 0usize;
    let err = sm3.try_update_weight(|_, _, _, _| {
        calls += 1;
        if calls == 2 {
            Err("boom")
        } else {
            Ok(42.0)
        }
    });
    assert_eq!(err, Err("boom"));
    assert_eq!(sm3.w, before_w);

    // try_update_weight_from_array rejects bad shape, NaN, and negative weights atomically.
    assert!(sm3
        .try_update_weight_from_array(Array2::zeros((2, 2)).view())
        .is_err());
    assert!(sm3
        .try_update_weight_from_array(
            arr2(&[[0.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]).view()
        )
        .is_err());
    assert!(sm3
        .try_update_weight_from_array(
            arr2(&[[0.0, f64::NAN, 1.0], [f64::NAN, 0.0, 1.0], [1.0, 1.0, 0.0]]).view()
        )
        .is_err());
    assert_eq!(sm3.w, before_w);

    // Disconnected graph remains finite across apply/run (no 0.0 * inf = NaN).
    let mut g_disc: UnGraph<(), ()> = UnGraph::new_undirected();
    let c0 = g_disc.add_node(());
    let c1 = g_disc.add_node(());
    let c2 = g_disc.add_node(());
    let c3 = g_disc.add_node(());
    g_disc.add_edge(c0, c1, ());
    g_disc.add_edge(c2, c3, ());
    let mut d_disc: DrawingEuclidean2d<_, f64> = DrawingEuclidean2d::initial_placement(&g_disc);
    let mut sm_disc = StressMajorization::try_new(&g_disc, &d_disc, |_| 1.0).unwrap();
    let diff = sm_disc.try_apply(&mut d_disc).unwrap();
    assert!(diff.is_finite());
    sm_disc.try_run(&mut d_disc).unwrap();
    for u in [c0, c1, c2, c3] {
        assert!(d_disc.x(u).unwrap().is_finite());
        assert!(d_disc.y(u).unwrap().is_finite());
    }
    let _ = &mut d3;
}
