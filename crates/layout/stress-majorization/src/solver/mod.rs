mod weights;

use crate::conjugate_gradient;
use ndarray::prelude::*;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount};
use petgraph_algorithm_shortest_path::{all_sources_dijkstra, DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{
    Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue, MetricEuclidean2d,
};

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
pub(super) fn stress<S: DrawingValue>(
    x: &Array1<S>,
    y: &Array1<S>,
    w: &Array2<S>,
    d: &Array2<S>,
) -> S {
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
    pub(crate) d: Array2<S>,
    pub(crate) w: Array2<S>,
    pub(crate) l_w: Array2<S>,
    l_z: Array2<S>,
    b: Array1<S>,
    pub(crate) stress: S,
    pub(crate) x_x: Array1<S>,
    pub(crate) x_y: Array1<S>,
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
}
