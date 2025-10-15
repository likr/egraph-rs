//! RdMds (Resistance-distance MDS) implementation for computing spectral embeddings.

use crate::eigenvalue::eigendecomposition;
use ndarray::{Array1, Array2};
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::Rng;

/// RdMds (Resistance-distance MDS) for computing spectral embeddings from graph Laplacians.
///
/// This structure computes d-dimensional spectral coordinates by finding the smallest
/// non-zero eigenvalues and eigenvectors of the graph Laplacian matrix.
#[derive(Debug, Clone)]
pub struct RdMds<S> {
    /// Number of spectral dimensions
    pub d: usize,
    /// Shift parameter for creating positive definite matrix L + cI
    pub shift: S,
    /// Maximum number of iterations for eigenvalue computation using inverse power method
    pub eigenvalue_max_iterations: usize,
    /// Maximum number of iterations for CG method
    pub cg_max_iterations: usize,
    /// Convergence tolerance for eigenvalue computation
    pub eigenvalue_tolerance: S,
    /// Convergence tolerance for CG method
    pub cg_tolerance: S,
}

impl<S> RdMds<S>
where
    S: DrawingValue,
{
    /// Creates a new RdMds with default values.
    ///
    /// Default values:
    /// - d: 2 (spectral dimensions)
    /// - shift: 1e-3 (shift parameter for positive definite matrix)
    /// - eigenvalue_max_iterations: 1000 (eigenvalue solver)
    /// - cg_max_iterations: 100 (CG solver)
    /// - eigenvalue_tolerance: 1e-1 (eigenvalue convergence)
    /// - cg_tolerance: 1e-4 (CG convergence)
    pub fn new() -> Self {
        Self {
            d: 2,
            shift: S::from_f32(1e-3).unwrap(),
            eigenvalue_max_iterations: 1000,
            cg_max_iterations: 100,
            eigenvalue_tolerance: S::from_f32(1e-1).unwrap(),
            cg_tolerance: S::from_f32(1e-4).unwrap(),
        }
    }

    /// Sets the number of spectral dimensions.
    pub fn d(&mut self, d: usize) -> &mut Self {
        self.d = d;
        self
    }

    /// Sets the shift parameter for creating positive definite matrix L + cI.
    pub fn shift(&mut self, shift: S) -> &mut Self {
        self.shift = shift;
        self
    }

    /// Sets maximum iterations for eigenvalue computation using inverse power method.
    pub fn eigenvalue_max_iterations(&mut self, eigenvalue_max_iterations: usize) -> &mut Self {
        self.eigenvalue_max_iterations = eigenvalue_max_iterations;
        self
    }

    /// Sets maximum iterations for CG method.
    pub fn cg_max_iterations(&mut self, cg_max_iterations: usize) -> &mut Self {
        self.cg_max_iterations = cg_max_iterations;
        self
    }

    /// Sets convergence tolerance for eigenvalue computation.
    pub fn eigenvalue_tolerance(&mut self, eigenvalue_tolerance: S) -> &mut Self {
        self.eigenvalue_tolerance = eigenvalue_tolerance;
        self
    }

    /// Sets convergence tolerance for CG method.
    pub fn cg_tolerance(&mut self, cg_tolerance: S) -> &mut Self {
        self.cg_tolerance = cg_tolerance;
        self
    }

    /// Computes spectral coordinates (embedding) using the configured parameters.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `rng` - Random number generator for spectral coordinate computation
    ///
    /// # Returns
    /// An Array2 where embedding.row(i) contains the d-dimensional coordinate for node i
    pub fn embedding<G, F, R>(&self, graph: G, length: F, rng: &mut R) -> Array2<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let (embedding, _eigenvalues) = self.eigendecomposition(graph, length, rng);
        embedding
    }

    /// Computes spectral coordinates and eigenvalues using the configured parameters.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    /// * `rng` - Random number generator for spectral coordinate computation
    ///
    /// # Returns
    /// A tuple containing:
    /// - Array2 where embedding.row(i) contains the d-dimensional coordinate for node i
    /// - Array1 of eigenvalues (λ_1, λ_2, ..., λ_d)
    pub fn eigendecomposition<G, F, R>(
        &self,
        graph: G,
        length: F,
        rng: &mut R,
    ) -> (Array2<S>, Array1<S>)
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        eigendecomposition(
            graph,
            length,
            self.shift,
            self.eigenvalue_max_iterations,
            self.cg_max_iterations,
            self.eigenvalue_tolerance,
            self.cg_tolerance,
            self.d,
            rng,
        )
    }
}

impl<S> Default for RdMds<S>
where
    S: DrawingValue,
{
    fn default() -> Self {
        Self::new()
    }
}
