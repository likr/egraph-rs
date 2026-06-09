//! RdMds (Resistance-distance MDS) implementation for computing spectral embeddings.

use crate::eigenvalue::eigendecomposition;
use ndarray::{Array1, Array2};
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_distance::{Laplacian, StandardLaplacian};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::Rng;

/// RdMds (Resistance-distance MDS) for computing spectral embeddings from graph Laplacians.
#[derive(Debug, Clone)]
pub struct RdMds<S, L = StandardLaplacian> {
    /// Number of spectral dimensions
    pub d: usize,
    /// Shift parameter for creating positive definite matrix L + cI
    pub shift: S,
    /// Maximum number of iterations for eigenvalue computation
    pub eigenvalue_max_iterations: usize,
    /// Maximum number of iterations for CG method
    pub cg_max_iterations: usize,
    /// Convergence tolerance for eigenvalue computation
    pub eigenvalue_tolerance: S,
    /// Convergence tolerance for CG method
    pub cg_tolerance: S,
    /// The builder to construct the Laplacian matrix
    pub laplacian_builder: L,
}

impl<S> RdMds<S, StandardLaplacian>
where
    S: DrawingValue + Default,
{
    /// Creates a new RdMds with default values.
    pub fn new() -> Self {
        Self {
            d: 2,
            shift: S::from_f32(1e-3).unwrap(),
            eigenvalue_max_iterations: 1000,
            cg_max_iterations: 100,
            eigenvalue_tolerance: S::from_f32(1e-1).unwrap(),
            cg_tolerance: S::from_f32(1e-4).unwrap(),
            laplacian_builder: StandardLaplacian,
        }
    }
}

impl<S, L> RdMds<S, L> {
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

    /// Sets maximum iterations for eigenvalue computation.
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

    /// Configures a custom Laplacian builder.
    pub fn laplacian_builder<L2>(self, laplacian_builder: L2) -> RdMds<S, L2> {
        RdMds {
            d: self.d,
            shift: self.shift,
            eigenvalue_max_iterations: self.eigenvalue_max_iterations,
            cg_max_iterations: self.cg_max_iterations,
            eigenvalue_tolerance: self.eigenvalue_tolerance,
            cg_tolerance: self.cg_tolerance,
            laplacian_builder,
        }
    }
}

impl<S, L> RdMds<S, L>
where
    S: DrawingValue + Default,
{
    /// Computes spectral coordinates (embedding) using the configured parameters.
    pub fn embedding<G, F, R>(&self, graph: G, length: F, rng: &mut R) -> Array2<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        L: Laplacian<G, S> + Copy,
    {
        let (embedding, _eigenvalues) = self.eigendecomposition(graph, length, rng);
        embedding
    }

    /// Computes spectral coordinates and eigenvalues using the configured parameters.
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
        L: Laplacian<G, S> + Copy,
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
            self.laplacian_builder,
            rng,
        )
    }
}

impl<S> Default for RdMds<S, StandardLaplacian>
where
    S: DrawingValue + Default,
{
    fn default() -> Self {
        Self::new()
    }
}
