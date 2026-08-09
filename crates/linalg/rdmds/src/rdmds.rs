//! RdMds (Resistance-distance MDS) implementation for computing spectral embeddings.

use crate::eigenvalue::{
    eigendecomposition, eigendecomposition_random_walk_normalized,
    eigendecomposition_symmetric_normalized, EigendecompositionResult,
};
use crate::solvers::LinearSolverBuilder;
use ndarray::{Array1, Array2};
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_distance::Laplacian;
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::Rng;

/// RdMds (Resistance-distance MDS) for computing spectral embeddings from graph Laplacians.
#[derive(Debug, Clone)]
pub struct RdMds<S> {
    /// Number of spectral dimensions
    pub d: usize,
    /// Shift parameter for creating positive definite matrix L + cI
    pub shift: S,
    /// Maximum number of iterations for eigenvalue computation
    pub eigenvalue_max_iterations: usize,
    /// Convergence tolerance for eigenvalue computation
    pub eigenvalue_tolerance: S,
}

impl<S> RdMds<S>
where
    S: DrawingValue + Default,
{
    /// Creates a new RdMds with default values.
    pub fn new() -> Self {
        Self {
            d: 2,
            shift: S::from_f32(1e-3).unwrap(),
            eigenvalue_max_iterations: 1000,
            eigenvalue_tolerance: S::from_f32(1e-1).unwrap(),
        }
    }
}

impl<S> RdMds<S> {
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

    /// Sets convergence tolerance for eigenvalue computation.
    pub fn eigenvalue_tolerance(&mut self, eigenvalue_tolerance: S) -> &mut Self {
        self.eigenvalue_tolerance = eigenvalue_tolerance;
        self
    }
}

impl<S> RdMds<S>
where
    S: DrawingValue + Default,
{
    /// Computes spectral coordinates (embedding) using the configured parameters for Standard Laplacian.
    pub fn embedding<G, F, R, Builder>(&self, graph: G, length: F, builder: &Builder, rng: &mut R) -> Array2<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        Builder: LinearSolverBuilder<S>,
    {
        let mut result = self.eigendecomposition(graph, length, builder, rng);
        let d = result.eigenvalues.len();
        for dim in 0..d {
            let mut eigenvector = result.eigenvectors.column_mut(dim);
            eigenvector /= result.eigenvalues[dim].max(S::zero()).sqrt();
        }
        result.eigenvectors
    }

    /// Computes spectral coordinates (embedding) for Symmetric Normalized Laplacian.
    pub fn embedding_symmetric_normalized<G, F, R, Builder>(&self, graph: G, length: F, builder: &Builder, rng: &mut R) -> Array2<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        Builder: LinearSolverBuilder<S>,
    {
        let mut result = self.eigendecomposition_symmetric_normalized(graph, length, builder, rng);
        let d = result.eigenvalues.len();
        for dim in 0..d {
            let mut eigenvector = result.eigenvectors.column_mut(dim);
            eigenvector /= result.eigenvalues[dim].max(S::zero()).sqrt();
        }
        result.eigenvectors
    }

    /// Computes spectral coordinates (embedding) for Random Walk Normalized Laplacian.
    pub fn embedding_random_walk_normalized<G, F, R, Builder>(&self, graph: G, length: F, builder: &Builder, rng: &mut R) -> Array2<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        Builder: LinearSolverBuilder<S>,
    {
        let mut result = self.eigendecomposition_random_walk_normalized(graph, length, builder, rng);
        let d = result.eigenvalues.len();
        for dim in 0..d {
            let mut eigenvector = result.eigenvectors.column_mut(dim);
            eigenvector /= result.eigenvalues[dim].max(S::zero()).sqrt();
        }
        result.eigenvectors
    }

    /// Computes spectral coordinates and eigenvalues for Standard Laplacian.
    pub fn eigendecomposition<G, F, R, Builder>(
        &self,
        graph: G,
        mut length: F,
        builder: &Builder,
        rng: &mut R,
    ) -> EigendecompositionResult<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        Builder: LinearSolverBuilder<S>,
    {
        let laplacian = petgraph_distance::StandardLaplacian
            .build(graph, &mut length)
            .scale_and_shift(S::one(), -self.shift);
        
        let solver = builder.build(laplacian);
        
        eigendecomposition(
            self.shift,
            self.eigenvalue_max_iterations,
            self.eigenvalue_tolerance,
            self.d,
            &solver,
            rng,
        )
    }

    /// Computes spectral coordinates and eigenvalues for Symmetric Normalized Laplacian.
    pub fn eigendecomposition_symmetric_normalized<G, F, R, Builder>(
        &self,
        graph: G,
        mut length: F,
        builder: &Builder,
        rng: &mut R,
    ) -> EigendecompositionResult<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        Builder: LinearSolverBuilder<S>,
    {
        let laplacian = petgraph_distance::SymmetricNormalizedLaplacian
            .build(graph, &mut length)
            .scale_and_shift(S::one(), -self.shift);
            
        let solver = builder.build(laplacian);
        
        eigendecomposition_symmetric_normalized(
            self.shift,
            self.eigenvalue_max_iterations,
            self.eigenvalue_tolerance,
            self.d,
            &solver,
            rng,
        )
    }

    /// Computes spectral coordinates and eigenvalues for Random Walk Normalized Laplacian.
    pub fn eigendecomposition_random_walk_normalized<G, F, R, Builder>(
        &self,
        graph: G,
        mut length: F,
        builder: &Builder,
        rng: &mut R,
    ) -> EigendecompositionResult<S>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        Builder: LinearSolverBuilder<S>,
    {
        let laplacian = petgraph_distance::StandardLaplacian
            .build(graph, &mut length)
            .scale_and_shift(S::one(), -self.shift);
            
        let solver = builder.build(laplacian);
        
        eigendecomposition_random_walk_normalized(
            self.shift,
            self.eigenvalue_max_iterations,
            self.eigenvalue_tolerance,
            self.d,
            &solver,
            rng,
        )
    }
}

impl<S> Default for RdMds<S>
where
    S: DrawingValue + Default,
{
    fn default() -> Self {
        Self::new()
    }
}
