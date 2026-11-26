//! Continuous force-directed placement solvers (`ForceAtlas2` and multilevel `Yifan Hu`).

mod coarsening;
mod config;
mod force_atlas2;
mod force_atlas2_nd;
mod force_atlas2_step;
mod overlap_quadtree;
mod yifan_hu;
mod yifan_hu_nd;
mod yifan_hu_step;

#[cfg(test)]
mod tests;

pub use config::{ForceAtlas2Config, YifanHuConfig};
pub use force_atlas2::ForceAtlas2Solver;
pub use yifan_hu::YifanHuSolver;

use egraph_core::{
    CoreResult, CsrGraph, IterativeSolver, Layout2d, LayoutNd, Scalar, SolverStatus,
};

/// Unified continuous force-directed placement solver wrapping [`ForceAtlas2Solver`] and [`YifanHuSolver`].
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ContinuousFdp<S: Scalar = f64> {
    /// Continuous ForceAtlas2 solver.
    ForceAtlas2(ForceAtlas2Solver<S>),
    /// Multilevel Yifan Hu solver.
    YifanHu(YifanHuSolver<S>),
}

impl<S: Scalar> ContinuousFdp<S> {
    /// Creates a [`ContinuousFdp::ForceAtlas2`] solver from `graph` and `config`.
    pub fn force_atlas2<W: Scalar>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
    ) -> CoreResult<Self> {
        Ok(Self::ForceAtlas2(ForceAtlas2Solver::new(graph, config)?))
    }

    /// Creates a [`ContinuousFdp::YifanHu`] solver from `graph` and `config`.
    pub fn yifan_hu<W: Scalar>(graph: &CsrGraph<W>, config: YifanHuConfig) -> CoreResult<Self> {
        Ok(Self::YifanHu(YifanHuSolver::new(graph, config)?))
    }

    /// Returns the expected number of vertices in [`Layout2d`] or [`LayoutNd`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        match self {
            Self::ForceAtlas2(solver) => solver.num_nodes(),
            Self::YifanHu(solver) => solver.num_nodes(),
        }
    }

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::ForceAtlas2(solver) => solver.status(),
            Self::YifanHu(solver) => solver.status(),
        }
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        match self {
            Self::ForceAtlas2(solver) => solver.iteration(),
            Self::YifanHu(solver) => solver.iteration(),
        }
    }

    /// Returns the scheduled maximum number of iterations.
    #[inline]
    pub fn total_iterations(&self) -> usize {
        match self {
            Self::ForceAtlas2(solver) => solver.total_iterations(),
            Self::YifanHu(solver) => solver.total_iterations(),
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for ContinuousFdp<S> {
    #[inline]
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        match self {
            Self::ForceAtlas2(solver) => solver.step(layout),
            Self::YifanHu(solver) => solver.step(layout),
        }
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status()
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iteration()
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.total_iterations()
    }
}

impl<S: Scalar> IterativeSolver<S, LayoutNd<S>> for ContinuousFdp<S> {
    #[inline]
    fn step(&mut self, layout: &mut LayoutNd<S>) -> CoreResult<SolverStatus> {
        match self {
            Self::ForceAtlas2(solver) => solver.step(layout),
            Self::YifanHu(solver) => solver.step(layout),
        }
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status()
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iteration()
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.total_iterations()
    }
}
