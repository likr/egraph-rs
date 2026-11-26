//! True Noack (2007/2009) LinLog and generalized $(a, r)$-PolyLog energy layout solver.
//!
//! Implements Noack's $(a, r)$ energy model (`noack2007_linlog`, `noack2009_modularity_layout`):
//! $$U_{(a, r)}(p) = \sum_{\{u,v\} \in E} w_{\{u,v\}} \frac{\|p_u - p_v\|^{a+1}}{a+1}
//!     - \sum_{\{u,v\} \subset V} m_u m_v \frac{\|p_u - p_v\|^{r+1}}{r+1}
//!     + U_{\text{grav}}(p)$$
//! with $a > r$ (where exponent $-1$ yields $\ln \|p_u - p_v\|$), supporting both
//! [`LinLogRepulsion::EdgeRepulsion`] ($m_u = \deg_w(u)$) and
//! [`LinLogRepulsion::NodeRepulsion`] ($m_u = 1$), mass-barycenter gravity, and
//! $O(N \log N)$ Barnes-Hut acceleration via [`ArenaQuadtree2d`].

mod config;
mod quadtree;
mod solver;
mod step;

#[cfg(test)]
mod tests;

#[cfg(doc)]
use egraph_spatial_trees::ArenaQuadtree2d;

pub use config::{LinLogConfig, LinLogRepulsion};
pub use solver::LinLogSolver;

pub(super) const ISOLATED_NODE_MASS: f64 = 1e-3;
