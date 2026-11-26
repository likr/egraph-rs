//! GEM (Graph Embedder) adaptive local-temperature force-directed layout solver.
//!
//! Implements Frick, Ludwig & Mehldau (1995), "A Fast Adaptive Layout Algorithm for
//! Undirected Graphs (Extended Abstract and System Demonstration)" (*GD '94*, LNCS 894,
//! 388–403):
//! - Sequential randomized vertex updates per round (`DeterministicRng::shuffle`).
//! - Barycenter gravity scaled by vertex degree inertia $\Phi(v) = 1 + \deg(v)/2$.
//! - Random symmetry-breaking impulse perturbation in $[-\text{perturbation}, \text{perturbation}]^2$.
//! - Inverse-square repulsion and weighted cubic-spring attraction.
//! - Per-vertex local temperature $v.t$, oscillation detection ($\alpha_o, \sigma_o$),
//!   and rotation skew-gauge damping ($\alpha_r, \sigma_r$) with non-negativity clamps
//!   (`1 - |v.d| >= 0`, `1 + sigma_o * cos_b >= 0`).

mod config;
mod solver;

#[cfg(test)]
mod tests;

pub use config::GemConfig;
pub use solver::GemSolver;
