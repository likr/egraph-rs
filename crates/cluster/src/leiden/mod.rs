//! Pure-Rust Leiden community detection (Traag, Waltman, & van Eck, 2019, Algorithm A.2)
//! supporting both Modularity and Constant Potts Model (CPM) objectives over flat CSR
//! super-graphs and `VisitStamp` accumulators (zero `HashMap` in inner loops or aggregation).

mod config;
mod level_graph;
mod local_move;
mod refine;
mod solver;

#[cfg(test)]
mod tests;

pub use config::{LeidenConfig, LeidenObjective, LeidenResult};
pub use solver::leiden;
