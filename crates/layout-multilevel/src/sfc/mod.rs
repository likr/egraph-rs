//! Canonical community-ordered Space-Filling Curve (`SFC`) graph layout baseline.
//!
//! # Literature Note (`muelder2008_space_filling` `MISSING_PDF`)
//!
//! Muelder & Ma (*"Rapid Graph Layout Using Space Filling Curves"*, IEEE TVCG 2008) is currently
//! marked `MISSING_PDF` in the project literature archive, so exact clustering, ordering, and
//! post-refinement parameters from that paper cannot be verified or claimed. This module
//! implements a canonical **community-ordered Hilbert/Morton space-filling curve baseline**
//! powered by [`egraph_clustering::louvain_coarsen`] and strictly injective 2D curve mapping,
//! guaranteeing zero vertex co-location in both `f64` and `f32`.

mod config;
mod ordering;
#[cfg(test)]
mod tests;

pub use config::{SfcConfig, SfcCurveKind};

use config::MAX_SFC_ORDER;
use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use ordering::{canonical_undirected_graph, compute_community_vertex_order};

/// Cooperative community-ordered Space-Filling Curve (`SFC`) layout solver.
///
/// Precomputes a hierarchical Louvain community vertex ordering and maps ranks injectively onto
/// a $2^k \times 2^k$ Hilbert or Morton grid, writing the resulting coordinates in a single
/// [`IterativeSolver::step`] (`total_iterations() == 1`).
#[derive(Debug, Clone)]
pub struct SfcLayoutSolver<S: Scalar = f64> {
    config: SfcConfig,
    num_nodes: usize,
    vertex_order: Vec<u32>,
    coords_x: Vec<S>,
    coords_y: Vec<S>,
    iter: usize,
    status: SolverStatus,
}

impl<S: Scalar> SfcLayoutSolver<S> {
    /// Constructs an [`SfcLayoutSolver`] for `graph` using [`SfcConfig`].
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: SfcConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr(
                "SfcLayoutSolver node count exceeds u32::MAX",
            ));
        }
        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite("SfcLayoutSolver::graph::weights"));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(
                        "SfcLayoutSolver requires non-negative edge weights",
                    ));
                }
            }
        }

        let order = match config.resolution_order {
            Some(ord) => {
                let total_cells = 1u64 << (2 * ord);
                if (n as u64) > total_cells {
                    return Err(CoreError::invalid_input(format!(
                        "SfcConfig::resolution_order={ord} provides {total_cells} cells, insufficient for {n} vertices without co-location"
                    )));
                }
                ord
            }
            None => auto_order(n)?,
        };

        if n == 0 {
            return Ok(Self {
                config,
                num_nodes: 0,
                vertex_order: Vec::new(),
                coords_x: Vec::new(),
                coords_y: Vec::new(),
                iter: 0,
                status: SolverStatus::Initialized,
            });
        }
        if n == 1 {
            return Ok(Self {
                config,
                num_nodes: 1,
                vertex_order: vec![0],
                coords_x: vec![S::ZERO],
                coords_y: vec![S::ZERO],
                iter: 0,
                status: SolverStatus::Initialized,
            });
        }

        let undirected = canonical_undirected_graph(graph)?;
        let vertex_order = if undirected.num_edges() == 0 {
            (0..n as u32).collect()
        } else {
            compute_community_vertex_order(&undirected, config.seed)?
        };

        let side = 1u64 << order;
        let total_cells = side * side;
        let side_f = side as f64;
        let mut coords_x = vec![S::ZERO; n];
        let mut coords_y = vec![S::ZERO; n];

        for (rank, &v_u32) in vertex_order.iter().enumerate() {
            // Since total_cells >= n, d(rank) = floor(rank * total_cells / n) is strictly
            // increasing in rank, guaranteeing unique grid cells and zero co-location.
            let d = ((rank as u128 * (total_cells as u128)) / (n as u128)) as u64;
            let (ix, iy) = match config.curve {
                SfcCurveKind::Hilbert => hilbert_d2xy(order, d),
                SfcCurveKind::Morton => morton_d2xy(order, d),
            };
            let xf = (((ix as f64) + 0.5) / side_f - 0.5) * config.scale;
            let yf = (((iy as f64) + 0.5) / side_f - 0.5) * config.scale;
            let idx = v_u32 as usize;
            coords_x[idx] = S::from_f64(xf);
            coords_y[idx] = S::from_f64(yf);
        }

        Ok(Self {
            config,
            num_nodes: n,
            vertex_order,
            coords_x,
            coords_y,
            iter: 0,
            status: SolverStatus::Initialized,
        })
    }

    /// Returns the community-ordered permutation of vertex indices mapped along the curve.
    #[inline]
    pub fn vertex_order(&self) -> &[u32] {
        &self.vertex_order
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &SfcConfig {
        &self.config
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }
}

impl<S: Scalar> IterativeSolver<S> for SfcLayoutSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SfcLayoutSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_nodes > 1 {
            let (xs_mut, ys_mut) = layout.xy_mut();
            xs_mut.copy_from_slice(&self.coords_x);
            ys_mut.copy_from_slice(&self.coords_y);
        }

        self.iter = 1;
        self.status = SolverStatus::Converged;
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        1
    }
}

fn auto_order(n: usize) -> CoreResult<u32> {
    let target = (n.max(1)) as u64;
    for ord in 1..=MAX_SFC_ORDER {
        if (1u64 << (2 * ord)) >= target {
            return Ok(ord);
        }
    }
    Err(CoreError::invalid_input(format!(
        "graph node count {n} exceeds maximum SFC grid capacity 2^{}",
        2 * MAX_SFC_ORDER
    )))
}

#[inline]
fn hilbert_d2xy(order: u32, mut d: u64) -> (u64, u64) {
    let n = 1u64 << order;
    let mut x = 0u64;
    let mut y = 0u64;
    let mut s = 1u64;
    while s < n {
        let rx = (d >> 1) & 1;
        let ry = (d ^ rx) & 1;
        if ry == 0 {
            if rx == 1 {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        x += s * rx;
        y += s * ry;
        d >>= 2;
        s <<= 1;
    }
    (x, y)
}

#[inline]
fn morton_d2xy(order: u32, d: u64) -> (u64, u64) {
    let mut x = 0u64;
    let mut y = 0u64;
    for bit in 0..order {
        x |= ((d >> (2 * bit)) & 1) << bit;
        y |= ((d >> (2 * bit + 1)) & 1) << bit;
    }
    (x, y)
}
