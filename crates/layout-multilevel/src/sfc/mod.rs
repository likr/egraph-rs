//! Canonical community-ordered Space-Filling Curve (`SFC`) graph layout baseline.
//!
//! Inspired by Muelder & Ma (*"Rapid Graph Layout Using Space Filling Curves"*, *IEEE TVCG* 2008),
//! this module implements a community-ordered Hilbert/Morton proxy baseline powered by
//! [`egraph_clustering::louvain_coarsen`] and strictly injective 2D curve mapping, guaranteeing
//! zero vertex co-location in both `f64` and `f32`.

mod config;
mod ordering;
#[cfg(test)]
mod tests;

pub use config::{SfcConfig, SfcCurveKind};

use std::marker::PhantomData;

use config::MAX_SFC_ORDER;
use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, LayoutGenerator, Scalar};
use ordering::{canonical_undirected_graph, compute_community_vertex_order};

/// One-shot community-ordered Space-Filling Curve (`SFC`) layout generator implementing
/// [`LayoutGenerator`] with `Output = Layout2d<S>`.
///
/// Computes a hierarchical Louvain community vertex ordering and maps ranks injectively onto
/// a $2^k \times 2^k$ Hilbert or Morton grid, returning the resulting [`Layout2d`] directly
/// from [`LayoutGenerator::generate`].
#[non_exhaustive]
#[derive(Debug)]
pub struct SfcLayout<'a, S: Scalar = f64, W = f64> {
    graph: &'a CsrGraph<W>,
    config: SfcConfig,
    vertex_order: Vec<u32>,
    _scalar: PhantomData<fn() -> S>,
}

impl<S: Scalar, W> Clone for SfcLayout<'_, S, W> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph,
            config: self.config.clone(),
            vertex_order: self.vertex_order.clone(),
            _scalar: PhantomData,
        }
    }
}

impl<'a, S: Scalar, W: Scalar> SfcLayout<'a, S, W> {
    /// Creates an [`SfcLayout`] generator with default [`SfcConfig`] on `graph`.
    pub fn new(graph: &'a CsrGraph<W>) -> Self {
        Self {
            graph,
            config: SfcConfig::default(),
            vertex_order: Vec::new(),
            _scalar: PhantomData,
        }
    }

    /// Creates an [`SfcLayout`] generator on `graph` with an explicit [`SfcConfig`].
    pub fn from_config(graph: &'a CsrGraph<W>, config: SfcConfig) -> Self {
        Self {
            graph,
            config,
            vertex_order: Vec::new(),
            _scalar: PhantomData,
        }
    }
}

impl<S: Scalar, W> SfcLayout<'_, S, W> {
    /// Replaces the configuration on this generator.
    #[inline]
    pub fn with_config(mut self, config: SfcConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the space-filling curve family `curve`.
    #[inline]
    pub fn with_curve(mut self, curve: SfcCurveKind) -> Self {
        self.config.curve = curve;
        self
    }

    /// Sets the target bounding square side length `scale`.
    #[inline]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.config.scale = scale;
        self
    }

    /// Sets the optional explicit curve resolution order `resolution_order`.
    #[inline]
    pub fn with_resolution_order(mut self, resolution_order: Option<u32>) -> Self {
        self.config.resolution_order = resolution_order;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Returns the community-ordered permutation of vertex indices computed by the most recent
    /// [`LayoutGenerator::generate`] call (empty before generation).
    #[inline]
    pub fn vertex_order(&self) -> &[u32] {
        &self.vertex_order
    }

    /// Returns the generator configuration.
    #[inline]
    pub fn config(&self) -> &SfcConfig {
        &self.config
    }

    /// Returns the number of vertices in the source graph.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }
}

impl<S: Scalar, W: Scalar> LayoutGenerator for SfcLayout<'_, S, W> {
    type Output = Layout2d<S>;

    fn generate(&mut self) -> CoreResult<Self::Output> {
        self.config.validate()?;
        let n = self.graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr(
                "SfcLayout node count exceeds u32::MAX",
            ));
        }
        if let Some(ws) = self.graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite("SfcLayout::graph::weights"));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(
                        "SfcLayout requires non-negative edge weights",
                    ));
                }
            }
        }

        let order = match self.config.resolution_order {
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
            self.vertex_order.clear();
            return Ok(Layout2d::zeros(0));
        }
        if n == 1 {
            self.vertex_order = vec![0];
            return Ok(Layout2d::zeros(1));
        }

        let undirected = canonical_undirected_graph(self.graph)?;
        let vertex_order = if undirected.num_edges() == 0 {
            (0..n as u32).collect()
        } else {
            compute_community_vertex_order(&undirected, self.config.seed)?
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
            let (ix, iy) = match self.config.curve {
                SfcCurveKind::Hilbert => hilbert_d2xy(order, d),
                SfcCurveKind::Morton => morton_d2xy(order, d),
            };
            let xf = (((ix as f64) + 0.5) / side_f - 0.5) * self.config.scale;
            let yf = (((iy as f64) + 0.5) / side_f - 0.5) * self.config.scale;
            let idx = v_u32 as usize;
            coords_x[idx] = S::from_f64(xf);
            coords_y[idx] = S::from_f64(yf);
        }

        let layout = Layout2d::from_xy(coords_x, coords_y)?;
        if layout.validate_finite().is_err() {
            return Err(CoreError::numerical_breakdown(
                "SfcLayout coordinates overflowed output scalar",
            ));
        }
        self.vertex_order = vertex_order;
        Ok(layout)
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
