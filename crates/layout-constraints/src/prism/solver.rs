use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
#[cfg(not(feature = "delaunay"))]
use egraph_spatial_trees::KdQueryScratch;

#[cfg(not(feature = "delaunay"))]
use super::K_FALLBACK_NEIGHBORS;
use super::pcg::SparsePcgWorkspace;
use super::proximity::PrismEdge;
use super::sap::{SapRect, sap_find_overlaps};
use super::{PrismConfig, STALL_NO_IMPROVE_STREAK};

/// Cooperative Gansner-Hu (2010) PRISM proximity-stress overlap removal solver.
///
/// Each [`IterativeSolver::step`] performs one PRISM proximity-graph + fused 2-RHS sparse
/// Symmetric SSOR-PCG stress-majorization round and updates
/// [`PrismOverlapRemoval::last_overlap_count`].
#[derive(Debug, Clone)]
pub struct PrismOverlapRemoval<S: Scalar = f64> {
    pub(super) config: PrismConfig,
    pub(super) num_nodes: usize,
    pub(super) eff_hw_f64: Vec<f64>,
    pub(super) eff_hh_f64: Vec<f64>,
    pub(super) eff_hw_s: Vec<S>,
    pub(super) eff_hh_s: Vec<S>,
    pub(super) rng: DeterministicRng,
    pub(super) pcg: SparsePcgWorkspace,
    // Reusable per-round scratch buffers.
    pub(super) work_layout: Layout2d<f64>,
    pub(super) x_next: Vec<f64>,
    pub(super) y_next: Vec<f64>,
    pub(super) rhs_x: Vec<f64>,
    pub(super) rhs_y: Vec<f64>,
    last_out_x: Vec<f64>,
    last_out_y: Vec<f64>,
    pub(super) active_overlaps: Vec<(u32, u32)>,
    sap_bounds: Vec<SapRect>,
    sap_order: Vec<u32>,
    sap_sorted: Vec<SapRect>,
    pub(super) raw_edges: Vec<(u32, u32)>,
    pub(super) prism_edges: Vec<PrismEdge>,
    pub(super) diag: Vec<f64>,
    pub(super) diag_inv: Vec<f64>,
    pub(super) l_deg: Vec<u32>,
    pub(super) u_deg: Vec<u32>,
    pub(super) l_pos: Vec<u32>,
    pub(super) u_pos: Vec<u32>,
    pub(super) l_indptr: Vec<u32>,
    pub(super) l_indices: Vec<u32>,
    pub(super) l_values: Vec<f64>,
    pub(super) u_indptr: Vec<u32>,
    pub(super) u_indices: Vec<u32>,
    pub(super) u_values: Vec<f64>,
    pub(super) sort_pts: Vec<(f64, f64, u32)>,
    #[cfg(not(feature = "delaunay"))]
    pub(super) kd_scratch: KdQueryScratch,
    #[cfg(not(feature = "delaunay"))]
    pub(super) knn_out: Vec<(usize, f64)>,
    iter: usize,
    last_overlap_count: usize,
    best_overlap_count: usize,
    no_improve_streak: usize,
    status: SolverStatus,
    _marker: PhantomData<S>,
}

impl<S: Scalar> PrismOverlapRemoval<S> {
    /// Creates a [`PrismOverlapRemoval`] solver for rectangular nodes with per-node half-widths
    /// and half-heights.
    pub fn new(half_widths: Vec<S>, half_heights: Vec<S>, config: PrismConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = half_widths.len();
        if half_heights.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                half_heights.len(),
                "PrismOverlapRemoval::new::half_heights",
            ));
        }
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "PrismOverlapRemoval node count exceeds u32::MAX",
            ));
        }

        let pad_half = 0.5 * config.padding;
        let mut eff_hw_f64 = Vec::with_capacity(n);
        let mut eff_hh_f64 = Vec::with_capacity(n);
        let mut eff_hw_s = Vec::with_capacity(n);
        let mut eff_hh_s = Vec::with_capacity(n);

        for i in 0..n {
            let hw = half_widths[i];
            let hh = half_heights[i];
            if !hw.is_finite() || !hh.is_finite() {
                return Err(CoreError::non_finite("PrismOverlapRemoval::new::extents"));
            }
            if hw < S::ZERO || hh < S::ZERO {
                return Err(CoreError::invalid_input(
                    "half_widths and half_heights must be >= 0",
                ));
            }
            let hw_f = hw.to_f64() + pad_half;
            let hh_f = hh.to_f64() + pad_half;
            eff_hw_f64.push(hw_f);
            eff_hh_f64.push(hh_f);
            eff_hw_s.push(S::from_f64(hw_f));
            eff_hh_s.push(S::from_f64(hh_f));
        }

        let status = if config.max_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            eff_hw_f64,
            eff_hh_f64,
            eff_hw_s,
            eff_hh_s,
            rng: DeterministicRng::from_seed_u64(config.seed),
            pcg: SparsePcgWorkspace::new(n),
            work_layout: Layout2d::zeros(n),
            x_next: vec![0.0; n],
            y_next: vec![0.0; n],
            rhs_x: vec![0.0; n],
            rhs_y: vec![0.0; n],
            last_out_x: vec![0.0; n],
            last_out_y: vec![0.0; n],
            active_overlaps: Vec::new(),
            sap_bounds: Vec::with_capacity(n),
            sap_order: (0..n as u32).collect(),
            sap_sorted: Vec::with_capacity(n),
            raw_edges: Vec::new(),
            prism_edges: Vec::new(),
            diag: vec![0.0; n],
            diag_inv: vec![0.0; n],
            l_deg: vec![0; n],
            u_deg: vec![0; n],
            l_pos: vec![0; n],
            u_pos: vec![0; n],
            l_indptr: vec![0; n + 1],
            l_indices: Vec::new(),
            l_values: Vec::new(),
            u_indptr: vec![0; n + 1],
            u_indices: Vec::new(),
            u_values: Vec::new(),
            sort_pts: Vec::with_capacity(n),
            #[cfg(not(feature = "delaunay"))]
            kd_scratch: KdQueryScratch::with_capacity(K_FALLBACK_NEIGHBORS),
            #[cfg(not(feature = "delaunay"))]
            knn_out: Vec::with_capacity(K_FALLBACK_NEIGHBORS),
            iter: 0,
            last_overlap_count: 0,
            best_overlap_count: usize::MAX,
            no_improve_streak: 0,
            status,
            _marker: PhantomData,
        })
    }

    /// Creates a [`PrismOverlapRemoval`] solver for circular/square nodes with equal half-width
    /// and half-height given by `radii`.
    pub fn from_radii(radii: Vec<S>, config: PrismConfig) -> CoreResult<Self> {
        let half_heights = radii.clone();
        Self::new(radii, half_heights, config)
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &PrismConfig {
        &self.config
    }

    /// Returns the expected number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of overlapping rectangle pairs recorded at the most recent step.
    #[inline]
    pub fn last_overlap_count(&self) -> usize {
        self.last_overlap_count
    }
}

impl<S: Scalar> IterativeSolver<S> for PrismOverlapRemoval<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "PrismOverlapRemoval::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let cache_hit = self.iter > 0 && {
            let xs = layout.x();
            let ys = layout.y();
            xs.iter()
                .zip(self.last_out_x.iter())
                .all(|(&a, &b)| a.to_f64() == b)
                && ys
                    .iter()
                    .zip(self.last_out_y.iter())
                    .all(|(&a, &b)| a.to_f64() == b)
        };

        if !cache_hit {
            sap_find_overlaps(
                layout.x(),
                layout.y(),
                &self.eff_hw_s,
                &self.eff_hh_s,
                &mut self.sap_bounds,
                &mut self.sap_order,
                &mut self.sap_sorted,
                &mut self.active_overlaps,
            );
        }

        if self.active_overlaps.is_empty() {
            self.last_overlap_count = 0;
            self.best_overlap_count = 0;
            self.no_improve_streak = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if self.iter == 0 {
            self.best_overlap_count = self.active_overlaps.len();
            self.no_improve_streak = 0;
        }

        {
            let xs = layout.x();
            let ys = layout.y();
            let (wx, wy) = self.work_layout.xy_mut();
            for i in 0..n {
                wx[i] = xs[i].to_f64();
                wy[i] = ys[i].to_f64();
            }
        }

        if self.jitter_coincident_points() {
            sap_find_overlaps(
                self.work_layout.x(),
                self.work_layout.y(),
                &self.eff_hw_f64,
                &self.eff_hh_f64,
                &mut self.sap_bounds,
                &mut self.sap_order,
                &mut self.sap_sorted,
                &mut self.active_overlaps,
            );
        }

        self.build_proximity_edges()?;
        self.build_laplacian_csr()?;
        self.run_majorization_round()?;

        {
            let (xs_mut, ys_mut) = layout.xy_mut();
            for i in 0..n {
                let nx = S::from_f64(self.x_next[i]);
                let ny = S::from_f64(self.y_next[i]);
                if nx.is_finite() {
                    xs_mut[i] = nx;
                }
                if ny.is_finite() {
                    ys_mut[i] = ny;
                }
                self.last_out_x[i] = xs_mut[i].to_f64();
                self.last_out_y[i] = ys_mut[i].to_f64();
            }
        }

        sap_find_overlaps(
            layout.x(),
            layout.y(),
            &self.eff_hw_s,
            &self.eff_hh_s,
            &mut self.sap_bounds,
            &mut self.sap_order,
            &mut self.sap_sorted,
            &mut self.active_overlaps,
        );
        self.last_overlap_count = self.active_overlaps.len();
        self.iter += 1;

        if self.last_overlap_count < self.best_overlap_count {
            self.best_overlap_count = self.last_overlap_count;
            self.no_improve_streak = 0;
        } else if !self.active_overlaps.is_empty() {
            self.no_improve_streak += 1;
        }

        // Fall back to deterministic 2D scanline QPSC projection if PRISM stalls/oscillates
        // (e.g. extreme aspect-ratio limit cycles) or reaches max_iterations with residual overlaps.
        if !self.active_overlaps.is_empty()
            && (self.no_improve_streak >= STALL_NO_IMPROVE_STREAK
                || self.iter >= self.config.max_iterations)
        {
            crate::constraints::project_rectangle_no_overlap_2d(
                layout,
                &self.eff_hw_s,
                &self.eff_hh_s,
            )?;
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                self.last_out_x[i] = xs[i].to_f64();
                self.last_out_y[i] = ys[i].to_f64();
            }
            sap_find_overlaps(
                layout.x(),
                layout.y(),
                &self.eff_hw_s,
                &self.eff_hh_s,
                &mut self.sap_bounds,
                &mut self.sap_order,
                &mut self.sap_sorted,
                &mut self.active_overlaps,
            );
            self.last_overlap_count = self.active_overlaps.len();
            self.best_overlap_count = self.best_overlap_count.min(self.last_overlap_count);
            self.no_improve_streak = 0;
        }

        self.status = if self.last_overlap_count == 0 {
            SolverStatus::Converged
        } else if self.iter >= self.config.max_iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };

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
        self.config.max_iterations
    }
}
