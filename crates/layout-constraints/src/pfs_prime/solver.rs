use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_spatial_trees::RectangleOverlapSweep;

use super::config::PfsPrimeConfig;
use super::verify::{
    count_union_pairs, find_f64_exact_overlaps, has_orthogonal_order_inversion, normalize_neg_zero,
};

/// Cooperative Hayashi et al. (1998) `PFS'` two-pass scan-line node overlap removal solver.
///
/// Executes the deterministic horizontal and vertical scan-line passes in a single
/// [`IterativeSolver::step`] (`total_iterations() == 1`), strictly preserving orthogonal order
/// (`oo_nni == 0`) while guaranteeing `0` overlapping rectangles in both `f64`-exact and
/// `S`-precision sweep checks.
#[derive(Debug, Clone)]
pub struct PfsPrimeOverlapRemoval<S: Scalar = f64> {
    config: PfsPrimeConfig,
    pub(super) num_nodes: usize,
    pub(super) eff_hw_s: Vec<S>,
    pub(super) eff_hh_s: Vec<S>,
    pub(super) eff_hw_req: Vec<f64>,
    pub(super) eff_hh_req: Vec<f64>,
    pub(super) rng: DeterministicRng,
    pub(super) orig_x: Vec<f64>,
    pub(super) orig_y: Vec<f64>,
    pub(super) work_x: Vec<f64>,
    pub(super) work_y: Vec<f64>,
    pub(super) up_gamma_x: Vec<f64>,
    pub(super) up_gamma_y: Vec<f64>,
    pub(super) order: Vec<usize>,
    pub(super) sort_pts: Vec<(f64, f64, u32)>,
    pub(super) gap_scratch: Vec<f64>,
    iter: usize,
    last_overlap_count: usize,
    status: SolverStatus,
    _marker: PhantomData<S>,
}

impl<S: Scalar> PfsPrimeOverlapRemoval<S> {
    /// Creates a [`PfsPrimeOverlapRemoval`] solver for rectangular nodes with per-node
    /// half-widths and half-heights.
    pub fn new(
        half_widths: Vec<S>,
        half_heights: Vec<S>,
        config: PfsPrimeConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = half_widths.len();
        if half_heights.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                half_heights.len(),
                "PfsPrimeOverlapRemoval::new::half_heights",
            ));
        }
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "PfsPrimeOverlapRemoval node count exceeds u32::MAX",
            ));
        }

        let pad_half = 0.5 * config.padding;
        let mut eff_hw_s = Vec::with_capacity(n);
        let mut eff_hh_s = Vec::with_capacity(n);
        let mut eff_hw_req = Vec::with_capacity(n);
        let mut eff_hh_req = Vec::with_capacity(n);

        for i in 0..n {
            let hw = half_widths[i];
            let hh = half_heights[i];
            if !hw.is_finite() || !hh.is_finite() {
                return Err(CoreError::non_finite(
                    "PfsPrimeOverlapRemoval::new::extents",
                ));
            }
            if hw < S::ZERO || hh < S::ZERO {
                return Err(CoreError::invalid_input(
                    "half_widths and half_heights must be >= 0",
                ));
            }
            let hw_f = hw.to_f64() + pad_half;
            let hh_f = hh.to_f64() + pad_half;
            let hw_s = S::from_f64(hw_f);
            let hh_s = S::from_f64(hh_f);
            if !hw_s.is_finite() || !hh_s.is_finite() {
                return Err(CoreError::non_finite(
                    "PfsPrimeOverlapRemoval::new::padded_extents",
                ));
            }
            eff_hw_req.push(hw_f.max(hw_s.to_f64()));
            eff_hh_req.push(hh_f.max(hh_s.to_f64()));
            eff_hw_s.push(hw_s);
            eff_hh_s.push(hh_s);
        }

        Ok(Self {
            config,
            num_nodes: n,
            eff_hw_s,
            eff_hh_s,
            eff_hw_req,
            eff_hh_req,
            rng: DeterministicRng::from_seed_u64(config.seed),
            orig_x: vec![0.0; n],
            orig_y: vec![0.0; n],
            work_x: vec![0.0; n],
            work_y: vec![0.0; n],
            up_gamma_x: vec![0.0; n],
            up_gamma_y: vec![0.0; n],
            order: (0..n).collect(),
            sort_pts: Vec::with_capacity(n),
            gap_scratch: Vec::with_capacity(n),
            iter: 0,
            last_overlap_count: 0,
            status: SolverStatus::Initialized,
            _marker: PhantomData,
        })
    }

    /// Creates a [`PfsPrimeOverlapRemoval`] solver for circular/square nodes with equal
    /// half-width and half-height given by `radii`.
    pub fn from_radii(radii: Vec<S>, config: PfsPrimeConfig) -> CoreResult<Self> {
        let half_heights = radii.clone();
        Self::new(radii, half_heights, config)
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &PfsPrimeConfig {
        &self.config
    }

    /// Returns the number of vertices.
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

impl<S: Scalar> IterativeSolver<S> for PfsPrimeOverlapRemoval<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "PfsPrimeOverlapRemoval::step",
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

        let initial_f64_ov = find_f64_exact_overlaps(layout, &self.eff_hw_req, &self.eff_hh_req)?;
        let initial_s_ov =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?;
        if initial_f64_ov.is_empty() && initial_s_ov.is_empty() {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                let xi = normalize_neg_zero(xs[i].to_f64());
                let yi = normalize_neg_zero(ys[i].to_f64());
                self.orig_x[i] = xi;
                self.orig_y[i] = yi;
                self.work_x[i] = xi;
                self.work_y[i] = yi;
            }
        }
        self.jitter_coincident_points();
        self.up_gamma_x.fill(0.0);
        self.up_gamma_y.fill(0.0);

        let max_abs_x = self.orig_x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let max_abs_y = self.orig_y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let sum_hw: f64 = self.eff_hw_req.iter().sum();
        let sum_hh: f64 = self.eff_hh_req.iter().sum();
        let s_eps = S::EPSILON.to_f64();
        let sep_eps = 1.0 + 1e-6_f64.max(16.0 * s_eps);
        let slack_x = 8.0 * (max_abs_x + sum_hw) * s_eps;
        let slack_y = 8.0 * (max_abs_y + sum_hh) * s_eps;

        self.horizontal_scan(sep_eps, slack_x, slack_y);
        // Per agorajs/agora-pfsp, verticalScan evaluates force() on original X coordinates
        // before committing up_gamma_x / up_gamma_y back to node centers.
        self.vertical_scan(sep_eps, slack_x, slack_y);

        {
            let (xs_mut, ys_mut) = layout.xy_mut();
            for i in 0..n {
                let nx = S::from_f64(self.work_x[i] + self.up_gamma_x[i]);
                let ny = S::from_f64(self.work_y[i] + self.up_gamma_y[i]);
                if nx.is_finite() {
                    xs_mut[i] = nx;
                }
                if ny.is_finite() {
                    ys_mut[i] = ny;
                }
            }
        }

        let mut f64_overlaps = find_f64_exact_overlaps(layout, &self.eff_hw_req, &self.eff_hh_req)?;
        let mut s_overlaps =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?;
        if !f64_overlaps.is_empty()
            || !s_overlaps.is_empty()
            || has_orthogonal_order_inversion(&self.orig_x, &self.orig_y, layout)
        {
            self.resolve_residual_overlaps_order_preserving(layout);
            f64_overlaps = find_f64_exact_overlaps(layout, &self.eff_hw_req, &self.eff_hh_req)?;
            s_overlaps = RectangleOverlapSweep::find_layout_overlaps(
                layout,
                &self.eff_hw_s,
                &self.eff_hh_s,
            )?;
        }
        let ov_count = count_union_pairs(&f64_overlaps, &s_overlaps);

        self.iter = 1;
        self.last_overlap_count = ov_count;
        self.status = if ov_count == 0 {
            SolverStatus::Converged
        } else {
            SolverStatus::Completed
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
        1
    }
}
