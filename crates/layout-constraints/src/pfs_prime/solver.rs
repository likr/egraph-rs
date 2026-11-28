use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, Layout2d, LayoutTransform, NodeExtents2d, Scalar,
};
use egraph_spatial_trees::RectangleOverlapSweep;

use super::config::PfsPrimeConfig;
use super::verify::{
    count_union_pairs, find_f64_exact_overlaps, has_orthogonal_order_inversion, normalize_neg_zero,
};

/// In-place Hayashi et al. (1998) `PFS'` two-pass scan-line node overlap removal transform.
///
/// Executes the deterministic horizontal and vertical scan-line passes in a single
/// [`LayoutTransform::apply`] call, strictly preserving orthogonal order (`oo_nni == 0`) while
/// guaranteeing `0` overlapping rectangles in both `f64`-exact and `S`-precision sweep checks.
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
    last_overlap_count: usize,
    _marker: PhantomData<S>,
}

impl<S: Scalar> PfsPrimeOverlapRemoval<S> {
    /// Creates a [`PfsPrimeOverlapRemoval`] transform for rectangular nodes with per-node
    /// half-extents.
    pub fn new(extents: NodeExtents2d<'_, S>, config: PfsPrimeConfig) -> CoreResult<Self> {
        config.validate()?;
        extents.validate()?;
        let n = extents.len();
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
            let (hw, hh) = extents.extents(i)?;
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

        let seed = config.seed;
        Ok(Self {
            config,
            num_nodes: n,
            eff_hw_s,
            eff_hh_s,
            eff_hw_req,
            eff_hh_req,
            rng: DeterministicRng::from_seed_u64(seed),
            orig_x: vec![0.0; n],
            orig_y: vec![0.0; n],
            work_x: vec![0.0; n],
            work_y: vec![0.0; n],
            up_gamma_x: vec![0.0; n],
            up_gamma_y: vec![0.0; n],
            order: (0..n).collect(),
            sort_pts: Vec::with_capacity(n),
            gap_scratch: Vec::with_capacity(n),
            last_overlap_count: 0,
            _marker: PhantomData,
        })
    }

    /// Creates a [`PfsPrimeOverlapRemoval`] transform for circular/square nodes with equal
    /// half-width and half-height given by `radii`.
    pub fn from_radii(radii: &[S], config: PfsPrimeConfig) -> CoreResult<Self> {
        Self::new(NodeExtents2d::from_radii(radii)?, config)
    }

    /// Returns the transform configuration.
    #[inline]
    pub fn config(&self) -> &PfsPrimeConfig {
        &self.config
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of overlapping rectangle pairs recorded at the most recent `apply` call.
    #[inline]
    pub fn last_overlap_count(&self) -> usize {
        self.last_overlap_count
    }
}

impl<S: Scalar> LayoutTransform for PfsPrimeOverlapRemoval<S> {
    type Target = Layout2d<S>;

    fn apply(&mut self, target: &mut Self::Target) -> CoreResult<()> {
        if target.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                target.len(),
                "PfsPrimeOverlapRemoval::apply",
            ));
        }
        target.validate_finite()?;
        self.rng = DeterministicRng::from_seed_u64(self.config.seed);

        let n = self.num_nodes;
        if n <= 1 {
            self.last_overlap_count = 0;
            return Ok(());
        }

        let initial_f64_ov = find_f64_exact_overlaps(target, &self.eff_hw_req, &self.eff_hh_req)?;
        let initial_s_ov =
            RectangleOverlapSweep::find_layout_overlaps(target, &self.eff_hw_s, &self.eff_hh_s)?;
        if initial_f64_ov.is_empty() && initial_s_ov.is_empty() {
            self.last_overlap_count = 0;
            return Ok(());
        }

        {
            let xs = target.x();
            let ys = target.y();
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
            let (xs_mut, ys_mut) = target.xy_mut();
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

        let mut f64_overlaps = find_f64_exact_overlaps(target, &self.eff_hw_req, &self.eff_hh_req)?;
        let mut s_overlaps =
            RectangleOverlapSweep::find_layout_overlaps(target, &self.eff_hw_s, &self.eff_hh_s)?;
        if !f64_overlaps.is_empty()
            || !s_overlaps.is_empty()
            || has_orthogonal_order_inversion(&self.orig_x, &self.orig_y, target)
        {
            self.resolve_residual_overlaps_order_preserving(target);
            f64_overlaps = find_f64_exact_overlaps(target, &self.eff_hw_req, &self.eff_hh_req)?;
            s_overlaps = RectangleOverlapSweep::find_layout_overlaps(
                target,
                &self.eff_hw_s,
                &self.eff_hh_s,
            )?;
        }
        self.last_overlap_count = count_union_pairs(&f64_overlaps, &s_overlaps);
        Ok(())
    }
}
