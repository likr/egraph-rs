use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, NodeExtents2d, Scalar,
    SolverStatus,
};
use egraph_spatial_trees::RectangleOverlapSweep;

use super::config::{ForbidConfig, SgdPairItem};

/// Cooperative Giovannangeli et al. (2022) FORBID node overlap removal solver.
///
/// Each [`IterativeSolver::step`] executes one binary-search scaling + `SGD2_StressOPT`
/// pass and updates [`ForbidOverlapRemoval::last_overlap_count`]. Reaching `max_scale_iters`
/// with residual overlaps restores the best overlap-free scaled layout (with deterministic QPSC
/// fallback if needed) so converged outputs strictly have `0` overlapping rectangles.
#[derive(Debug, Clone)]
pub struct ForbidOverlapRemoval<S: Scalar = f64> {
    pub(super) config: ForbidConfig,
    pub(super) num_nodes: usize,
    pub(super) eff_hw_f64: Vec<f64>,
    pub(super) eff_hh_f64: Vec<f64>,
    pub(super) eff_hw_s: Vec<S>,
    pub(super) eff_hh_s: Vec<S>,
    pub(super) rng: DeterministicRng,
    pub(super) initialized_state: bool,
    pub(super) orig_x: Vec<f64>,
    pub(super) orig_y: Vec<f64>,
    pub(super) orig_cx: f64,
    pub(super) orig_cy: f64,
    pub(super) work_layout: Layout2d<f64>,
    pub(super) best_free_x: Vec<f64>,
    pub(super) best_free_y: Vec<f64>,
    pub(super) has_best_free: bool,
    pub(super) base_pairs: Vec<(u32, u32)>,
    pub(super) active_overlaps: Vec<(u32, u32)>,
    pub(super) sgd_pairs: Vec<SgdPairItem>,
    pub(super) sort_pts: Vec<(f64, f64, u32)>,
    pub(super) low_scale: f64,
    pub(super) up_scale: f64,
    pub(super) cur_scale: f64,
    pub(super) prev_scale: f64,
    pub(super) iter: usize,
    last_overlap_count: usize,
    status: SolverStatus,
    _marker: PhantomData<S>,
}

impl<S: Scalar> ForbidOverlapRemoval<S> {
    /// Creates a [`ForbidOverlapRemoval`] solver for rectangular nodes with per-node half-extents.
    pub fn new(extents: NodeExtents2d<'_, S>, config: ForbidConfig) -> CoreResult<Self> {
        config.validate()?;
        extents.validate()?;
        let n = extents.len();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "ForbidOverlapRemoval node count exceeds u32::MAX",
            ));
        }

        let pad_half = 0.5 * config.padding;
        let mut eff_hw_f64 = Vec::with_capacity(n);
        let mut eff_hh_f64 = Vec::with_capacity(n);
        let mut eff_hw_s = Vec::with_capacity(n);
        let mut eff_hh_s = Vec::with_capacity(n);

        for i in 0..n {
            let (hw, hh) = extents.extents(i)?;
            let hw_f = hw.to_f64() + pad_half;
            let hh_f = hh.to_f64() + pad_half;
            let hw_s = S::from_f64(hw_f);
            let hh_s = S::from_f64(hh_f);
            if !hw_s.is_finite() || !hh_s.is_finite() {
                return Err(CoreError::non_finite(
                    "ForbidOverlapRemoval::new::padded_extents",
                ));
            }
            eff_hw_f64.push(hw_f);
            eff_hh_f64.push(hh_f);
            eff_hw_s.push(hw_s);
            eff_hh_s.push(hh_s);
        }

        let seed = config.seed;
        let status = if config.max_scale_iters == 0 {
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
            rng: DeterministicRng::from_seed_u64(seed),
            initialized_state: false,
            orig_x: vec![0.0; n],
            orig_y: vec![0.0; n],
            orig_cx: 0.0,
            orig_cy: 0.0,
            work_layout: Layout2d::zeros(n),
            best_free_x: vec![0.0; n],
            best_free_y: vec![0.0; n],
            has_best_free: false,
            base_pairs: Vec::new(),
            active_overlaps: Vec::new(),
            sgd_pairs: Vec::new(),
            sort_pts: Vec::with_capacity(n),
            low_scale: 1.0,
            up_scale: 1.0,
            cur_scale: 1.0,
            prev_scale: 1.0,
            iter: 0,
            last_overlap_count: 0,
            status,
            _marker: PhantomData,
        })
    }

    /// Creates a [`ForbidOverlapRemoval`] solver for circular/square nodes with equal half-width
    /// and half-height given by `radii`.
    pub fn from_radii(radii: &[S], config: ForbidConfig) -> CoreResult<Self> {
        Self::new(NodeExtents2d::from_radii(radii)?, config)
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &ForbidConfig {
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

    fn execute_step_inner(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        let n = self.num_nodes;
        if n <= 1 {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if !self.initialized_state {
            let initial_overlaps = RectangleOverlapSweep::find_layout_overlaps(
                layout,
                &self.eff_hw_s,
                &self.eff_hh_s,
            )?;
            if initial_overlaps.is_empty() {
                self.last_overlap_count = 0;
                self.status = SolverStatus::Converged;
                return Ok(self.status);
            }
            self.initialize_from_layout(layout)?;
        }

        self.apply_scale();
        self.sgd2_stress_opt()?;
        self.copy_work_to_layout(layout)?;

        let mut ov_count =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?
                .len();

        let there_is_overlap = ov_count > 0;
        if !there_is_overlap {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                self.best_free_x[i] = xs[i].to_f64();
                self.best_free_y[i] = ys[i].to_f64();
            }
            self.has_best_free = true;
            self.up_scale = self.cur_scale;
        } else {
            self.low_scale = self.cur_scale;
        }
        self.cur_scale = 0.5 * (self.low_scale + self.up_scale);
        self.iter += 1;

        if !there_is_overlap && (self.up_scale - self.low_scale <= self.config.scale_step) {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if self.iter >= self.config.max_scale_iters {
            if there_is_overlap {
                if self.has_best_free {
                    let (xs_mut, ys_mut) = layout.xy_mut();
                    for i in 0..n {
                        xs_mut[i] = S::from_f64(self.best_free_x[i]);
                        ys_mut[i] = S::from_f64(self.best_free_y[i]);
                    }
                    ov_count = RectangleOverlapSweep::find_layout_overlaps(
                        layout,
                        &self.eff_hw_s,
                        &self.eff_hh_s,
                    )?
                    .len();
                }
                if ov_count > 0 {
                    let eff_extents = NodeExtents2d::new(&self.eff_hw_s, &self.eff_hh_s)?;
                    crate::constraints::project_rectangle_no_overlap_2d(layout, eff_extents)?;
                    ov_count = RectangleOverlapSweep::find_layout_overlaps(
                        layout,
                        &self.eff_hw_s,
                        &self.eff_hh_s,
                    )?
                    .len();
                }
            }
            self.last_overlap_count = ov_count;
            self.status = if ov_count == 0 {
                SolverStatus::Converged
            } else {
                SolverStatus::Completed
            };
        } else {
            self.last_overlap_count = ov_count;
            self.status = SolverStatus::Running;
        }

        Ok(self.status)
    }
}

impl<S: Scalar> IterativeSolver for ForbidOverlapRemoval<S> {
    type Target = Layout2d<S>;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        if target.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                target.len(),
                "ForbidOverlapRemoval::validate_target",
            ));
        }
        target.validate_finite()
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        match self.execute_step_inner(target) {
            Ok(status) => Ok(status),
            Err(err) => {
                if matches!(
                    err,
                    CoreError::NumericalBreakdown(_) | CoreError::NonFiniteValue { .. }
                ) {
                    self.status = SolverStatus::Failed;
                }
                Err(err)
            }
        }
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
    fn total_iterations(&self) -> Option<usize> {
        Some(self.config.max_scale_iters)
    }
}
