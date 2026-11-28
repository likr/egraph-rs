use std::any::Any;

use egraph_core::{
    CoreError, CoreResult, Layout2d, Layout3d, LayoutNd, LayoutStorage, ManifoldKind,
    ManifoldLayout2d, RiemannianLayout, Scalar, SolverStatus,
};
use egraph_spatial_fft::{FitField2d, FitFieldOutput, FitFieldStats};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use super::config::{TsNetBackend, TsNetConfig};
use super::dim3::{TsNet3dScratch, step_3d};
use super::manifold::{TsNetManifoldScratch, step_manifold};
use super::nd::{TsNetNdScratch, step_nd};
use super::riemannian::TsNetRiemannianScratch;
use super::step2d::step_2d;

#[derive(Debug, Clone)]
pub(super) struct TsNetState<S: Scalar> {
    pub(super) config: TsNetConfig,
    pub(super) resolved_backend: TsNetBackend,
    pub(super) num_nodes: usize,
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) p_vals: Vec<f64>,
    pub(super) inv_jacobi: Vec<f64>,
    pub(super) attr_x: Vec<f64>,
    pub(super) attr_y: Vec<f64>,
    pub(super) vel_x: Vec<f64>,
    pub(super) vel_y: Vec<f64>,
    pub(super) rep_x: Vec<S>,
    pub(super) rep_y: Vec<S>,
    pub(super) ent_x: Vec<S>,
    pub(super) ent_y: Vec<S>,
    pub(super) fit_field: Option<FitField2d<S>>,
    pub(super) fit_out: FitFieldOutput<S>,
    pub(super) quadtree: ArenaQuadtree2d<S>,
    pub(super) octree: ArenaOctree3d<S>,
    pub(super) dim3_scratch: Option<TsNet3dScratch<S>>,
    pub(super) nd_scratch: Option<TsNetNdScratch>,
    pub(super) manifold_scratch: Option<TsNetManifoldScratch>,
    pub(super) riemannian_scratch: Option<TsNetRiemannianScratch>,
    pub(super) locked_nd_dims: Option<usize>,
    pub(super) locked_manifold: Option<ManifoldKind>,
    pub(super) status: SolverStatus,
    pub(super) iteration: usize,
    pub(super) total_iterations: usize,
    pub(super) stage_idx: usize,
    pub(super) stage_iter: usize,
    pub(super) stage_entered: bool,
    pub(super) prev_stage_lr: f64,
    pub(super) active_stage_lr: f64,
    pub(super) last_fft_stats: Option<FitFieldStats>,
    pub(super) total_fft_2d_calls: u64,
}

impl<S: Scalar> TsNetState<S> {
    pub(super) fn advance_stage_counters(&mut self) {
        self.iteration += 1;
        self.stage_iter += 1;
        if self.stage_iter < self.config.stages[self.stage_idx].iterations {
            self.status = SolverStatus::Running;
            return;
        }

        self.stage_iter = 0;
        self.stage_entered = false;
        if let Some(rel_next) = self.config.stages[self.stage_idx + 1..]
            .iter()
            .position(|s| s.iterations > 0)
        {
            self.stage_idx += 1 + rel_next;
            self.status = SolverStatus::Running;
        } else {
            self.status = SolverStatus::Completed;
        }
    }
}

pub(super) type TsNetValidateFn<L> =
    fn(&TsNetState<<L as LayoutStorage>::Scalar>, &L) -> CoreResult<()>;
pub(super) type TsNetStepFn<L> =
    fn(&mut TsNetState<<L as LayoutStorage>::Scalar>, &mut L) -> CoreResult<()>;

pub(super) fn resolve_backend_for_target(
    backend: TsNetBackend,
    num_nodes: usize,
    is_2d: bool,
    is_3d: bool,
    is_nd: bool,
    is_manifold: bool,
) -> CoreResult<TsNetBackend> {
    if is_2d {
        return Ok(match backend {
            TsNetBackend::Auto => {
                if num_nodes <= 256 {
                    TsNetBackend::Exact
                } else {
                    TsNetBackend::LinearFft
                }
            }
            other => other,
        });
    }
    if is_3d {
        return match backend {
            TsNetBackend::Exact => Ok(TsNetBackend::Exact),
            TsNetBackend::BarnesHut => Ok(TsNetBackend::BarnesHut),
            TsNetBackend::Auto => Ok(if num_nodes <= 256 {
                TsNetBackend::Exact
            } else {
                TsNetBackend::BarnesHut
            }),
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft => {
                Err(CoreError::unsupported(
                    "FFT interpolation backends are only supported on Layout2d; use Exact or BarnesHut on Layout3d",
                ))
            }
        };
    }
    if is_nd {
        return match backend {
            TsNetBackend::Exact | TsNetBackend::Auto => Ok(TsNetBackend::Exact),
            TsNetBackend::BarnesHut => Err(CoreError::unsupported(
                "TsNetBackend::BarnesHut is not supported on LayoutNd; use Layout3d for 3D Barnes-Hut, Layout2d for 2D, or Exact for arbitrary N-D",
            )),
            TsNetBackend::FftInterpolation | TsNetBackend::LinearFft => {
                Err(CoreError::unsupported(
                    "FFT interpolation backends are only supported on Layout2d; use Exact on LayoutNd",
                ))
            }
        };
    }
    if is_manifold {
        return match backend {
            TsNetBackend::Exact | TsNetBackend::Auto => Ok(TsNetBackend::Exact),
            other @ (TsNetBackend::BarnesHut
            | TsNetBackend::FftInterpolation
            | TsNetBackend::LinearFft) => Err(CoreError::unsupported(format!(
                "TsNetBackend::{other:?} is only supported on Euclidean layouts; use Exact or Auto on ManifoldLayout2d"
            ))),
        };
    }
    match backend {
        TsNetBackend::Exact | TsNetBackend::Auto => Ok(TsNetBackend::Exact),
        other => Err(CoreError::unsupported(format!(
            "TsNetBackend::{other:?} is only supported on Euclidean layouts; use Exact or Auto on general RiemannianLayout targets"
        ))),
    }
}

pub(super) fn validate_basic<L: RiemannianLayout>(
    state: &TsNetState<L::Scalar>,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "TsNetSolver::validate_target",
        ));
    }
    target.validate_finite()
}

pub(super) fn validate_nd<L: RiemannianLayout + 'static>(
    state: &TsNetState<L::Scalar>,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "TsNetSolver::validate_target",
        ));
    }
    let layout = (target as &dyn Any)
        .downcast_ref::<LayoutNd<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("TsNetSolver target type mismatch"))?;
    let dims = layout.dims();
    if dims == 0 {
        return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
    }
    if let Some(prev) = state.locked_nd_dims
        && prev != dims
    {
        return Err(CoreError::dimension_mismatch(
            prev,
            dims,
            "TsNetSolver::validate_target::dims",
        ));
    }
    layout.validate_finite()
}

pub(super) fn validate_manifold<L: RiemannianLayout + 'static>(
    state: &TsNetState<L::Scalar>,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "TsNetSolver::validate_target",
        ));
    }
    let layout = (target as &dyn Any)
        .downcast_ref::<ManifoldLayout2d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("TsNetSolver target type mismatch"))?;
    if let Some(prev) = state.locked_manifold
        && prev != layout.kind()
    {
        return Err(CoreError::invalid_input(
            "TsNetSolver::validate_target: cannot switch ManifoldKind on stateful solver",
        ));
    }
    layout.validate_finite()
}

pub(super) fn validate_riemannian<L: RiemannianLayout>(
    state: &TsNetState<L::Scalar>,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "TsNetSolver::validate_target",
        ));
    }
    if target.tangent_dim() == 0 && state.num_nodes > 0 {
        return Err(CoreError::invalid_input("tangent_dim must be >= 1"));
    }
    target.validate_finite()
}

pub(super) fn step_any_2d<L: RiemannianLayout + 'static>(
    state: &mut TsNetState<L::Scalar>,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout2d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("TsNetSolver target type mismatch"))?;
    step_2d(state, layout)
}

pub(super) fn step_any_3d<L: RiemannianLayout + 'static>(
    state: &mut TsNetState<L::Scalar>,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout3d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("TsNetSolver target type mismatch"))?;
    step_3d(state, layout)
}

pub(super) fn step_any_nd<L: RiemannianLayout + 'static>(
    state: &mut TsNetState<L::Scalar>,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<LayoutNd<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("TsNetSolver target type mismatch"))?;
    step_nd(state, layout)
}

pub(super) fn step_any_manifold<L: RiemannianLayout + 'static>(
    state: &mut TsNetState<L::Scalar>,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<ManifoldLayout2d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("TsNetSolver target type mismatch"))?;
    step_manifold(state, layout)
}
