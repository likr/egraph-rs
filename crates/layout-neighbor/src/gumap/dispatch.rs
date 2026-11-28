use std::any::Any;

use egraph_core::{
    CoreError, CoreResult, Layout2d, Layout3d, LayoutNd, LayoutStorage, ManifoldLayout2d,
    RiemannianLayout,
};

use super::manifold::step_manifold;
use super::nd::step_nd;
use super::solver::GuMapState;
use super::step2d::step_2d;
use super::step3d::step_3d;

pub(super) type GuMapValidateFn<L> =
    fn(&GuMapState<<L as LayoutStorage>::Scalar>, &L) -> CoreResult<()>;
pub(super) type GuMapStepFn<L> =
    fn(&mut GuMapState<<L as LayoutStorage>::Scalar>, &mut L) -> CoreResult<()>;

pub(super) fn validate_basic<L: RiemannianLayout>(
    state: &GuMapState<L::Scalar>,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "GuMapSolver::validate_target",
        ));
    }
    target.validate_finite()
}

pub(super) fn validate_nd<L: RiemannianLayout + 'static>(
    state: &GuMapState<L::Scalar>,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "GuMapSolver::validate_target",
        ));
    }
    let layout = (target as &dyn Any)
        .downcast_ref::<LayoutNd<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("GuMapSolver target type mismatch"))?;
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
            "GuMapSolver::validate_target::dims",
        ));
    }
    layout.validate_finite()
}

pub(super) fn validate_manifold<L: RiemannianLayout + 'static>(
    state: &GuMapState<L::Scalar>,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "GuMapSolver::validate_target",
        ));
    }
    let layout = (target as &dyn Any)
        .downcast_ref::<ManifoldLayout2d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("GuMapSolver target type mismatch"))?;
    if let Some(prev) = state.locked_manifold
        && prev != layout.kind()
    {
        return Err(CoreError::invalid_input(
            "GuMapSolver::validate_target: cannot switch ManifoldKind on stateful solver",
        ));
    }
    layout.validate_finite()
}

pub(super) fn validate_riemannian<L: RiemannianLayout>(
    state: &GuMapState<L::Scalar>,
    target: &L,
) -> CoreResult<()> {
    if target.len() != state.num_nodes {
        return Err(CoreError::dimension_mismatch(
            state.num_nodes,
            target.len(),
            "GuMapSolver::validate_target",
        ));
    }
    if target.tangent_dim() == 0 && state.num_nodes > 0 {
        return Err(CoreError::invalid_input("tangent_dim must be >= 1"));
    }
    target.validate_finite()
}

pub(super) fn step_any_2d<L: RiemannianLayout + 'static>(
    state: &mut GuMapState<L::Scalar>,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout2d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("GuMapSolver target type mismatch"))?;
    step_2d(state, layout)
}

pub(super) fn step_any_3d<L: RiemannianLayout + 'static>(
    state: &mut GuMapState<L::Scalar>,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<Layout3d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("GuMapSolver target type mismatch"))?;
    step_3d(state, layout)
}

pub(super) fn step_any_nd<L: RiemannianLayout + 'static>(
    state: &mut GuMapState<L::Scalar>,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<LayoutNd<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("GuMapSolver target type mismatch"))?;
    step_nd(state, layout)
}

pub(super) fn step_any_manifold<L: RiemannianLayout + 'static>(
    state: &mut GuMapState<L::Scalar>,
    target: &mut L,
) -> CoreResult<()> {
    let layout = (target as &mut dyn Any)
        .downcast_mut::<ManifoldLayout2d<L::Scalar>>()
        .ok_or_else(|| CoreError::numerical_breakdown("GuMapSolver target type mismatch"))?;
    step_manifold(state, layout)
}
