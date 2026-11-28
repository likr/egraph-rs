use egraph_core::{CoreError, CoreResult, LayoutNd, LayoutStorage, Scalar};

use super::config::{MIN_LAMBDA_SUM, TsNetStage};
use super::repulsion::{TsNetCore, TsNetRepulsion, TsNetSpaceKind, TsNetWorkspace};

/// Lazily allocated N-D scratch buffers for [`TsNetSolver`](super::TsNetSolver) on [`LayoutNd`].
#[derive(Debug, Clone)]
pub(super) struct TsNetNdWorkspace {
    locked_dims: Option<usize>,
    any_entropy: bool,
    attr: Vec<f64>,
    vel: Vec<f64>,
    rep: Vec<f64>,
    ent: Vec<f64>,
    mean: Vec<f64>,
}

impl TsNetNdWorkspace {
    pub(super) fn new(any_entropy: bool) -> Self {
        Self {
            locked_dims: None,
            any_entropy,
            attr: Vec::new(),
            vel: Vec::new(),
            rep: Vec::new(),
            ent: Vec::new(),
            mean: Vec::new(),
        }
    }

    fn ensure_allocated(&mut self, n: usize, dims: usize) -> CoreResult<()> {
        if self.locked_dims.is_some() {
            return Ok(());
        }
        let total = n
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("LayoutNd size overflow"))?;
        self.locked_dims = Some(dims);
        self.attr.resize(total, 0.0);
        self.vel.resize(total, 0.0);
        self.rep.resize(total, 0.0);
        if self.any_entropy {
            self.ent.resize(total, 0.0);
        }
        self.mean.resize(dims, 0.0);
        Ok(())
    }
}

pub(super) fn validate_nd<S: Scalar>(
    core: &TsNetCore,
    workspace: &TsNetWorkspace,
    layout: &LayoutNd<S>,
) -> CoreResult<()> {
    if layout.len() != core.num_nodes {
        return Err(CoreError::dimension_mismatch(
            core.num_nodes,
            layout.len(),
            "TsNetSolver::validate_target",
        ));
    }
    let dims = layout.dims();
    if dims == 0 {
        return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
    }
    if let TsNetSpaceKind::EuclideanNd(ws) = &workspace.0
        && let Some(prev) = ws.locked_dims
        && prev != dims
    {
        return Err(CoreError::dimension_mismatch(
            prev,
            dims,
            "TsNetSolver::validate_target::dims",
        ));
    }
    LayoutStorage::validate_finite(layout)
}

fn compute_attractions_nd<S: Scalar>(
    core: &TsNetCore,
    ws: &mut TsNetNdWorkspace,
    coords: &[S],
    dims: usize,
    exaggeration: f64,
) {
    let n = core.num_nodes;
    for i in 0..n {
        let i_base = i * dims;
        ws.attr[i_base..i_base + dims].fill(0.0);
        let start = core.indptr[i] as usize;
        let end = core.indptr[i + 1] as usize;
        for pos in start..end {
            let j = core.indices[pos] as usize;
            if j == i {
                continue;
            }
            let j_base = j * dims;
            let mut dist_sq = 0.0_f64;
            for d in 0..dims {
                let dx = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                dist_sq += dx * dx;
            }
            let q_unnorm = 1.0 / (1.0 + dist_sq);
            let mult = (core.p_vals[pos] * exaggeration) * q_unnorm;
            for d in 0..dims {
                let dx = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                ws.attr[i_base + d] += mult * dx;
            }
        }
    }
}

fn apply_nd_update<S: Scalar>(
    core: &TsNetCore,
    ws: &mut TsNetNdWorkspace,
    layout: &mut LayoutNd<S>,
    dims: usize,
    stage: &TsNetStage,
) {
    let n = core.num_nodes;
    let n_f64 = n as f64;
    let n_sq_f64 = n_f64 * n_f64;
    let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
    let stage_lr = core.active_stage_lr;
    let momentum = core.config.momentum;
    let use_jacobi = core.config.jacobi_precondition;
    let compute_entropy = stage.lambda_r > 0.0;

    ws.mean.fill(0.0);
    let coords = layout.coords_mut();
    for (i, &inv_j) in core.inv_jacobi.iter().enumerate().take(n) {
        let lr_i = if use_jacobi {
            stage_lr * inv_j
        } else {
            stage_lr
        };
        let i_base = i * dims;
        for d in 0..dims {
            let idx = i_base + d;
            let x_d = coords[idx].to_f64();
            let grad_kl = 4.0 * (ws.attr[idx] - ws.rep[idx]);
            let grad_c = x_d / n_f64;
            let grad_r = if compute_entropy {
                -ws.ent[idx] / n_sq_f64
            } else {
                0.0
            };

            let raw_g =
                (stage.lambda_kl * grad_kl + stage.lambda_c * grad_c + stage.lambda_r * grad_r)
                    / lambda_sum;
            let g = if raw_g.is_finite() { raw_g } else { 0.0 };

            let mut v = momentum * ws.vel[idx] - lr_i * g;
            if !v.is_finite() {
                v = 0.0;
            }
            ws.vel[idx] = v;

            let cand = x_d + v;
            let new_val = if cand.is_finite() { cand } else { x_d };
            ws.attr[idx] = new_val;
            ws.mean[d] += new_val;
        }
    }

    for mean in &mut ws.mean {
        let m = *mean / n_f64;
        *mean = if m.is_finite() { m } else { 0.0 };
    }

    for i in 0..n {
        let i_base = i * dims;
        for d in 0..dims {
            let idx = i_base + d;
            let centered = S::from_f64(ws.attr[idx] - ws.mean[d]);
            if centered.is_finite() {
                coords[idx] = centered;
            }
        }
    }
}

pub(super) fn step_nd<S: Scalar, B: TsNetRepulsion<LayoutNd<S>>>(
    layout: &mut LayoutNd<S>,
    core: &mut TsNetCore,
    workspace: &mut TsNetWorkspace,
    backend: &B,
    backend_state: &mut B::State,
) -> CoreResult<()> {
    let TsNetSpaceKind::EuclideanNd(ws) = &mut workspace.0 else {
        return Err(CoreError::numerical_breakdown(
            "TsNetNdWorkspace variant mismatch",
        ));
    };

    let dims = layout.dims();
    ws.ensure_allocated(core.num_nodes, dims)?;

    let stage = core.config.stages[core.stage_idx].clone();
    let compute_entropy = stage.lambda_r > 0.0;

    if !core.stage_entered {
        let stage_lr = core.config.compute_stage_lr(&stage, core.num_nodes);
        if stage_lr < core.prev_stage_lr {
            let vel_scale = stage_lr / core.prev_stage_lr;
            for v in &mut ws.vel {
                *v *= vel_scale;
            }
        }
        core.prev_stage_lr = stage_lr;
        core.active_stage_lr = stage_lr;
        backend.on_stage_enter(backend_state, core.config.epsilon_r, compute_entropy)?;
        core.stage_entered = true;
    }

    compute_attractions_nd(core, ws, layout.coords(), dims, stage.exaggeration);
    backend.compute_repulsion(
        backend_state,
        layout,
        core.config.epsilon_r,
        compute_entropy,
        &mut ws.rep,
        &mut ws.ent,
    )?;
    apply_nd_update(core, ws, layout, dims, &stage);
    Ok(())
}
