use egraph_core::{CoreError, CoreResult, RiemannianLayout};

use super::config::{MIN_LAMBDA_SUM, compute_stage_lr};
use super::repulsion::{TsNetCore, TsNetRepulsion, TsNetSpaceKind, TsNetWorkspace};

#[derive(Debug, Clone)]
pub(super) struct TsNetRiemannianWorkspace {
    any_entropy: bool,
    dims: Option<usize>,
    attr: Vec<f64>,
    vel: Vec<f64>,
    rep: Vec<f64>,
    ent: Vec<f64>,
    log_buf: Vec<f64>,
    delta_buf: Vec<f64>,
    vel_buf: Vec<f64>,
}

impl TsNetRiemannianWorkspace {
    pub(super) fn new(num_nodes: usize, any_entropy: bool) -> Self {
        let _ = num_nodes;
        Self {
            any_entropy,
            dims: None,
            attr: Vec::new(),
            vel: Vec::new(),
            rep: Vec::new(),
            ent: Vec::new(),
            log_buf: Vec::new(),
            delta_buf: Vec::new(),
            vel_buf: Vec::new(),
        }
    }

    fn ensure_allocated(&mut self, n: usize, dims: usize) -> CoreResult<()> {
        if let Some(prev) = self.dims {
            if prev != dims {
                return Err(CoreError::dimension_mismatch(
                    prev,
                    dims,
                    "TsNetSolver::riemannian::tangent_dim",
                ));
            }
            return Ok(());
        }
        let total = n
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("RiemannianLayout scratch size overflow"))?;
        self.dims = Some(dims);
        self.attr.resize(total, 0.0);
        self.vel.resize(total, 0.0);
        self.rep.resize(total, 0.0);
        if self.any_entropy {
            self.ent.resize(total, 0.0);
        }
        self.log_buf.resize(dims, 0.0);
        self.delta_buf.resize(dims, 0.0);
        self.vel_buf.resize(dims, 0.0);
        Ok(())
    }
}

pub(super) fn validate_riemannian<L: RiemannianLayout + ?Sized>(
    core: &TsNetCore,
    workspace: &TsNetWorkspace,
    target: &L,
) -> CoreResult<()> {
    if target.len() != core.num_nodes {
        return Err(CoreError::dimension_mismatch(
            core.num_nodes,
            target.len(),
            "TsNetSolver::validate_target",
        ));
    }
    let dims = target.tangent_dim();
    if dims == 0 && core.num_nodes > 0 {
        return Err(CoreError::invalid_input("tangent_dim must be >= 1"));
    }
    if let TsNetSpaceKind::Riemannian(ws) = &workspace.0
        && let Some(prev) = ws.dims
        && prev != dims
    {
        return Err(CoreError::dimension_mismatch(
            prev,
            dims,
            "TsNetSolver::riemannian::tangent_dim",
        ));
    }
    target.validate_finite()
}

pub(super) fn step_riemannian<L: RiemannianLayout + ?Sized, B: TsNetRepulsion<L>>(
    layout: &mut L,
    core: &mut TsNetCore,
    workspace: &mut TsNetWorkspace,
    backend: &B,
    backend_state: &mut B::State,
) -> CoreResult<()> {
    let TsNetSpaceKind::Riemannian(ws) = &mut workspace.0 else {
        return Err(CoreError::numerical_breakdown(
            "TsNetRiemannianWorkspace variant mismatch",
        ));
    };

    let dims = layout.tangent_dim();
    ws.ensure_allocated(core.num_nodes, dims)?;

    let stage = core.config.stages[core.stage_idx].clone();
    let compute_entropy = stage.lambda_r > 0.0;

    if !core.stage_entered {
        let base_lr = core.config.resolve_learning_rate(core.num_nodes);
        let stage_lr = compute_stage_lr(&stage, core.num_nodes, base_lr, core.config.momentum);
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
            layout.log_map_into(i, j, &mut ws.log_buf)?;
            let mut dist_sq = 0.0_f64;
            for &v in &ws.log_buf {
                dist_sq += v * v;
            }
            let q_unnorm = 1.0 / (1.0 + dist_sq);
            let mult = (core.p_vals[pos] * stage.exaggeration) * q_unnorm;
            for d in 0..dims {
                ws.attr[i_base + d] += mult * (-ws.log_buf[d]);
            }
        }
    }

    backend.compute_repulsion(
        backend_state,
        layout,
        core.config.epsilon_r,
        compute_entropy,
        &mut ws.rep,
        &mut ws.ent,
    )?;

    let n_sq_f64 = (n as f64) * (n as f64);
    let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
    let stage_lr = core.active_stage_lr;
    let momentum = core.config.momentum;
    let use_jacobi = core.config.jacobi_precondition;

    for (i, &inv_j) in core.inv_jacobi.iter().enumerate().take(n) {
        let i_base = i * dims;
        let lr_i = if use_jacobi {
            stage_lr * inv_j
        } else {
            stage_lr
        };
        let mut any_nonzero = false;
        for d in 0..dims {
            let idx = i_base + d;
            let grad_kl = 4.0 * (ws.attr[idx] - ws.rep[idx]);
            let grad_r = if compute_entropy {
                -ws.ent[idx] / n_sq_f64
            } else {
                0.0
            };
            let raw_g = (stage.lambda_kl * grad_kl + stage.lambda_r * grad_r) / lambda_sum;
            let g = if raw_g.is_finite() { raw_g } else { 0.0 };
            let mut v = momentum * ws.vel[idx] - lr_i * g;
            if !v.is_finite() {
                v = 0.0;
            }
            ws.vel[idx] = v;
            ws.delta_buf[d] = v;
            ws.vel_buf[d] = v;
            if v != 0.0 {
                any_nonzero = true;
            }
        }
        if any_nonzero {
            layout.exp_map_and_transport(i, &ws.delta_buf, &mut ws.vel_buf)?;
            for d in 0..dims {
                let transported = ws.vel_buf[d];
                ws.vel[i_base + d] = if transported.is_finite() {
                    transported
                } else {
                    0.0
                };
            }
        }
    }
    Ok(())
}
