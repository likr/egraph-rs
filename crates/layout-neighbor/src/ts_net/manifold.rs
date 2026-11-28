use std::f64::consts::TAU;

use egraph_core::{
    CoreError, CoreResult, LayoutStorage, ManifoldKind, ManifoldLayout2d, RiemannianLayout, Scalar,
    cartesian_to_spherical, lorentz_log_map, poincare_log_map, spherical_log_map,
    spherical_to_cartesian, torus_delta,
};

use crate::manifold_utils::{clamp_manifold_tangent_step, guard_spherical_boundary};

use super::config::MIN_LAMBDA_SUM;
use super::repulsion::{TsNetCore, TsNetRepulsion, TsNetSpaceKind, TsNetWorkspace};

/// Geodesic tangent-space scratch buffers for [`TsNetSolver`](super::TsNetSolver) on [`ManifoldLayout2d`].
#[derive(Debug, Clone)]
pub(super) struct TsNetManifoldWorkspace {
    pub(super) locked_kind: Option<ManifoldKind>,
    pub(super) attr_x: Vec<f64>,
    pub(super) attr_y: Vec<f64>,
    pub(super) vel_x: Vec<f64>,
    pub(super) vel_y: Vec<f64>,
    pub(super) rep: Vec<f64>,
    pub(super) ent: Vec<f64>,
}

impl TsNetManifoldWorkspace {
    pub(super) fn new(num_nodes: usize, any_entropy: bool) -> Self {
        let ent_len = if any_entropy { num_nodes * 2 } else { 0 };
        Self {
            locked_kind: None,
            attr_x: vec![0.0; num_nodes],
            attr_y: vec![0.0; num_nodes],
            vel_x: vec![0.0; num_nodes],
            vel_y: vec![0.0; num_nodes],
            rep: vec![0.0; num_nodes * 2],
            ent: vec![0.0; ent_len],
        }
    }
}

pub(super) fn validate_manifold<S: Scalar>(
    core: &TsNetCore,
    workspace: &TsNetWorkspace,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<()> {
    if layout.len() != core.num_nodes {
        return Err(CoreError::dimension_mismatch(
            core.num_nodes,
            layout.len(),
            "TsNetSolver::validate_target",
        ));
    }
    if let TsNetSpaceKind::Manifold2d(ws) = &workspace.0
        && let Some(prev) = ws.locked_kind
        && prev != layout.kind()
    {
        return Err(CoreError::invalid_input(
            "TsNetSolver::validate_target: cannot switch ManifoldKind on stateful solver",
        ));
    }
    LayoutStorage::validate_finite(layout)
}

enum CompressionCenter {
    Origin,
    SphericalMean(Option<(f64, f64)>),
    TorusMean { mx: Option<f64>, my: Option<f64> },
    None,
}

impl CompressionCenter {
    fn compute<S: Scalar>(layout: &ManifoldLayout2d<S>) -> Self {
        let n = layout.len();
        if n == 0 {
            return Self::None;
        }
        let n_f64 = n as f64;
        match layout.kind() {
            ManifoldKind::HyperbolicPoincare | ManifoldKind::HyperbolicLorentz => Self::Origin,
            ManifoldKind::Spherical => {
                let mut sx = 0.0_f64;
                let mut sy = 0.0_f64;
                let mut sz = 0.0_f64;
                for (&lon, &colat) in layout.x().iter().zip(layout.y().iter()) {
                    let p = spherical_to_cartesian(lon.to_f64(), colat.to_f64());
                    sx += p[0];
                    sy += p[1];
                    sz += p[2];
                }
                let norm = (sx * sx + sy * sy + sz * sz).sqrt();
                if norm > 1e-12 * n_f64 && norm.is_finite() {
                    let (mlon, mcolat) = cartesian_to_spherical(sx, sy, sz);
                    Self::SphericalMean(Some((mlon, mcolat)))
                } else {
                    Self::SphericalMean(None)
                }
            }
            ManifoldKind::Torus2d => {
                let mut cx = 0.0_f64;
                let mut sx = 0.0_f64;
                let mut cy = 0.0_f64;
                let mut sy = 0.0_f64;
                for (&x, &y) in layout.x().iter().zip(layout.y().iter()) {
                    let (sin_x, cos_x) = (TAU * x.to_f64()).sin_cos();
                    let (sin_y, cos_y) = (TAU * y.to_f64()).sin_cos();
                    cx += cos_x;
                    sx += sin_x;
                    cy += cos_y;
                    sy += sin_y;
                }
                let mx = if cx.hypot(sx) > 1e-12 * n_f64 {
                    Some((sx.atan2(cx) / TAU).rem_euclid(1.0))
                } else {
                    None
                };
                let my = if cy.hypot(sy) > 1e-12 * n_f64 {
                    Some((sy.atan2(cy) / TAU).rem_euclid(1.0))
                } else {
                    None
                };
                Self::TorusMean { mx, my }
            }
            _ => Self::None,
        }
    }

    fn gradient_at<S: Scalar>(&self, layout: &ManifoldLayout2d<S>, idx: usize) -> (f64, f64) {
        let n_f64 = layout.len() as f64;
        let xi = layout.x()[idx].to_f64();
        let yi = layout.y()[idx].to_f64();
        let scale = layout.scale();
        match (self, layout.kind()) {
            (Self::Origin, ManifoldKind::HyperbolicPoincare) => {
                let (vx, vy) = poincare_log_map(xi, yi, 0.0, 0.0);
                (-scale * vx / n_f64, -scale * vy / n_f64)
            }
            (Self::Origin, ManifoldKind::HyperbolicLorentz) => {
                let (vx, vy) = lorentz_log_map(xi, yi, 0.0, 0.0);
                (-scale * vx / n_f64, -scale * vy / n_f64)
            }
            (Self::SphericalMean(Some((mlon, mcolat))), ManifoldKind::Spherical) => {
                let (vx, vy) = spherical_log_map(xi, yi, *mlon, *mcolat);
                (-scale * vx / n_f64, -scale * vy / n_f64)
            }
            (Self::TorusMean { mx, my }, ManifoldKind::Torus2d) => {
                let lx = scale * layout.aspect();
                let ly = scale;
                let dx = mx.map_or(0.0, |m| torus_delta(m, 0.0, xi, 0.0).0);
                let dy = my.map_or(0.0, |m| torus_delta(0.0, m, 0.0, yi).1);
                (-lx * dx / n_f64, -ly * dy / n_f64)
            }
            _ => (0.0, 0.0),
        }
    }
}

fn compute_attractions_manifold<S: Scalar>(
    core: &TsNetCore,
    ws: &mut TsNetManifoldWorkspace,
    layout: &ManifoldLayout2d<S>,
    exaggeration: f64,
) -> CoreResult<()> {
    let n = core.num_nodes;
    for i in 0..n {
        let start = core.indptr[i] as usize;
        let end = core.indptr[i + 1] as usize;
        let mut ax = 0.0_f64;
        let mut ay = 0.0_f64;
        for pos in start..end {
            let j = core.indices[pos] as usize;
            if j == i {
                continue;
            }
            let (tx_s, ty_s) = layout.tangent_delta(i, j)?;
            let dx = -tx_s.to_f64();
            let dy = -ty_s.to_f64();
            let dist_sq = dx * dx + dy * dy;
            let q_unnorm = 1.0 / (1.0 + dist_sq);
            let mult = (core.p_vals[pos] * exaggeration) * q_unnorm;
            ax += mult * dx;
            ay += mult * dy;
        }
        ws.attr_x[i] = ax;
        ws.attr_y[i] = ay;
    }
    Ok(())
}

pub(super) fn step_manifold<S: Scalar, B: TsNetRepulsion<ManifoldLayout2d<S>>>(
    layout: &mut ManifoldLayout2d<S>,
    core: &mut TsNetCore,
    workspace: &mut TsNetWorkspace,
    backend: &B,
    backend_state: &mut B::State,
) -> CoreResult<()> {
    let TsNetSpaceKind::Manifold2d(ws) = &mut workspace.0 else {
        return Err(CoreError::numerical_breakdown(
            "TsNetManifoldWorkspace variant mismatch",
        ));
    };

    if ws.locked_kind.is_none() {
        ws.locked_kind = Some(layout.kind());
    }

    let stage = core.config.stages[core.stage_idx].clone();
    let compute_entropy = stage.lambda_r > 0.0;

    if !core.stage_entered {
        let stage_lr = core.config.compute_stage_lr(&stage, core.num_nodes);
        if stage_lr < core.prev_stage_lr {
            let vel_scale = stage_lr / core.prev_stage_lr;
            for v in &mut ws.vel_x {
                *v *= vel_scale;
            }
            for v in &mut ws.vel_y {
                *v *= vel_scale;
            }
        }
        core.prev_stage_lr = stage_lr;
        core.active_stage_lr = stage_lr;
        backend.on_stage_enter(backend_state, core.config.epsilon_r, compute_entropy)?;
        core.stage_entered = true;
    }

    compute_attractions_manifold(core, ws, layout, stage.exaggeration)?;
    backend.compute_repulsion(
        backend_state,
        layout,
        core.config.epsilon_r,
        compute_entropy,
        &mut ws.rep,
        &mut ws.ent,
    )?;

    let n = core.num_nodes;
    let n_sq_f64 = (n as f64) * (n as f64);
    let lambda_sum = (stage.lambda_kl + stage.lambda_c + stage.lambda_r).max(MIN_LAMBDA_SUM);
    let stage_lr = core.active_stage_lr;
    let momentum = core.config.momentum;
    let use_jacobi = core.config.jacobi_precondition;
    let center = if stage.lambda_c > 0.0 {
        CompressionCenter::compute(layout)
    } else {
        CompressionCenter::None
    };

    for i in 0..n {
        let grad_kl_x = 4.0 * (ws.attr_x[i] - ws.rep[2 * i]);
        let grad_kl_y = 4.0 * (ws.attr_y[i] - ws.rep[2 * i + 1]);
        let (grad_c_x, grad_c_y) = if stage.lambda_c > 0.0 {
            center.gradient_at(layout, i)
        } else {
            (0.0, 0.0)
        };
        let (grad_r_x, grad_r_y) = if compute_entropy {
            (-ws.ent[2 * i] / n_sq_f64, -ws.ent[2 * i + 1] / n_sq_f64)
        } else {
            (0.0, 0.0)
        };

        let raw_gx =
            (stage.lambda_kl * grad_kl_x + stage.lambda_c * grad_c_x + stage.lambda_r * grad_r_x)
                / lambda_sum;
        let raw_gy =
            (stage.lambda_kl * grad_kl_y + stage.lambda_c * grad_c_y + stage.lambda_r * grad_r_y)
                / lambda_sum;
        let gx = if raw_gx.is_finite() { raw_gx } else { 0.0 };
        let gy = if raw_gy.is_finite() { raw_gy } else { 0.0 };

        let lr_i = if use_jacobi {
            stage_lr * core.inv_jacobi[i]
        } else {
            stage_lr
        };
        let raw_vx = momentum * ws.vel_x[i] - lr_i * gx;
        let raw_vy = momentum * ws.vel_y[i] - lr_i * gy;
        let (cvx, cvy) = clamp_manifold_tangent_step(layout, raw_vx, raw_vy);
        ws.vel_x[i] = cvx;
        ws.vel_y[i] = cvy;
    }

    for i in 0..n {
        let vx = ws.vel_x[i];
        let vy = ws.vel_y[i];
        if vx != 0.0 || vy != 0.0 {
            let delta = [vx, vy];
            let mut vel = [vx, vy];
            layout.exp_map_and_transport(i, &delta, &mut vel)?;
            guard_spherical_boundary(layout, i)?;
            ws.vel_x[i] = if vel[0].is_finite() { vel[0] } else { 0.0 };
            ws.vel_y[i] = if vel[1].is_finite() { vel[1] } else { 0.0 };
        }
    }

    Ok(())
}
