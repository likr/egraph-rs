use egraph_core::{CoreError, CoreResult, Layout2d, LayoutStorage, Scalar};

use super::ib_umap_repulsion::{IbUmapCore, IbUmapRepulsion, IbUmapSpaceKind, IbUmapWorkspace};

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;

#[derive(Debug, Clone)]
pub(super) struct IbUmap2dWorkspace {
    pub(super) attr_x: Vec<f64>,
    pub(super) attr_y: Vec<f64>,
    pub(super) next_x: Vec<f64>,
    pub(super) next_y: Vec<f64>,
    pub(super) rep: Vec<f64>,
}

impl IbUmap2dWorkspace {
    pub(super) fn new(num_nodes: usize) -> Self {
        Self {
            attr_x: vec![0.0; num_nodes],
            attr_y: vec![0.0; num_nodes],
            next_x: vec![0.0; num_nodes],
            next_y: vec![0.0; num_nodes],
            rep: vec![0.0; num_nodes * 2],
        }
    }
}

pub(super) fn validate_2d<S: Scalar>(core: &IbUmapCore, layout: &Layout2d<S>) -> CoreResult<()> {
    if layout.len() != core.num_nodes {
        return Err(CoreError::dimension_mismatch(
            core.num_nodes,
            layout.len(),
            "IbUmapSolver::validate_target",
        ));
    }
    LayoutStorage::validate_finite(layout)
}

pub(super) fn step_2d<S: Scalar, B: IbUmapRepulsion<Layout2d<S>>>(
    layout: &mut Layout2d<S>,
    core: &mut IbUmapCore,
    workspace: &mut IbUmapWorkspace,
    backend: &B,
    backend_state: &mut B::State,
) -> CoreResult<()> {
    let IbUmapSpaceKind::Euclidean2d(ws) = &mut workspace.0;

    let n = core.num_nodes;
    if core.iteration == 0 {
        let (xs, ys) = layout.xy_mut();
        if xs.iter().all(|&v| v == S::ZERO) && ys.iter().all(|&v| v == S::ZERO) {
            for idx in 0..n {
                xs[idx] = S::from_f64(core.rng.next_f64() * 20.0 - 10.0);
                ys[idx] = S::from_f64(core.rng.next_f64() * 20.0 - 10.0);
            }
        }
    }

    let two_ab = 2.0 * core.a * core.b;
    let a = core.a;
    let b = core.b;
    {
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let start = core.indptr[i] as usize;
            let end = core.indptr[i + 1] as usize;
            let mut fx = 0.0_f64;
            let mut fy = 0.0_f64;
            for pos in start..end {
                let j = core.indices[pos] as usize;
                if j == i {
                    continue;
                }
                let p_ij = core.edge_weights[pos] as f64;
                if p_ij <= 0.0 {
                    continue;
                }
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                let d2 = (dx * dx + dy * dy).max(MIN_DIST_SQ_CLAMP);
                let d_2b = d2.powf(b);
                let d_pow = d_2b / d2;
                let coeff = -two_ab * p_ij * d_pow / (1.0 + a * d_2b);
                if coeff.is_finite() {
                    fx += coeff * dx;
                    fy += coeff * dy;
                }
            }
            ws.attr_x[i] = fx;
            ws.attr_y[i] = fy;
        }
    }

    backend.compute_repulsion(backend_state, layout, a, b, core.config.c_k, &mut ws.rep)?;

    let alpha_t = core.config.learning_rate
        * (1.0 - (core.iteration as f64) / (core.total_iterations as f64));
    let tau_t = core.config.tau_scale * alpha_t;
    let inv_n = 1.0 / (n as f64);

    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    {
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let raw_fx = ws.attr_x[i] + ws.rep[2 * i] * inv_n;
            let raw_fy = ws.attr_y[i] + ws.rep[2 * i + 1] * inv_n;
            let fx = if raw_fx.is_finite() { raw_fx } else { 0.0 };
            let fy = if raw_fy.is_finite() { raw_fy } else { 0.0 };

            let step_scale = alpha_t * core.hub_eta[i];
            let mut dx = step_scale * fx;
            let mut dy = step_scale * fy;
            if !dx.is_finite() {
                dx = 0.0;
            }
            if !dy.is_finite() {
                dy = 0.0;
            }
            let norm = dx.hypot(dy);
            if norm > tau_t && norm > 0.0 {
                let clip = tau_t / norm;
                dx *= clip;
                dy *= clip;
            }

            let cand_x = xi + dx;
            let cand_y = yi + dy;
            let nx_val = if cand_x.is_finite() { cand_x } else { xi };
            let ny_val = if cand_y.is_finite() { cand_y } else { yi };
            ws.next_x[i] = nx_val;
            ws.next_y[i] = ny_val;
            mean_x += nx_val;
            mean_y += ny_val;
        }
    }

    mean_x *= inv_n;
    mean_y *= inv_n;
    let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
    let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };

    let (xs_mut, ys_mut) = layout.xy_mut();
    for i in 0..n {
        let cx = S::from_f64(ws.next_x[i] - shift_x);
        let cy = S::from_f64(ws.next_y[i] - shift_y);
        if cx.is_finite() {
            xs_mut[i] = cx;
        }
        if cy.is_finite() {
            ys_mut[i] = cy;
        }
    }
    Ok(())
}
