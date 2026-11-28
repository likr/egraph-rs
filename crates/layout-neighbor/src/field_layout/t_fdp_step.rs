use egraph_core::{CoreError, CoreResult, Layout2d, LayoutNd, LayoutStorage, Scalar};

use super::repulsion::{TFdpCore, TFdpRepulsion, TFdpSpaceKind, TFdpWorkspace};

#[derive(Debug, Clone)]
pub(super) struct TFdp2dWorkspace {
    pub(super) attr_x: Vec<f64>,
    pub(super) attr_y: Vec<f64>,
    pub(super) vel_x: Vec<f64>,
    pub(super) vel_y: Vec<f64>,
    pub(super) next_x: Vec<f64>,
    pub(super) next_y: Vec<f64>,
    pub(super) rep: Vec<f64>,
}

impl TFdp2dWorkspace {
    pub(super) fn new(num_nodes: usize) -> Self {
        Self {
            attr_x: vec![0.0; num_nodes],
            attr_y: vec![0.0; num_nodes],
            vel_x: vec![0.0; num_nodes],
            vel_y: vec![0.0; num_nodes],
            next_x: vec![0.0; num_nodes],
            next_y: vec![0.0; num_nodes],
            rep: vec![0.0; num_nodes * 2],
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct TFdpNdWorkspace {
    locked_dims: Option<usize>,
    pub(super) attr: Vec<f64>,
    pub(super) vel: Vec<f64>,
    pub(super) next: Vec<f64>,
    pub(super) rep: Vec<f64>,
    mean: Vec<f64>,
}

impl TFdpNdWorkspace {
    pub(super) fn new() -> Self {
        Self {
            locked_dims: None,
            attr: Vec::new(),
            vel: Vec::new(),
            next: Vec::new(),
            rep: Vec::new(),
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
        self.next.resize(total, 0.0);
        self.rep.resize(total, 0.0);
        self.mean.resize(dims, 0.0);
        Ok(())
    }
}

pub(super) fn validate_2d<S: Scalar>(core: &TFdpCore, layout: &Layout2d<S>) -> CoreResult<()> {
    if layout.len() != core.num_nodes {
        return Err(CoreError::dimension_mismatch(
            core.num_nodes,
            layout.len(),
            "TFdpSolver::validate_target",
        ));
    }
    LayoutStorage::validate_finite(layout)
}

pub(super) fn validate_nd<S: Scalar>(
    core: &TFdpCore,
    workspace: &TFdpWorkspace,
    layout: &LayoutNd<S>,
) -> CoreResult<()> {
    if layout.len() != core.num_nodes {
        return Err(CoreError::dimension_mismatch(
            core.num_nodes,
            layout.len(),
            "TFdpSolver::validate_target",
        ));
    }
    let dims = layout.dims();
    if dims == 0 {
        return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
    }
    if let TFdpSpaceKind::EuclideanNd(ws) = &workspace.0
        && let Some(prev) = ws.locked_dims
        && prev != dims
    {
        return Err(CoreError::dimension_mismatch(
            prev,
            dims,
            "TFdpSolver::validate_target::dims",
        ));
    }
    LayoutStorage::validate_finite(layout)
}

pub(super) fn step_2d<S: Scalar, B: TFdpRepulsion<Layout2d<S>>>(
    layout: &mut Layout2d<S>,
    core: &mut TFdpCore,
    workspace: &mut TFdpWorkspace,
    backend: &B,
    backend_state: &mut B::State,
) -> CoreResult<()> {
    let TFdpSpaceKind::Euclidean2d(ws) = &mut workspace.0 else {
        return Err(CoreError::numerical_breakdown(
            "TFdp2dWorkspace variant mismatch",
        ));
    };

    let n = core.num_nodes;
    if core.iteration == 0 {
        let (xs, ys) = layout.xy_mut();
        if xs.iter().all(|&v| v == S::ZERO) && ys.iter().all(|&v| v == S::ZERO) {
            for idx in 0..n {
                xs[idx] = S::from_f64(core.rng.next_f64() * 2.0 - 1.0);
                ys[idx] = S::from_f64(core.rng.next_f64() * 2.0 - 1.0);
            }
        }
    }

    let inv_alpha_sq = 1.0 / (core.config.alpha * core.config.alpha);
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
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                let d2 = dx * dx + dy * dy;
                let mult = (core.edge_weights[pos] as f64) / (1.0 + d2 * inv_alpha_sq);
                fx -= mult * dx;
                fy -= mult * dy;
            }
            ws.attr_x[i] = fx;
            ws.attr_y[i] = fy;
        }
    }

    backend.compute_repulsion(backend_state, layout, core.config.gamma, &mut ws.rep)?;

    let rho = core.current_rho() * core.rep_scale;
    let step_lr = core.config.learning_rate
        * (1.0 - (core.iteration as f64) / (core.total_iterations as f64));
    let momentum = core.config.momentum;
    let max_step = 5.0 * core.config.alpha.max(1.0) * core.config.learning_rate.max(1.0);

    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    {
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let raw_fx = ws.attr_x[i] + rho * ws.rep[2 * i];
            let raw_fy = ws.attr_y[i] + rho * ws.rep[2 * i + 1];
            let fx = if raw_fx.is_finite() { raw_fx } else { 0.0 };
            let fy = if raw_fy.is_finite() { raw_fy } else { 0.0 };

            let mut vx = momentum * ws.vel_x[i] + step_lr * fx;
            let mut vy = momentum * ws.vel_y[i] + step_lr * fy;
            if !vx.is_finite() {
                vx = 0.0;
            }
            if !vy.is_finite() {
                vy = 0.0;
            }
            let v_norm = vx.hypot(vy);
            if v_norm > max_step && v_norm > 0.0 {
                let scale = max_step / v_norm;
                vx *= scale;
                vy *= scale;
            }
            ws.vel_x[i] = vx;
            ws.vel_y[i] = vy;

            let cand_x = xi + vx;
            let cand_y = yi + vy;
            let nx_val = if cand_x.is_finite() { cand_x } else { xi };
            let ny_val = if cand_y.is_finite() { cand_y } else { yi };
            ws.next_x[i] = nx_val;
            ws.next_y[i] = ny_val;
            mean_x += nx_val;
            mean_y += ny_val;
        }
    }

    let inv_n = 1.0 / (n as f64);
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

pub(super) fn step_nd<S: Scalar, B: TFdpRepulsion<LayoutNd<S>>>(
    layout: &mut LayoutNd<S>,
    core: &mut TFdpCore,
    workspace: &mut TFdpWorkspace,
    backend: &B,
    backend_state: &mut B::State,
) -> CoreResult<()> {
    let TFdpSpaceKind::EuclideanNd(ws) = &mut workspace.0 else {
        return Err(CoreError::numerical_breakdown(
            "TFdpNdWorkspace variant mismatch",
        ));
    };

    let n = core.num_nodes;
    let dims = layout.dims();
    ws.ensure_allocated(n, dims)?;

    if core.iteration == 0 {
        let coords = layout.coords_mut();
        if coords.iter().all(|&v| v == S::ZERO) {
            for v in coords.iter_mut() {
                *v = S::from_f64(core.rng.next_f64() * 2.0 - 1.0);
            }
        }
    }

    let rho = core.current_rho() * core.rep_scale;
    let step_lr = core.config.learning_rate
        * (1.0 - (core.iteration as f64) / (core.total_iterations as f64));
    let momentum = core.config.momentum;
    let max_step = 5.0 * core.config.alpha.max(1.0) * core.config.learning_rate.max(1.0);
    let inv_alpha_sq = 1.0 / (core.config.alpha * core.config.alpha);

    {
        let coords = layout.coords();
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
                let mut d2 = 0.0_f64;
                for d in 0..dims {
                    let diff = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                    d2 += diff * diff;
                }
                let mult = (core.edge_weights[pos] as f64) / (1.0 + d2 * inv_alpha_sq);
                for d in 0..dims {
                    let diff = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                    ws.attr[i_base + d] -= mult * diff;
                }
            }
        }
    }

    backend.compute_repulsion(backend_state, layout, core.config.gamma, &mut ws.rep)?;

    ws.mean.fill(0.0);
    {
        let coords = layout.coords();
        for i in 0..n {
            let i_base = i * dims;
            let mut v_sq = 0.0_f64;
            for d in 0..dims {
                let idx = i_base + d;
                let raw_f = ws.attr[idx] + rho * ws.rep[idx];
                let f = if raw_f.is_finite() { raw_f } else { 0.0 };
                let mut v = momentum * ws.vel[idx] + step_lr * f;
                if !v.is_finite() {
                    v = 0.0;
                }
                ws.vel[idx] = v;
                v_sq += v * v;
            }
            let v_norm = v_sq.sqrt();
            if v_norm > max_step && v_norm > 0.0 {
                let scale = max_step / v_norm;
                for d in 0..dims {
                    ws.vel[i_base + d] *= scale;
                }
            }
            for d in 0..dims {
                let idx = i_base + d;
                let xi = coords[idx].to_f64();
                let cand = xi + ws.vel[idx];
                let nx = if cand.is_finite() { cand } else { xi };
                ws.next[idx] = nx;
                ws.mean[d] += nx;
            }
        }
    }

    let inv_n = 1.0 / (n as f64);
    for mean in &mut ws.mean {
        let m = *mean * inv_n;
        *mean = if m.is_finite() { m } else { 0.0 };
    }

    let coords_mut = layout.coords_mut();
    for i in 0..n {
        let i_base = i * dims;
        for d in 0..dims {
            let idx = i_base + d;
            let centered = S::from_f64(ws.next[idx] - ws.mean[d]);
            if centered.is_finite() {
                coords_mut[idx] = centered;
            }
        }
    }
    Ok(())
}
