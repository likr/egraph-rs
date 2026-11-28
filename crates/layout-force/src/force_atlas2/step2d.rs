use egraph_core::{
    CoreError, CoreResult, DeterministicRng, EuclideanLayout, EuclideanStorageMut, Scalar,
};

use super::repulsion::{ForceRepulsion, ForceRepulsionParams};
use super::solver::{COULOMB_EPS_SQ, ForceAtlas2State, MIN_FA2_SPEED};

#[derive(Debug, Clone)]
pub(super) struct Fa2Buf2d {
    forces: Vec<f64>,
    prev_forces: Vec<f64>,
}

impl Fa2Buf2d {
    pub(super) fn new(num_nodes: usize) -> Self {
        let len = num_nodes * 2;
        Self {
            forces: vec![0.0; len],
            prev_forces: vec![0.0; len],
        }
    }
}

pub(super) fn step_2d<L: EuclideanLayout, B: ForceRepulsion<L>>(
    state: &mut ForceAtlas2State,
    buf: &mut Fa2Buf2d,
    backend: &B,
    rep_state: &mut B::State,
    layout: &mut L,
) -> CoreResult<()> {
    let n = state.num_nodes;
    if state.iter == 0
        && let EuclideanStorageMut::Soa2d { x, y } = layout.as_euclidean_storage_mut()
    {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for i in 0..n {
            let xf = x[i].to_f64();
            let yf = y[i].to_f64();
            min_x = min_x.min(xf);
            max_x = max_x.max(xf);
            min_y = min_y.min(yf);
            max_y = max_y.max(yf);
        }
        if (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
            let mut rng = DeterministicRng::from_seed_u64(state.config.seed);
            for i in 0..n {
                let nx = L::Scalar::from_f64(x[i].to_f64() + 1e-4 * (rng.next_f64() - 0.5));
                let ny = L::Scalar::from_f64(y[i].to_f64() + 1e-4 * (rng.next_f64() - 0.5));
                if !nx.is_finite() || !ny.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "ForceAtlas2Solver 2D scatter coordinates became non-finite",
                    ));
                }
                x[i] = nx;
                y[i] = ny;
            }
        }
    }

    buf.forces.fill(0.0);
    let params = ForceRepulsionParams::new(
        &state.masses,
        &state.node_radii,
        state.config.scaling_ratio,
        COULOMB_EPS_SQ,
        state.config.prevent_overlap,
        state.config.seed,
        state.iter,
    );
    backend.accumulate_repulsion(rep_state, layout, &params, &mut buf.forces)?;

    let EuclideanStorageMut::Soa2d {
        x: xs_mut,
        y: ys_mut,
    } = layout.as_euclidean_storage_mut()
    else {
        return Err(CoreError::numerical_breakdown(
            "ForceAtlas2Solver 2D storage mismatch",
        ));
    };

    let prevent_overlap = state.config.prevent_overlap;
    let linlog = state.config.linlog_mode;
    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let dx = xs_mut[v].to_f64() - xs_mut[u].to_f64();
        let dy = ys_mut[v].to_f64() - ys_mut[u].to_f64();
        let factor = if prevent_overlap {
            let d = dx.hypot(dy);
            let d_prime = d - (state.node_radii[u] + state.node_radii[v]);
            if d_prime > 0.0 && d > 1e-12 {
                if linlog {
                    w * d_prime.ln_1p() / d
                } else {
                    w * d_prime / d
                }
            } else {
                0.0
            }
        } else if linlog {
            let d_sq = dx * dx + dy * dy;
            if d_sq > 1e-24 {
                let d = d_sq.sqrt();
                w * d.ln_1p() / d
            } else {
                0.0
            }
        } else {
            w
        };
        if factor == 0.0 {
            continue;
        }
        let ax = factor * dx;
        let ay = factor * dy;
        buf.forces[2 * u] += ax;
        buf.forces[2 * u + 1] += ay;
        buf.forces[2 * v] -= ax;
        buf.forces[2 * v + 1] -= ay;
    }

    if state.config.gravity > 0.0 {
        let g = state.config.gravity;
        if state.config.strong_gravity {
            for i in 0..n {
                let gf = g * state.masses[i];
                buf.forces[2 * i] -= gf * xs_mut[i].to_f64();
                buf.forces[2 * i + 1] -= gf * ys_mut[i].to_f64();
            }
        } else {
            for i in 0..n {
                let xi = xs_mut[i].to_f64();
                let yi = ys_mut[i].to_f64();
                let d = xi.hypot(yi);
                if d > 1e-12 {
                    let gf = (g * state.masses[i]) / d;
                    buf.forces[2 * i] -= gf * xi;
                    buf.forces[2 * i + 1] -= gf * yi;
                }
            }
        }
    }

    let mut total_swing = 0.0_f64;
    let mut total_traction = 0.0_f64;
    for i in 0..n {
        let fxi = if buf.forces[2 * i].is_finite() {
            buf.forces[2 * i]
        } else {
            0.0
        };
        let fyi = if buf.forces[2 * i + 1].is_finite() {
            buf.forces[2 * i + 1]
        } else {
            0.0
        };
        buf.forces[2 * i] = fxi;
        buf.forces[2 * i + 1] = fyi;

        let pfx = buf.prev_forces[2 * i];
        let pfy = buf.prev_forces[2 * i + 1];
        let swg = (fxi - pfx).hypot(fyi - pfy);
        let tra = 0.5 * (fxi + pfx).hypot(fyi + pfy);
        state.swing_buf[i] = swg;
        let m = state.masses[i];
        total_swing += m * swg;
        total_traction += m * tra;
    }

    let jt = state.config.jitter_tolerance;
    let max_speed = 10.0 * jt.max(1.0);
    let target_speed = if total_swing > 1e-12 {
        (jt * jt * total_traction / total_swing).clamp(MIN_FA2_SPEED, max_speed)
    } else {
        (state.global_speed * 1.5).min(max_speed)
    };
    if state.iter > 0 {
        let max_rise = 0.5 * state.global_speed;
        state.global_speed += (target_speed - state.global_speed).min(max_rise);
    } else {
        state.global_speed = target_speed;
    }
    state.global_speed = state.global_speed.clamp(MIN_FA2_SPEED, max_speed);

    let speed = state.global_speed;
    let speed_scale = if prevent_overlap { 0.1 } else { 1.0 };
    let max_disp = state.max_displacement;
    for i in 0..n {
        let fxi = buf.forces[2 * i];
        let fyi = buf.forces[2 * i + 1];
        let local_factor = speed_scale * (speed / (1.0 + (speed * state.swing_buf[i]).sqrt()));
        let mut dx = fxi * local_factor;
        let mut dy = fyi * local_factor;
        let disp = dx.hypot(dy);
        if disp > max_disp {
            let cap = max_disp / disp;
            dx *= cap;
            dy *= cap;
        }
        let nx = L::Scalar::from_f64(xs_mut[i].to_f64() + dx);
        let ny = L::Scalar::from_f64(ys_mut[i].to_f64() + dy);
        if !nx.is_finite() || !ny.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "ForceAtlas2Solver 2D coordinates became non-finite",
            ));
        }
        xs_mut[i] = nx;
        ys_mut[i] = ny;
        buf.prev_forces[2 * i] = fxi;
        buf.prev_forces[2 * i + 1] = fyi;
    }

    Ok(())
}
