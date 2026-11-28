use egraph_core::{
    CoreError, CoreResult, DeterministicRng, EuclideanLayout, EuclideanStorageMut,
    EuclideanStorageRef, Scalar,
};

use super::repulsion::{ForceRepulsion, ForceRepulsionParams};
use super::solver::{COULOMB_EPS_SQ, ForceAtlas2State, MIN_FA2_SPEED};

#[derive(Debug, Clone)]
pub(super) struct Fa2Buf3d {
    forces: Vec<f64>,
    prev_forces: Vec<f64>,
}

impl Fa2Buf3d {
    pub(super) fn new(num_nodes: usize) -> Self {
        let len = num_nodes * 3;
        Self {
            forces: vec![0.0; len],
            prev_forces: vec![0.0; len],
        }
    }
}

pub(super) fn step_3d<L: EuclideanLayout, B: ForceRepulsion<L>>(
    state: &mut ForceAtlas2State,
    buf: &mut Fa2Buf3d,
    backend: &B,
    rep_state: &mut B::State,
    layout: &mut L,
) -> CoreResult<()> {
    let n = state.num_nodes;
    let mut staged: Option<L> = None;
    if state.iter == 0
        && let EuclideanStorageRef::Soa3d { x, y, z } = layout.as_euclidean_storage()
    {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;
        for i in 0..n {
            let xf = x[i].to_f64();
            let yf = y[i].to_f64();
            let zf = z[i].to_f64();
            min_x = min_x.min(xf);
            max_x = max_x.max(xf);
            min_y = min_y.min(yf);
            max_y = max_y.max(yf);
            min_z = min_z.min(zf);
            max_z = max_z.max(zf);
        }
        if (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 && (max_z - min_z) < 1e-12 {
            let mut copy = layout.zeros_like(n)?;
            let mut rng = DeterministicRng::from_seed_u64(state.config.seed);
            if let EuclideanStorageMut::Soa3d {
                x: sx,
                y: sy,
                z: sz,
            } = copy.as_euclidean_storage_mut()
            {
                for i in 0..n {
                    let nx = L::Scalar::from_f64(x[i].to_f64() + 1e-4 * (rng.next_f64() - 0.5));
                    let ny = L::Scalar::from_f64(y[i].to_f64() + 1e-4 * (rng.next_f64() - 0.5));
                    let nz = L::Scalar::from_f64(z[i].to_f64() + 1e-4 * (rng.next_f64() - 0.5));
                    if !nx.is_finite() || !ny.is_finite() || !nz.is_finite() {
                        return Err(CoreError::numerical_breakdown(
                            "ForceAtlas2Solver 3D scatter coordinates became non-finite",
                        ));
                    }
                    sx[i] = nx;
                    sy[i] = ny;
                    sz[i] = nz;
                }
            }
            staged = Some(copy);
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
    backend.accumulate_repulsion(
        rep_state,
        staged.as_ref().unwrap_or(layout),
        &params,
        &mut buf.forces,
    )?;

    let dest_layout = staged.as_mut().unwrap_or(layout);
    let EuclideanStorageMut::Soa3d {
        x: xs_mut,
        y: ys_mut,
        z: zs_mut,
    } = dest_layout.as_euclidean_storage_mut()
    else {
        return Err(CoreError::numerical_breakdown(
            "ForceAtlas2Solver 3D storage mismatch",
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
        let dz = zs_mut[v].to_f64() - zs_mut[u].to_f64();
        let factor = if prevent_overlap {
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
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
            let d_sq = dx * dx + dy * dy + dz * dz;
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
        let az = factor * dz;
        buf.forces[3 * u] += ax;
        buf.forces[3 * u + 1] += ay;
        buf.forces[3 * u + 2] += az;
        buf.forces[3 * v] -= ax;
        buf.forces[3 * v + 1] -= ay;
        buf.forces[3 * v + 2] -= az;
    }

    if state.config.gravity > 0.0 {
        let g = state.config.gravity;
        if state.config.strong_gravity {
            for i in 0..n {
                let gf = g * state.masses[i];
                buf.forces[3 * i] -= gf * xs_mut[i].to_f64();
                buf.forces[3 * i + 1] -= gf * ys_mut[i].to_f64();
                buf.forces[3 * i + 2] -= gf * zs_mut[i].to_f64();
            }
        } else {
            for i in 0..n {
                let xi = xs_mut[i].to_f64();
                let yi = ys_mut[i].to_f64();
                let zi = zs_mut[i].to_f64();
                let d = (xi * xi + yi * yi + zi * zi).sqrt();
                if d > 1e-12 {
                    let gf = (g * state.masses[i]) / d;
                    buf.forces[3 * i] -= gf * xi;
                    buf.forces[3 * i + 1] -= gf * yi;
                    buf.forces[3 * i + 2] -= gf * zi;
                }
            }
        }
    }

    let mut total_swing = 0.0_f64;
    let mut total_traction = 0.0_f64;
    for i in 0..n {
        let fxi = if buf.forces[3 * i].is_finite() {
            buf.forces[3 * i]
        } else {
            0.0
        };
        let fyi = if buf.forces[3 * i + 1].is_finite() {
            buf.forces[3 * i + 1]
        } else {
            0.0
        };
        let fzi = if buf.forces[3 * i + 2].is_finite() {
            buf.forces[3 * i + 2]
        } else {
            0.0
        };
        buf.forces[3 * i] = fxi;
        buf.forces[3 * i + 1] = fyi;
        buf.forces[3 * i + 2] = fzi;

        let pfx = buf.prev_forces[3 * i];
        let pfy = buf.prev_forces[3 * i + 1];
        let pfz = buf.prev_forces[3 * i + 2];
        let dx_s = fxi - pfx;
        let dy_s = fyi - pfy;
        let dz_s = fzi - pfz;
        let swg = (dx_s * dx_s + dy_s * dy_s + dz_s * dz_s).sqrt();
        let dx_t = fxi + pfx;
        let dy_t = fyi + pfy;
        let dz_t = fzi + pfz;
        let tra = 0.5 * (dx_t * dx_t + dy_t * dy_t + dz_t * dz_t).sqrt();
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
        let fxi = buf.forces[3 * i];
        let fyi = buf.forces[3 * i + 1];
        let fzi = buf.forces[3 * i + 2];
        let local_factor = speed_scale * (speed / (1.0 + (speed * state.swing_buf[i]).sqrt()));
        let mut dx = fxi * local_factor;
        let mut dy = fyi * local_factor;
        let mut dz = fzi * local_factor;
        let disp = (dx * dx + dy * dy + dz * dz).sqrt();
        if disp > max_disp {
            let cap = max_disp / disp;
            dx *= cap;
            dy *= cap;
            dz *= cap;
        }
        let nx = L::Scalar::from_f64(xs_mut[i].to_f64() + dx);
        let ny = L::Scalar::from_f64(ys_mut[i].to_f64() + dy);
        let nz = L::Scalar::from_f64(zs_mut[i].to_f64() + dz);
        if !nx.is_finite() || !ny.is_finite() || !nz.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "ForceAtlas2Solver 3D coordinates became non-finite",
            ));
        }
        xs_mut[i] = nx;
        ys_mut[i] = ny;
        zs_mut[i] = nz;
        buf.prev_forces[3 * i] = fxi;
        buf.prev_forces[3 * i + 1] = fyi;
        buf.prev_forces[3 * i + 2] = fzi;
    }

    if let Some(copy) = staged
        && let (
            EuclideanStorageRef::Soa3d {
                x: sx,
                y: sy,
                z: sz,
            },
            EuclideanStorageMut::Soa3d {
                x: dx,
                y: dy,
                z: dz,
            },
        ) = (
            copy.as_euclidean_storage(),
            layout.as_euclidean_storage_mut(),
        )
    {
        dx.copy_from_slice(sx);
        dy.copy_from_slice(sy);
        dz.copy_from_slice(sz);
    }

    Ok(())
}
