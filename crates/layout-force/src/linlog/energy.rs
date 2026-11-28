use egraph_core::{CoreResult, EuclideanLayout, EuclideanStorageRef, Scalar};

use super::solver::LinLogState;

pub(super) fn compute_energy<L: EuclideanLayout>(
    state: &LinLogState,
    layout: &L,
) -> CoreResult<f64> {
    let n = state.num_nodes;
    if n <= 1 {
        return Ok(0.0);
    }

    let a_plus_1 = state.config.attraction_exponent + 1.0;
    let r_plus_1 = state.config.repulsion_exponent + 1.0;

    match layout.as_euclidean_storage() {
        EuclideanStorageRef::Soa2d { x, y } => compute_energy_2d(state, x, y, a_plus_1, r_plus_1),
        EuclideanStorageRef::Soa3d { x, y, z } => {
            compute_energy_3d(state, x, y, z, a_plus_1, r_plus_1)
        }
        EuclideanStorageRef::FlatNd { coords, dims, .. } => {
            compute_energy_flat_nd(state, coords, dims, a_plus_1, r_plus_1)
        }
    }
}

#[inline]
fn polylog_term(d: f64, exp_plus_1: f64) -> f64 {
    let d_safe = d.max(1e-12);
    if exp_plus_1.abs() <= 1e-12 {
        d_safe.ln()
    } else {
        d_safe.powf(exp_plus_1) / exp_plus_1
    }
}

fn compute_energy_2d<S: Scalar>(
    state: &LinLogState,
    xs: &[S],
    ys: &[S],
    a_plus_1: f64,
    r_plus_1: f64,
) -> CoreResult<f64> {
    let n = state.num_nodes;
    let mut u_att = 0.0_f64;
    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let d = (xs[u].to_f64() - xs[v].to_f64()).hypot(ys[u].to_f64() - ys[v].to_f64());
        u_att += w * polylog_term(d, a_plus_1);
    }

    let mut u_rep = 0.0_f64;
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let mi = state.masses[i];
        for j in (i + 1)..n {
            let d = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            u_rep -= mi * state.masses[j] * polylog_term(d, r_plus_1);
        }
    }

    let mut u_grav = 0.0_f64;
    if state.config.gravity > 0.0 && state.total_mass > 0.0 {
        let mut bx = 0.0_f64;
        let mut by = 0.0_f64;
        for i in 0..n {
            bx += state.masses[i] * xs[i].to_f64();
            by += state.masses[i] * ys[i].to_f64();
        }
        bx /= state.total_mass;
        by /= state.total_mass;
        for i in 0..n {
            let d = (xs[i].to_f64() - bx).hypot(ys[i].to_f64() - by);
            u_grav += state.config.gravity * state.masses[i] * polylog_term(d, a_plus_1);
        }
    }

    Ok(u_att + u_rep + u_grav)
}

fn compute_energy_3d<S: Scalar>(
    state: &LinLogState,
    xs: &[S],
    ys: &[S],
    zs: &[S],
    a_plus_1: f64,
    r_plus_1: f64,
) -> CoreResult<f64> {
    let n = state.num_nodes;
    let mut u_att = 0.0_f64;
    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let dx = xs[u].to_f64() - xs[v].to_f64();
        let dy = ys[u].to_f64() - ys[v].to_f64();
        let dz = zs[u].to_f64() - zs[v].to_f64();
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        u_att += w * polylog_term(d, a_plus_1);
    }

    let mut u_rep = 0.0_f64;
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let zi = zs[i].to_f64();
        let mi = state.masses[i];
        for j in (i + 1)..n {
            let dx = xi - xs[j].to_f64();
            let dy = yi - ys[j].to_f64();
            let dz = zi - zs[j].to_f64();
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            u_rep -= mi * state.masses[j] * polylog_term(d, r_plus_1);
        }
    }

    let mut u_grav = 0.0_f64;
    if state.config.gravity > 0.0 && state.total_mass > 0.0 {
        let mut bx = 0.0_f64;
        let mut by = 0.0_f64;
        let mut bz = 0.0_f64;
        for i in 0..n {
            let m = state.masses[i];
            bx += m * xs[i].to_f64();
            by += m * ys[i].to_f64();
            bz += m * zs[i].to_f64();
        }
        bx /= state.total_mass;
        by /= state.total_mass;
        bz /= state.total_mass;
        for i in 0..n {
            let dx = xs[i].to_f64() - bx;
            let dy = ys[i].to_f64() - by;
            let dz = zs[i].to_f64() - bz;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            u_grav += state.config.gravity * state.masses[i] * polylog_term(d, a_plus_1);
        }
    }

    Ok(u_att + u_rep + u_grav)
}

fn compute_energy_flat_nd<S: Scalar>(
    state: &LinLogState,
    coords: &[S],
    dims: usize,
    a_plus_1: f64,
    r_plus_1: f64,
) -> CoreResult<f64> {
    let n = state.num_nodes;
    let mut u_att = 0.0_f64;
    for idx in 0..state.edges_u.len() {
        let u = state.edges_u[idx] as usize;
        let v = state.edges_v[idx] as usize;
        let w = state.edges_w[idx];
        if w == 0.0 {
            continue;
        }
        let base_u = u * dims;
        let base_v = v * dims;
        let mut d_sq = 0.0_f64;
        for axis in 0..dims {
            let delta = coords[base_u + axis].to_f64() - coords[base_v + axis].to_f64();
            d_sq += delta * delta;
        }
        u_att += w * polylog_term(d_sq.sqrt(), a_plus_1);
    }

    let mut u_rep = 0.0_f64;
    for i in 0..n {
        let base_i = i * dims;
        let mi = state.masses[i];
        for (j, &mj) in state.masses.iter().enumerate().take(n).skip(i + 1) {
            let base_j = j * dims;
            let mut d_sq = 0.0_f64;
            for axis in 0..dims {
                let delta = coords[base_i + axis].to_f64() - coords[base_j + axis].to_f64();
                d_sq += delta * delta;
            }
            u_rep -= mi * mj * polylog_term(d_sq.sqrt(), r_plus_1);
        }
    }

    let mut u_grav = 0.0_f64;
    if state.config.gravity > 0.0 && state.total_mass > 0.0 {
        let mut barycenter = vec![0.0_f64; dims];
        for (i, &m) in state.masses.iter().enumerate().take(n) {
            let base = i * dims;
            for axis in 0..dims {
                barycenter[axis] += m * coords[base + axis].to_f64();
            }
        }
        for b in &mut barycenter {
            *b /= state.total_mass;
        }
        for i in 0..n {
            let base = i * dims;
            let mut d_sq = 0.0_f64;
            for axis in 0..dims {
                let delta = coords[base + axis].to_f64() - barycenter[axis];
                d_sq += delta * delta;
            }
            u_grav += state.config.gravity * state.masses[i] * polylog_term(d_sq.sqrt(), a_plus_1);
        }
    }

    Ok(u_att + u_rep + u_grav)
}
