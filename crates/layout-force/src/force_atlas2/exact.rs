use egraph_core::Scalar;

use super::overlap_quadtree::{OVERLAP_DPRIME_MIN, OVERLAP_REPULSION_K, overlap_repulsion_pair};
use super::repulsion::ForceRepulsionParams;

pub(super) fn exact_2d<S: Scalar>(
    x: &[S],
    y: &[S],
    params: &ForceRepulsionParams<'_>,
    forces: &mut [f64],
) {
    let n = x.len();
    let sr = params.scaling_ratio;
    if params.prevent_overlap {
        let seed = params.seed;
        let iter = params.iter;
        for i in 0..n {
            let xi = x[i].to_f64();
            let yi = y[i].to_f64();
            let mi = params.masses[i];
            let ri = params.node_radii[i];
            let mut acc_x = 0.0_f64;
            let mut acc_y = 0.0_f64;
            for j in (i + 1)..n {
                let (fx_ij, fy_ij) = overlap_repulsion_pair(
                    xi - x[j].to_f64(),
                    yi - y[j].to_f64(),
                    mi,
                    params.masses[j],
                    ri + params.node_radii[j],
                    sr,
                    seed,
                    iter,
                    i,
                    j,
                );
                acc_x += fx_ij;
                acc_y += fy_ij;
                forces[2 * j] -= fx_ij;
                forces[2 * j + 1] -= fy_ij;
            }
            forces[2 * i] += acc_x;
            forces[2 * i + 1] += acc_y;
        }
    } else {
        let eps_sq = params.eps_sq;
        for i in 0..n {
            let xi = x[i].to_f64();
            let yi = y[i].to_f64();
            let mi_sr = sr * params.masses[i];
            let mut acc_x = 0.0_f64;
            let mut acc_y = 0.0_f64;
            for j in (i + 1)..n {
                let dx = xi - x[j].to_f64();
                let dy = yi - y[j].to_f64();
                let mult = (mi_sr * params.masses[j]) / (dx * dx + dy * dy + eps_sq);
                let fx_ij = mult * dx;
                let fy_ij = mult * dy;
                acc_x += fx_ij;
                acc_y += fy_ij;
                forces[2 * j] -= fx_ij;
                forces[2 * j + 1] -= fy_ij;
            }
            forces[2 * i] += acc_x;
            forces[2 * i + 1] += acc_y;
        }
    }
}

pub(super) fn exact_3d<S: Scalar>(
    x: &[S],
    y: &[S],
    z: &[S],
    params: &ForceRepulsionParams<'_>,
    forces: &mut [f64],
) {
    let n = x.len();
    let sr = params.scaling_ratio;
    if params.prevent_overlap {
        for i in 0..n {
            let xi = x[i].to_f64();
            let yi = y[i].to_f64();
            let zi = z[i].to_f64();
            let mi = params.masses[i];
            let ri = params.node_radii[i];
            for j in (i + 1)..n {
                let mj = params.masses[j];
                let r_sum = ri + params.node_radii[j];
                let dx = xi - x[j].to_f64();
                let dy = yi - y[j].to_f64();
                let dz = zi - z[j].to_f64();
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let d_prime = dist - r_sum;
                if d_prime > 0.0 {
                    let d_prime_safe = d_prime.max(OVERLAP_DPRIME_MIN);
                    let mult = (sr * mi * mj) / (dist * d_prime_safe);
                    let fx = mult * dx;
                    let fy = mult * dy;
                    let fz = mult * dz;
                    forces[3 * i] += fx;
                    forces[3 * i + 1] += fy;
                    forces[3 * i + 2] += fz;
                    forces[3 * j] -= fx;
                    forces[3 * j + 1] -= fy;
                    forces[3 * j + 2] -= fz;
                } else if d_prime < 0.0 {
                    let mag = OVERLAP_REPULSION_K * mi * mj;
                    if dist > 1e-12 {
                        let scale = mag / dist;
                        let fx = scale * dx;
                        let fy = scale * dy;
                        let fz = scale * dz;
                        forces[3 * i] += fx;
                        forces[3 * i + 1] += fy;
                        forces[3 * i + 2] += fz;
                        forces[3 * j] -= fx;
                        forces[3 * j + 1] -= fy;
                        forces[3 * j + 2] -= fz;
                    } else {
                        forces[3 * i] += mag;
                        forces[3 * j] -= mag;
                    }
                }
            }
        }
    } else {
        let eps_sq = params.eps_sq;
        for i in 0..n {
            let xi = x[i].to_f64();
            let yi = y[i].to_f64();
            let zi = z[i].to_f64();
            let mi_sr = sr * params.masses[i];
            let mut acc_x = 0.0_f64;
            let mut acc_y = 0.0_f64;
            let mut acc_z = 0.0_f64;
            for j in (i + 1)..n {
                let dx = xi - x[j].to_f64();
                let dy = yi - y[j].to_f64();
                let dz = zi - z[j].to_f64();
                let mult = (mi_sr * params.masses[j]) / (dx * dx + dy * dy + dz * dz + eps_sq);
                let fx_ij = mult * dx;
                let fy_ij = mult * dy;
                let fz_ij = mult * dz;
                acc_x += fx_ij;
                acc_y += fy_ij;
                acc_z += fz_ij;
                forces[3 * j] -= fx_ij;
                forces[3 * j + 1] -= fy_ij;
                forces[3 * j + 2] -= fz_ij;
            }
            forces[3 * i] += acc_x;
            forces[3 * i + 1] += acc_y;
            forces[3 * i + 2] += acc_z;
        }
    }
}

pub(super) fn exact_nd<S: Scalar>(
    coords: &[S],
    n: usize,
    dims: usize,
    params: &ForceRepulsionParams<'_>,
    forces: &mut [f64],
) {
    let sr = params.scaling_ratio;
    if params.prevent_overlap {
        for i in 0..n {
            let base_i = i * dims;
            let mi = params.masses[i];
            let ri = params.node_radii[i];
            for j in (i + 1)..n {
                let base_j = j * dims;
                let mj = params.masses[j];
                let r_sum = ri + params.node_radii[j];
                let mut dist_sq = 0.0_f64;
                for axis in 0..dims {
                    let delta = coords[base_i + axis].to_f64() - coords[base_j + axis].to_f64();
                    dist_sq += delta * delta;
                }
                let dist = dist_sq.sqrt();
                let d_prime = dist - r_sum;
                if d_prime > 0.0 {
                    let d_prime_safe = d_prime.max(OVERLAP_DPRIME_MIN);
                    let mult = (sr * mi * mj) / (dist * d_prime_safe);
                    for axis in 0..dims {
                        let f = mult
                            * (coords[base_i + axis].to_f64() - coords[base_j + axis].to_f64());
                        forces[base_i + axis] += f;
                        forces[base_j + axis] -= f;
                    }
                } else if d_prime < 0.0 {
                    let mag = OVERLAP_REPULSION_K * mi * mj;
                    if dist > 1e-12 {
                        let scale = mag / dist;
                        for axis in 0..dims {
                            let f = scale
                                * (coords[base_i + axis].to_f64() - coords[base_j + axis].to_f64());
                            forces[base_i + axis] += f;
                            forces[base_j + axis] -= f;
                        }
                    } else {
                        forces[base_i] += mag;
                        forces[base_j] -= mag;
                    }
                }
            }
        }
    } else {
        let eps_sq = params.eps_sq;
        for i in 0..n {
            let base_i = i * dims;
            let mi_sr = sr * params.masses[i];
            for (j, &mj) in params.masses.iter().enumerate().take(n).skip(i + 1) {
                let base_j = j * dims;
                let mut dist_sq = 0.0_f64;
                for axis in 0..dims {
                    let delta = coords[base_i + axis].to_f64() - coords[base_j + axis].to_f64();
                    dist_sq += delta * delta;
                }
                let mult = (mi_sr * mj) / (dist_sq + eps_sq);
                for axis in 0..dims {
                    let f =
                        mult * (coords[base_i + axis].to_f64() - coords[base_j + axis].to_f64());
                    forces[base_i + axis] += f;
                    forces[base_j + axis] -= f;
                }
            }
        }
    }
}
