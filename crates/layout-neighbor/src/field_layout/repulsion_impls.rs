use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageRef, Layout2d, Layout3d, Scalar,
};
use egraph_spatial_fft::{Fft2dWorkspace, IbFft2d};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use crate::backends::{BarnesHut, Exact, IbFft};

use super::config::FieldFftStats;
use super::repulsion::TFdpRepulsion;

/// Workspace state for [`Exact`] repulsion in [`TFdpSolver`](super::TFdpSolver).
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct ExactTFdpState;

/// Workspace state for 2D [`BarnesHut`] repulsion in [`TFdpSolver`](super::TFdpSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TFdpBarnesHut2dState<S: Scalar> {
    quadtree: ArenaQuadtree2d<S>,
    rep_x: Vec<S>,
    rep_y: Vec<S>,
}

/// Workspace state for 3D [`BarnesHut`] repulsion in [`TFdpSolver`](super::TFdpSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TFdpBarnesHut3dState<S: Scalar> {
    octree: ArenaOctree3d<S>,
    rep_x: Vec<S>,
    rep_y: Vec<S>,
    rep_z: Vec<S>,
}

/// Workspace state for [`IbFft`] repulsion in [`TFdpSolver`](super::TFdpSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TFdpIbFft2dState<S: Scalar> {
    ibfft: IbFft2d<S>,
    fft_ws: Fft2dWorkspace,
    scaled_x: Vec<S>,
    scaled_y: Vec<S>,
    phi_1: Vec<S>,
    rep_x: Vec<S>,
    rep_y: Vec<S>,
    last_fft_stats: Option<FieldFftStats>,
}

pub(super) fn check_rep_slice(
    n: usize,
    dims: usize,
    rep_forces: &[f64],
    ctx: &'static str,
) -> CoreResult<()> {
    let expected = n
        .checked_mul(dims)
        .ok_or_else(|| CoreError::invalid_input("repulsion force buffer size overflow"))?;
    if rep_forces.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            rep_forces.len(),
            ctx,
        ));
    }
    Ok(())
}

impl<L: EuclideanLayout> TFdpRepulsion<L> for Exact {
    type State = ExactTFdpState;

    fn validate_config(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(ExactTFdpState)
    }

    fn compute_repulsion(
        &self,
        _state: &mut Self::State,
        layout: &L,
        gamma: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let dims = layout.dims();
        check_rep_slice(n, dims, rep_forces, "Exact::compute_repulsion(TFdp)")?;
        if n < 2 || dims == 0 {
            rep_forces.fill(0.0);
            return Ok(());
        }

        match layout.as_euclidean_storage() {
            EuclideanStorageRef::Soa2d { x, y } => {
                for i in 0..n {
                    let xi = x[i].to_f64();
                    let yi = y[i].to_f64();
                    let mut fx = 0.0_f64;
                    let mut fy = 0.0_f64;
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let dx = xi - x[j].to_f64();
                        let dy = yi - y[j].to_f64();
                        let d2 = dx * dx + dy * dy;
                        let k = 1.0 / (1.0 + d2).powf(gamma);
                        fx += k * dx;
                        fy += k * dy;
                    }
                    rep_forces[2 * i] = fx;
                    rep_forces[2 * i + 1] = fy;
                }
            }
            EuclideanStorageRef::Soa3d { x, y, z } => {
                for i in 0..n {
                    let xi = x[i].to_f64();
                    let yi = y[i].to_f64();
                    let zi = z[i].to_f64();
                    let mut fx = 0.0_f64;
                    let mut fy = 0.0_f64;
                    let mut fz = 0.0_f64;
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let dx = xi - x[j].to_f64();
                        let dy = yi - y[j].to_f64();
                        let dz = zi - z[j].to_f64();
                        let d2 = dx * dx + dy * dy + dz * dz;
                        let k = 1.0 / (1.0 + d2).powf(gamma);
                        fx += k * dx;
                        fy += k * dy;
                        fz += k * dz;
                    }
                    rep_forces[3 * i] = fx;
                    rep_forces[3 * i + 1] = fy;
                    rep_forces[3 * i + 2] = fz;
                }
            }
            EuclideanStorageRef::FlatNd {
                coords,
                num_nodes,
                dims: nd_dims,
            } => {
                for i in 0..num_nodes {
                    let i_base = i * nd_dims;
                    rep_forces[i_base..i_base + nd_dims].fill(0.0);
                    for j in 0..num_nodes {
                        if j == i {
                            continue;
                        }
                        let j_base = j * nd_dims;
                        let mut d2 = 0.0_f64;
                        for d in 0..nd_dims {
                            let diff = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                            d2 += diff * diff;
                        }
                        let k = 1.0 / (1.0 + d2).powf(gamma);
                        for d in 0..nd_dims {
                            let diff = coords[i_base + d].to_f64() - coords[j_base + d].to_f64();
                            rep_forces[i_base + d] += k * diff;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl<S: Scalar> TFdpRepulsion<Layout2d<S>> for BarnesHut {
    type State = TFdpBarnesHut2dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(TFdpBarnesHut2dState {
            quadtree: ArenaQuadtree2d::new(),
            rep_x: vec![S::ZERO; num_nodes],
            rep_y: vec![S::ZERO; num_nodes],
        })
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        gamma: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        check_rep_slice(n, 2, rep_forces, "BarnesHut::compute_repulsion(TFdp 2d)")?;
        if n < 2 {
            rep_forces.fill(0.0);
            return Ok(());
        }
        if state.rep_x.len() != n {
            state.rep_x.resize(n, S::ZERO);
            state.rep_y.resize(n, S::ZERO);
        }
        state.quadtree.build(layout)?;
        let theta_s = S::from_f64(self.theta);
        if (gamma - 1.0).abs() <= 1e-12 {
            state.quadtree.compute_coulomb_repulsion(
                theta_s,
                S::ONE,
                &mut state.rep_x,
                &mut state.rep_y,
            )?;
            for i in 0..n {
                rep_forces[2 * i] = state.rep_x[i].to_f64();
                rep_forces[2 * i + 1] = state.rep_y[i].to_f64();
            }
        } else if (gamma - 2.0).abs() <= 1e-12 {
            let total_z =
                state
                    .quadtree
                    .compute_kl_repulsion(theta_s, &mut state.rep_x, &mut state.rep_y)?;
            let z_f64 = total_z.to_f64();
            for i in 0..n {
                rep_forces[2 * i] = state.rep_x[i].to_f64() * z_f64;
                rep_forces[2 * i + 1] = state.rep_y[i].to_f64() * z_f64;
            }
        } else {
            let q = 2.0 * gamma - 2.0;
            state.quadtree.compute_maxent_repulsion(
                theta_s,
                q,
                S::ONE,
                &mut state.rep_x,
                &mut state.rep_y,
            )?;
            for i in 0..n {
                rep_forces[2 * i] = state.rep_x[i].to_f64();
                rep_forces[2 * i + 1] = state.rep_y[i].to_f64();
            }
        }
        Ok(())
    }
}

impl<S: Scalar> TFdpRepulsion<Layout3d<S>> for BarnesHut {
    type State = TFdpBarnesHut3dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(TFdpBarnesHut3dState {
            octree: ArenaOctree3d::new(),
            rep_x: vec![S::ZERO; num_nodes],
            rep_y: vec![S::ZERO; num_nodes],
            rep_z: vec![S::ZERO; num_nodes],
        })
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout3d<S>,
        gamma: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        check_rep_slice(n, 3, rep_forces, "BarnesHut::compute_repulsion(TFdp 3d)")?;
        if n < 2 {
            rep_forces.fill(0.0);
            return Ok(());
        }
        if state.rep_x.len() != n {
            state.rep_x.resize(n, S::ZERO);
            state.rep_y.resize(n, S::ZERO);
            state.rep_z.resize(n, S::ZERO);
        }
        state.octree.build_3d(layout)?;
        let theta_s = S::from_f64(self.theta);
        if (gamma - 1.0).abs() <= 1e-12 {
            state.octree.compute_coulomb_repulsion(
                theta_s,
                S::ONE,
                &mut state.rep_x,
                &mut state.rep_y,
                &mut state.rep_z,
            )?;
            for i in 0..n {
                rep_forces[3 * i] = state.rep_x[i].to_f64();
                rep_forces[3 * i + 1] = state.rep_y[i].to_f64();
                rep_forces[3 * i + 2] = state.rep_z[i].to_f64();
            }
        } else if (gamma - 2.0).abs() <= 1e-12 {
            let total_z = state.octree.compute_kl_repulsion(
                theta_s,
                &mut state.rep_x,
                &mut state.rep_y,
                &mut state.rep_z,
            )?;
            let z_f64 = total_z.to_f64();
            for i in 0..n {
                rep_forces[3 * i] = state.rep_x[i].to_f64() * z_f64;
                rep_forces[3 * i + 1] = state.rep_y[i].to_f64() * z_f64;
                rep_forces[3 * i + 2] = state.rep_z[i].to_f64() * z_f64;
            }
        } else {
            let q = 2.0 * gamma - 2.0;
            state.octree.compute_maxent_repulsion(
                theta_s,
                q,
                S::ONE,
                &mut state.rep_x,
                &mut state.rep_y,
                &mut state.rep_z,
            )?;
            for i in 0..n {
                rep_forces[3 * i] = state.rep_x[i].to_f64();
                rep_forces[3 * i + 1] = state.rep_y[i].to_f64();
                rep_forces[3 * i + 2] = state.rep_z[i].to_f64();
            }
        }
        Ok(())
    }
}

impl<S: Scalar> TFdpRepulsion<Layout2d<S>> for IbFft {
    type State = TFdpIbFft2dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(TFdpIbFft2dState {
            ibfft: IbFft2d::new(self.ibfft.clone())?,
            fft_ws: Fft2dWorkspace::new(),
            scaled_x: vec![S::ZERO; num_nodes],
            scaled_y: vec![S::ZERO; num_nodes],
            phi_1: vec![S::ZERO; num_nodes],
            rep_x: vec![S::ZERO; num_nodes],
            rep_y: vec![S::ZERO; num_nodes],
            last_fft_stats: None,
        })
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        gamma: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        check_rep_slice(n, 2, rep_forces, "IbFft::compute_repulsion(TFdp 2d)")?;
        if n < 2 {
            rep_forces.fill(0.0);
            return Ok(());
        }
        let xs = layout.x();
        let ys = layout.y();
        let mut min_x = xs[0].to_f64();
        let mut max_x = min_x;
        let mut min_y = ys[0].to_f64();
        let mut max_y = min_y;
        for i in 1..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        let span_cap = (state.ibfft.config().max_grid_dim as f64) * 0.4;
        let max_span = (max_x - min_x).max(max_y - min_y);
        let before_calls = state.fft_ws.fft_2d_call_count();
        let particle_fwd;
        if max_span > span_cap && max_span.is_finite() && span_cap > 0.0 {
            let cx = 0.5 * (min_x + max_x);
            let cy = 0.5 * (min_y + max_y);
            let scale = span_cap / max_span;
            let inv_scale = max_span / span_cap;
            let inv_scale_sq = inv_scale * inv_scale;
            for i in 0..n {
                state.scaled_x[i] = S::from_f64((xs[i].to_f64() - cx) * scale);
                state.scaled_y[i] = S::from_f64((ys[i].to_f64() - cy) * scale);
            }
            particle_fwd =
                state
                    .ibfft
                    .prepare_xy(&mut state.fft_ws, &state.scaled_x, &state.scaled_y)?;
            let soft = state
                .ibfft
                .grid_spec()
                .map_or(0.0, |spec| spec.softening_sq);
            state.ibfft.evaluate_isotropic_kernel(
                &mut state.fft_ws,
                |d2| 1.0 / (1.0 + soft + d2 * inv_scale_sq).powf(gamma),
                &mut state.phi_1,
                &mut state.rep_x,
                &mut state.rep_y,
            )?;
            for i in 0..n {
                rep_forces[2 * i] = state.rep_x[i].to_f64() * inv_scale;
                rep_forces[2 * i + 1] = state.rep_y[i].to_f64() * inv_scale;
            }
        } else {
            particle_fwd = state.ibfft.prepare_xy(&mut state.fft_ws, xs, ys)?;
            let soft = state
                .ibfft
                .grid_spec()
                .map_or(0.0, |spec| spec.softening_sq);
            state.ibfft.evaluate_isotropic_kernel(
                &mut state.fft_ws,
                |d2| 1.0 / (1.0 + soft + d2).powf(gamma),
                &mut state.phi_1,
                &mut state.rep_x,
                &mut state.rep_y,
            )?;
            for i in 0..n {
                rep_forces[2 * i] = state.rep_x[i].to_f64();
                rep_forces[2 * i + 1] = state.rep_y[i].to_f64();
            }
        }
        let total_calls = state
            .fft_ws
            .fft_2d_call_count()
            .saturating_sub(before_calls);
        let spec = state.ibfft.grid_spec();
        state.last_fft_stats = Some(FieldFftStats {
            particle_forward_ffts: particle_fwd,
            kernel_forward_ffts: u64::from(total_calls > particle_fwd),
            inverse_ffts: total_calls.saturating_sub(particle_fwd + 1),
            total_fft_2d_calls: total_calls,
            nx: spec.map_or(0, |s| s.nx),
            ny: spec.map_or(0, |s| s.ny),
        });
        Ok(())
    }

    #[inline]
    fn last_fft_stats<'a>(&self, state: &'a Self::State) -> Option<&'a FieldFftStats> {
        state.last_fft_stats.as_ref()
    }

    #[inline]
    fn total_fft_2d_calls(&self, state: &Self::State) -> u64 {
        state.fft_ws.fft_2d_call_count()
    }
}
