use std::fmt;

use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageRef, Layout2d, Layout3d, LayoutStorage,
    Scalar,
};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use crate::backend::{BarnesHut, Exact, ForceBackend};

use super::quadtree::GenRepulsionQuadtree;

/// Open repulsion backend trait for [`LinLogSolver`](super::LinLogSolver).
///
/// Implementations are provided for:
/// - [`Exact`] across all [`EuclideanLayout`] targets ([`Layout2d`], [`Layout3d`], [`LayoutNd`](egraph_core::LayoutNd)).
/// - [`BarnesHut`] and [`ForceBackend`] strictly on [`Layout2d`] and [`Layout3d`]
///   (passing [`BarnesHut`] or [`ForceBackend`] with [`LayoutNd`](egraph_core::LayoutNd) is
///   rejected at compile time).
///
/// External crates (such as `egraph-surface`) may implement this trait for custom spaces and
/// geodesic repulsion backends without modifying `egraph-layout-force`.
pub trait LinLogRepulsionBackend<L: LayoutStorage + ?Sized> {
    /// Backend-owned scratch state (e.g. spatial trees; `()` for [`Exact`]).
    type State: fmt::Debug + Clone;

    /// Validates backend configuration before constructing a solver.
    fn validate(&self) -> CoreResult<()>;

    /// Initializes backend-specific scratch state for `num_nodes` vertices.
    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State>;

    /// Accumulates repulsive forces into `forces` (row-major `num_nodes * dims`).
    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &L,
        masses: &[f64],
        repulsion_exponent: f64,
        eps_sq: f64,
        forces: &mut [f64],
    ) -> CoreResult<()>;
}

/// Scratch state for 2D [`BarnesHut`] repulsion in [`LinLogSolver`](super::LinLogSolver).
#[derive(Debug, Clone)]
pub struct LinLogBarnesHut2dState {
    quadtree: ArenaQuadtree2d<f64>,
    gen_quadtree: GenRepulsionQuadtree,
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
}

/// Scratch state for 3D [`BarnesHut`] repulsion in [`LinLogSolver`](super::LinLogSolver).
#[derive(Debug, Clone)]
pub struct LinLogBarnesHut3dState {
    octree: ArenaOctree3d<f64>,
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    z_buf: Vec<f64>,
    rep_3d_flat: Vec<f64>,
}

/// Scratch state for [`ForceBackend`] in 2D [`LinLogSolver`](super::LinLogSolver).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LinLogDynamic2dState {
    Exact,
    BarnesHut(Box<LinLogBarnesHut2dState>),
}

/// Scratch state for [`ForceBackend`] in 3D [`LinLogSolver`](super::LinLogSolver).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LinLogDynamic3dState {
    Exact,
    BarnesHut(Box<LinLogBarnesHut3dState>),
}

impl<L: EuclideanLayout> LinLogRepulsionBackend<L> for Exact {
    type State = ();

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    #[inline]
    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(())
    }

    fn accumulate_repulsion(
        &self,
        _state: &mut Self::State,
        layout: &L,
        masses: &[f64],
        repulsion_exponent: f64,
        eps_sq: f64,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let dims = layout.dims();
        let expected = n
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("LinLog repulsion buffer overflow"))?;
        if forces.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                forces.len(),
                "LinLogRepulsionBackend<Exact>::accumulate_repulsion",
            ));
        }
        if n < 2 {
            return Ok(());
        }

        let is_coulomb = (repulsion_exponent - (-1.0)).abs() <= 1e-12;
        let pow_exp = 0.5 * (repulsion_exponent - 1.0);

        match layout.as_euclidean_storage() {
            EuclideanStorageRef::Soa2d { x, y } => {
                for i in 0..n {
                    let xi = x[i].to_f64();
                    let yi = y[i].to_f64();
                    let mi = masses[i];
                    let mut acc_x = 0.0_f64;
                    let mut acc_y = 0.0_f64;
                    for j in (i + 1)..n {
                        let dx = xi - x[j].to_f64();
                        let dy = yi - y[j].to_f64();
                        let d_sq_reg = dx * dx + dy * dy + eps_sq;
                        let mult = if is_coulomb {
                            (mi * masses[j]) / d_sq_reg
                        } else {
                            (mi * masses[j]) * d_sq_reg.powf(pow_exp)
                        };
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
            EuclideanStorageRef::Soa3d { x, y, z } => {
                for i in 0..n {
                    let xi = x[i].to_f64();
                    let yi = y[i].to_f64();
                    let zi = z[i].to_f64();
                    let mi = masses[i];
                    let mut acc_x = 0.0_f64;
                    let mut acc_y = 0.0_f64;
                    let mut acc_z = 0.0_f64;
                    for j in (i + 1)..n {
                        let dx = xi - x[j].to_f64();
                        let dy = yi - y[j].to_f64();
                        let dz = zi - z[j].to_f64();
                        let d_sq_reg = dx * dx + dy * dy + dz * dz + eps_sq;
                        let mult = if is_coulomb {
                            (mi * masses[j]) / d_sq_reg
                        } else {
                            (mi * masses[j]) * d_sq_reg.powf(pow_exp)
                        };
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
            EuclideanStorageRef::FlatNd { coords, dims, .. } => {
                for i in 0..n {
                    let base_i = i * dims;
                    let mi = masses[i];
                    for (j, &mj) in masses.iter().enumerate().take(n).skip(i + 1) {
                        let base_j = j * dims;
                        let mut d_sq = 0.0_f64;
                        for axis in 0..dims {
                            let delta =
                                coords[base_i + axis].to_f64() - coords[base_j + axis].to_f64();
                            d_sq += delta * delta;
                        }
                        let d_sq_reg = d_sq + eps_sq;
                        let mult = if is_coulomb {
                            (mi * mj) / d_sq_reg
                        } else {
                            (mi * mj) * d_sq_reg.powf(pow_exp)
                        };
                        for axis in 0..dims {
                            let f = mult
                                * (coords[base_i + axis].to_f64() - coords[base_j + axis].to_f64());
                            forces[base_i + axis] += f;
                            forces[base_j + axis] -= f;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl<S: Scalar> LinLogRepulsionBackend<Layout2d<S>> for BarnesHut {
    type State = LinLogBarnesHut2dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        BarnesHut::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(LinLogBarnesHut2dState {
            quadtree: ArenaQuadtree2d::new(),
            gen_quadtree: GenRepulsionQuadtree::default(),
            x_buf: vec![0.0; num_nodes],
            y_buf: vec![0.0; num_nodes],
            rep_x: vec![0.0; num_nodes],
            rep_y: vec![0.0; num_nodes],
        })
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        masses: &[f64],
        repulsion_exponent: f64,
        eps_sq: f64,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let expected = n
            .checked_mul(2)
            .ok_or_else(|| CoreError::invalid_input("LinLog 2D buffer overflow"))?;
        if forces.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                forces.len(),
                "LinLogRepulsionBackend<BarnesHut, Layout2d>::accumulate_repulsion",
            ));
        }
        if n < 2 {
            return Ok(());
        }

        let xs = layout.x();
        let ys = layout.y();
        state.x_buf.resize(n, 0.0);
        state.y_buf.resize(n, 0.0);
        for i in 0..n {
            state.x_buf[i] = xs[i].to_f64();
            state.y_buf[i] = ys[i].to_f64();
        }
        state.rep_x.resize(n, 0.0);
        state.rep_y.resize(n, 0.0);

        let is_coulomb = (repulsion_exponent - (-1.0)).abs() <= 1e-12;
        if is_coulomb {
            state
                .quadtree
                .build_weighted_xy(&state.x_buf, &state.y_buf, masses)?;
            state.quadtree.compute_coulomb_repulsion(
                self.theta(),
                eps_sq,
                &mut state.rep_x,
                &mut state.rep_y,
            )?;
        } else {
            state.gen_quadtree.build(&state.x_buf, &state.y_buf, masses);
            state.gen_quadtree.compute_general_repulsion(
                &state.x_buf,
                &state.y_buf,
                masses,
                self.theta(),
                repulsion_exponent,
                eps_sq,
                &mut state.rep_x,
                &mut state.rep_y,
            );
        }
        for i in 0..n {
            let mi = masses[i];
            forces[2 * i] += mi * state.rep_x[i];
            forces[2 * i + 1] += mi * state.rep_y[i];
        }
        Ok(())
    }
}

impl<S: Scalar> LinLogRepulsionBackend<Layout3d<S>> for BarnesHut {
    type State = LinLogBarnesHut3dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        BarnesHut::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(LinLogBarnesHut3dState {
            octree: ArenaOctree3d::new(),
            x_buf: vec![0.0; num_nodes],
            y_buf: vec![0.0; num_nodes],
            z_buf: vec![0.0; num_nodes],
            rep_3d_flat: vec![0.0; num_nodes * 3],
        })
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout3d<S>,
        masses: &[f64],
        repulsion_exponent: f64,
        eps_sq: f64,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let expected = n
            .checked_mul(3)
            .ok_or_else(|| CoreError::invalid_input("LinLog 3D buffer overflow"))?;
        if forces.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                forces.len(),
                "LinLogRepulsionBackend<BarnesHut, Layout3d>::accumulate_repulsion",
            ));
        }
        if n < 2 {
            return Ok(());
        }

        let xs = layout.x();
        let ys = layout.y();
        let zs = layout.z();
        state.x_buf.resize(n, 0.0);
        state.y_buf.resize(n, 0.0);
        state.z_buf.resize(n, 0.0);
        for i in 0..n {
            state.x_buf[i] = xs[i].to_f64();
            state.y_buf[i] = ys[i].to_f64();
            state.z_buf[i] = zs[i].to_f64();
        }
        state.rep_3d_flat.resize(expected, 0.0);

        state
            .octree
            .build_weighted_xyz(&state.x_buf, &state.y_buf, &state.z_buf, masses)?;
        state.octree.compute_power_repulsion_flat(
            self.theta(),
            repulsion_exponent,
            eps_sq,
            &mut state.rep_3d_flat,
        )?;
        for (i, &mi) in masses.iter().enumerate().take(n) {
            let base = i * 3;
            forces[base] += mi * state.rep_3d_flat[base];
            forces[base + 1] += mi * state.rep_3d_flat[base + 1];
            forces[base + 2] += mi * state.rep_3d_flat[base + 2];
        }
        Ok(())
    }
}

impl<S: Scalar> LinLogRepulsionBackend<Layout2d<S>> for ForceBackend {
    type State = LinLogDynamic2dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        ForceBackend::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        match self {
            Self::Exact(ex) => {
                <Exact as LinLogRepulsionBackend<Layout2d<S>>>::init_state(ex, num_nodes)?;
                Ok(LinLogDynamic2dState::Exact)
            }
            Self::BarnesHut(bh) => Ok(LinLogDynamic2dState::BarnesHut(Box::new(
                <BarnesHut as LinLogRepulsionBackend<Layout2d<S>>>::init_state(bh, num_nodes)?,
            ))),
        }
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        masses: &[f64],
        repulsion_exponent: f64,
        eps_sq: f64,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        match (self, state) {
            (Self::Exact(ex), LinLogDynamic2dState::Exact) => {
                ex.accumulate_repulsion(&mut (), layout, masses, repulsion_exponent, eps_sq, forces)
            }
            (Self::BarnesHut(bh), LinLogDynamic2dState::BarnesHut(st)) => bh.accumulate_repulsion(
                st.as_mut(),
                layout,
                masses,
                repulsion_exponent,
                eps_sq,
                forces,
            ),
            _ => Err(CoreError::invalid_input(
                "ForceBackend state mismatch in LinLog 2D",
            )),
        }
    }
}

impl<S: Scalar> LinLogRepulsionBackend<Layout3d<S>> for ForceBackend {
    type State = LinLogDynamic3dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        ForceBackend::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        match self {
            Self::Exact(ex) => {
                <Exact as LinLogRepulsionBackend<Layout3d<S>>>::init_state(ex, num_nodes)?;
                Ok(LinLogDynamic3dState::Exact)
            }
            Self::BarnesHut(bh) => Ok(LinLogDynamic3dState::BarnesHut(Box::new(
                <BarnesHut as LinLogRepulsionBackend<Layout3d<S>>>::init_state(bh, num_nodes)?,
            ))),
        }
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout3d<S>,
        masses: &[f64],
        repulsion_exponent: f64,
        eps_sq: f64,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        match (self, state) {
            (Self::Exact(ex), LinLogDynamic3dState::Exact) => {
                ex.accumulate_repulsion(&mut (), layout, masses, repulsion_exponent, eps_sq, forces)
            }
            (Self::BarnesHut(bh), LinLogDynamic3dState::BarnesHut(st)) => bh.accumulate_repulsion(
                st.as_mut(),
                layout,
                masses,
                repulsion_exponent,
                eps_sq,
                forces,
            ),
            _ => Err(CoreError::invalid_input(
                "ForceBackend state mismatch in LinLog 3D",
            )),
        }
    }
}
