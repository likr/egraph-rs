use std::fmt;

use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, Layout2d, Layout3d, LayoutStorage, Scalar,
};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use crate::backend::{BarnesHut, Exact, ForceBackend};

/// Active-level coordinate view passed to [`YifanHuRepulsion::accumulate_level_repulsion`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum YifanHuLevelCoords<'a> {
    /// Active 2D SoA level coordinates of length `ln`.
    Soa2d { x: &'a [f64], y: &'a [f64] },
    /// Active 3D SoA level coordinates of length `ln`.
    Soa3d {
        x: &'a [f64],
        y: &'a [f64],
        z: &'a [f64],
    },
    /// Active row-major N-D level coordinates of length `num_active * dims`.
    FlatNd {
        coords: &'a [f64],
        num_active: usize,
        dims: usize,
    },
}

/// Open repulsion backend trait for multilevel [`YifanHuSolver`](super::YifanHuSolver).
///
/// Implementations are provided for:
/// - [`Exact`] across all [`EuclideanLayout`] targets ([`Layout2d`], [`Layout3d`], [`LayoutNd`](egraph_core::LayoutNd)).
/// - [`BarnesHut`] and [`ForceBackend`] strictly on [`Layout2d`] and [`Layout3d`]
///   (passing [`BarnesHut`] or [`ForceBackend`] with [`LayoutNd`](egraph_core::LayoutNd) is
///   rejected at compile time).
///
/// External crates may implement this trait for custom spaces and geodesic repulsion backends
/// without modifying `egraph-layout-force`.
pub trait YifanHuRepulsion<L: LayoutStorage + ?Sized> {
    /// Backend-owned scratch state (e.g. spatial trees; `()` for [`Exact`]).
    type State: fmt::Debug;

    /// Validates backend configuration before constructing a solver.
    fn validate(&self) -> CoreResult<()>;

    /// Initializes backend-specific scratch state for `num_nodes` finest-level vertices.
    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State>;

    /// Accumulates active-level repulsive forces scaled by `rep_scale` into `forces`
    /// (row-major `num_active * dims`).
    fn accumulate_level_repulsion(
        &self,
        state: &mut Self::State,
        coords: &YifanHuLevelCoords<'_>,
        eps_sq: f64,
        rep_scale: f64,
        forces: &mut [f64],
    ) -> CoreResult<()>;
}

/// Scratch state for 2D [`BarnesHut`] repulsion in [`YifanHuSolver`](super::YifanHuSolver).
#[derive(Debug, Clone)]
pub struct YhBarnesHut2dState {
    quadtree: ArenaQuadtree2d<f64>,
    unit_masses: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
}

/// Scratch state for 3D [`BarnesHut`] repulsion in [`YifanHuSolver`](super::YifanHuSolver).
#[derive(Debug, Clone)]
pub struct YhBarnesHut3dState {
    octree: ArenaOctree3d<f64>,
    unit_masses: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
    rep_z: Vec<f64>,
}

/// Scratch state for [`ForceBackend`] in 2D [`YifanHuSolver`](super::YifanHuSolver).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum YhDynamic2dState {
    Exact,
    BarnesHut(Box<YhBarnesHut2dState>),
}

/// Scratch state for [`ForceBackend`] in 3D [`YifanHuSolver`](super::YifanHuSolver).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum YhDynamic3dState {
    Exact,
    BarnesHut(Box<YhBarnesHut3dState>),
}

impl<L: EuclideanLayout> YifanHuRepulsion<L> for Exact {
    type State = ();

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    #[inline]
    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(())
    }

    fn accumulate_level_repulsion(
        &self,
        _state: &mut Self::State,
        coords: &YifanHuLevelCoords<'_>,
        eps_sq: f64,
        rep_scale: f64,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        match *coords {
            YifanHuLevelCoords::Soa2d { x, y } => {
                let ln = x.len();
                if forces.len() != ln * 2 {
                    return Err(CoreError::dimension_mismatch(
                        ln * 2,
                        forces.len(),
                        "YifanHuRepulsion<Exact>::accumulate_level_repulsion",
                    ));
                }
                for i in 0..ln {
                    let xi = x[i];
                    let yi = y[i];
                    let mut acc_x = 0.0_f64;
                    let mut acc_y = 0.0_f64;
                    for j in (i + 1)..ln {
                        let dx = xi - x[j];
                        let dy = yi - y[j];
                        let mult = rep_scale / (dx * dx + dy * dy + eps_sq);
                        let fx = mult * dx;
                        let fy = mult * dy;
                        acc_x += fx;
                        acc_y += fy;
                        forces[2 * j] -= fx;
                        forces[2 * j + 1] -= fy;
                    }
                    forces[2 * i] += acc_x;
                    forces[2 * i + 1] += acc_y;
                }
            }
            YifanHuLevelCoords::Soa3d { x, y, z } => {
                let ln = x.len();
                if forces.len() != ln * 3 {
                    return Err(CoreError::dimension_mismatch(
                        ln * 3,
                        forces.len(),
                        "YifanHuRepulsion<Exact>::accumulate_level_repulsion",
                    ));
                }
                for i in 0..ln {
                    let xi = x[i];
                    let yi = y[i];
                    let zi = z[i];
                    let mut acc_x = 0.0_f64;
                    let mut acc_y = 0.0_f64;
                    let mut acc_z = 0.0_f64;
                    for j in (i + 1)..ln {
                        let dx = xi - x[j];
                        let dy = yi - y[j];
                        let dz = zi - z[j];
                        let mult = rep_scale / (dx * dx + dy * dy + dz * dz + eps_sq);
                        let fx = mult * dx;
                        let fy = mult * dy;
                        let fz = mult * dz;
                        acc_x += fx;
                        acc_y += fy;
                        acc_z += fz;
                        forces[3 * j] -= fx;
                        forces[3 * j + 1] -= fy;
                        forces[3 * j + 2] -= fz;
                    }
                    forces[3 * i] += acc_x;
                    forces[3 * i + 1] += acc_y;
                    forces[3 * i + 2] += acc_z;
                }
            }
            YifanHuLevelCoords::FlatNd {
                coords: c,
                num_active,
                dims,
            } => {
                let expected = num_active * dims;
                if forces.len() != expected {
                    return Err(CoreError::dimension_mismatch(
                        expected,
                        forces.len(),
                        "YifanHuRepulsion<Exact>::accumulate_level_repulsion",
                    ));
                }
                for i in 0..num_active {
                    let base_i = i * dims;
                    for j in (i + 1)..num_active {
                        let base_j = j * dims;
                        let mut dist_sq = 0.0_f64;
                        for axis in 0..dims {
                            let delta = c[base_i + axis] - c[base_j + axis];
                            dist_sq += delta * delta;
                        }
                        let mult = rep_scale / (dist_sq + eps_sq);
                        for axis in 0..dims {
                            let f = mult * (c[base_i + axis] - c[base_j + axis]);
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

impl<S: Scalar> YifanHuRepulsion<Layout2d<S>> for BarnesHut {
    type State = YhBarnesHut2dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        BarnesHut::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(YhBarnesHut2dState {
            quadtree: ArenaQuadtree2d::new(),
            unit_masses: vec![1.0; num_nodes],
            rep_x: vec![0.0; num_nodes],
            rep_y: vec![0.0; num_nodes],
        })
    }

    fn accumulate_level_repulsion(
        &self,
        state: &mut Self::State,
        coords: &YifanHuLevelCoords<'_>,
        eps_sq: f64,
        rep_scale: f64,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        let YifanHuLevelCoords::Soa2d { x, y } = *coords else {
            return Err(CoreError::invalid_input(
                "BarnesHut<Layout2d> expects 2D level coordinates",
            ));
        };
        let ln = x.len();
        if forces.len() != ln * 2 {
            return Err(CoreError::dimension_mismatch(
                ln * 2,
                forces.len(),
                "YifanHuRepulsion<BarnesHut, Layout2d>::accumulate_level_repulsion",
            ));
        }
        if ln < 2 {
            return Ok(());
        }
        state.unit_masses.resize(ln, 1.0);
        state.rep_x.resize(ln, 0.0);
        state.rep_y.resize(ln, 0.0);
        state
            .quadtree
            .build_weighted_xy(x, y, &state.unit_masses[..ln])?;
        state.quadtree.compute_coulomb_repulsion(
            self.theta(),
            eps_sq,
            &mut state.rep_x[..ln],
            &mut state.rep_y[..ln],
        )?;
        for i in 0..ln {
            forces[2 * i] += rep_scale * state.rep_x[i];
            forces[2 * i + 1] += rep_scale * state.rep_y[i];
        }
        Ok(())
    }
}

impl<S: Scalar> YifanHuRepulsion<Layout3d<S>> for BarnesHut {
    type State = YhBarnesHut3dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        BarnesHut::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(YhBarnesHut3dState {
            octree: ArenaOctree3d::new(),
            unit_masses: vec![1.0; num_nodes],
            rep_x: vec![0.0; num_nodes],
            rep_y: vec![0.0; num_nodes],
            rep_z: vec![0.0; num_nodes],
        })
    }

    fn accumulate_level_repulsion(
        &self,
        state: &mut Self::State,
        coords: &YifanHuLevelCoords<'_>,
        eps_sq: f64,
        rep_scale: f64,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        let YifanHuLevelCoords::Soa3d { x, y, z } = *coords else {
            return Err(CoreError::invalid_input(
                "BarnesHut<Layout3d> expects 3D level coordinates",
            ));
        };
        let ln = x.len();
        if forces.len() != ln * 3 {
            return Err(CoreError::dimension_mismatch(
                ln * 3,
                forces.len(),
                "YifanHuRepulsion<BarnesHut, Layout3d>::accumulate_level_repulsion",
            ));
        }
        if ln < 2 {
            return Ok(());
        }
        state.unit_masses.resize(ln, 1.0);
        state.rep_x.resize(ln, 0.0);
        state.rep_y.resize(ln, 0.0);
        state.rep_z.resize(ln, 0.0);
        state
            .octree
            .build_weighted_xyz(x, y, z, &state.unit_masses[..ln])?;
        state.octree.compute_coulomb_repulsion(
            self.theta(),
            eps_sq,
            &mut state.rep_x[..ln],
            &mut state.rep_y[..ln],
            &mut state.rep_z[..ln],
        )?;
        for i in 0..ln {
            forces[3 * i] += rep_scale * state.rep_x[i];
            forces[3 * i + 1] += rep_scale * state.rep_y[i];
            forces[3 * i + 2] += rep_scale * state.rep_z[i];
        }
        Ok(())
    }
}

impl<S: Scalar> YifanHuRepulsion<Layout2d<S>> for ForceBackend {
    type State = YhDynamic2dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        ForceBackend::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        match self {
            Self::Exact(ex) => {
                <Exact as YifanHuRepulsion<Layout2d<S>>>::init_state(ex, num_nodes)?;
                Ok(YhDynamic2dState::Exact)
            }
            Self::BarnesHut(bh) => Ok(YhDynamic2dState::BarnesHut(Box::new(
                <BarnesHut as YifanHuRepulsion<Layout2d<S>>>::init_state(bh, num_nodes)?,
            ))),
        }
    }

    fn accumulate_level_repulsion(
        &self,
        state: &mut Self::State,
        coords: &YifanHuLevelCoords<'_>,
        eps_sq: f64,
        rep_scale: f64,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        match (self, state) {
            (Self::Exact(ex), YhDynamic2dState::Exact) => {
                <Exact as YifanHuRepulsion<Layout2d<S>>>::accumulate_level_repulsion(
                    ex,
                    &mut (),
                    coords,
                    eps_sq,
                    rep_scale,
                    forces,
                )
            }
            (Self::BarnesHut(bh), YhDynamic2dState::BarnesHut(st)) => {
                <BarnesHut as YifanHuRepulsion<Layout2d<S>>>::accumulate_level_repulsion(
                    bh,
                    st.as_mut(),
                    coords,
                    eps_sq,
                    rep_scale,
                    forces,
                )
            }
            _ => Err(CoreError::invalid_input(
                "ForceBackend state mismatch in YifanHu 2D",
            )),
        }
    }
}

impl<S: Scalar> YifanHuRepulsion<Layout3d<S>> for ForceBackend {
    type State = YhDynamic3dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        ForceBackend::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        match self {
            Self::Exact(ex) => {
                <Exact as YifanHuRepulsion<Layout3d<S>>>::init_state(ex, num_nodes)?;
                Ok(YhDynamic3dState::Exact)
            }
            Self::BarnesHut(bh) => Ok(YhDynamic3dState::BarnesHut(Box::new(
                <BarnesHut as YifanHuRepulsion<Layout3d<S>>>::init_state(bh, num_nodes)?,
            ))),
        }
    }

    fn accumulate_level_repulsion(
        &self,
        state: &mut Self::State,
        coords: &YifanHuLevelCoords<'_>,
        eps_sq: f64,
        rep_scale: f64,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        match (self, state) {
            (Self::Exact(ex), YhDynamic3dState::Exact) => {
                <Exact as YifanHuRepulsion<Layout3d<S>>>::accumulate_level_repulsion(
                    ex,
                    &mut (),
                    coords,
                    eps_sq,
                    rep_scale,
                    forces,
                )
            }
            (Self::BarnesHut(bh), YhDynamic3dState::BarnesHut(st)) => {
                <BarnesHut as YifanHuRepulsion<Layout3d<S>>>::accumulate_level_repulsion(
                    bh,
                    st.as_mut(),
                    coords,
                    eps_sq,
                    rep_scale,
                    forces,
                )
            }
            _ => Err(CoreError::invalid_input(
                "ForceBackend state mismatch in YifanHu 3D",
            )),
        }
    }
}
