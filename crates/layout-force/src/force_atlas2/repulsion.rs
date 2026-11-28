use std::fmt;

use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, EuclideanStorageRef, Layout2d, Layout3d, LayoutStorage,
    Scalar,
};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use crate::backend::{BarnesHut, Exact, ForceBackend};

use super::exact::{exact_2d, exact_3d, exact_nd};
use super::overlap_quadtree::OverlapQuadtree;

/// Parameters passed to [`ForceRepulsion::accumulate_repulsion`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ForceRepulsionParams<'a> {
    /// Per-node repulsion masses (`deg(u) + 1`).
    pub masses: &'a [f64],
    /// Per-node collision radii used when `prevent_overlap` is `true`.
    pub node_radii: &'a [f64],
    /// Repulsion scaling ratio $k_r$.
    pub scaling_ratio: f64,
    /// Softening squared-distance parameter $\varepsilon^2 > 0$.
    pub eps_sq: f64,
    /// Whether Jacomy (2014) border-to-border overlap prevention is active.
    pub prevent_overlap: bool,
    /// Deterministic RNG seed for coincident overlap tie-breaking.
    pub seed: u64,
    /// Current iteration index.
    pub iter: usize,
}

impl<'a> ForceRepulsionParams<'a> {
    /// Creates a new [`ForceRepulsionParams`].
    #[inline]
    pub fn new(
        masses: &'a [f64],
        node_radii: &'a [f64],
        scaling_ratio: f64,
        eps_sq: f64,
        prevent_overlap: bool,
        seed: u64,
        iter: usize,
    ) -> Self {
        Self {
            masses,
            node_radii,
            scaling_ratio,
            eps_sq,
            prevent_overlap,
            seed,
            iter,
        }
    }
}

/// Open repulsion backend trait for [`ForceAtlas2Solver`](super::ForceAtlas2Solver).
///
/// Implementations are provided for:
/// - [`Exact`] across all [`EuclideanLayout`] targets ([`Layout2d`], [`Layout3d`], [`LayoutNd`](egraph_core::LayoutNd)).
/// - [`BarnesHut`] and [`ForceBackend`] strictly on [`Layout2d`] and [`Layout3d`]
///   (passing [`BarnesHut`] or [`ForceBackend`] with [`LayoutNd`](egraph_core::LayoutNd) is
///   rejected at compile time).
///
/// External crates (such as `egraph-surface`) may implement this trait for custom spaces and
/// geodesic repulsion backends without modifying `egraph-layout-force`.
pub trait ForceRepulsion<L: LayoutStorage + ?Sized> {
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
        params: &ForceRepulsionParams<'_>,
        forces: &mut [f64],
    ) -> CoreResult<()>;
}

/// Scratch state for 2D [`BarnesHut`] repulsion in [`ForceAtlas2Solver`](super::ForceAtlas2Solver).
#[derive(Debug, Clone)]
pub struct Fa2BarnesHut2dState {
    quadtree: ArenaQuadtree2d<f64>,
    overlap_tree: OverlapQuadtree,
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
}

/// Scratch state for 3D [`BarnesHut`] repulsion in [`ForceAtlas2Solver`](super::ForceAtlas2Solver).
#[derive(Debug, Clone)]
pub struct Fa2BarnesHut3dState {
    octree: ArenaOctree3d<f64>,
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    z_buf: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
    rep_z: Vec<f64>,
}

/// Scratch state for [`ForceBackend`] in 2D [`ForceAtlas2Solver`](super::ForceAtlas2Solver).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum Fa2Dynamic2dState {
    Exact,
    BarnesHut(Box<Fa2BarnesHut2dState>),
}

/// Scratch state for [`ForceBackend`] in 3D [`ForceAtlas2Solver`](super::ForceAtlas2Solver).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum Fa2Dynamic3dState {
    Exact,
    BarnesHut(Box<Fa2BarnesHut3dState>),
}

impl<L: EuclideanLayout> ForceRepulsion<L> for Exact {
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
        params: &ForceRepulsionParams<'_>,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let dims = layout.dims();
        let expected = n
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("ForceAtlas2 repulsion buffer overflow"))?;
        if forces.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                forces.len(),
                "ForceRepulsion<Exact>::accumulate_repulsion",
            ));
        }
        if n < 2 {
            return Ok(());
        }

        match layout.as_euclidean_storage() {
            EuclideanStorageRef::Soa2d { x, y } => exact_2d(x, y, params, forces),
            EuclideanStorageRef::Soa3d { x, y, z } => exact_3d(x, y, z, params, forces),
            EuclideanStorageRef::FlatNd { coords, dims, .. } => {
                exact_nd(coords, n, dims, params, forces)
            }
        }
        Ok(())
    }
}

impl<S: Scalar> ForceRepulsion<Layout2d<S>> for BarnesHut {
    type State = Fa2BarnesHut2dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        BarnesHut::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(Fa2BarnesHut2dState {
            quadtree: ArenaQuadtree2d::new(),
            overlap_tree: OverlapQuadtree::default(),
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
        params: &ForceRepulsionParams<'_>,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let expected = n
            .checked_mul(2)
            .ok_or_else(|| CoreError::invalid_input("ForceAtlas2 2D buffer overflow"))?;
        if forces.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                forces.len(),
                "ForceRepulsion<BarnesHut, Layout2d>::accumulate_repulsion",
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

        if params.prevent_overlap {
            state
                .overlap_tree
                .build(&state.x_buf, &state.y_buf, params.masses, params.node_radii);
            state.overlap_tree.compute_repulsion(
                &state.x_buf,
                &state.y_buf,
                params.masses,
                params.node_radii,
                self.theta(),
                params.scaling_ratio,
                params.seed,
                params.iter,
                &mut state.rep_x,
                &mut state.rep_y,
            );
            for i in 0..n {
                forces[2 * i] += state.rep_x[i];
                forces[2 * i + 1] += state.rep_y[i];
            }
        } else {
            state
                .quadtree
                .build_weighted_xy(&state.x_buf, &state.y_buf, params.masses)?;
            state.quadtree.compute_coulomb_repulsion(
                self.theta(),
                params.eps_sq,
                &mut state.rep_x,
                &mut state.rep_y,
            )?;
            let sr = params.scaling_ratio;
            for i in 0..n {
                let scale = sr * params.masses[i];
                forces[2 * i] += scale * state.rep_x[i];
                forces[2 * i + 1] += scale * state.rep_y[i];
            }
        }
        Ok(())
    }
}

impl<S: Scalar> ForceRepulsion<Layout3d<S>> for BarnesHut {
    type State = Fa2BarnesHut3dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        BarnesHut::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(Fa2BarnesHut3dState {
            octree: ArenaOctree3d::new(),
            x_buf: vec![0.0; num_nodes],
            y_buf: vec![0.0; num_nodes],
            z_buf: vec![0.0; num_nodes],
            rep_x: vec![0.0; num_nodes],
            rep_y: vec![0.0; num_nodes],
            rep_z: vec![0.0; num_nodes],
        })
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout3d<S>,
        params: &ForceRepulsionParams<'_>,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let expected = n
            .checked_mul(3)
            .ok_or_else(|| CoreError::invalid_input("ForceAtlas2 3D buffer overflow"))?;
        if forces.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                forces.len(),
                "ForceRepulsion<BarnesHut, Layout3d>::accumulate_repulsion",
            ));
        }
        if n < 2 {
            return Ok(());
        }

        if params.prevent_overlap {
            exact_3d(layout.x(), layout.y(), layout.z(), params, forces);
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
        state.rep_x.resize(n, 0.0);
        state.rep_y.resize(n, 0.0);
        state.rep_z.resize(n, 0.0);

        state
            .octree
            .build_weighted_xyz(&state.x_buf, &state.y_buf, &state.z_buf, params.masses)?;
        state.octree.compute_coulomb_repulsion(
            self.theta(),
            params.eps_sq,
            &mut state.rep_x,
            &mut state.rep_y,
            &mut state.rep_z,
        )?;
        let sr = params.scaling_ratio;
        for i in 0..n {
            let scale = sr * params.masses[i];
            forces[3 * i] += scale * state.rep_x[i];
            forces[3 * i + 1] += scale * state.rep_y[i];
            forces[3 * i + 2] += scale * state.rep_z[i];
        }
        Ok(())
    }
}

impl<S: Scalar> ForceRepulsion<Layout2d<S>> for ForceBackend {
    type State = Fa2Dynamic2dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        ForceBackend::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        match self {
            Self::Exact(ex) => {
                <Exact as ForceRepulsion<Layout2d<S>>>::init_state(ex, num_nodes)?;
                Ok(Fa2Dynamic2dState::Exact)
            }
            Self::BarnesHut(bh) => Ok(Fa2Dynamic2dState::BarnesHut(Box::new(
                <BarnesHut as ForceRepulsion<Layout2d<S>>>::init_state(bh, num_nodes)?,
            ))),
        }
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        params: &ForceRepulsionParams<'_>,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        match (self, state) {
            (Self::Exact(ex), Fa2Dynamic2dState::Exact) => {
                ex.accumulate_repulsion(&mut (), layout, params, forces)
            }
            (Self::BarnesHut(bh), Fa2Dynamic2dState::BarnesHut(st)) => {
                bh.accumulate_repulsion(st.as_mut(), layout, params, forces)
            }
            _ => Err(CoreError::invalid_input(
                "ForceBackend state mismatch in ForceAtlas2 2D",
            )),
        }
    }
}

impl<S: Scalar> ForceRepulsion<Layout3d<S>> for ForceBackend {
    type State = Fa2Dynamic3dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        ForceBackend::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        match self {
            Self::Exact(ex) => {
                <Exact as ForceRepulsion<Layout3d<S>>>::init_state(ex, num_nodes)?;
                Ok(Fa2Dynamic3dState::Exact)
            }
            Self::BarnesHut(bh) => Ok(Fa2Dynamic3dState::BarnesHut(Box::new(
                <BarnesHut as ForceRepulsion<Layout3d<S>>>::init_state(bh, num_nodes)?,
            ))),
        }
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout3d<S>,
        params: &ForceRepulsionParams<'_>,
        forces: &mut [f64],
    ) -> CoreResult<()> {
        match (self, state) {
            (Self::Exact(ex), Fa2Dynamic3dState::Exact) => {
                ex.accumulate_repulsion(&mut (), layout, params, forces)
            }
            (Self::BarnesHut(bh), Fa2Dynamic3dState::BarnesHut(st)) => {
                bh.accumulate_repulsion(st.as_mut(), layout, params, forces)
            }
            _ => Err(CoreError::invalid_input(
                "ForceBackend state mismatch in ForceAtlas2 3D",
            )),
        }
    }
}
