use egraph_core::{CoreError, CoreResult, DeterministicRng, Layout2d, Layout3d, Scalar};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use crate::backend::BarnesHut;

use super::repulsion::FrRepulsion;
use super::solver::BH_EPS_SQ;

/// Scratch state for 2D [`BarnesHut`] repulsion in [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver).
#[derive(Debug, Clone)]
pub struct FrBarnesHut2dState {
    quadtree: ArenaQuadtree2d<f64>,
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
}

/// Scratch state for 3D [`BarnesHut`] repulsion in [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver).
#[derive(Debug, Clone)]
pub struct FrBarnesHut3dState {
    octree: ArenaOctree3d<f64>,
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    z_buf: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
    rep_z: Vec<f64>,
}

impl<S: Scalar> FrRepulsion<Layout2d<S>> for BarnesHut {
    type State = FrBarnesHut2dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        BarnesHut::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(FrBarnesHut2dState {
            quadtree: ArenaQuadtree2d::new(),
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
        k: f64,
        iter: usize,
        rng: &mut DeterministicRng,
        disp: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let expected = n
            .checked_mul(2)
            .ok_or_else(|| CoreError::invalid_input("FR 2D buffer size overflow"))?;
        if disp.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                disp.len(),
                "FrRepulsion<BarnesHut, Layout2d>::accumulate_repulsion",
            ));
        }
        if n < 2 {
            return Ok(());
        }

        let xs = layout.x();
        let ys = layout.y();
        state.x_buf.resize(n, 0.0);
        state.y_buf.resize(n, 0.0);
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            state.x_buf[i] = x;
            state.y_buf[i] = y;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        if iter == 0 && (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
            let spread = 0.5 * k;
            for i in 0..n {
                state.x_buf[i] += spread * (rng.next_f64() - 0.5);
                state.y_buf[i] += spread * (rng.next_f64() - 0.5);
            }
        }

        state.rep_x.resize(n, 0.0);
        state.rep_y.resize(n, 0.0);
        state.quadtree.build_xy(&state.x_buf, &state.y_buf)?;
        state.quadtree.compute_coulomb_repulsion(
            self.theta(),
            BH_EPS_SQ,
            &mut state.rep_x,
            &mut state.rep_y,
        )?;
        let k_sq = k * k;
        for i in 0..n {
            disp[2 * i] += k_sq * state.rep_x[i];
            disp[2 * i + 1] += k_sq * state.rep_y[i];
        }
        Ok(())
    }
}

impl<S: Scalar> FrRepulsion<Layout3d<S>> for BarnesHut {
    type State = FrBarnesHut3dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        BarnesHut::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(FrBarnesHut3dState {
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
        k: f64,
        iter: usize,
        rng: &mut DeterministicRng,
        disp: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let expected = n
            .checked_mul(3)
            .ok_or_else(|| CoreError::invalid_input("FR 3D buffer size overflow"))?;
        if disp.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                disp.len(),
                "FrRepulsion<BarnesHut, Layout3d>::accumulate_repulsion",
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
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;
        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            let z = zs[i].to_f64();
            state.x_buf[i] = x;
            state.y_buf[i] = y;
            state.z_buf[i] = z;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
        if iter == 0
            && (max_x - min_x) < 1e-12
            && (max_y - min_y) < 1e-12
            && (max_z - min_z) < 1e-12
        {
            let spread = 0.5 * k;
            for i in 0..n {
                state.x_buf[i] += spread * (rng.next_f64() - 0.5);
                state.y_buf[i] += spread * (rng.next_f64() - 0.5);
                state.z_buf[i] += spread * (rng.next_f64() - 0.5);
            }
        }

        state.rep_x.resize(n, 0.0);
        state.rep_y.resize(n, 0.0);
        state.rep_z.resize(n, 0.0);
        state
            .octree
            .build_xyz(&state.x_buf, &state.y_buf, &state.z_buf)?;
        state.octree.compute_coulomb_repulsion(
            self.theta(),
            BH_EPS_SQ,
            &mut state.rep_x,
            &mut state.rep_y,
            &mut state.rep_z,
        )?;
        let k_sq = k * k;
        for i in 0..n {
            disp[3 * i] += k_sq * state.rep_x[i];
            disp[3 * i + 1] += k_sq * state.rep_y[i];
            disp[3 * i + 2] += k_sq * state.rep_z[i];
        }
        Ok(())
    }
}
