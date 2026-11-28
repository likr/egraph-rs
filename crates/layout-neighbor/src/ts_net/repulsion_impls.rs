use egraph_core::{CoreError, CoreResult, Layout2d, Layout3d, RiemannianLayout, Scalar};
use egraph_spatial_trees::{ArenaOctree3d, ArenaQuadtree2d};

use crate::backends::{BarnesHut, Exact};

use super::repulsion::TsNetRepulsion;

const MIN_TOTAL_Z: f64 = 1e-12;
const EXACT_ENTROPY_MIN_DIST: f64 = 1e-12;

/// Workspace state for [`Exact`] repulsion in [`TsNetSolver`](super::TsNetSolver).
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct ExactTsNetState {
    dims: usize,
    log_buf: Vec<f64>,
    rf_buf: Vec<f64>,
    ef_buf: Vec<f64>,
}

/// Workspace state for 2D [`BarnesHut`] repulsion in [`TsNetSolver`](super::TsNetSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct BarnesHut2dState<S: Scalar> {
    quadtree: ArenaQuadtree2d<S>,
    rep_x: Vec<S>,
    rep_y: Vec<S>,
    ent_x: Vec<S>,
    ent_y: Vec<S>,
}

/// Workspace state for 3D [`BarnesHut`] repulsion in [`TsNetSolver`](super::TsNetSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct BarnesHut3dState<S: Scalar> {
    octree: ArenaOctree3d<S>,
    rep_x: Vec<S>,
    rep_y: Vec<S>,
    rep_z: Vec<S>,
    ent_x: Vec<S>,
    ent_y: Vec<S>,
    ent_z: Vec<S>,
}

pub(super) fn check_force_slices(
    n: usize,
    dims: usize,
    compute_entropy: bool,
    rep_forces: &[f64],
    ent_forces: &[f64],
    ctx: &'static str,
) -> CoreResult<usize> {
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
    if compute_entropy && ent_forces.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            ent_forces.len(),
            ctx,
        ));
    }
    Ok(expected)
}

impl<L: RiemannianLayout> TsNetRepulsion<L> for Exact {
    type State = ExactTsNetState;

    fn validate_config(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(
        &self,
        _num_nodes: usize,
        _epsilon_r: f64,
        _any_entropy: bool,
        _init_entropy: bool,
    ) -> CoreResult<Self::State> {
        Ok(ExactTsNetState::default())
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &L,
        epsilon_r: f64,
        compute_entropy: bool,
        rep_forces: &mut [f64],
        ent_forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let dims = layout.tangent_dim();
        check_force_slices(
            n,
            dims,
            compute_entropy,
            rep_forces,
            ent_forces,
            "Exact::compute_repulsion",
        )?;
        if n < 2 || dims == 0 {
            rep_forces.fill(0.0);
            if compute_entropy {
                ent_forces.fill(0.0);
            }
            return Ok(());
        }
        if state.dims != dims {
            state.dims = dims;
            state.log_buf.resize(dims, 0.0);
            state.rf_buf.resize(dims, 0.0);
            state.ef_buf.resize(dims, 0.0);
        }

        let mut total_z = 0.0_f64;
        for i in 0..n {
            let i_base = i * dims;
            let mut z_i = 0.0_f64;
            state.rf_buf.fill(0.0);
            if compute_entropy {
                state.ef_buf.fill(0.0);
            }
            for j in 0..n {
                if j == i {
                    continue;
                }
                layout.log_map_into(i, j, &mut state.log_buf)?;
                let mut dist_sq = 0.0_f64;
                for &v in &state.log_buf {
                    dist_sq += v * v;
                }
                let w = 1.0 / (1.0 + dist_sq);
                z_i += w;
                let w2 = w * w;
                let ent_mult = if compute_entropy {
                    let dist = dist_sq.sqrt();
                    if dist > EXACT_ENTROPY_MIN_DIST {
                        Some(1.0 / (dist * (dist + epsilon_r)))
                    } else {
                        None
                    }
                } else {
                    None
                };
                for d in 0..dims {
                    let dx = -state.log_buf[d];
                    state.rf_buf[d] += w2 * dx;
                    if let Some(mult) = ent_mult {
                        state.ef_buf[d] += mult * dx;
                    }
                }
            }
            total_z += z_i;
            for d in 0..dims {
                rep_forces[i_base + d] = state.rf_buf[d];
                if compute_entropy {
                    ent_forces[i_base + d] = state.ef_buf[d];
                }
            }
        }
        let denom = total_z.max(MIN_TOTAL_Z);
        for v in rep_forces.iter_mut() {
            *v /= denom;
        }
        Ok(())
    }
}

impl<S: Scalar> TsNetRepulsion<Layout2d<S>> for BarnesHut {
    type State = BarnesHut2dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(
        &self,
        num_nodes: usize,
        _epsilon_r: f64,
        any_entropy: bool,
        _init_entropy: bool,
    ) -> CoreResult<Self::State> {
        let ent_len = if any_entropy { num_nodes } else { 0 };
        Ok(BarnesHut2dState {
            quadtree: ArenaQuadtree2d::new(),
            rep_x: vec![S::ZERO; num_nodes],
            rep_y: vec![S::ZERO; num_nodes],
            ent_x: vec![S::ZERO; ent_len],
            ent_y: vec![S::ZERO; ent_len],
        })
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        epsilon_r: f64,
        compute_entropy: bool,
        rep_forces: &mut [f64],
        ent_forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        check_force_slices(
            n,
            2,
            compute_entropy,
            rep_forces,
            ent_forces,
            "BarnesHut::compute_repulsion(2d)",
        )?;
        if n < 2 {
            rep_forces.fill(0.0);
            if compute_entropy {
                ent_forces.fill(0.0);
            }
            return Ok(());
        }
        if state.rep_x.len() != n {
            state.rep_x.resize(n, S::ZERO);
            state.rep_y.resize(n, S::ZERO);
        }
        if compute_entropy && state.ent_x.len() != n {
            state.ent_x.resize(n, S::ZERO);
            state.ent_y.resize(n, S::ZERO);
        }

        state.quadtree.build(layout)?;
        let theta_s = S::from_f64(self.theta);
        state
            .quadtree
            .compute_kl_repulsion(theta_s, &mut state.rep_x, &mut state.rep_y)?;
        for i in 0..n {
            rep_forces[2 * i] = state.rep_x[i].to_f64();
            rep_forces[2 * i + 1] = state.rep_y[i].to_f64();
        }
        if compute_entropy {
            let eps_s = S::from_f64(epsilon_r).max(S::MIN_POSITIVE);
            state.quadtree.compute_entropy_repulsion(
                theta_s,
                eps_s,
                &mut state.ent_x,
                &mut state.ent_y,
            )?;
            for i in 0..n {
                ent_forces[2 * i] = state.ent_x[i].to_f64();
                ent_forces[2 * i + 1] = state.ent_y[i].to_f64();
            }
        }
        Ok(())
    }
}

impl<S: Scalar> TsNetRepulsion<Layout3d<S>> for BarnesHut {
    type State = BarnesHut3dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(
        &self,
        num_nodes: usize,
        _epsilon_r: f64,
        any_entropy: bool,
        _init_entropy: bool,
    ) -> CoreResult<Self::State> {
        let ent_len = if any_entropy { num_nodes } else { 0 };
        Ok(BarnesHut3dState {
            octree: ArenaOctree3d::new(),
            rep_x: vec![S::ZERO; num_nodes],
            rep_y: vec![S::ZERO; num_nodes],
            rep_z: vec![S::ZERO; num_nodes],
            ent_x: vec![S::ZERO; ent_len],
            ent_y: vec![S::ZERO; ent_len],
            ent_z: vec![S::ZERO; ent_len],
        })
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout3d<S>,
        epsilon_r: f64,
        compute_entropy: bool,
        rep_forces: &mut [f64],
        ent_forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        check_force_slices(
            n,
            3,
            compute_entropy,
            rep_forces,
            ent_forces,
            "BarnesHut::compute_repulsion(3d)",
        )?;
        if n < 2 {
            rep_forces.fill(0.0);
            if compute_entropy {
                ent_forces.fill(0.0);
            }
            return Ok(());
        }
        if state.rep_x.len() != n {
            state.rep_x.resize(n, S::ZERO);
            state.rep_y.resize(n, S::ZERO);
            state.rep_z.resize(n, S::ZERO);
        }
        if compute_entropy && state.ent_x.len() != n {
            state.ent_x.resize(n, S::ZERO);
            state.ent_y.resize(n, S::ZERO);
            state.ent_z.resize(n, S::ZERO);
        }

        state.octree.build_3d(layout)?;
        let theta_s = S::from_f64(self.theta);
        state.octree.compute_kl_repulsion(
            theta_s,
            &mut state.rep_x,
            &mut state.rep_y,
            &mut state.rep_z,
        )?;
        for i in 0..n {
            rep_forces[3 * i] = state.rep_x[i].to_f64();
            rep_forces[3 * i + 1] = state.rep_y[i].to_f64();
            rep_forces[3 * i + 2] = state.rep_z[i].to_f64();
        }
        if compute_entropy {
            let eps_s = S::from_f64(epsilon_r).max(S::MIN_POSITIVE);
            state.octree.compute_entropy_repulsion(
                theta_s,
                eps_s,
                &mut state.ent_x,
                &mut state.ent_y,
                &mut state.ent_z,
            )?;
            for i in 0..n {
                ent_forces[3 * i] = state.ent_x[i].to_f64();
                ent_forces[3 * i + 1] = state.ent_y[i].to_f64();
                ent_forces[3 * i + 2] = state.ent_z[i].to_f64();
            }
        }
        Ok(())
    }
}
