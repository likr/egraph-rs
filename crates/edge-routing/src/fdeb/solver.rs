use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, EdgePolylines, EuclideanLayout, EuclideanStorageMut,
    EuclideanStorageRef, Layout2d, Scalar, SolverStatus,
};

use super::MIN_EDGE_LEN;
use super::compatibility::{edge_compatibility, norm_diff};
use super::config::FdebConfig;

#[derive(Debug, Clone, Copy)]
pub(super) struct EdgeMeta {
    pub(super) length: f64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct EdgePair {
    pub(super) p: u32,
    pub(super) q: u32,
    pub(super) compatibility: f64,
    pub(super) same_dir: bool,
}

/// Cooperative Force-Directed Edge Bundling (`FDEB`) solver mutating an [`EdgePolylines`]
/// buffer across [`Layout2d`], [`egraph_core::Layout3d`], and [`egraph_core::LayoutNd`].
#[derive(Debug, Clone)]
pub struct FdebSolver<L: EuclideanLayout = Layout2d<f64>> {
    pub(super) config: FdebConfig,
    pub(super) dims: usize,
    pub(super) num_edges: usize,
    pub(super) expected_total_points: usize,
    pub(super) expected_offsets: Vec<u32>,
    pub(super) edges: Vec<EdgeMeta>,
    pub(super) edge_pairs: Vec<EdgePair>,
    pub(super) cycle_iters: Vec<usize>,
    pub(super) total_iterations: usize,
    pub(super) current_cycle: usize,
    pub(super) cycle_iter: usize,
    pub(super) iter: usize,
    pub(super) alpha: f64,
    pub(super) pts_per_edge: usize,
    pub(super) work: Vec<f64>,
    pub(super) scratch: Vec<f64>,
    pub(super) vel: Vec<f64>,
    pub(super) status: SolverStatus,
    _layout: PhantomData<fn() -> L>,
}

impl<L: EuclideanLayout> FdebSolver<L> {
    /// Constructs an [`FdebSolver`] from an initial [`EdgePolylines`] buffer and [`FdebConfig`].
    ///
    /// Expects each polyline in `polylines` to contain a uniform number of control points
    /// (`>= 2`, typically created via [`EdgePolylines::straight_line_from_graph`]).
    pub fn new(polylines: &EdgePolylines<L>, config: FdebConfig) -> CoreResult<Self> {
        config.validate()?;
        polylines.validate_finite()?;

        let dims = polylines.points().dims();
        if dims == 0 {
            return Err(CoreError::invalid_input(
                "FdebSolver requires coordinate dimensionality >= 1",
            ));
        }

        let offsets = polylines.edge_offsets();
        let num_edges = polylines.num_edges();
        let total_points = polylines.total_points();

        let mut pts_per_edge = 0usize;
        for e in 0..num_edges {
            let count = (offsets[e + 1] - offsets[e]) as usize;
            if count < 2 {
                return Err(CoreError::invalid_input(format!(
                    "FdebSolver requires at least 2 points per edge, got {count} at edge {e}"
                )));
            }
            if e == 0 {
                pts_per_edge = count;
            } else if count != pts_per_edge {
                return Err(CoreError::invalid_input(format!(
                    "FdebSolver requires uniform point counts per edge: expected {pts_per_edge}, got {count} at edge {e}"
                )));
            }
        }

        if num_edges > 0 {
            let factor = 1usize
                .checked_shl(config.cycles as u32)
                .ok_or_else(|| CoreError::invalid_input("FDEB subdivision factor overflow"))?;
            let final_pts_per_edge = (pts_per_edge - 1)
                .checked_mul(factor)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| CoreError::invalid_input("FDEB point count overflow"))?;
            let max_total_pts = num_edges
                .checked_mul(final_pts_per_edge)
                .ok_or_else(|| CoreError::invalid_input("FDEB point buffer size overflow"))?;
            if u32::try_from(max_total_pts).is_err() {
                return Err(CoreError::invalid_input(
                    "FDEB total point count exceeds u32::MAX",
                ));
            }
            if max_total_pts.checked_mul(dims).is_none() {
                return Err(CoreError::invalid_input(
                    "FDEB flat coordinate buffer size overflow",
                ));
            }
        }

        let mut cycle_iters = Vec::with_capacity(config.cycles);
        let mut cur_iter = config.steps_per_cycle;
        let mut total_iterations = 0usize;
        for _ in 0..config.cycles {
            cycle_iters.push(cur_iter);
            total_iterations = total_iterations.saturating_add(cur_iter);
            cur_iter = ((cur_iter as f64) * config.i_step) as usize;
        }

        let mut work = vec![0.0_f64; total_points * dims];
        read_storage_to_work(polylines.points().as_euclidean_storage(), dims, &mut work)?;

        let mut edges = Vec::with_capacity(num_edges);
        for e in 0..num_edges {
            let start = offsets[e] as usize;
            let end = offsets[e + 1] as usize;
            let p1 = &work[start * dims..(start + 1) * dims];
            let p2 = &work[(end - 1) * dims..end * dims];
            let length = norm_diff(p1, p2);
            if !length.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "FDEB edge length is not finite",
                ));
            }
            edges.push(EdgeMeta { length });
        }

        let mut edge_pairs = Vec::new();
        if total_iterations > 0 && num_edges > 1 {
            for p in 0..num_edges {
                let ep = edges[p];
                if ep.length <= MIN_EDGE_LEN {
                    continue;
                }
                let p_start = offsets[p] as usize;
                let p_end = offsets[p + 1] as usize;
                let p1 = &work[p_start * dims..(p_start + 1) * dims];
                let p2 = &work[(p_end - 1) * dims..p_end * dims];

                for (q, &eq) in edges.iter().enumerate().skip(p + 1) {
                    if eq.length <= MIN_EDGE_LEN {
                        continue;
                    }
                    let q_start = offsets[q] as usize;
                    let q_end = offsets[q + 1] as usize;
                    let q1 = &work[q_start * dims..(q_start + 1) * dims];
                    let q2 = &work[(q_end - 1) * dims..q_end * dims];

                    let (c_e, cos_theta) = edge_compatibility(p1, p2, ep.length, q1, q2, eq.length);
                    if !c_e.is_finite() || !cos_theta.is_finite() {
                        return Err(CoreError::numerical_breakdown(
                            "FDEB edge compatibility became non-finite",
                        ));
                    }
                    if c_e >= config.minimum_edge_compatibility {
                        let theta = cos_theta.clamp(-1.0, 1.0).acos();
                        edge_pairs.push(EdgePair {
                            p: p as u32,
                            q: q as u32,
                            compatibility: c_e,
                            same_dir: theta < std::f64::consts::FRAC_PI_2,
                        });
                    }
                }
            }
        }

        let alpha = config.s0;
        let status = if total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            dims,
            num_edges,
            expected_total_points: total_points,
            expected_offsets: offsets.to_vec(),
            edges,
            edge_pairs,
            cycle_iters,
            total_iterations,
            current_cycle: 0,
            cycle_iter: 0,
            iter: 0,
            alpha,
            pts_per_edge,
            work,
            scratch: Vec::new(),
            vel: vec![0.0_f64; total_points * dims],
            status,
            _layout: PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &FdebConfig {
        &self.config
    }

    /// Returns the Euclidean coordinate dimensionality (`2`, `3`, or `LayoutNd::dims()`).
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Returns the number of polylines (edges) managed by the solver.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Returns the number of compatible edge pairs interacting during bundling.
    #[inline]
    pub fn num_compatible_pairs(&self) -> usize {
        self.edge_pairs.len()
    }

    /// Returns the current number of control points per edge (`0` when `num_edges() == 0`).
    #[inline]
    pub fn pts_per_edge(&self) -> usize {
        self.pts_per_edge
    }

    /// Returns the current subdivision cycle index.
    #[inline]
    pub fn current_cycle(&self) -> usize {
        self.current_cycle
    }

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.iter
    }

    /// Returns the total scheduled iteration count.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.total_iterations)
    }

    /// Returns `true` if the solver has reached a terminal status.
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }
}

pub(super) fn read_storage_to_work<S: Scalar>(
    storage: EuclideanStorageRef<'_, S>,
    expected_dims: usize,
    work: &mut [f64],
) -> CoreResult<()> {
    match storage {
        EuclideanStorageRef::Soa2d { x, y } if expected_dims == 2 => {
            for (i, (&vx, &vy)) in x.iter().zip(y.iter()).enumerate() {
                work[2 * i] = vx.to_f64();
                work[2 * i + 1] = vy.to_f64();
            }
            Ok(())
        }
        EuclideanStorageRef::Soa3d { x, y, z } if expected_dims == 3 => {
            for i in 0..x.len() {
                work[3 * i] = x[i].to_f64();
                work[3 * i + 1] = y[i].to_f64();
                work[3 * i + 2] = z[i].to_f64();
            }
            Ok(())
        }
        EuclideanStorageRef::FlatNd { coords, dims, .. } if dims == expected_dims => {
            for (dst, &src) in work.iter_mut().zip(coords.iter()) {
                *dst = src.to_f64();
            }
            Ok(())
        }
        _ => Err(CoreError::invalid_input(
            "mismatched Euclidean storage representation in FdebSolver",
        )),
    }
}

pub(super) fn validate_work_for_scalar<S: Scalar>(work: &[f64]) -> CoreResult<()> {
    for &val in work {
        if !val.is_finite() || !S::from_f64(val).is_finite() {
            return Err(CoreError::numerical_breakdown(
                "FDEB coordinates became non-finite for target scalar",
            ));
        }
    }
    Ok(())
}

pub(super) fn write_work_to_storage<S: Scalar>(
    work: &[f64],
    expected_dims: usize,
    storage: EuclideanStorageMut<'_, S>,
) -> CoreResult<()> {
    match storage {
        EuclideanStorageMut::Soa2d { x, y } if expected_dims == 2 => {
            for i in 0..x.len() {
                x[i] = S::from_f64(work[2 * i]);
                y[i] = S::from_f64(work[2 * i + 1]);
            }
            Ok(())
        }
        EuclideanStorageMut::Soa3d { x, y, z } if expected_dims == 3 => {
            for i in 0..x.len() {
                x[i] = S::from_f64(work[3 * i]);
                y[i] = S::from_f64(work[3 * i + 1]);
                z[i] = S::from_f64(work[3 * i + 2]);
            }
            Ok(())
        }
        EuclideanStorageMut::FlatNd { coords, dims, .. } if dims == expected_dims => {
            for (dst, &src) in coords.iter_mut().zip(work.iter()) {
                *dst = S::from_f64(src);
            }
            Ok(())
        }
        _ => Err(CoreError::invalid_input(
            "mismatched Euclidean storage representation in FdebSolver",
        )),
    }
}
