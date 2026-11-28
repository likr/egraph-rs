//! Four-layer geometric trait hierarchy (`LayoutStorage`, `MetricSpace`, `RiemannianLayout`,
//! `EuclideanLayout`) and repulsion backend abstraction (`RepulsionBackend`).

mod euclidean_impls;
mod layout2d_impl;
mod layout_nd_impl;

#[cfg(test)]
mod tests;

use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;

/// Minimal storage contract shared by all node coordinate containers.
pub trait LayoutStorage {
    /// Coordinate scalar type (`f32` or `f64`).
    type Scalar: Scalar;

    /// Returns the number of vertices stored in the layout.
    fn len(&self) -> usize;

    /// Returns `true` if the layout contains zero vertices.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validates that every coordinate stored in the layout is finite.
    fn validate_finite(&self) -> CoreResult<()>;
}

/// Embedding space endowed with a pairwise distance function accumulated in `f64`.
pub trait MetricSpace: LayoutStorage {
    /// Pairwise distance between vertices `i` and `j` in the embedding metric (`f64`).
    fn distance(&self, i: usize, j: usize) -> f64;
}

/// Smooth Riemannian embedding space supporting logarithmic/exponential maps and parallel transport.
pub trait RiemannianLayout: MetricSpace {
    /// Dimension of each vertex's tangent space.
    fn tangent_dim(&self) -> usize;

    /// Writes the tangent vector $\log_i(j)$ at vertex `i` pointing toward `j` into
    /// `out[..self.tangent_dim()]`.
    fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()>;

    /// Updates vertex `i` via the Riemannian exponential map $\exp_i(\delta)$.
    fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()>;

    /// Parallel-transports tangent vector `vec` along the geodesic step `delta` starting at
    /// pre-step vertex `i`.
    ///
    /// Callers must invoke this **before** mutating vertex `i` via [`Self::exp_map`], or use
    /// [`Self::exp_map_and_transport`] which shares the geodesic evaluation atomically.
    fn parallel_transport_into(&self, i: usize, delta: &[f64], vec: &mut [f64]) -> CoreResult<()>;

    /// Atomically transports `vec` along `delta` and advances vertex `i` via `exp_i(delta)`.
    fn exp_map_and_transport(
        &mut self,
        i: usize,
        delta: &[f64],
        vec: &mut [f64],
    ) -> CoreResult<()> {
        self.parallel_transport_into(i, delta, vec)?;
        self.exp_map(i, delta)
    }

    /// Performs a simultaneous pairwise radial update along the geodesic connecting `i` and `j`.
    ///
    /// Positive `delta_r_i` moves vertex `i` along the unit geodesic tangent toward `j`, and
    /// positive `delta_r_j` moves vertex `j` along the unit geodesic tangent toward `i`
    /// (negative values move each endpoint away from the other).
    ///
    /// # Coincident-point contract
    ///
    /// After validating vertex indices and argument finiteness, if `i == j` or
    /// `self.distance(i, j) <= 1e-15`, this method returns `Ok(())` without mutating coordinates
    /// because no canonical unit geodesic tangent exists at zero separation. Solvers that separate
    /// coincident vertices (such as repulsive or hinge solvers) must explicitly jitter collapsed
    /// pairs using their own [`DeterministicRng`](crate::DeterministicRng).
    fn move_pair_radial(
        &mut self,
        i: usize,
        j: usize,
        delta_r_i: f64,
        delta_r_j: f64,
    ) -> CoreResult<()> {
        let n = self.len();
        validate_radial_args(
            n,
            i,
            j,
            delta_r_i,
            delta_r_j,
            "RiemannianLayout::move_pair_radial",
        )?;
        if i == j {
            return Ok(());
        }
        let d = self.distance(i, j);
        if !d.is_finite() {
            return Err(CoreError::non_finite("RiemannianLayout::move_pair_radial"));
        }
        if d <= 1e-15 || (delta_r_i == 0.0 && delta_r_j == 0.0) {
            return Ok(());
        }
        let si = delta_r_i / d;
        let sj = delta_r_j / d;
        if !si.is_finite() || !sj.is_finite() {
            return Err(CoreError::non_finite("RiemannianLayout::move_pair_radial"));
        }
        const STACK_DIM: usize = 8;
        let dim = self.tangent_dim();
        let mut stack_i = [0.0_f64; STACK_DIM];
        let mut stack_j = [0.0_f64; STACK_DIM];
        let mut heap_i = Vec::new();
        let mut heap_j = Vec::new();

        let delta_i: &[f64] = if delta_r_i != 0.0 {
            let buf = if dim <= STACK_DIM {
                &mut stack_i[..dim]
            } else {
                heap_i.resize(dim, 0.0_f64);
                heap_i.as_mut_slice()
            };
            self.log_map_into(i, j, buf)?;
            for v in buf.iter_mut() {
                *v *= si;
                if !v.is_finite() {
                    return Err(CoreError::non_finite("RiemannianLayout::move_pair_radial"));
                }
            }
            buf
        } else {
            &[]
        };

        let delta_j: &[f64] = if delta_r_j != 0.0 {
            let buf = if dim <= STACK_DIM {
                &mut stack_j[..dim]
            } else {
                heap_j.resize(dim, 0.0_f64);
                heap_j.as_mut_slice()
            };
            self.log_map_into(j, i, buf)?;
            for v in buf.iter_mut() {
                *v *= sj;
                if !v.is_finite() {
                    return Err(CoreError::non_finite("RiemannianLayout::move_pair_radial"));
                }
            }
            buf
        } else {
            &[]
        };

        if delta_r_i != 0.0 {
            self.exp_map(i, delta_i)?;
        }
        if delta_r_j != 0.0 {
            self.exp_map(j, delta_j)?;
        }
        Ok(())
    }

    /// Performs a simultaneous pairwise SGD update along the geodesic connecting `i` and `j`.
    ///
    /// Follows the same coincident-point contract as [`Self::move_pair_radial`]: after validating
    /// indices and parameter finiteness, if `i == j` or `self.distance(i, j) <= 1e-15`, this
    /// method returns `Ok(())` without mutating coordinates.
    fn move_pair_toward(
        &mut self,
        i: usize,
        j: usize,
        target_dist: f64,
        step_weight_i: f64,
        step_weight_j: f64,
    ) -> CoreResult<()> {
        let n = self.len();
        validate_pair_and_weights(
            n,
            i,
            j,
            target_dist,
            step_weight_i,
            step_weight_j,
            "RiemannianLayout::move_pair_toward",
        )?;
        if i == j {
            return Ok(());
        }
        let d = self.distance(i, j);
        if !d.is_finite() {
            return Err(CoreError::non_finite("RiemannianLayout::move_pair_toward"));
        }
        if d <= 1e-15 {
            return Ok(());
        }
        let diff = d - target_dist;
        let delta_r_i = step_weight_i * diff;
        let delta_r_j = step_weight_j * diff;
        if !delta_r_i.is_finite() || !delta_r_j.is_finite() {
            return Err(CoreError::non_finite("RiemannianLayout::move_pair_toward"));
        }
        self.move_pair_radial(i, j, delta_r_i, delta_r_j)
    }
}

mod sealed {
    pub trait Sealed {}
    impl<S: super::Scalar> Sealed for crate::layout::Layout2d<S> {}
    impl<S: super::Scalar> Sealed for crate::layout::Layout3d<S> {}
    impl<S: super::Scalar> Sealed for crate::layout::LayoutNd<S> {}
}

/// Zero-copy immutable storage view over sealed Euclidean layouts (`Layout2d`, `Layout3d`, `LayoutNd`).
#[doc(hidden)]
pub enum EuclideanStorageRef<'a, S: Scalar> {
    Soa2d {
        x: &'a [S],
        y: &'a [S],
    },
    Soa3d {
        x: &'a [S],
        y: &'a [S],
        z: &'a [S],
    },
    FlatNd {
        coords: &'a [S],
        num_nodes: usize,
        dims: usize,
    },
}

/// Zero-copy mutable storage view over sealed Euclidean layouts (`Layout2d`, `Layout3d`, `LayoutNd`).
#[doc(hidden)]
pub enum EuclideanStorageMut<'a, S: Scalar> {
    Soa2d {
        x: &'a mut [S],
        y: &'a mut [S],
    },
    Soa3d {
        x: &'a mut [S],
        y: &'a mut [S],
        z: &'a mut [S],
    },
    FlatNd {
        coords: &'a mut [S],
        num_nodes: usize,
        dims: usize,
    },
}

/// Sealed flat Euclidean layout trait restricted to [`crate::Layout2d`], [`crate::Layout3d`], and
/// [`crate::LayoutNd`].
///
/// Sealing permits simultaneous multi-axis `&mut` borrows and zero-overhead spatial tree/FFT
/// access via `EuclideanStorageRef` / `EuclideanStorageMut` without exposing internal layout
/// padding or SIMD alignment contracts.
pub trait EuclideanLayout: RiemannianLayout + sealed::Sealed {
    /// Returns the number of Cartesian coordinate axes.
    fn dims(&self) -> usize;

    /// Returns the coordinate value for `(node, axis)` as `f64`.
    fn coord(&self, node: usize, axis: usize) -> CoreResult<f64>;

    /// Sets the coordinate value for `(node, axis)` after validating bounds and finiteness.
    fn set_coord(&mut self, node: usize, axis: usize, value: f64) -> CoreResult<()>;

    /// Returns an immutable zero-copy storage view for internal spatial kernels.
    #[doc(hidden)]
    fn as_euclidean_storage(&self) -> EuclideanStorageRef<'_, Self::Scalar>;

    /// Returns a mutable zero-copy storage view for internal spatial kernels.
    #[doc(hidden)]
    fn as_euclidean_storage_mut(&mut self) -> EuclideanStorageMut<'_, Self::Scalar>;

    /// Allocates a zero-initialized storage container of the same geometry for `num_nodes`.
    #[doc(hidden)]
    fn zeros_like(&self, num_nodes: usize) -> CoreResult<Self>
    where
        Self: Sized;
}

/// Pluggable single-channel repulsion evaluator for scalar-objective solvers.
///
/// This trait targets solvers whose repulsive step accumulates a single row-major gradient/force
/// buffer (`forces`) alongside an optional scalar objective contribution such as the partition
/// function $Z$ (for example, `MaxentRepulsion::Evaluator` in `egraph-layout-stress` for
/// `MaxentStressSolver`).
///
/// Solvers requiring multi-channel force outputs, solver-specific per-step hyperparameters,
/// multilevel level views, or telemetry instead define dedicated repulsion traits in their solver
/// crates (`TsNetRepulsion`, `TFdpRepulsion`, `ForceRepulsion`, `LinLogRepulsionBackend`,
/// `YifanHuRepulsion`, and `FrRepulsion`).
pub trait RepulsionBackend<L: LayoutStorage + ?Sized> {
    /// Accumulates single-channel repulsive forces/gradients into `forces` (row-major
    /// `layout.len() * dim`) and returns an optional scalar objective contribution
    /// (e.g., partition function $Z$).
    fn accumulate_repulsion(&mut self, layout: &L, forces: &mut [f64]) -> CoreResult<Option<f64>>;
}

impl<L: LayoutStorage + ?Sized, T: RepulsionBackend<L> + ?Sized> RepulsionBackend<L> for &mut T {
    #[inline]
    fn accumulate_repulsion(&mut self, layout: &L, forces: &mut [f64]) -> CoreResult<Option<f64>> {
        (**self).accumulate_repulsion(layout, forces)
    }
}

impl<L: LayoutStorage + ?Sized, T: RepulsionBackend<L> + ?Sized> RepulsionBackend<L> for Box<T> {
    #[inline]
    fn accumulate_repulsion(&mut self, layout: &L, forces: &mut [f64]) -> CoreResult<Option<f64>> {
        (**self).accumulate_repulsion(layout, forces)
    }
}

#[inline]
pub(super) fn validate_radial_args(
    n: usize,
    i: usize,
    j: usize,
    delta_r_i: f64,
    delta_r_j: f64,
    ctx: &'static str,
) -> CoreResult<()> {
    if i >= n {
        return Err(CoreError::index_out_of_bounds(i, n, ctx));
    }
    if j >= n {
        return Err(CoreError::index_out_of_bounds(j, n, ctx));
    }
    if !delta_r_i.is_finite() || !delta_r_j.is_finite() {
        return Err(CoreError::non_finite(ctx));
    }
    Ok(())
}

#[inline]
pub(super) fn validate_pair_and_weights(
    n: usize,
    i: usize,
    j: usize,
    target_dist: f64,
    step_weight_i: f64,
    step_weight_j: f64,
    ctx: &'static str,
) -> CoreResult<()> {
    if i >= n {
        return Err(CoreError::index_out_of_bounds(i, n, ctx));
    }
    if j >= n {
        return Err(CoreError::index_out_of_bounds(j, n, ctx));
    }
    if !target_dist.is_finite() || !step_weight_i.is_finite() || !step_weight_j.is_finite() {
        return Err(CoreError::non_finite(ctx));
    }
    Ok(())
}

#[inline]
pub(super) fn validate_euclidean_tangent_args(
    n: usize,
    dim: usize,
    i: usize,
    delta: &[f64],
    ctx: &'static str,
) -> CoreResult<()> {
    if i >= n {
        return Err(CoreError::index_out_of_bounds(i, n, ctx));
    }
    if delta.len() != dim {
        return Err(CoreError::dimension_mismatch(dim, delta.len(), ctx));
    }
    if !delta.iter().all(|v| v.is_finite()) {
        return Err(CoreError::non_finite(ctx));
    }
    Ok(())
}
