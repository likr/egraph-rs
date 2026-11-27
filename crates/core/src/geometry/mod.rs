//! Four-layer geometric trait hierarchy (`LayoutStorage`, `MetricSpace`, `RiemannianLayout`,
//! `EuclideanLayout`) and repulsion backend abstraction (`RepulsionBackend`).

mod euclidean_impls;

#[cfg(test)]
mod tests;

use crate::error::{CoreError, CoreResult};
use crate::layout::Layout2d;
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

    /// Performs a simultaneous pairwise SGD update along the geodesic connecting `i` and `j`.
    fn move_pair_toward(
        &mut self,
        i: usize,
        j: usize,
        target_dist: f64,
        step_weight_i: f64,
        step_weight_j: f64,
    ) -> CoreResult<()> {
        let n = self.len();
        if i >= n {
            return Err(CoreError::index_out_of_bounds(
                i,
                n,
                "RiemannianLayout::move_pair_toward",
            ));
        }
        if j >= n {
            return Err(CoreError::index_out_of_bounds(
                j,
                n,
                "RiemannianLayout::move_pair_toward",
            ));
        }
        if !target_dist.is_finite() || !step_weight_i.is_finite() || !step_weight_j.is_finite() {
            return Err(CoreError::non_finite("RiemannianLayout::move_pair_toward"));
        }
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
        let dim = self.tangent_dim();
        let mut delta_i = vec![0.0_f64; dim];
        let mut delta_j = vec![0.0_f64; dim];
        self.log_map_into(i, j, &mut delta_i)?;
        self.log_map_into(j, i, &mut delta_j)?;
        let coeff = (d - target_dist) / d;
        let si = step_weight_i * coeff;
        let sj = step_weight_j * coeff;
        for v in &mut delta_i {
            *v *= si;
        }
        for v in &mut delta_j {
            *v *= sj;
        }
        self.exp_map(i, &delta_i)?;
        self.exp_map(j, &delta_j)
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

/// Sealed flat Euclidean layout trait restricted to [`Layout2d`], [`crate::Layout3d`], and
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

/// Pluggable repulsion/gradient backend for iterative force-directed and neighbor-embedding solvers.
pub trait RepulsionBackend<L: LayoutStorage + ?Sized> {
    /// Accumulates repulsive forces/gradients into `forces` (row-major `layout.len() * dim`)
    /// and returns an optional scalar objective contribution (e.g., partition function $Z$).
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
fn validate_pair_and_weights(
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
fn validate_euclidean_tangent_args(
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

impl<S: Scalar> LayoutStorage for Layout2d<S> {
    type Scalar = S;

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn validate_finite(&self) -> CoreResult<()> {
        self.validate_finite()
    }
}

impl<S: Scalar> MetricSpace for Layout2d<S> {
    #[inline]
    fn distance(&self, i: usize, j: usize) -> f64 {
        let dx = self.x()[i].to_f64() - self.x()[j].to_f64();
        let dy = self.y()[i].to_f64() - self.y()[j].to_f64();
        dx.hypot(dy)
    }
}

impl<S: Scalar> RiemannianLayout for Layout2d<S> {
    #[inline]
    fn tangent_dim(&self) -> usize {
        2
    }

    fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
        let n = self.len();
        if i >= n {
            return Err(CoreError::index_out_of_bounds(
                i,
                n,
                "Layout2d::log_map_into",
            ));
        }
        if j >= n {
            return Err(CoreError::index_out_of_bounds(
                j,
                n,
                "Layout2d::log_map_into",
            ));
        }
        if out.len() < 2 {
            return Err(CoreError::dimension_mismatch(
                2,
                out.len(),
                "Layout2d::log_map_into",
            ));
        }
        out[0] = self.x()[j].to_f64() - self.x()[i].to_f64();
        out[1] = self.y()[j].to_f64() - self.y()[i].to_f64();
        Ok(())
    }

    fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
        validate_euclidean_tangent_args(self.len(), 2, i, delta, "Layout2d::exp_map")?;
        let nx = S::from_f64(self.x()[i].to_f64() + delta[0]);
        let ny = S::from_f64(self.y()[i].to_f64() + delta[1]);
        if !nx.is_finite() || !ny.is_finite() {
            return Err(CoreError::non_finite("Layout2d::exp_map"));
        }
        self.x_mut()[i] = nx;
        self.y_mut()[i] = ny;
        Ok(())
    }

    fn parallel_transport_into(&self, i: usize, delta: &[f64], vec: &mut [f64]) -> CoreResult<()> {
        validate_euclidean_tangent_args(
            self.len(),
            2,
            i,
            delta,
            "Layout2d::parallel_transport_into",
        )?;
        if vec.len() != 2 {
            return Err(CoreError::dimension_mismatch(
                2,
                vec.len(),
                "Layout2d::parallel_transport_into",
            ));
        }
        if !vec.iter().all(|v| v.is_finite()) {
            return Err(CoreError::non_finite("Layout2d::parallel_transport_into"));
        }
        Ok(())
    }

    #[inline]
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
            "Layout2d::move_pair_toward",
        )?;
        let (xs, ys) = self.xy_mut();
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let xj = xs[j].to_f64();
        let yj = ys[j].to_f64();
        if !xi.is_finite() || !yi.is_finite() || !xj.is_finite() || !yj.is_finite() {
            return Err(CoreError::non_finite("Layout2d::move_pair_toward"));
        }
        if i == j {
            return Ok(());
        }
        let dx = xj - xi;
        let dy = yj - yi;
        let d = dx.hypot(dy);
        if !d.is_finite() {
            return Err(CoreError::non_finite("Layout2d::move_pair_toward"));
        }
        if d <= 1e-15 {
            return Ok(());
        }
        let coeff = (d - target_dist) / d;
        let si = step_weight_i * coeff;
        let sj = step_weight_j * coeff;
        let nxi = S::from_f64(xi + si * dx);
        let nyi = S::from_f64(yi + si * dy);
        let nxj = S::from_f64(xj - sj * dx);
        let nyj = S::from_f64(yj - sj * dy);
        if !nxi.is_finite() || !nyi.is_finite() || !nxj.is_finite() || !nyj.is_finite() {
            return Err(CoreError::non_finite("Layout2d::move_pair_toward"));
        }
        xs[i] = nxi;
        ys[i] = nyi;
        xs[j] = nxj;
        ys[j] = nyj;
        Ok(())
    }
}

impl<S: Scalar> EuclideanLayout for Layout2d<S> {
    #[inline]
    fn dims(&self) -> usize {
        2
    }

    #[inline]
    fn coord(&self, node: usize, axis: usize) -> CoreResult<f64> {
        if node >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.len(),
                "Layout2d::coord",
            ));
        }
        match axis {
            0 => Ok(self.x()[node].to_f64()),
            1 => Ok(self.y()[node].to_f64()),
            _ => Err(CoreError::index_out_of_bounds(axis, 2, "Layout2d::coord")),
        }
    }

    #[inline]
    fn set_coord(&mut self, node: usize, axis: usize, value: f64) -> CoreResult<()> {
        if node >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.len(),
                "Layout2d::set_coord",
            ));
        }
        if axis >= 2 {
            return Err(CoreError::index_out_of_bounds(
                axis,
                2,
                "Layout2d::set_coord",
            ));
        }
        let converted = S::from_f64(value);
        if !value.is_finite() || !converted.is_finite() {
            return Err(CoreError::non_finite("Layout2d::set_coord"));
        }
        if axis == 0 {
            self.x_mut()[node] = converted;
        } else {
            self.y_mut()[node] = converted;
        }
        Ok(())
    }

    #[inline]
    fn as_euclidean_storage(&self) -> EuclideanStorageRef<'_, Self::Scalar> {
        EuclideanStorageRef::Soa2d {
            x: self.x(),
            y: self.y(),
        }
    }

    #[inline]
    fn as_euclidean_storage_mut(&mut self) -> EuclideanStorageMut<'_, Self::Scalar> {
        let (x, y) = self.xy_mut();
        EuclideanStorageMut::Soa2d { x, y }
    }

    #[inline]
    fn zeros_like(&self, num_nodes: usize) -> CoreResult<Self> {
        Ok(Self::zeros(num_nodes))
    }
}
