use egraph_core::CoreResult;

use super::KdQueryScratch;

/// Common spatial nearest-neighbor and radius-query interface across static, dynamic, and
/// future metric-tree indices.
///
/// Query results written into `out` are `(point_index, dist_sq)` pairs using `f64` squared
/// Euclidean distances, sorted deterministically by ascending `(dist_sq, point_index)`.
pub trait NearestNeighborIndex {
    /// Returns the coordinate dimensionality $D \ge 1$ indexed by `self`.
    fn dims(&self) -> usize;

    /// Returns the number of points stored in the index.
    fn len(&self) -> usize;

    /// Returns `true` if the index stores zero points.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reuses `out` to return the `k` nearest points `(index, dist_sq)` to `point`.
    fn query_knn(&self, point: &[f64], k: usize, out: &mut Vec<(usize, f64)>) -> CoreResult<()>;

    /// Reuses `scratch` and `out` to return the `k` nearest points `(index, dist_sq)` to `point`.
    fn query_knn_with_scratch(
        &self,
        point: &[f64],
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let _ = scratch;
        self.query_knn(point, k, out)
    }

    /// Reuses `out` to return the `k` nearest points `(index, dist_sq)` to stored vertex
    /// `point_idx`, excluding `point_idx` itself.
    fn query_knn_excluding_self(
        &self,
        point_idx: usize,
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()>;

    /// Reuses `scratch` and `out` to return the `k` nearest points `(index, dist_sq)` to
    /// stored vertex `point_idx`, excluding `point_idx` itself.
    fn query_knn_excluding_self_with_scratch(
        &self,
        point_idx: usize,
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let _ = scratch;
        self.query_knn_excluding_self(point_idx, k, out)
    }

    /// Returns the single nearest point `(index, dist_sq)` to stored vertex `point_idx`,
    /// excluding `point_idx` itself.
    fn nearest_excluding_self(&self, point_idx: usize) -> CoreResult<Option<(usize, f64)>>;

    /// Reuses `scratch` to return the single nearest point `(index, dist_sq)` to stored
    /// vertex `point_idx`, excluding `point_idx` itself.
    fn nearest_excluding_self_with_scratch(
        &self,
        point_idx: usize,
        scratch: &mut KdQueryScratch,
    ) -> CoreResult<Option<(usize, f64)>> {
        let _ = scratch;
        self.nearest_excluding_self(point_idx)
    }

    /// Reuses `out` to return all points `(index, dist_sq)` within Euclidean `radius >= 0.0`
    /// of `point`.
    fn query_radius(
        &self,
        point: &[f64],
        radius: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()>;

    /// Reuses `out` to return all points `(index, dist_sq)` with `dist_sq <= radius_sq` of `point`.
    fn query_radius_sq(
        &self,
        point: &[f64],
        radius_sq: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()>;

    /// Reuses `scratch` and `out` to return all points `(index, dist_sq)` with
    /// `dist_sq <= radius_sq` of `point`.
    fn query_radius_sq_with_scratch(
        &self,
        point: &[f64],
        radius_sq: f64,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let _ = scratch;
        self.query_radius_sq(point, radius_sq, out)
    }
}

/// Alias matching the unified spatial-index trait contract.
pub use NearestNeighborIndex as SpatialIndex;

macro_rules! impl_nn_index_forward {
    ($ty:ty) => {
        impl<T: NearestNeighborIndex + ?Sized> NearestNeighborIndex for $ty {
            #[inline]
            fn dims(&self) -> usize {
                (**self).dims()
            }

            #[inline]
            fn len(&self) -> usize {
                (**self).len()
            }

            #[inline]
            fn query_knn(
                &self,
                point: &[f64],
                k: usize,
                out: &mut Vec<(usize, f64)>,
            ) -> CoreResult<()> {
                (**self).query_knn(point, k, out)
            }

            #[inline]
            fn query_knn_with_scratch(
                &self,
                point: &[f64],
                k: usize,
                scratch: &mut KdQueryScratch,
                out: &mut Vec<(usize, f64)>,
            ) -> CoreResult<()> {
                (**self).query_knn_with_scratch(point, k, scratch, out)
            }

            #[inline]
            fn query_knn_excluding_self(
                &self,
                point_idx: usize,
                k: usize,
                out: &mut Vec<(usize, f64)>,
            ) -> CoreResult<()> {
                (**self).query_knn_excluding_self(point_idx, k, out)
            }

            #[inline]
            fn query_knn_excluding_self_with_scratch(
                &self,
                point_idx: usize,
                k: usize,
                scratch: &mut KdQueryScratch,
                out: &mut Vec<(usize, f64)>,
            ) -> CoreResult<()> {
                (**self).query_knn_excluding_self_with_scratch(point_idx, k, scratch, out)
            }

            #[inline]
            fn nearest_excluding_self(&self, point_idx: usize) -> CoreResult<Option<(usize, f64)>> {
                (**self).nearest_excluding_self(point_idx)
            }

            #[inline]
            fn nearest_excluding_self_with_scratch(
                &self,
                point_idx: usize,
                scratch: &mut KdQueryScratch,
            ) -> CoreResult<Option<(usize, f64)>> {
                (**self).nearest_excluding_self_with_scratch(point_idx, scratch)
            }

            #[inline]
            fn query_radius(
                &self,
                point: &[f64],
                radius: f64,
                out: &mut Vec<(usize, f64)>,
            ) -> CoreResult<()> {
                (**self).query_radius(point, radius, out)
            }

            #[inline]
            fn query_radius_sq(
                &self,
                point: &[f64],
                radius_sq: f64,
                out: &mut Vec<(usize, f64)>,
            ) -> CoreResult<()> {
                (**self).query_radius_sq(point, radius_sq, out)
            }

            #[inline]
            fn query_radius_sq_with_scratch(
                &self,
                point: &[f64],
                radius_sq: f64,
                scratch: &mut KdQueryScratch,
                out: &mut Vec<(usize, f64)>,
            ) -> CoreResult<()> {
                (**self).query_radius_sq_with_scratch(point, radius_sq, scratch, out)
            }
        }
    };
}

impl_nn_index_forward!(&T);
impl_nn_index_forward!(&mut T);
impl_nn_index_forward!(Box<T>);
