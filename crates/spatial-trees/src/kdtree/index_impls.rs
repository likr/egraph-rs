use egraph_core::{CoreError, CoreResult, Scalar};

use super::{DynamicKdTree, KdQueryScratch, NearestNeighborIndex, StaticKdTree};

#[inline]
fn slice_to_finite_array<const D: usize>(
    point: &[f64],
    context: &'static str,
) -> CoreResult<[f64; D]> {
    if point.len() != D {
        return Err(CoreError::dimension_mismatch(D, point.len(), context));
    }
    let mut out = [0.0_f64; D];
    for d in 0..D {
        if !point[d].is_finite() {
            return Err(CoreError::non_finite(context));
        }
        out[d] = point[d];
    }
    Ok(out)
}

impl<const D: usize, S: Scalar> NearestNeighborIndex for StaticKdTree<D, S> {
    #[inline]
    fn dims(&self) -> usize {
        D
    }

    #[inline]
    fn len(&self) -> usize {
        StaticKdTree::len(self)
    }

    fn query_knn(&self, point: &[f64], k: usize, out: &mut Vec<(usize, f64)>) -> CoreResult<()> {
        let mut scratch = KdQueryScratch::with_capacity(k);
        self.query_knn_with_scratch(point, k, &mut scratch, out)
    }

    fn query_knn_with_scratch(
        &self,
        point: &[f64],
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let q = slice_to_finite_array::<D>(point, "StaticKdTree::query_knn")?;
        self.k_nearest_f64_with_scratch(&q, k, scratch, out);
        Ok(())
    }

    #[inline]
    fn query_knn_excluding_self(
        &self,
        point_idx: usize,
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        self.k_nearest_excluding_self_into(point_idx, k, out)
    }

    #[inline]
    fn query_knn_excluding_self_with_scratch(
        &self,
        point_idx: usize,
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        self.k_nearest_excluding_self_with_scratch(point_idx, k, scratch, out)
    }

    #[inline]
    fn nearest_excluding_self(&self, point_idx: usize) -> CoreResult<Option<(usize, f64)>> {
        StaticKdTree::nearest_excluding_self(self, point_idx)
    }

    #[inline]
    fn nearest_excluding_self_with_scratch(
        &self,
        point_idx: usize,
        scratch: &mut KdQueryScratch,
    ) -> CoreResult<Option<(usize, f64)>> {
        StaticKdTree::nearest_excluding_self_with_scratch(self, point_idx, scratch)
    }

    fn query_radius(
        &self,
        point: &[f64],
        radius: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let q = slice_to_finite_array::<D>(point, "StaticKdTree::query_radius")?;
        if !radius.is_finite() {
            return Err(CoreError::non_finite("StaticKdTree::query_radius"));
        }
        if radius < 0.0 {
            return Err(CoreError::invalid_input("radius must be >= 0.0"));
        }
        let mut scratch = KdQueryScratch::with_capacity(0);
        self.within_radius_sq_f64_with_scratch(&q, radius * radius, &mut scratch, out)
    }

    fn query_radius_sq(
        &self,
        point: &[f64],
        radius_sq: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let mut scratch = KdQueryScratch::with_capacity(0);
        self.query_radius_sq_with_scratch(point, radius_sq, &mut scratch, out)
    }

    fn query_radius_sq_with_scratch(
        &self,
        point: &[f64],
        radius_sq: f64,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let q = slice_to_finite_array::<D>(point, "StaticKdTree::query_radius_sq")?;
        self.within_radius_sq_f64_with_scratch(&q, radius_sq, scratch, out)
    }
}

impl<S: Scalar> NearestNeighborIndex for DynamicKdTree<S> {
    #[inline]
    fn dims(&self) -> usize {
        DynamicKdTree::dims(self)
    }

    #[inline]
    fn len(&self) -> usize {
        DynamicKdTree::len(self)
    }

    fn query_knn(&self, point: &[f64], k: usize, out: &mut Vec<(usize, f64)>) -> CoreResult<()> {
        let mut scratch = KdQueryScratch::with_capacity(k);
        self.query_knn_with_scratch(point, k, &mut scratch, out)
    }

    fn query_knn_with_scratch(
        &self,
        point: &[f64],
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        self.k_nearest_f64_with_scratch(point, k, scratch, out)
    }

    #[inline]
    fn query_knn_excluding_self(
        &self,
        point_idx: usize,
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        self.k_nearest_excluding_self_into(point_idx, k, out)
    }

    #[inline]
    fn query_knn_excluding_self_with_scratch(
        &self,
        point_idx: usize,
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        self.k_nearest_excluding_self_with_scratch(point_idx, k, scratch, out)
    }

    #[inline]
    fn nearest_excluding_self(&self, point_idx: usize) -> CoreResult<Option<(usize, f64)>> {
        DynamicKdTree::nearest_excluding_self(self, point_idx)
    }

    #[inline]
    fn nearest_excluding_self_with_scratch(
        &self,
        point_idx: usize,
        scratch: &mut KdQueryScratch,
    ) -> CoreResult<Option<(usize, f64)>> {
        DynamicKdTree::nearest_excluding_self_with_scratch(self, point_idx, scratch)
    }

    fn query_radius(
        &self,
        point: &[f64],
        radius: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let mut scratch = KdQueryScratch::with_capacity(0);
        if point.len() != self.dims() {
            return Err(CoreError::dimension_mismatch(
                self.dims(),
                point.len(),
                "DynamicKdTree::query_radius",
            ));
        }
        if !point.iter().all(|v| v.is_finite()) || !radius.is_finite() {
            return Err(CoreError::non_finite("DynamicKdTree::query_radius"));
        }
        if radius < 0.0 {
            return Err(CoreError::invalid_input("radius must be >= 0.0"));
        }
        self.within_radius_sq_f64_with_scratch(point, radius * radius, &mut scratch, out)
    }

    fn query_radius_sq(
        &self,
        point: &[f64],
        radius_sq: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let mut scratch = KdQueryScratch::with_capacity(0);
        self.within_radius_sq_f64_with_scratch(point, radius_sq, &mut scratch, out)
    }

    fn query_radius_sq_with_scratch(
        &self,
        point: &[f64],
        radius_sq: f64,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        self.within_radius_sq_f64_with_scratch(point, radius_sq, scratch, out)
    }
}
