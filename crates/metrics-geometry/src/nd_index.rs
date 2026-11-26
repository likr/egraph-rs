use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, LayoutNd, Scalar};
use egraph_spatial_trees::{DynamicKdTree, KdQueryScratch, StaticKdTree};

use crate::proximity::exact_predicates::euclidean_diameter_2d;

/// Runtime-dimension spatial index over [`LayoutNd`] coordinates in `f64`.
///
/// Uses monomorphized [`StaticKdTree<D, f64>`] for dimensions `1..=8` and [`DynamicKdTree<f64>`]
/// for higher dimensions, preserving deterministic `(dist_sq, index)` tie-breaking.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum NdSpatialIndex {
    D1(StaticKdTree<1, f64>),
    D2(StaticKdTree<2, f64>),
    D3(StaticKdTree<3, f64>),
    D4(StaticKdTree<4, f64>),
    D5(StaticKdTree<5, f64>),
    D6(StaticKdTree<6, f64>),
    D7(StaticKdTree<7, f64>),
    D8(StaticKdTree<8, f64>),
    Dynamic(DynamicKdTree<f64>),
}

impl NdSpatialIndex {
    /// Returns the coordinate dimensionality stored in the index.
    #[inline]
    pub fn dims(&self) -> usize {
        match self {
            Self::D1(_) => 1,
            Self::D2(_) => 2,
            Self::D3(_) => 3,
            Self::D4(_) => 4,
            Self::D5(_) => 5,
            Self::D6(_) => 6,
            Self::D7(_) => 7,
            Self::D8(_) => 8,
            Self::Dynamic(t) => t.dims(),
        }
    }

    /// Returns the number of indexed points.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::D1(t) => t.len(),
            Self::D2(t) => t.len(),
            Self::D3(t) => t.len(),
            Self::D4(t) => t.len(),
            Self::D5(t) => t.len(),
            Self::D6(t) => t.len(),
            Self::D7(t) => t.len(),
            Self::D8(t) => t.len(),
            Self::Dynamic(t) => t.len(),
        }
    }

    /// Returns `true` if the index contains zero points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reuses `out` to return the `k` nearest points `(index, dist_sq)` to `point_idx`
    /// excluding `point_idx` itself.
    pub fn k_nearest_excluding_self_into(
        &self,
        point_idx: usize,
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        match self {
            Self::D1(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D2(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D3(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D4(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D5(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D6(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D7(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::D8(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
            Self::Dynamic(t) => t.k_nearest_excluding_self_into(point_idx, k, out),
        }
    }

    /// Reuses `scratch` and `out` to return the `k` nearest points `(index, dist_sq)` to
    /// `point_idx` excluding `point_idx` itself.
    pub fn k_nearest_excluding_self_with_scratch(
        &self,
        point_idx: usize,
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        match self {
            Self::D1(t) => t.k_nearest_excluding_self_with_scratch(point_idx, k, scratch, out),
            Self::D2(t) => t.k_nearest_excluding_self_with_scratch(point_idx, k, scratch, out),
            Self::D3(t) => t.k_nearest_excluding_self_with_scratch(point_idx, k, scratch, out),
            Self::D4(t) => t.k_nearest_excluding_self_with_scratch(point_idx, k, scratch, out),
            Self::D5(t) => t.k_nearest_excluding_self_with_scratch(point_idx, k, scratch, out),
            Self::D6(t) => t.k_nearest_excluding_self_with_scratch(point_idx, k, scratch, out),
            Self::D7(t) => t.k_nearest_excluding_self_with_scratch(point_idx, k, scratch, out),
            Self::D8(t) => t.k_nearest_excluding_self_with_scratch(point_idx, k, scratch, out),
            Self::Dynamic(t) => t.k_nearest_excluding_self_with_scratch(point_idx, k, scratch, out),
        }
    }

    /// Returns the single nearest point `(index, dist_sq)` to `point_idx` excluding itself.
    pub fn nearest_excluding_self(&self, point_idx: usize) -> CoreResult<Option<(usize, f64)>> {
        match self {
            Self::D1(t) => t.nearest_excluding_self(point_idx),
            Self::D2(t) => t.nearest_excluding_self(point_idx),
            Self::D3(t) => t.nearest_excluding_self(point_idx),
            Self::D4(t) => t.nearest_excluding_self(point_idx),
            Self::D5(t) => t.nearest_excluding_self(point_idx),
            Self::D6(t) => t.nearest_excluding_self(point_idx),
            Self::D7(t) => t.nearest_excluding_self(point_idx),
            Self::D8(t) => t.nearest_excluding_self(point_idx),
            Self::Dynamic(t) => t.nearest_excluding_self(point_idx),
        }
    }

    /// Reuses `scratch` to return the single nearest point `(index, dist_sq)` to `point_idx`
    /// excluding itself.
    pub fn nearest_excluding_self_with_scratch(
        &self,
        point_idx: usize,
        scratch: &mut KdQueryScratch,
    ) -> CoreResult<Option<(usize, f64)>> {
        match self {
            Self::D1(t) => t.nearest_excluding_self(point_idx),
            Self::D2(t) => t.nearest_excluding_self(point_idx),
            Self::D3(t) => t.nearest_excluding_self(point_idx),
            Self::D4(t) => t.nearest_excluding_self(point_idx),
            Self::D5(t) => t.nearest_excluding_self(point_idx),
            Self::D6(t) => t.nearest_excluding_self(point_idx),
            Self::D7(t) => t.nearest_excluding_self(point_idx),
            Self::D8(t) => t.nearest_excluding_self(point_idx),
            Self::Dynamic(t) => t.nearest_excluding_self_with_scratch(point_idx, scratch),
        }
    }

    /// Reuses `out` to return all points `(index, dist_sq)` within Euclidean `radius >= 0.0`.
    pub fn within_radius_into(
        &self,
        query: &[f64],
        radius: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let d = self.dims();
        if query.len() != d {
            return Err(CoreError::dimension_mismatch(
                d,
                query.len(),
                "NdSpatialIndex::within_radius_into",
            ));
        }
        match self {
            Self::D1(t) => t.within_radius_into(&slice_to_array::<1>(query), radius, out),
            Self::D2(t) => t.within_radius_into(&slice_to_array::<2>(query), radius, out),
            Self::D3(t) => t.within_radius_into(&slice_to_array::<3>(query), radius, out),
            Self::D4(t) => t.within_radius_into(&slice_to_array::<4>(query), radius, out),
            Self::D5(t) => t.within_radius_into(&slice_to_array::<5>(query), radius, out),
            Self::D6(t) => t.within_radius_into(&slice_to_array::<6>(query), radius, out),
            Self::D7(t) => t.within_radius_into(&slice_to_array::<7>(query), radius, out),
            Self::D8(t) => t.within_radius_into(&slice_to_array::<8>(query), radius, out),
            Self::Dynamic(t) => t.within_radius_into(query, radius, out),
        }
    }

    /// Reuses `out` to return all points `(index, dist_sq)` with `dist_sq <= radius_sq`.
    pub fn within_radius_sq_into(
        &self,
        query: &[f64],
        radius_sq: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let d = self.dims();
        if query.len() != d {
            return Err(CoreError::dimension_mismatch(
                d,
                query.len(),
                "NdSpatialIndex::within_radius_sq_into",
            ));
        }
        match self {
            Self::D1(t) => t.within_radius_sq_into(&slice_to_array::<1>(query), radius_sq, out),
            Self::D2(t) => t.within_radius_sq_into(&slice_to_array::<2>(query), radius_sq, out),
            Self::D3(t) => t.within_radius_sq_into(&slice_to_array::<3>(query), radius_sq, out),
            Self::D4(t) => t.within_radius_sq_into(&slice_to_array::<4>(query), radius_sq, out),
            Self::D5(t) => t.within_radius_sq_into(&slice_to_array::<5>(query), radius_sq, out),
            Self::D6(t) => t.within_radius_sq_into(&slice_to_array::<6>(query), radius_sq, out),
            Self::D7(t) => t.within_radius_sq_into(&slice_to_array::<7>(query), radius_sq, out),
            Self::D8(t) => t.within_radius_sq_into(&slice_to_array::<8>(query), radius_sq, out),
            Self::Dynamic(t) => t.within_radius_sq_into(query, radius_sq, out),
        }
    }
}

#[inline]
fn slice_to_array<const D: usize>(slice: &[f64]) -> [f64; D] {
    let mut arr = [0.0_f64; D];
    arr.copy_from_slice(&slice[..D]);
    arr
}

fn build_static_tree_f64<const D: usize, S: Scalar>(
    layout: &LayoutNd<S>,
) -> CoreResult<StaticKdTree<D, f64>> {
    let n = layout.num_nodes();
    let coords = layout.coords();
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * D;
        let mut p = [0.0_f64; D];
        for d in 0..D {
            p[d] = coords[base + d].to_f64();
        }
        points.push(p);
    }
    StaticKdTree::<D, f64>::build(&points)
}

/// Validates that `layout` has positive dimensionality, consistent buffer length, and finite coordinates.
#[doc(hidden)]
#[inline]
pub fn validate_layout_nd<S: Scalar>(layout: &LayoutNd<S>) -> CoreResult<()> {
    let dims = layout.dims();
    if dims == 0 {
        return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
    }
    let expected = layout
        .num_nodes()
        .checked_mul(dims)
        .ok_or_else(|| CoreError::invalid_input("LayoutNd size overflow"))?;
    if layout.coords().len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            layout.coords().len(),
            "validate_layout_nd",
        ));
    }
    layout.validate_finite()?;
    Ok(())
}

/// Validates matching vertex counts between `graph` and `layout`, plus finite coordinates in `layout`.
#[doc(hidden)]
#[inline]
pub fn validate_graph_layout_nd<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
) -> CoreResult<()> {
    if graph.num_nodes() != layout.num_nodes() {
        return Err(CoreError::dimension_mismatch(
            graph.num_nodes(),
            layout.num_nodes(),
            "validate_graph_layout_nd",
        ));
    }
    validate_layout_nd(layout)
}

/// Validates that `index` matches both node count and coordinate dimensionality of `layout`.
#[doc(hidden)]
#[inline]
pub fn validate_index_layout_nd<S: Scalar>(
    index: &NdSpatialIndex,
    layout: &LayoutNd<S>,
    context: &'static str,
) -> CoreResult<()> {
    if index.len() != layout.num_nodes() {
        return Err(CoreError::dimension_mismatch(
            layout.num_nodes(),
            index.len(),
            context,
        ));
    }
    if index.dims() != layout.dims() {
        return Err(CoreError::dimension_mismatch(
            layout.dims(),
            index.dims(),
            context,
        ));
    }
    Ok(())
}

/// Builds an [`NdSpatialIndex`] in `f64` from `layout`.
#[doc(hidden)]
pub fn build_nd_spatial_index<S: Scalar>(layout: &LayoutNd<S>) -> CoreResult<NdSpatialIndex> {
    validate_layout_nd(layout)?;
    match layout.dims() {
        1 => Ok(NdSpatialIndex::D1(build_static_tree_f64::<1, S>(layout)?)),
        2 => Ok(NdSpatialIndex::D2(build_static_tree_f64::<2, S>(layout)?)),
        3 => Ok(NdSpatialIndex::D3(build_static_tree_f64::<3, S>(layout)?)),
        4 => Ok(NdSpatialIndex::D4(build_static_tree_f64::<4, S>(layout)?)),
        5 => Ok(NdSpatialIndex::D5(build_static_tree_f64::<5, S>(layout)?)),
        6 => Ok(NdSpatialIndex::D6(build_static_tree_f64::<6, S>(layout)?)),
        7 => Ok(NdSpatialIndex::D7(build_static_tree_f64::<7, S>(layout)?)),
        8 => Ok(NdSpatialIndex::D8(build_static_tree_f64::<8, S>(layout)?)),
        dims => {
            let coords_f64: Vec<f64> = layout.coords().iter().map(|v| v.to_f64()).collect();
            Ok(NdSpatialIndex::Dynamic(DynamicKdTree::<f64>::from_flat(
                &coords_f64,
                layout.num_nodes(),
                dims,
            )?))
        }
    }
}

/// Computes exact Euclidean distance between two equal-length coordinate slices in `f64`,
/// matching `hypot` for 2D and guarding against overflow/underflow for arbitrary `dims`.
#[inline]
pub(crate) fn slice_dist_nd<S: Scalar>(a: &[S], b: &[S]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.len() == 2 {
        return (a[0].to_f64() - b[0].to_f64()).hypot(a[1].to_f64() - b[1].to_f64());
    }
    let mut sum_sq = 0.0_f64;
    let mut any_nonzero = false;
    for k in 0..a.len() {
        let d = a[k].to_f64() - b[k].to_f64();
        if d != 0.0 {
            any_nonzero = true;
        }
        sum_sq += d * d;
    }
    if sum_sq.is_finite() && (sum_sq > 0.0 || !any_nonzero) {
        return sum_sq.sqrt();
    }
    let mut max_abs = 0.0_f64;
    for k in 0..a.len() {
        max_abs = max_abs.max((a[k].to_f64() - b[k].to_f64()).abs());
    }
    if max_abs == 0.0 || !max_abs.is_finite() {
        return max_abs;
    }
    let inv = 1.0 / max_abs;
    let mut scaled_sq = 0.0_f64;
    for k in 0..a.len() {
        let s = (a[k].to_f64() - b[k].to_f64()) * inv;
        scaled_sq += s * s;
    }
    max_abs * scaled_sq.sqrt()
}

/// Computes squared Euclidean distance between two `f64` coordinate slices.
#[inline]
pub(crate) fn dist_sq_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f64;
    for d in 0..a.len() {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sum
}

/// Computes the exact Euclidean diameter $\max_{i, j} \|p_i - p_j\|_2$ of `layout`.
///
/// Uses $O(N \log N)$ rotating calipers when `dims == 2`, and exact $O(N^2 D)$ evaluation
/// with overflow/underflow scaling for other dimensions.
#[doc(hidden)]
pub fn euclidean_diameter_nd<S: Scalar>(layout: &LayoutNd<S>) -> CoreResult<f64> {
    validate_layout_nd(layout)?;
    let n = layout.num_nodes();
    if n <= 1 {
        return Ok(0.0);
    }
    let dims = layout.dims();
    if dims == 2 {
        let layout2d = Layout2d::from_interleaved(layout.coords())?;
        return Ok(euclidean_diameter_2d(&layout2d));
    }

    let coords = layout.coords();
    let mut max_abs = 0.0_f64;
    for &v in coords {
        max_abs = max_abs.max(v.to_f64().abs());
    }
    if max_abs == 0.0 {
        return Ok(0.0);
    }

    let scale = if max_abs > 1e150 || max_abs < 1e-150 {
        1.0 / max_abs
    } else {
        1.0
    };
    let pts: Vec<f64> = coords.iter().map(|v| v.to_f64() * scale).collect();

    let mut best_sq = 0.0_f64;
    for i in 0..n {
        let pi = &pts[i * dims..(i + 1) * dims];
        for j in (i + 1)..n {
            let pj = &pts[j * dims..(j + 1) * dims];
            let d_sq = dist_sq_f64(pi, pj);
            if d_sq > best_sq {
                best_sq = d_sq;
            }
        }
    }
    Ok(best_sq.sqrt() / scale)
}
