use std::any::Any;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EuclideanLayout, EuclideanStorageRef, Layout2d, Layout3d,
    LayoutNd, LayoutStorage, ManifoldLayout2d, MetricSpace, Scalar,
};
use egraph_spatial_trees::{DynamicKdTree, NearestNeighborIndex, StaticKdTree};

/// Validates that `layout` and `labels` have equal length and all layout coordinates are finite.
#[inline]
pub(super) fn validate_metric_labels<L: LayoutStorage + ?Sized>(
    layout: &L,
    labels: &[u32],
    context: &'static str,
) -> CoreResult<()> {
    if layout.len() != labels.len() {
        return Err(CoreError::dimension_mismatch(
            layout.len(),
            labels.len(),
            context,
        ));
    }
    layout.validate_finite()?;
    Ok(())
}

/// Validates that `graph` and `layout` have identical vertex counts and finite coordinates.
#[inline]
pub(super) fn validate_graph_layout<W, L: LayoutStorage + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
    context: &'static str,
) -> CoreResult<()> {
    if graph.num_nodes() != layout.len() {
        return Err(CoreError::dimension_mismatch(
            graph.num_nodes(),
            layout.len(),
            context,
        ));
    }
    layout.validate_finite()?;
    Ok(())
}

/// Extracts unique non-self-loop graph edges `(u, v)`.
///
/// For undirected graphs, returns canonical pairs with `u < v`. For directed graphs,
/// returns every stored `(u, v)` edge (ignoring self-loops).
pub(super) fn extract_unique_edges<W>(graph: &CsrGraph<W>) -> Vec<(u32, u32)> {
    let n = graph.num_nodes();
    let directed = graph.is_directed();
    let mut edges = Vec::with_capacity(graph.num_edges());
    for u in 0..n {
        let u32_u = u as u32;
        for &v in graph.neighbors(u) {
            if v != u32_u && (directed || u32_u < v) {
                edges.push((u32_u, v));
            }
        }
    }
    edges
}

/// Validates that `index` matches both vertex count and coordinate dimensionality of `layout`.
#[inline]
pub(super) fn validate_index_layout<
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    index: &I,
    layout: &L,
    context: &'static str,
) -> CoreResult<()> {
    if index.len() != layout.len() {
        return Err(CoreError::dimension_mismatch(
            layout.len(),
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

/// Builds a boxed [`NearestNeighborIndex`] from any sealed [`EuclideanLayout`].
pub(super) fn build_spatial_index<L: EuclideanLayout + ?Sized>(
    layout: &L,
) -> CoreResult<Box<dyn NearestNeighborIndex>> {
    layout.validate_finite()?;
    match layout.as_euclidean_storage() {
        EuclideanStorageRef::Soa2d { .. } => Ok(Box::new(
            StaticKdTree::<2, L::Scalar>::from_euclidean(layout)?,
        )),
        EuclideanStorageRef::Soa3d { .. } => Ok(Box::new(
            StaticKdTree::<3, L::Scalar>::from_euclidean(layout)?,
        )),
        EuclideanStorageRef::FlatNd { dims, .. } => match dims {
            1 => Ok(Box::new(StaticKdTree::<1, L::Scalar>::from_euclidean(
                layout,
            )?)),
            2 => Ok(Box::new(StaticKdTree::<2, L::Scalar>::from_euclidean(
                layout,
            )?)),
            3 => Ok(Box::new(StaticKdTree::<3, L::Scalar>::from_euclidean(
                layout,
            )?)),
            4 => Ok(Box::new(StaticKdTree::<4, L::Scalar>::from_euclidean(
                layout,
            )?)),
            5 => Ok(Box::new(StaticKdTree::<5, L::Scalar>::from_euclidean(
                layout,
            )?)),
            6 => Ok(Box::new(StaticKdTree::<6, L::Scalar>::from_euclidean(
                layout,
            )?)),
            7 => Ok(Box::new(StaticKdTree::<7, L::Scalar>::from_euclidean(
                layout,
            )?)),
            8 => Ok(Box::new(StaticKdTree::<8, L::Scalar>::from_euclidean(
                layout,
            )?)),
            _ => Ok(Box::new(DynamicKdTree::<L::Scalar>::from_euclidean(
                layout,
            )?)),
        },
    }
}

/// Attempts to downcast `layout` to a concrete [`EuclideanLayout`] and build a
/// [`NearestNeighborIndex`]. Returns `Ok(None)` for non-Euclidean [`MetricSpace`] types.
pub(super) fn try_build_spatial_index<L: MetricSpace + 'static>(
    layout: &L,
) -> CoreResult<Option<Box<dyn NearestNeighborIndex>>> {
    let any: &dyn Any = layout;
    if let Some(l) = any.downcast_ref::<Layout2d<f64>>() {
        return build_spatial_index(l).map(Some);
    }
    if let Some(l) = any.downcast_ref::<Layout2d<f32>>() {
        return build_spatial_index(l).map(Some);
    }
    if let Some(l) = any.downcast_ref::<Layout3d<f64>>() {
        return build_spatial_index(l).map(Some);
    }
    if let Some(l) = any.downcast_ref::<Layout3d<f32>>() {
        return build_spatial_index(l).map(Some);
    }
    if let Some(l) = any.downcast_ref::<LayoutNd<f64>>() {
        return build_spatial_index(l).map(Some);
    }
    if let Some(l) = any.downcast_ref::<LayoutNd<f32>>() {
        return build_spatial_index(l).map(Some);
    }
    Ok(None)
}

/// Copies row-major `f64` coordinates (`len * dims`) and returns `(coords, dims)`.
pub(super) fn copy_euclidean_coords_f64<L: EuclideanLayout + ?Sized>(
    layout: &L,
) -> (Vec<f64>, usize) {
    let n = layout.len();
    let dims = layout.dims();
    let mut pts = Vec::with_capacity(n * dims);
    match layout.as_euclidean_storage() {
        EuclideanStorageRef::Soa2d { x, y } => {
            for i in 0..n {
                pts.push(x[i].to_f64());
                pts.push(y[i].to_f64());
            }
        }
        EuclideanStorageRef::Soa3d { x, y, z } => {
            for i in 0..n {
                pts.push(x[i].to_f64());
                pts.push(y[i].to_f64());
                pts.push(z[i].to_f64());
            }
        }
        EuclideanStorageRef::FlatNd { coords, .. } => {
            pts.extend(coords.iter().map(|v| v.to_f64()));
        }
    }
    (pts, dims)
}

/// Squared Euclidean distance between two equal-length `f64` slices.
#[inline]
pub(super) fn dist_sq_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f64;
    for axis in 0..a.len() {
        let diff = a[axis] - b[axis];
        sum += diff * diff;
    }
    sum
}

/// Rescaled Euclidean norm of `a - b` avoiding squaring overflow/underflow.
#[inline]
pub(super) fn euclidean_dist_slice(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.len() == 2 {
        return (a[0] - b[0]).hypot(a[1] - b[1]);
    }
    let mut max_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for axis in 0..a.len() {
        let diff = (a[axis] - b[axis]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        sum_sq += diff * diff;
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    if sum_sq == 0.0 || !sum_sq.is_finite() {
        let inv = 1.0 / max_abs;
        let mut scaled_sq = 0.0_f64;
        for axis in 0..a.len() {
            let s = (a[axis] - b[axis]) * inv;
            scaled_sq += s * s;
        }
        return max_abs * scaled_sq.sqrt();
    }
    sum_sq.sqrt()
}

/// Exact diameter $\max_{i < j} d(i, j)$ for an arbitrary [`MetricSpace`].
pub(super) fn metric_diameter<L: MetricSpace + ?Sized>(layout: &L) -> f64 {
    let n = layout.len();
    if n <= 1 {
        return 0.0;
    }
    let mut d_max = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            d_max = d_max.max(layout.distance(i, j));
        }
    }
    d_max
}

/// Resolved concrete space for coordinate/centroid/grid cluster metrics.
pub(super) enum SpaceKind<'a> {
    Euclidean { coords: Vec<f64>, dims: usize },
    ManifoldF64(&'a ManifoldLayout2d<f64>),
    ManifoldF32(&'a ManifoldLayout2d<f32>),
}

/// Classifies `layout` into a concrete coordinate/manifold space or fails fast with
/// [`CoreError::UnsupportedOperation`].
pub(super) fn classify_space<'a, L: MetricSpace + 'static>(
    layout: &'a L,
    context: &'static str,
) -> CoreResult<SpaceKind<'a>> {
    let any: &dyn Any = layout;
    if let Some(l) = any.downcast_ref::<Layout2d<f64>>() {
        let (coords, dims) = copy_euclidean_coords_f64(l);
        return Ok(SpaceKind::Euclidean { coords, dims });
    }
    if let Some(l) = any.downcast_ref::<Layout2d<f32>>() {
        let (coords, dims) = copy_euclidean_coords_f64(l);
        return Ok(SpaceKind::Euclidean { coords, dims });
    }
    if let Some(l) = any.downcast_ref::<Layout3d<f64>>() {
        let (coords, dims) = copy_euclidean_coords_f64(l);
        return Ok(SpaceKind::Euclidean { coords, dims });
    }
    if let Some(l) = any.downcast_ref::<Layout3d<f32>>() {
        let (coords, dims) = copy_euclidean_coords_f64(l);
        return Ok(SpaceKind::Euclidean { coords, dims });
    }
    if let Some(l) = any.downcast_ref::<LayoutNd<f64>>() {
        let (coords, dims) = copy_euclidean_coords_f64(l);
        return Ok(SpaceKind::Euclidean { coords, dims });
    }
    if let Some(l) = any.downcast_ref::<LayoutNd<f32>>() {
        let (coords, dims) = copy_euclidean_coords_f64(l);
        return Ok(SpaceKind::Euclidean { coords, dims });
    }
    if let Some(ml) = any.downcast_ref::<ManifoldLayout2d<f64>>() {
        return Ok(SpaceKind::ManifoldF64(ml));
    }
    if let Some(ml) = any.downcast_ref::<ManifoldLayout2d<f32>>() {
        return Ok(SpaceKind::ManifoldF32(ml));
    }
    Err(CoreError::unsupported(format!(
        "{context} requires a EuclideanLayout or ManifoldLayout2d"
    )))
}
