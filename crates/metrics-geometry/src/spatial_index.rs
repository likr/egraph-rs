use std::any::Any;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EuclideanLayout, EuclideanStorageRef, Layout2d, Layout3d,
    LayoutNd, LayoutStorage, MetricSpace, Scalar,
};
use egraph_spatial_trees::{DynamicKdTree, NearestNeighborIndex, StaticKdTree};

use crate::proximity::exact_predicates::euclidean_diameter_2d_slices;

/// Validates that `graph` and `layout` have identical vertex counts and finite coordinates.
#[inline]
pub(crate) fn validate_graph_layout<W, L: LayoutStorage + ?Sized>(
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

/// Validates that `index` matches both vertex count and coordinate dimensionality of `layout`.
#[inline]
pub(crate) fn validate_index_layout<
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
pub(crate) fn build_spatial_index<L: EuclideanLayout + ?Sized>(
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

/// Attempts to downcast `layout` to a concrete Euclidean layout and build a [`NearestNeighborIndex`].
///
/// Returns `Ok(None)` for non-Euclidean [`MetricSpace`] types such as `ManifoldLayout2d` or
/// external metric spaces.
pub(crate) fn try_build_spatial_index<L: MetricSpace + 'static>(
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

/// Copies row-major `f64` coordinates from any [`EuclideanLayout`] (`len * dims`).
pub(crate) fn copy_euclidean_coords_f64<L: EuclideanLayout + ?Sized>(layout: &L) -> Vec<f64> {
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
    pts
}

/// Computes squared Euclidean distance between two equal-length `f64` coordinate slices.
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

/// Computes exact diameter $\max_{i < j} d(i, j)$ for an arbitrary [`MetricSpace`].
pub(crate) fn metric_diameter<L: MetricSpace + ?Sized>(layout: &L) -> f64 {
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

/// Computes the exact Euclidean diameter $\max_{i, j} \|p_i - p_j\|_2$ of `layout`.
///
/// Uses $O(N \log N)$ rotating calipers when `dims() == 2`, and exact $O(N^2 D)$ evaluation
/// with overflow/underflow scaling for other dimensions.
pub(crate) fn euclidean_diameter<L: EuclideanLayout + ?Sized>(layout: &L) -> CoreResult<f64> {
    layout.validate_finite()?;
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }
    match layout.as_euclidean_storage() {
        EuclideanStorageRef::Soa2d { x, y } => {
            return Ok(euclidean_diameter_2d_slices(x, y));
        }
        EuclideanStorageRef::FlatNd {
            coords, dims: 2, ..
        } => {
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            for i in 0..n {
                xs.push(coords[2 * i].to_f64());
                ys.push(coords[2 * i + 1].to_f64());
            }
            return Ok(euclidean_diameter_2d_slices(&xs, &ys));
        }
        _ => {}
    }

    let pts = copy_euclidean_coords_f64(layout);
    let dims = layout.dims();
    let mut max_abs = 0.0_f64;
    for &v in &pts {
        max_abs = max_abs.max(v.abs());
    }
    if max_abs == 0.0 {
        return Ok(0.0);
    }
    let scale = if max_abs > 1e150 || max_abs < 1e-150 {
        1.0 / max_abs
    } else {
        1.0
    };
    let scaled: Vec<f64> = if scale != 1.0 {
        pts.into_iter().map(|v| v * scale).collect()
    } else {
        pts
    };

    let mut best_sq = 0.0_f64;
    for i in 0..n {
        let pi = &scaled[i * dims..(i + 1) * dims];
        for j in (i + 1)..n {
            let pj = &scaled[j * dims..(j + 1) * dims];
            let d_sq = dist_sq_f64(pi, pj);
            if d_sq > best_sq {
                best_sq = d_sq;
            }
        }
    }
    Ok(best_sq.sqrt() / scale)
}
