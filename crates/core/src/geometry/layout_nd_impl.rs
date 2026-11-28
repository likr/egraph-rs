use crate::error::{CoreError, CoreResult};
use crate::layout::LayoutNd;
use crate::scalar::Scalar;

use super::{
    EuclideanLayout, EuclideanStorageMut, EuclideanStorageRef, LayoutStorage, MetricSpace,
    RiemannianLayout, validate_euclidean_tangent_args, validate_pair_and_weights,
    validate_radial_args,
};

#[inline]
fn norm_slice_diff_f64<S: Scalar>(a: &[S], b: &[S]) -> f64 {
    let mut m = 0.0_f64;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        let d = (va.to_f64() - vb.to_f64()).abs();
        if !d.is_finite() {
            return d;
        }
        if d > m {
            m = d;
        }
    }
    if m == 0.0 {
        return 0.0;
    }
    let mut sum_sq = 0.0_f64;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        let s = (va.to_f64() - vb.to_f64()) / m;
        sum_sq += s * s;
    }
    m * sum_sq.sqrt()
}

#[inline]
fn move_pair_nd_with<S: Scalar>(
    layout: &mut LayoutNd<S>,
    i: usize,
    j: usize,
    ctx: &'static str,
    compute_scales: impl FnOnce(f64) -> (f64, f64),
) -> CoreResult<()> {
    let d = layout.dims();
    let coords = layout.coords_mut();
    let si = i * d;
    let sj = j * d;
    if !coords[si..si + d].iter().all(|v| v.is_finite())
        || !coords[sj..sj + d].iter().all(|v| v.is_finite())
    {
        return Err(CoreError::non_finite(ctx));
    }
    if i == j {
        return Ok(());
    }
    let dist = norm_slice_diff_f64(&coords[si..si + d], &coords[sj..sj + d]);
    if !dist.is_finite() {
        return Err(CoreError::non_finite(ctx));
    }
    if dist <= 1e-15 {
        return Ok(());
    }
    let (wi, wj) = compute_scales(dist);
    if !wi.is_finite() || !wj.is_finite() {
        return Err(CoreError::non_finite(ctx));
    }
    if wi == 0.0 && wj == 0.0 {
        return Ok(());
    }
    // Two-pass validation ensures atomic no-mutation semantics if any axis overflows `S`.
    for axis in 0..d {
        let xi = coords[si + axis].to_f64();
        let xj = coords[sj + axis].to_f64();
        let delta = xj - xi;
        if wi != 0.0 && !S::from_f64(xi + wi * delta).is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
        if wj != 0.0 && !S::from_f64(xj - wj * delta).is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
    }
    for axis in 0..d {
        let xi = coords[si + axis].to_f64();
        let xj = coords[sj + axis].to_f64();
        let delta = xj - xi;
        if wi != 0.0 {
            coords[si + axis] = S::from_f64(xi + wi * delta);
        }
        if wj != 0.0 {
            coords[sj + axis] = S::from_f64(xj - wj * delta);
        }
    }
    Ok(())
}

impl<S: Scalar> LayoutStorage for LayoutNd<S> {
    type Scalar = S;

    #[inline]
    fn len(&self) -> usize {
        self.num_nodes()
    }

    #[inline]
    fn validate_finite(&self) -> CoreResult<()> {
        self.validate_finite()
    }
}

impl<S: Scalar> MetricSpace for LayoutNd<S> {
    #[inline]
    fn distance(&self, i: usize, j: usize) -> f64 {
        let d = self.dims();
        let coords = self.coords();
        let si = i * d;
        let sj = j * d;
        norm_slice_diff_f64(&coords[si..si + d], &coords[sj..sj + d])
    }
}

impl<S: Scalar> RiemannianLayout for LayoutNd<S> {
    #[inline]
    fn tangent_dim(&self) -> usize {
        self.dims()
    }

    fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
        let n = self.num_nodes();
        let d = self.dims();
        if i >= n {
            return Err(CoreError::index_out_of_bounds(
                i,
                n,
                "LayoutNd::log_map_into",
            ));
        }
        if j >= n {
            return Err(CoreError::index_out_of_bounds(
                j,
                n,
                "LayoutNd::log_map_into",
            ));
        }
        if out.len() < d {
            return Err(CoreError::dimension_mismatch(
                d,
                out.len(),
                "LayoutNd::log_map_into",
            ));
        }
        let coords = self.coords();
        let si = i * d;
        let sj = j * d;
        for axis in 0..d {
            out[axis] = coords[sj + axis].to_f64() - coords[si + axis].to_f64();
        }
        Ok(())
    }

    fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
        let n = self.num_nodes();
        let d = self.dims();
        validate_euclidean_tangent_args(n, d, i, delta, "LayoutNd::exp_map")?;
        let base = i * d;
        let coords = self.coords_mut();
        for axis in 0..d {
            let next = S::from_f64(coords[base + axis].to_f64() + delta[axis]);
            if !next.is_finite() {
                return Err(CoreError::non_finite("LayoutNd::exp_map"));
            }
        }
        for (axis, &delta_val) in delta.iter().enumerate() {
            coords[base + axis] = S::from_f64(coords[base + axis].to_f64() + delta_val);
        }
        Ok(())
    }

    fn parallel_transport_into(&self, i: usize, delta: &[f64], vec: &mut [f64]) -> CoreResult<()> {
        let n = self.num_nodes();
        let d = self.dims();
        validate_euclidean_tangent_args(n, d, i, delta, "LayoutNd::parallel_transport_into")?;
        if vec.len() != d {
            return Err(CoreError::dimension_mismatch(
                d,
                vec.len(),
                "LayoutNd::parallel_transport_into",
            ));
        }
        if !vec.iter().all(|v| v.is_finite()) {
            return Err(CoreError::non_finite("LayoutNd::parallel_transport_into"));
        }
        Ok(())
    }

    fn move_pair_radial(
        &mut self,
        i: usize,
        j: usize,
        delta_r_i: f64,
        delta_r_j: f64,
    ) -> CoreResult<()> {
        validate_radial_args(
            self.num_nodes(),
            i,
            j,
            delta_r_i,
            delta_r_j,
            "LayoutNd::move_pair_radial",
        )?;
        move_pair_nd_with(self, i, j, "LayoutNd::move_pair_radial", |dist| {
            (delta_r_i / dist, delta_r_j / dist)
        })
    }

    fn move_pair_toward(
        &mut self,
        i: usize,
        j: usize,
        target_dist: f64,
        step_weight_i: f64,
        step_weight_j: f64,
    ) -> CoreResult<()> {
        validate_pair_and_weights(
            self.num_nodes(),
            i,
            j,
            target_dist,
            step_weight_i,
            step_weight_j,
            "LayoutNd::move_pair_toward",
        )?;
        move_pair_nd_with(self, i, j, "LayoutNd::move_pair_toward", |dist| {
            let coeff = (dist - target_dist) / dist;
            (step_weight_i * coeff, step_weight_j * coeff)
        })
    }
}

impl<S: Scalar> EuclideanLayout for LayoutNd<S> {
    #[inline]
    fn dims(&self) -> usize {
        self.dims()
    }

    #[inline]
    fn coord(&self, node: usize, axis: usize) -> CoreResult<f64> {
        self.get(node, axis).map(Scalar::to_f64)
    }

    #[inline]
    fn set_coord(&mut self, node: usize, axis: usize, value: f64) -> CoreResult<()> {
        let converted = S::from_f64(value);
        if !value.is_finite() || !converted.is_finite() {
            return Err(CoreError::non_finite("LayoutNd::set_coord"));
        }
        self.set(node, axis, converted)
    }

    #[inline]
    fn as_euclidean_storage(&self) -> EuclideanStorageRef<'_, Self::Scalar> {
        EuclideanStorageRef::FlatNd {
            coords: self.coords(),
            num_nodes: self.num_nodes(),
            dims: self.dims(),
        }
    }

    #[inline]
    fn as_euclidean_storage_mut(&mut self) -> EuclideanStorageMut<'_, Self::Scalar> {
        let num_nodes = self.num_nodes();
        let dims = self.dims();
        EuclideanStorageMut::FlatNd {
            coords: self.coords_mut(),
            num_nodes,
            dims,
        }
    }

    #[inline]
    fn zeros_like(&self, num_nodes: usize) -> CoreResult<Self> {
        Self::zeros(num_nodes, self.dims())
    }
}
