use crate::error::{CoreError, CoreResult};
use crate::layout::{Layout3d, LayoutNd};
use crate::scalar::Scalar;

use super::{
    EuclideanLayout, EuclideanStorageMut, EuclideanStorageRef, LayoutStorage, MetricSpace,
    RiemannianLayout, validate_euclidean_tangent_args, validate_pair_and_weights,
};

#[inline]
fn norm3_f64(dx: f64, dy: f64, dz: f64) -> f64 {
    let m = dx.abs().max(dy.abs()).max(dz.abs());
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let sx = dx / m;
    let sy = dy / m;
    let sz = dz / m;
    m * (sx * sx + sy * sy + sz * sz).sqrt()
}

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

impl<S: Scalar> LayoutStorage for Layout3d<S> {
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

impl<S: Scalar> MetricSpace for Layout3d<S> {
    #[inline]
    fn distance(&self, i: usize, j: usize) -> f64 {
        let dx = self.x()[i].to_f64() - self.x()[j].to_f64();
        let dy = self.y()[i].to_f64() - self.y()[j].to_f64();
        let dz = self.z()[i].to_f64() - self.z()[j].to_f64();
        norm3_f64(dx, dy, dz)
    }
}

impl<S: Scalar> RiemannianLayout for Layout3d<S> {
    #[inline]
    fn tangent_dim(&self) -> usize {
        3
    }

    fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
        let n = self.len();
        if i >= n {
            return Err(CoreError::index_out_of_bounds(
                i,
                n,
                "Layout3d::log_map_into",
            ));
        }
        if j >= n {
            return Err(CoreError::index_out_of_bounds(
                j,
                n,
                "Layout3d::log_map_into",
            ));
        }
        if out.len() < 3 {
            return Err(CoreError::dimension_mismatch(
                3,
                out.len(),
                "Layout3d::log_map_into",
            ));
        }
        out[0] = self.x()[j].to_f64() - self.x()[i].to_f64();
        out[1] = self.y()[j].to_f64() - self.y()[i].to_f64();
        out[2] = self.z()[j].to_f64() - self.z()[i].to_f64();
        Ok(())
    }

    fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
        validate_euclidean_tangent_args(self.len(), 3, i, delta, "Layout3d::exp_map")?;
        let nx = S::from_f64(self.x()[i].to_f64() + delta[0]);
        let ny = S::from_f64(self.y()[i].to_f64() + delta[1]);
        let nz = S::from_f64(self.z()[i].to_f64() + delta[2]);
        if !nx.is_finite() || !ny.is_finite() || !nz.is_finite() {
            return Err(CoreError::non_finite("Layout3d::exp_map"));
        }
        self.x_mut()[i] = nx;
        self.y_mut()[i] = ny;
        self.z_mut()[i] = nz;
        Ok(())
    }

    fn parallel_transport_into(&self, i: usize, delta: &[f64], vec: &mut [f64]) -> CoreResult<()> {
        validate_euclidean_tangent_args(
            self.len(),
            3,
            i,
            delta,
            "Layout3d::parallel_transport_into",
        )?;
        if vec.len() != 3 {
            return Err(CoreError::dimension_mismatch(
                3,
                vec.len(),
                "Layout3d::parallel_transport_into",
            ));
        }
        if !vec.iter().all(|v| v.is_finite()) {
            return Err(CoreError::non_finite("Layout3d::parallel_transport_into"));
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
            "Layout3d::move_pair_toward",
        )?;
        let (xs, ys, zs) = self.xyz_mut();
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let zi = zs[i].to_f64();
        let xj = xs[j].to_f64();
        let yj = ys[j].to_f64();
        let zj = zs[j].to_f64();
        if !xi.is_finite()
            || !yi.is_finite()
            || !zi.is_finite()
            || !xj.is_finite()
            || !yj.is_finite()
            || !zj.is_finite()
        {
            return Err(CoreError::non_finite("Layout3d::move_pair_toward"));
        }
        if i == j {
            return Ok(());
        }
        let dx = xj - xi;
        let dy = yj - yi;
        let dz = zj - zi;
        let d = norm3_f64(dx, dy, dz);
        if !d.is_finite() {
            return Err(CoreError::non_finite("Layout3d::move_pair_toward"));
        }
        if d <= 1e-15 {
            return Ok(());
        }
        let coeff = (d - target_dist) / d;
        let si = step_weight_i * coeff;
        let sj = step_weight_j * coeff;
        let nxi = S::from_f64(xi + si * dx);
        let nyi = S::from_f64(yi + si * dy);
        let nzi = S::from_f64(zi + si * dz);
        let nxj = S::from_f64(xj - sj * dx);
        let nyj = S::from_f64(yj - sj * dy);
        let nzj = S::from_f64(zj - sj * dz);
        if !nxi.is_finite()
            || !nyi.is_finite()
            || !nzi.is_finite()
            || !nxj.is_finite()
            || !nyj.is_finite()
            || !nzj.is_finite()
        {
            return Err(CoreError::non_finite("Layout3d::move_pair_toward"));
        }
        xs[i] = nxi;
        ys[i] = nyi;
        zs[i] = nzi;
        xs[j] = nxj;
        ys[j] = nyj;
        zs[j] = nzj;
        Ok(())
    }
}

impl<S: Scalar> EuclideanLayout for Layout3d<S> {
    #[inline]
    fn dims(&self) -> usize {
        3
    }

    #[inline]
    fn coord(&self, node: usize, axis: usize) -> CoreResult<f64> {
        if node >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.len(),
                "Layout3d::coord",
            ));
        }
        match axis {
            0 => Ok(self.x()[node].to_f64()),
            1 => Ok(self.y()[node].to_f64()),
            2 => Ok(self.z()[node].to_f64()),
            _ => Err(CoreError::index_out_of_bounds(axis, 3, "Layout3d::coord")),
        }
    }

    #[inline]
    fn set_coord(&mut self, node: usize, axis: usize, value: f64) -> CoreResult<()> {
        if node >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.len(),
                "Layout3d::set_coord",
            ));
        }
        if axis >= 3 {
            return Err(CoreError::index_out_of_bounds(
                axis,
                3,
                "Layout3d::set_coord",
            ));
        }
        let converted = S::from_f64(value);
        if !value.is_finite() || !converted.is_finite() {
            return Err(CoreError::non_finite("Layout3d::set_coord"));
        }
        match axis {
            0 => self.x_mut()[node] = converted,
            1 => self.y_mut()[node] = converted,
            _ => self.z_mut()[node] = converted,
        }
        Ok(())
    }

    #[inline]
    fn as_euclidean_storage(&self) -> EuclideanStorageRef<'_, Self::Scalar> {
        EuclideanStorageRef::Soa3d {
            x: self.x(),
            y: self.y(),
            z: self.z(),
        }
    }

    #[inline]
    fn as_euclidean_storage_mut(&mut self) -> EuclideanStorageMut<'_, Self::Scalar> {
        let (x, y, z) = self.xyz_mut();
        EuclideanStorageMut::Soa3d { x, y, z }
    }

    #[inline]
    fn zeros_like(&self, num_nodes: usize) -> CoreResult<Self> {
        Ok(Self::zeros(num_nodes))
    }
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
            coords[base + axis] = next;
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

    fn move_pair_toward(
        &mut self,
        i: usize,
        j: usize,
        target_dist: f64,
        step_weight_i: f64,
        step_weight_j: f64,
    ) -> CoreResult<()> {
        let n = self.num_nodes();
        let d = self.dims();
        validate_pair_and_weights(
            n,
            i,
            j,
            target_dist,
            step_weight_i,
            step_weight_j,
            "LayoutNd::move_pair_toward",
        )?;
        let coords = self.coords_mut();
        let si = i * d;
        let sj = j * d;
        if !coords[si..si + d].iter().all(|v| v.is_finite())
            || !coords[sj..sj + d].iter().all(|v| v.is_finite())
        {
            return Err(CoreError::non_finite("LayoutNd::move_pair_toward"));
        }
        if i == j {
            return Ok(());
        }
        let dist = norm_slice_diff_f64(&coords[si..si + d], &coords[sj..sj + d]);
        if !dist.is_finite() {
            return Err(CoreError::non_finite("LayoutNd::move_pair_toward"));
        }
        if dist <= 1e-15 {
            return Ok(());
        }
        let coeff = (dist - target_dist) / dist;
        let wi = step_weight_i * coeff;
        let wj = step_weight_j * coeff;
        for axis in 0..d {
            let xi = coords[si + axis].to_f64();
            let xj = coords[sj + axis].to_f64();
            let delta = xj - xi;
            let nxi = S::from_f64(xi + wi * delta);
            let nxj = S::from_f64(xj - wj * delta);
            if !nxi.is_finite() || !nxj.is_finite() {
                return Err(CoreError::non_finite("LayoutNd::move_pair_toward"));
            }
            coords[si + axis] = nxi;
            coords[sj + axis] = nxj;
        }
        Ok(())
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
