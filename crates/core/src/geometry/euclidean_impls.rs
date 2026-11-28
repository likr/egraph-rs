use crate::error::{CoreError, CoreResult};
use crate::layout::Layout3d;
use crate::scalar::Scalar;

use super::{
    EuclideanLayout, EuclideanStorageMut, EuclideanStorageRef, LayoutStorage, MetricSpace,
    RiemannianLayout, validate_euclidean_tangent_args, validate_pair_and_weights,
    validate_radial_args,
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
fn move_pair_3d_with<S: Scalar>(
    layout: &mut Layout3d<S>,
    i: usize,
    j: usize,
    ctx: &'static str,
    compute_scales: impl FnOnce(f64) -> (f64, f64),
) -> CoreResult<()> {
    let (xs, ys, zs) = layout.xyz_mut();
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
        return Err(CoreError::non_finite(ctx));
    }
    if i == j {
        return Ok(());
    }
    let dx = xj - xi;
    let dy = yj - yi;
    let dz = zj - zi;
    let d = norm3_f64(dx, dy, dz);
    if !d.is_finite() {
        return Err(CoreError::non_finite(ctx));
    }
    if d <= 1e-15 {
        return Ok(());
    }
    let (si, sj) = compute_scales(d);
    if !si.is_finite() || !sj.is_finite() {
        return Err(CoreError::non_finite(ctx));
    }
    if si == 0.0 && sj == 0.0 {
        return Ok(());
    }
    let next_i = if si != 0.0 {
        let nxi = S::from_f64(xi + si * dx);
        let nyi = S::from_f64(yi + si * dy);
        let nzi = S::from_f64(zi + si * dz);
        if !nxi.is_finite() || !nyi.is_finite() || !nzi.is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
        Some((nxi, nyi, nzi))
    } else {
        None
    };
    let next_j = if sj != 0.0 {
        let nxj = S::from_f64(xj - sj * dx);
        let nyj = S::from_f64(yj - sj * dy);
        let nzj = S::from_f64(zj - sj * dz);
        if !nxj.is_finite() || !nyj.is_finite() || !nzj.is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
        Some((nxj, nyj, nzj))
    } else {
        None
    };
    if let Some((nxi, nyi, nzi)) = next_i {
        xs[i] = nxi;
        ys[i] = nyi;
        zs[i] = nzi;
    }
    if let Some((nxj, nyj, nzj)) = next_j {
        xs[j] = nxj;
        ys[j] = nyj;
        zs[j] = nzj;
    }
    Ok(())
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
    fn move_pair_radial(
        &mut self,
        i: usize,
        j: usize,
        delta_r_i: f64,
        delta_r_j: f64,
    ) -> CoreResult<()> {
        validate_radial_args(
            self.len(),
            i,
            j,
            delta_r_i,
            delta_r_j,
            "Layout3d::move_pair_radial",
        )?;
        move_pair_3d_with(self, i, j, "Layout3d::move_pair_radial", |d| {
            (delta_r_i / d, delta_r_j / d)
        })
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
        validate_pair_and_weights(
            self.len(),
            i,
            j,
            target_dist,
            step_weight_i,
            step_weight_j,
            "Layout3d::move_pair_toward",
        )?;
        move_pair_3d_with(self, i, j, "Layout3d::move_pair_toward", |d| {
            let coeff = (d - target_dist) / d;
            (step_weight_i * coeff, step_weight_j * coeff)
        })
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
