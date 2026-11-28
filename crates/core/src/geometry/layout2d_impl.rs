use crate::error::{CoreError, CoreResult};
use crate::layout::Layout2d;
use crate::scalar::Scalar;

use super::{
    EuclideanLayout, EuclideanStorageMut, EuclideanStorageRef, LayoutStorage, MetricSpace,
    RiemannianLayout, validate_euclidean_tangent_args, validate_pair_and_weights,
    validate_radial_args,
};

#[inline]
fn move_pair_2d_with<S: Scalar>(
    layout: &mut Layout2d<S>,
    i: usize,
    j: usize,
    ctx: &'static str,
    compute_scales: impl FnOnce(f64) -> (f64, f64),
) -> CoreResult<()> {
    let (xs, ys) = layout.xy_mut();
    let xi = xs[i].to_f64();
    let yi = ys[i].to_f64();
    let xj = xs[j].to_f64();
    let yj = ys[j].to_f64();
    if !xi.is_finite() || !yi.is_finite() || !xj.is_finite() || !yj.is_finite() {
        return Err(CoreError::non_finite(ctx));
    }
    if i == j {
        return Ok(());
    }
    let dx = xj - xi;
    let dy = yj - yi;
    let d = dx.hypot(dy);
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
        if !nxi.is_finite() || !nyi.is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
        Some((nxi, nyi))
    } else {
        None
    };
    let next_j = if sj != 0.0 {
        let nxj = S::from_f64(xj - sj * dx);
        let nyj = S::from_f64(yj - sj * dy);
        if !nxj.is_finite() || !nyj.is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
        Some((nxj, nyj))
    } else {
        None
    };
    if let Some((nxi, nyi)) = next_i {
        xs[i] = nxi;
        ys[i] = nyi;
    }
    if let Some((nxj, nyj)) = next_j {
        xs[j] = nxj;
        ys[j] = nyj;
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
            "Layout2d::move_pair_radial",
        )?;
        move_pair_2d_with(self, i, j, "Layout2d::move_pair_radial", |d| {
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
            "Layout2d::move_pair_toward",
        )?;
        move_pair_2d_with(self, i, j, "Layout2d::move_pair_toward", |d| {
            let coeff = (d - target_dist) / d;
            (step_weight_i * coeff, step_weight_j * coeff)
        })
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
