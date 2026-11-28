use crate::error::{CoreError, CoreResult};
use crate::geometry::{LayoutStorage, MetricSpace, RiemannianLayout};
use crate::scalar::Scalar;

use super::lorentz::{lorentz_distance_f64, lorentz_log_map};
use super::lorentz_transport::lorentz_step_and_transport;
use super::poincare::{clamp_poincare_point, poincare_distance, poincare_log_map};
use super::spherical::{
    cartesian_to_spherical, project_spherical_coords, spherical_distance, spherical_log_map,
    unit_sphere_f64,
};
use super::torus::{torus_delta, wrap_torus_point};
use super::{ManifoldKind, ManifoldLayout2d};

#[inline]
fn validate_tangent_slice(slice: &[f64], expected_len: usize, ctx: &'static str) -> CoreResult<()> {
    if slice.len() != expected_len {
        return Err(CoreError::dimension_mismatch(
            expected_len,
            slice.len(),
            ctx,
        ));
    }
    if !slice[0].is_finite() || !slice[1].is_finite() {
        return Err(CoreError::non_finite(ctx));
    }
    Ok(())
}

#[inline]
fn validate_index_and_point<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    i: usize,
    ctx: &'static str,
) -> CoreResult<(S, S)> {
    let n = layout.len();
    if i >= n {
        return Err(CoreError::index_out_of_bounds(i, n, ctx));
    }
    let x = layout.x()[i];
    let y = layout.y()[i];
    if !x.is_finite() || !y.is_finite() {
        return Err(CoreError::non_finite(ctx));
    }
    Ok((x, y))
}

fn poincare_step_and_transport<S: Scalar>(
    x: S,
    y: S,
    d0: f64,
    d1: f64,
    vec_opt: Option<&mut [f64]>,
) -> (S, S) {
    let (cx, cy) = clamp_poincare_point(x, y);
    let theta = d0.hypot(d1);
    if theta <= 1e-15 {
        return (cx, cy);
    }
    let xx = cx.to_f64();
    let xy = cy.to_f64();
    let scale_u = (0.5 * theta).tanh() / theta;
    let ux = d0 * scale_u;
    let uy = d1 * scale_u;

    let a_r = 1.0 + xx * ux + xy * uy;
    let a_i = xy * ux - xx * uy;
    let a_sq = a_r * a_r + a_i * a_i;

    if let Some(vec) = vec_opt
        && a_sq > 1e-30
    {
        let cos_psi = (a_r * a_r - a_i * a_i) / a_sq;
        let sin_psi = (2.0 * a_r * a_i) / a_sq;
        let w0 = vec[0];
        let w1 = vec[1];
        vec[0] = cos_psi * w0 - sin_psi * w1;
        vec[1] = sin_psi * w0 + cos_psi * w1;
    }

    if a_sq <= 1e-24 {
        return clamp_poincare_point(S::from_f64(ux), S::from_f64(uy));
    }
    let num_x = xx + ux;
    let num_y = xy + uy;
    let rx = (num_x * a_r - num_y * a_i) / a_sq;
    let ry = (num_y * a_r + num_x * a_i) / a_sq;
    clamp_poincare_point(S::from_f64(rx), S::from_f64(ry))
}

fn spherical_step_and_transport<S: Scalar>(
    lon: S,
    colat: S,
    d0: f64,
    d1: f64,
    vec_opt: Option<&mut [f64]>,
) -> (S, S) {
    let theta = d0.hypot(d1);
    if theta <= 1e-15 {
        return project_spherical_coords(lon, colat);
    }

    let lon_f = lon.to_f64();
    let colat_f = colat.to_f64();
    let p = unit_sphere_f64(lon_f, colat_f);
    let (slon, clon) = lon_f.sin_cos();
    let (scolat, ccolat) = colat_f.sin_cos();
    let e_lon_p = [-slon, 0.0, clon];
    let e_colat_p = [clon * ccolat, -scolat, slon * ccolat];

    let u = [
        (d0 * e_lon_p[0] + d1 * e_colat_p[0]) / theta,
        (d0 * e_lon_p[1] + d1 * e_colat_p[1]) / theta,
        (d0 * e_lon_p[2] + d1 * e_colat_p[2]) / theta,
    ];
    let (st, ct) = theta.sin_cos();
    let q = [
        ct * p[0] + st * u[0],
        ct * p[1] + st * u[1],
        ct * p[2] + st * u[2],
    ];
    let (qlon_f, qcolat_f) = cartesian_to_spherical(q[0], q[1], q[2]);
    let (qlon_s, qcolat_s) = project_spherical_coords(S::from_f64(qlon_f), S::from_f64(qcolat_f));

    if let Some(vec) = vec_opt {
        let w = [
            vec[0] * e_lon_p[0] + vec[1] * e_colat_p[0],
            vec[0] * e_lon_p[1] + vec[1] * e_colat_p[1],
            vec[0] * e_lon_p[2] + vec[1] * e_colat_p[2],
        ];
        let w_dot_u = w[0] * u[0] + w[1] * u[1] + w[2] * u[2];
        let sin_half = (0.5 * theta).sin();
        let cos_m1 = -2.0 * sin_half * sin_half;
        let wt = [
            w[0] + w_dot_u * (cos_m1 * u[0] - st * p[0]),
            w[1] + w_dot_u * (cos_m1 * u[1] - st * p[1]),
            w[2] + w_dot_u * (cos_m1 * u[2] - st * p[2]),
        ];
        let (q_slon, q_clon) = qlon_s.to_f64().sin_cos();
        let (q_scolat, q_ccolat) = qcolat_s.to_f64().sin_cos();
        let e_lon_q = [-q_slon, 0.0, q_clon];
        let e_colat_q = [q_clon * q_ccolat, -q_scolat, q_slon * q_ccolat];
        vec[0] = wt[0] * e_lon_q[0] + wt[1] * e_lon_q[1] + wt[2] * e_lon_q[2];
        vec[1] = wt[0] * e_colat_q[0] + wt[1] * e_colat_q[1] + wt[2] * e_colat_q[2];
    }

    (qlon_s, qcolat_s)
}

fn step_point_f64<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    x: S,
    y: S,
    delta0: f64,
    delta1: f64,
    vec_opt: Option<&mut [f64]>,
) -> (S, S) {
    let m = layout.metric_ref();
    let s = m.scale;
    match layout.kind() {
        ManifoldKind::HyperbolicPoincare => {
            poincare_step_and_transport(x, y, delta0 / s, delta1 / s, vec_opt)
        }
        ManifoldKind::HyperbolicLorentz => {
            lorentz_step_and_transport(x, y, delta0 / s, delta1 / s, vec_opt, true)
        }
        ManifoldKind::Spherical => {
            spherical_step_and_transport(x, y, delta0 / s, delta1 / s, vec_opt)
        }
        ManifoldKind::Torus2d => {
            let lx = s * m.aspect;
            let ly = s;
            wrap_torus_point(
                S::from_f64(x.to_f64() + delta0 / lx),
                S::from_f64(y.to_f64() + delta1 / ly),
            )
        }
    }
}

impl<S: Scalar> LayoutStorage for ManifoldLayout2d<S> {
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

impl<S: Scalar> MetricSpace for ManifoldLayout2d<S> {
    fn distance(&self, i: usize, j: usize) -> f64 {
        let x1 = self.x()[i].to_f64();
        let y1 = self.y()[i].to_f64();
        let x2 = self.x()[j].to_f64();
        let y2 = self.y()[j].to_f64();
        let m = self.metric_ref();
        match self.kind() {
            ManifoldKind::HyperbolicPoincare => m.scale * poincare_distance(x1, y1, x2, y2),
            ManifoldKind::HyperbolicLorentz => m.scale * lorentz_distance_f64(x1, y1, x2, y2),
            ManifoldKind::Spherical => m.scale * spherical_distance(x1, y1, x2, y2),
            ManifoldKind::Torus2d => {
                let (dx_norm, dy_norm) = torus_delta(x2, y2, x1, y1);
                let lx = m.scale * m.aspect;
                let ly = m.scale;
                (lx * dx_norm).hypot(ly * dy_norm)
            }
        }
    }
}

#[inline]
fn move_pair_manifold_with<S: Scalar>(
    layout: &mut ManifoldLayout2d<S>,
    i: usize,
    j: usize,
    ctx: &'static str,
    compute_scales: impl FnOnce(f64) -> (f64, f64),
) -> CoreResult<()> {
    let (xi, yi) = validate_index_and_point(layout, i, ctx)?;
    let (xj, yj) = validate_index_and_point(layout, j, ctx)?;
    if i == j {
        return Ok(());
    }
    let d = layout.distance(i, j);
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
    let mut log_ij = [0.0_f64; 2];
    let mut log_ji = [0.0_f64; 2];
    let (di0, di1) = if si != 0.0 {
        layout.log_map_into(i, j, &mut log_ij)?;
        let d0 = si * log_ij[0];
        let d1 = si * log_ij[1];
        if !d0.is_finite() || !d1.is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
        (d0, d1)
    } else {
        (0.0, 0.0)
    };
    let (dj0, dj1) = if sj != 0.0 {
        layout.log_map_into(j, i, &mut log_ji)?;
        let d0 = sj * log_ji[0];
        let d1 = sj * log_ji[1];
        if !d0.is_finite() || !d1.is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
        (d0, d1)
    } else {
        (0.0, 0.0)
    };
    let next_i = if si != 0.0 {
        let (nxi, nyi) = step_point_f64(layout, xi, yi, di0, di1, None);
        if !nxi.is_finite() || !nyi.is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
        Some((nxi, nyi))
    } else {
        None
    };
    let next_j = if sj != 0.0 {
        let (nxj, nyj) = step_point_f64(layout, xj, yj, dj0, dj1, None);
        if !nxj.is_finite() || !nyj.is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
        Some((nxj, nyj))
    } else {
        None
    };
    let (xs, ys) = layout.xy_mut();
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

impl<S: Scalar> RiemannianLayout for ManifoldLayout2d<S> {
    #[inline]
    fn tangent_dim(&self) -> usize {
        2
    }

    fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
        let (x1s, y1s) = validate_index_and_point(self, i, "ManifoldLayout2d::log_map_into")?;
        let (x2s, y2s) = validate_index_and_point(self, j, "ManifoldLayout2d::log_map_into")?;
        if out.len() < 2 {
            return Err(CoreError::dimension_mismatch(
                2,
                out.len(),
                "ManifoldLayout2d::log_map_into",
            ));
        }
        let x1 = x1s.to_f64();
        let y1 = y1s.to_f64();
        let x2 = x2s.to_f64();
        let y2 = y2s.to_f64();
        let m = self.metric_ref();
        match self.kind() {
            ManifoldKind::HyperbolicPoincare => {
                let (vx, vy) = poincare_log_map(x1, y1, x2, y2);
                out[0] = m.scale * vx;
                out[1] = m.scale * vy;
            }
            ManifoldKind::HyperbolicLorentz => {
                let (vx, vy) = lorentz_log_map(x1, y1, x2, y2);
                out[0] = m.scale * vx;
                out[1] = m.scale * vy;
            }
            ManifoldKind::Spherical => {
                let (vx, vy) = spherical_log_map(x1, y1, x2, y2);
                out[0] = m.scale * vx;
                out[1] = m.scale * vy;
            }
            ManifoldKind::Torus2d => {
                let (dx_norm, dy_norm) = torus_delta(x2, y2, x1, y1);
                out[0] = m.scale * m.aspect * dx_norm;
                out[1] = m.scale * dy_norm;
            }
        }
        Ok(())
    }

    fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
        validate_tangent_slice(delta, 2, "ManifoldLayout2d::exp_map")?;
        let (x, y) = validate_index_and_point(self, i, "ManifoldLayout2d::exp_map")?;
        let (nx, ny) = step_point_f64(self, x, y, delta[0], delta[1], None);
        let (xs, ys) = self.xy_mut();
        xs[i] = nx;
        ys[i] = ny;
        Ok(())
    }

    fn parallel_transport_into(&self, i: usize, delta: &[f64], vec: &mut [f64]) -> CoreResult<()> {
        validate_tangent_slice(delta, 2, "ManifoldLayout2d::parallel_transport_into")?;
        validate_tangent_slice(vec, 2, "ManifoldLayout2d::parallel_transport_into")?;
        let (x, y) =
            validate_index_and_point(self, i, "ManifoldLayout2d::parallel_transport_into")?;
        let s = self.metric_ref().scale;
        match self.kind() {
            ManifoldKind::Torus2d => {}
            ManifoldKind::Spherical => {
                let _ = spherical_step_and_transport(x, y, delta[0] / s, delta[1] / s, Some(vec));
            }
            ManifoldKind::HyperbolicPoincare => {
                let _ = poincare_step_and_transport(x, y, delta[0] / s, delta[1] / s, Some(vec));
            }
            ManifoldKind::HyperbolicLorentz => {
                let _ =
                    lorentz_step_and_transport(x, y, delta[0] / s, delta[1] / s, Some(vec), false);
            }
        }
        Ok(())
    }

    fn exp_map_and_transport(
        &mut self,
        i: usize,
        delta: &[f64],
        vec: &mut [f64],
    ) -> CoreResult<()> {
        validate_tangent_slice(delta, 2, "ManifoldLayout2d::exp_map_and_transport")?;
        validate_tangent_slice(vec, 2, "ManifoldLayout2d::exp_map_and_transport")?;
        let (x, y) = validate_index_and_point(self, i, "ManifoldLayout2d::exp_map_and_transport")?;
        let (nx, ny) = step_point_f64(self, x, y, delta[0], delta[1], Some(vec));
        let (xs, ys) = self.xy_mut();
        xs[i] = nx;
        ys[i] = ny;
        Ok(())
    }

    fn move_pair_radial(
        &mut self,
        i: usize,
        j: usize,
        delta_r_i: f64,
        delta_r_j: f64,
    ) -> CoreResult<()> {
        let n = self.len();
        if i >= n {
            return Err(CoreError::index_out_of_bounds(
                i,
                n,
                "ManifoldLayout2d::move_pair_radial",
            ));
        }
        if j >= n {
            return Err(CoreError::index_out_of_bounds(
                j,
                n,
                "ManifoldLayout2d::move_pair_radial",
            ));
        }
        if !delta_r_i.is_finite() || !delta_r_j.is_finite() {
            return Err(CoreError::non_finite("ManifoldLayout2d::move_pair_radial"));
        }
        move_pair_manifold_with(self, i, j, "ManifoldLayout2d::move_pair_radial", |d| {
            (delta_r_i / d, delta_r_j / d)
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
        let n = self.len();
        if i >= n {
            return Err(CoreError::index_out_of_bounds(
                i,
                n,
                "ManifoldLayout2d::move_pair_toward",
            ));
        }
        if j >= n {
            return Err(CoreError::index_out_of_bounds(
                j,
                n,
                "ManifoldLayout2d::move_pair_toward",
            ));
        }
        if !target_dist.is_finite() || !step_weight_i.is_finite() || !step_weight_j.is_finite() {
            return Err(CoreError::non_finite("ManifoldLayout2d::move_pair_toward"));
        }
        move_pair_manifold_with(self, i, j, "ManifoldLayout2d::move_pair_toward", |d| {
            let coeff = (d - target_dist) / d;
            (step_weight_i * coeff, step_weight_j * coeff)
        })
    }
}
