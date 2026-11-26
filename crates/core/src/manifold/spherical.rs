use std::f64::consts::{PI, TAU};

use crate::scalar::Scalar;

/// Normalizes `(longitude, colatitude)` coordinates onto `[0, 2π) × [0, π]`.
///
/// Colatitudes outside `[0, π]` reflect across the north/south poles while shifting
/// longitude by `π`, preserving the 3D point on the unit sphere.
pub fn project_spherical_coords<S: Scalar>(lon: S, colat: S) -> (S, S) {
    let mut lon_f = lon.to_f64();
    let mut t = colat.to_f64().rem_euclid(TAU);
    if t > PI {
        t = TAU - t;
        lon_f += PI;
    }
    lon_f = lon_f.rem_euclid(TAU);
    if lon_f >= TAU {
        lon_f = 0.0;
    }
    (S::from_f64(lon_f), S::from_f64(t.clamp(0.0, PI)))
}

/// Converts `(longitude, colatitude)` to a 3D unit vector `(x, y, z)` on $S^2$.
///
/// Uses the `egraph-rs` convention where colatitude `0` is the north pole `(0, 1, 0)`.
#[inline]
pub fn spherical_to_cartesian<S: Scalar>(lon: S, colat: S) -> [S; 3] {
    let (slon, clon) = lon.to_f64().sin_cos();
    let (scolat, ccolat) = colat.to_f64().sin_cos();
    [
        S::from_f64(clon * scolat),
        S::from_f64(ccolat),
        S::from_f64(slon * scolat),
    ]
}

/// Projects a 3D vector `(x, y, z)` onto the unit sphere `(longitude, colatitude)` in `[0, 2π) × [0, π]`.
///
/// Zero or non-finite vectors map to the north pole `(0, 0)`.
pub fn cartesian_to_spherical<S: Scalar>(x: S, y: S, z: S) -> (S, S) {
    let xf = x.to_f64();
    let yf = y.to_f64();
    let zf = z.to_f64();
    let norm = (xf * xf + yf * yf + zf * zf).sqrt();
    if !norm.is_finite() || norm <= 1e-18 {
        return (S::ZERO, S::ZERO);
    }
    let nx = xf / norm;
    let ny = (yf / norm).clamp(-1.0, 1.0);
    let nz = zf / norm;
    let colat = ny.acos();
    let lon = if nx * nx + nz * nz <= 1e-24 {
        0.0
    } else {
        let mut l = nz.atan2(nx).rem_euclid(TAU);
        if l >= TAU {
            l = 0.0;
        }
        l
    };
    (S::from_f64(lon), S::from_f64(colat))
}

#[inline]
fn unit_sphere_f64(lon: f64, colat: f64) -> [f64; 3] {
    let (slon, clon) = lon.sin_cos();
    let (scolat, ccolat) = colat.sin_cos();
    [clon * scolat, ccolat, slon * scolat]
}

/// Computes the great-circle geodesic distance on the unit sphere in `[0, π]`.
///
/// Uses `atan2(||p1 × p2||, p1 · p2)` in `f64` for full precision near `0` and `π`.
pub fn spherical_distance<S: Scalar>(lon1: S, colat1: S, lon2: S, colat2: S) -> S {
    let p1 = unit_sphere_f64(lon1.to_f64(), colat1.to_f64());
    let p2 = unit_sphere_f64(lon2.to_f64(), colat2.to_f64());
    let dot = (p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]).clamp(-1.0, 1.0);
    let cx = p1[1] * p2[2] - p1[2] * p2[1];
    let cy = p1[2] * p2[0] - p1[0] * p2[2];
    let cz = p1[0] * p2[1] - p1[1] * p2[0];
    let cross_norm = (cx * cx + cy * cy + cz * cz).sqrt();
    S::from_f64(cross_norm.atan2(dot))
}

/// Computes the tangent vector at `(lon1, colat1)` pointing along the great circle toward `(lon2, colat2)`.
pub fn spherical_log_map<S: Scalar>(lon1: S, colat1: S, lon2: S, colat2: S) -> (S, S) {
    let lon1_f = lon1.to_f64();
    let colat1_f = colat1.to_f64();
    let p1 = unit_sphere_f64(lon1_f, colat1_f);
    let p2 = unit_sphere_f64(lon2.to_f64(), colat2.to_f64());

    let dot = (p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]).clamp(-1.0, 1.0);
    let cx = p1[1] * p2[2] - p1[2] * p2[1];
    let cy = p1[2] * p2[0] - p1[0] * p2[2];
    let cz = p1[0] * p2[1] - p1[1] * p2[0];
    let cross_norm = (cx * cx + cy * cy + cz * cz).sqrt();
    let angle = cross_norm.atan2(dot);
    if angle <= 1e-15 {
        return (S::ZERO, S::ZERO);
    }

    // Orthonormal tangent basis (e_lon, e_colat) at (lon1, colat1).
    let (slon, clon) = lon1_f.sin_cos();
    let (scolat, ccolat) = colat1_f.sin_cos();
    let e_lon = [-slon, 0.0, clon];
    let e_colat = [clon * ccolat, -scolat, slon * ccolat];

    // Tangent direction in R^3: p2 - (p1 · p2) p1.
    let tx = p2[0] - dot * p1[0];
    let ty = p2[1] - dot * p1[1];
    let tz = p2[2] - dot * p1[2];
    let t_norm = (tx * tx + ty * ty + tz * tz).sqrt();
    if t_norm <= 1e-15 {
        // Antipodal fallback: advance southward along the local meridian.
        return (S::ZERO, S::from_f64(angle));
    }
    let scale = angle / t_norm;
    let v_lon = scale * (tx * e_lon[0] + ty * e_lon[1] + tz * e_lon[2]);
    let v_colat = scale * (tx * e_colat[0] + ty * e_colat[1] + tz * e_colat[2]);
    (S::from_f64(v_lon), S::from_f64(v_colat))
}

/// Advances `(lon, colat)` along the great circle determined by tangent vector `(vx, vy)`.
pub fn spherical_exp_map<S: Scalar>(lon: S, colat: S, vx: S, vy: S) -> (S, S) {
    let lon_f = lon.to_f64();
    let colat_f = colat.to_f64();
    let vxf = vx.to_f64();
    let vyf = vy.to_f64();
    let theta = vxf.hypot(vyf);
    if theta <= 1e-15 {
        return project_spherical_coords(lon, colat);
    }

    let p = unit_sphere_f64(lon_f, colat_f);
    let (slon, clon) = lon_f.sin_cos();
    let (scolat, ccolat) = colat_f.sin_cos();
    let e_lon = [-slon, 0.0, clon];
    let e_colat = [clon * ccolat, -scolat, slon * ccolat];

    let dir = [
        (vxf * e_lon[0] + vyf * e_colat[0]) / theta,
        (vxf * e_lon[1] + vyf * e_colat[1]) / theta,
        (vxf * e_lon[2] + vyf * e_colat[2]) / theta,
    ];
    let (st, ct) = theta.sin_cos();
    let qx = ct * p[0] + st * dir[0];
    let qy = ct * p[1] + st * dir[1];
    let qz = ct * p[2] + st * dir[2];
    cartesian_to_spherical(S::from_f64(qx), S::from_f64(qy), S::from_f64(qz))
}
