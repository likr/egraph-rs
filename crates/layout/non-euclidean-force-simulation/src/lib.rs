use petgraph_layout_force_simulation::Point;

pub trait Map {
    fn to_tangent_space(x: (f32, f32), y: (f32, f32)) -> (f32, f32);
    fn from_tangent_space(x: (f32, f32), z: (f32, f32)) -> (f32, f32);

    fn map_to_tangent_space(i: usize, riemann_space: &[Point], tangent_space: &mut [Point]) {
        let n = riemann_space.len();
        let a = {
            let Point { x, y, .. } = riemann_space[i];
            (x, y)
        };
        for j in 0..n {
            let b = {
                let Point { x, y, .. } = riemann_space[j];
                (x, y)
            };
            let (x, y) = Self::to_tangent_space(a, b);
            tangent_space[j].x = x;
            tangent_space[j].y = y;
            tangent_space[j].vx = 0.;
            tangent_space[j].vy = 0.;
        }
    }

    fn update_position(
        i: usize,
        riemann_space: &mut [Point],
        tangent_space: &[Point],
        velocity_decay: f32,
    ) {
        let Point { vx, vy, .. } = tangent_space[i];
        let Point { x: x0, y: y0, .. } = riemann_space[i];
        let (x, y) = Self::from_tangent_space((x0, y0), (vx * velocity_decay, vy * velocity_decay));
        riemann_space[i].x = x;
        riemann_space[i].y = y;
    }
}

pub struct HyperbolicSpace;

impl Map for HyperbolicSpace {
    fn to_tangent_space(x: (f32, f32), y: (f32, f32)) -> (f32, f32) {
        let dx = y.0 - x.0;
        let dy = y.1 - x.1;
        let dr = 1. - x.0 * y.0 - x.1 * y.1;
        let di = x.1 * y.0 - x.0 * y.1;
        let d = dr * dr + di * di;
        let z = ((dr * dx + di * dy) / d, (dr * dy - di * dx) / d);
        let z_norm = (z.0 * z.0 + z.1 * z.1).sqrt();
        if z_norm < 1e-4 {
            return (0., 0.);
        }
        let e = ((1. + z_norm) / (1. - z_norm)).ln();
        if e.is_finite() {
            (z.0 / z_norm * e, z.1 / z_norm * e)
        } else {
            (z.0 / z_norm, z.1 / z_norm)
        }
    }

    fn from_tangent_space(x: (f32, f32), z: (f32, f32)) -> (f32, f32) {
        let z_norm = (z.0 * z.0 + z.1 * z.1).sqrt();
        let y = if z_norm < 1e-4 {
            (0., 0.)
        } else if z_norm.exp().is_infinite() {
            (z.0 / z_norm, z.1 / z_norm)
        } else {
            let e = ((1. - z_norm.exp()) / (1. + z_norm.exp())).abs();
            (z.0 / z_norm * e, z.1 / z_norm * e)
        };
        let dx = -y.0 - x.0;
        let dy = -y.1 - x.1;
        let dr = -1. - x.0 * y.0 - x.1 * y.1;
        let di = x.1 * y.0 - x.0 * y.1;
        let d = dr * dr + di * di;
        let yx = (dr * dx + di * dy) / d;
        let yy = (dr * dy - di * dx) / d;
        let t = 0.99;
        if (yx * yx + yy * yy).sqrt() < t {
            (yx, yy)
        } else {
            (yx * t, yy * t)
        }
    }
}

pub struct SphericalSpace;

impl Map for SphericalSpace {
    fn to_tangent_space(x: (f32, f32), y: (f32, f32)) -> (f32, f32) {
        let ux = (-x.0.sin() * x.1.sin(), 0., x.0.cos() * x.1.sin());
        let vx = (x.0.cos() * x.1.cos(), -x.1.sin(), x.0.sin() * x.1.cos());
        let ey = (y.0.cos() * y.1.sin(), y.1.cos(), y.0.sin() * y.1.sin());
        let d = (x.1.sin() * y.1.sin() * (y.0 - x.0).cos() + x.1.cos() * y.1.cos())
            .clamp(-1., 1.)
            .acos();
        (
            d * (ux.0 * ey.0 + ux.1 * ey.1 + ux.2 * ey.2),
            d * (vx.0 * ey.0 + vx.1 * ey.1 + vx.2 * ey.2),
        )
    }

    fn from_tangent_space(x: (f32, f32), z: (f32, f32)) -> (f32, f32) {
        let ux = (-x.0.sin() * x.1.sin(), 0., x.0.cos() * x.1.sin());
        let vx = (x.0.cos() * x.1.cos(), -x.1.sin(), x.0.sin() * x.1.cos());
        let p = (z.1, -z.0);
        let n = {
            let n = (
                p.0 * ux.0 + p.1 * vx.0,
                p.0 * ux.1 + p.1 * vx.1,
                p.0 * ux.2 + p.1 * vx.2,
            );
            let d = (n.0 * n.0 + n.1 * n.1 + n.2 * n.2).sqrt();
            (n.0 / d, n.1 / d, n.2 / d)
        };
        let ex = (x.0.cos() * x.1.sin(), x.1.cos(), x.0.sin() * x.1.sin());
        let t = -(z.0 * z.0 + z.1 * z.1).sqrt();
        let ey = (
            (n.0 * n.0 * (1. - t.cos()) + t.cos()) * ex.0
                + (n.0 * n.1 * (1. - t.cos()) - n.2 * t.sin()) * ex.1
                + (n.2 * n.0 * (1. - t.cos()) + n.1 * t.sin()) * ex.2,
            (n.0 * n.1 * (1. - t.cos()) + n.2 * t.sin()) * ex.0
                + (n.1 * n.1 * (1. - t.cos()) + t.cos()) * ex.1
                + (n.1 * n.2 * (1. - t.cos()) - n.0 * t.sin()) * ex.2,
            (n.2 * n.0 * (1. - t.cos()) - n.1 * t.sin()) * ex.0
                + (n.1 * n.2 * (1. - t.cos()) + n.0 * t.sin()) * ex.1
                + (n.2 * n.2 * (1. - t.cos()) + t.cos()) * ex.2,
        );
        (ey.2.atan2(ey.0), ey.1.acos())
    }
}
