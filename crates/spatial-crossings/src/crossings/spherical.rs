use std::f64::consts::{FRAC_PI_2, TAU};

use egraph_core::{
    CoreError, CoreResult, CsrGraph, ManifoldKind, ManifoldLayout2d, Scalar, spherical_to_cartesian,
};

use super::CrossingPair;
use super::geometry::extract_csr_edges;

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[derive(Debug, Clone, Copy)]
struct SphereArc {
    edge_id: usize,
    u: u32,
    v: u32,
    a: [f64; 3],
    b: [f64; 3],
    n: [f64; 3],
    sin_len: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
}

impl SphereArc {
    fn new(edge_id: usize, u: u32, v: u32, a: [f64; 3], b: [f64; 3]) -> Option<Self> {
        if u == v {
            return None;
        }
        let cross_ab = cross3(a, b);
        let sin_len = norm3(cross_ab);
        if !sin_len.is_finite() || sin_len <= 1e-12 {
            // Coincident or antipodal endpoints: shortest great-circle arc is degenerate/non-unique.
            return None;
        }
        let n = [
            cross_ab[0] / sin_len,
            cross_ab[1] / sin_len,
            cross_ab[2] / sin_len,
        ];
        let dot_ab = dot3(a, b).clamp(-1.0, 1.0);
        let theta = sin_len.atan2(dot_ab);
        // Orthogonal basis (a, e) in the great-circle plane for exact 3D AABB extrema.
        let e_raw = [
            b[0] - dot_ab * a[0],
            b[1] - dot_ab * a[1],
            b[2] - dot_ab * a[2],
        ];
        let e_norm = norm3(e_raw);
        let e = if e_norm > 1e-15 {
            [e_raw[0] / e_norm, e_raw[1] / e_norm, e_raw[2] / e_norm]
        } else {
            cross3(n, a)
        };

        let [x_min, x_max] = axis_bounds(a[0], e[0], b[0], theta);
        let [y_min, y_max] = axis_bounds(a[1], e[1], b[1], theta);
        let [z_min, z_max] = axis_bounds(a[2], e[2], b[2], theta);

        Some(Self {
            edge_id,
            u,
            v,
            a,
            b,
            n,
            sin_len,
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
        })
    }

    #[inline]
    fn shares_endpoint(&self, other: &Self) -> bool {
        self.u == other.u || self.u == other.v || self.v == other.u || self.v == other.v
    }
}

#[inline]
fn axis_bounds(a_c: f64, e_c: f64, b_c: f64, theta: f64) -> [f64; 2] {
    let mut lo = a_c.min(b_c);
    let mut hi = a_c.max(b_c);
    if e_c != 0.0 || a_c != 0.0 {
        let phi0 = e_c.atan2(a_c);
        for shift in [0.0_f64, std::f64::consts::PI] {
            let phi = (phi0 + shift).rem_euclid(TAU);
            if phi > 0.0 && phi < theta {
                let (sp, cp) = phi.sin_cos();
                let val = cp * a_c + sp * e_c;
                lo = lo.min(val);
                hi = hi.max(val);
            }
        }
    }
    const PAD: f64 = 1e-11;
    [lo - PAD, hi + PAD]
}

#[inline]
fn point_on_shortest_arc(pt: [f64; 3], arc: &SphereArc) -> bool {
    let tol = -1e-12 * arc.sin_len;
    let c1 = dot3(cross3(arc.a, pt), arc.n);
    let c2 = dot3(cross3(pt, arc.b), arc.n);
    c1 >= tol && c2 >= tol
}

#[inline]
fn sphere_arcs_cross(a: &SphereArc, b: &SphereArc) -> Option<(Option<f64>, f64)> {
    let line = cross3(a.n, b.n);
    let line_norm = norm3(line);
    if !line_norm.is_finite() || line_norm <= 1e-12 {
        // Coplanar great circles: no transverse intersection.
        return None;
    }
    let p_pos = [
        line[0] / line_norm,
        line[1] / line_norm,
        line[2] / line_norm,
    ];
    let p_neg = [-p_pos[0], -p_pos[1], -p_pos[2]];

    let hits = (point_on_shortest_arc(p_pos, a) && point_on_shortest_arc(p_pos, b))
        || (point_on_shortest_arc(p_neg, a) && point_on_shortest_arc(p_neg, b));
    if !hits {
        return None;
    }

    let cos_val = dot3(a.n, b.n).clamp(-1.0, 1.0);
    let acute = cos_val.abs().acos().min(FRAC_PI_2);
    let cos_sq = cos_val * cos_val;
    Some((
        Some(if acute == 0.0 { 0.0 } else { acute }),
        if cos_sq == 0.0 { 0.0 } else { cos_sq },
    ))
}

fn build_sphere_arcs<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    context: &'static str,
) -> CoreResult<Vec<SphereArc>> {
    if layout.kind() != ManifoldKind::Spherical {
        return Err(CoreError::unsupported(
            "spherical crossings require ManifoldKind::Spherical",
        ));
    }
    layout.validate_finite()?;
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let pts: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let p = spherical_to_cartesian(xs[i], ys[i]);
            [p[0].to_f64(), p[1].to_f64(), p[2].to_f64()]
        })
        .collect();

    let edges = extract_csr_edges(graph);
    let mut arcs = Vec::with_capacity(edges.len());
    for (edge_id, &(u, v)) in edges.iter().enumerate() {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= n {
            return Err(CoreError::index_out_of_bounds(ui, n, context));
        }
        if vi >= n {
            return Err(CoreError::index_out_of_bounds(vi, n, context));
        }
        if let Some(arc) = SphereArc::new(edge_id, u, v, pts[ui], pts[vi]) {
            arcs.push(arc);
        }
    }
    arcs.sort_unstable_by(|a, b| {
        a.x_min
            .total_cmp(&b.x_min)
            .then_with(|| a.edge_id.cmp(&b.edge_id))
    });
    Ok(arcs)
}

/// Counts exact shortest great-circle geodesic arc crossings on [`ManifoldKind::Spherical`].
///
/// Safely handles shared endpoints, coincident vertices, antipodal endpoints, and coplanar
/// great circles without `NaN` or panics.
pub fn spherical_crossing_number<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<usize> {
    let arcs = build_sphere_arcs(graph, layout, "spherical_crossing_number")?;
    let mut count = 0usize;
    for i in 0..arcs.len() {
        let ai = &arcs[i];
        for aj in &arcs[(i + 1)..] {
            if aj.x_min > ai.x_max {
                break;
            }
            if ai.y_max < aj.y_min
                || aj.y_max < ai.y_min
                || ai.z_max < aj.z_min
                || aj.z_max < ai.z_min
                || ai.shares_endpoint(aj)
            {
                continue;
            }
            if sphere_arcs_cross(ai, aj).is_some() {
                count += 1;
            }
        }
    }
    Ok(count)
}

/// Finds all shortest great-circle geodesic arc crossing pairs on [`ManifoldKind::Spherical`],
/// sorted by `(edge_a, edge_b)` with spherical tangent intersection angles.
pub fn crossing_edges_spherical<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<Vec<CrossingPair>> {
    let arcs = build_sphere_arcs(graph, layout, "crossing_edges_spherical")?;
    let mut out = Vec::new();
    for i in 0..arcs.len() {
        let ai = &arcs[i];
        for aj in &arcs[(i + 1)..] {
            if aj.x_min > ai.x_max {
                break;
            }
            if ai.y_max < aj.y_min
                || aj.y_max < ai.y_min
                || ai.z_max < aj.z_min
                || aj.z_max < ai.z_min
                || ai.shares_endpoint(aj)
            {
                continue;
            }
            if let Some((angle_rad, cos_sq)) = sphere_arcs_cross(ai, aj) {
                if ai.edge_id <= aj.edge_id {
                    out.push(CrossingPair {
                        edge_a: ai.edge_id,
                        edge_b: aj.edge_id,
                        u1: ai.u,
                        v1: ai.v,
                        u2: aj.u,
                        v2: aj.v,
                        angle_rad,
                        cos_sq,
                    });
                } else {
                    out.push(CrossingPair {
                        edge_a: aj.edge_id,
                        edge_b: ai.edge_id,
                        u1: aj.u,
                        v1: aj.v,
                        u2: ai.u,
                        v2: ai.v,
                        angle_rad,
                        cos_sq,
                    });
                }
            }
        }
    }
    out.sort_unstable_by(|a, b| {
        a.edge_a
            .cmp(&b.edge_a)
            .then_with(|| a.edge_b.cmp(&b.edge_b))
    });
    Ok(out)
}
