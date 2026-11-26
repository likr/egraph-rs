mod coherence_and_weights;
mod golden_and_boundary;
mod mec_and_ties;
mod numerical_stability;
mod purity_and_limits;
mod validation_and_invariance;

use std::f64::consts::PI;

use egraph_core::{DeterministicRng, Layout2d};

use super::{ClusterAreaDetail, ClusterAreaReport};

pub(super) fn sample_disk_points(
    rng: &mut DeterministicRng,
    cx: f64,
    cy: f64,
    radius: f64,
    count: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::with_capacity(count);
    let mut ys = Vec::with_capacity(count);
    for _ in 0..count {
        let r = radius * rng.next_f64().sqrt();
        let theta = 2.0 * PI * rng.next_f64();
        xs.push(cx + r * theta.cos());
        ys.push(cy + r * theta.sin());
    }
    (xs, ys)
}

pub(super) fn make_ideal_10_disks(
    n_per_cluster: usize,
    radius: f64,
    spacing: f64,
) -> (Layout2d<f64>, Vec<u32>) {
    let mut rng = DeterministicRng::from_seed_u64(42);
    let mut xs = Vec::with_capacity(10 * n_per_cluster);
    let mut ys = Vec::with_capacity(10 * n_per_cluster);
    let mut labels = Vec::with_capacity(10 * n_per_cluster);
    for k in 0..10usize {
        let gx = (k % 5) as f64 * spacing;
        let gy = (k / 5) as f64 * spacing;
        let (dx, dy) = sample_disk_points(&mut rng, gx, gy, radius, n_per_cluster);
        xs.extend(dx);
        ys.extend(dy);
        labels.extend(std::iter::repeat_n(k as u32, n_per_cluster));
    }
    (Layout2d::from_xy(xs, ys).unwrap(), labels)
}

pub(super) fn assert_detail_unit_interval(detail: &ClusterAreaDetail, ctx: &str) {
    let r = &detail.report;
    for (name, val) in [
        ("caf", r.caf),
        ("purity_adj", r.purity_adj),
        ("prop_vis", r.prop_vis),
        ("res_density", r.res_density),
        ("res_deconv", r.res_deconv),
        ("res_combined", r.res_combined),
        ("spatial_coherence", r.spatial_coherence),
        ("in_frac", r.in_frac),
    ] {
        assert!(
            val.is_finite() && (0.0..=1.0).contains(&val),
            "{ctx}: report.{name}={val:?} out of [0, 1]"
        );
    }
    for (vec_name, vec) in [
        ("p", &detail.p),
        ("q", &detail.q),
        ("purity", &detail.purity),
        ("resolution", &detail.resolution),
        ("res_density", &detail.res_density),
        ("res_deconv", &detail.res_deconv),
        ("caf_per_cluster", &detail.caf_per_cluster),
        ("coherence", &detail.coherence),
    ] {
        for (k, &val) in vec.iter().enumerate() {
            assert!(
                val.is_finite() && (0.0..=1.0).contains(&val),
                "{ctx}: detail.{vec_name}[{k}]={val:?} out of [0, 1]"
            );
        }
    }
}

pub(super) fn assert_report_close(
    a: &ClusterAreaReport,
    b: &ClusterAreaReport,
    tol: f64,
    ctx: &str,
) {
    assert_eq!(a.num_clusters, b.num_clusters, "{ctx}: num_clusters");
    assert_eq!(a.num_nodes, b.num_nodes, "{ctx}: num_nodes");
    assert_eq!(a.g_eff, b.g_eff, "{ctx}: g_eff");
    for (name, va, vb) in [
        ("caf", a.caf, b.caf),
        ("purity_adj", a.purity_adj, b.purity_adj),
        ("prop_vis", a.prop_vis, b.prop_vis),
        ("res_density", a.res_density, b.res_density),
        ("res_deconv", a.res_deconv, b.res_deconv),
        ("res_combined", a.res_combined, b.res_combined),
        (
            "spatial_coherence",
            a.spatial_coherence,
            b.spatial_coherence,
        ),
        ("in_frac", a.in_frac, b.in_frac),
        (
            "effective_boundary_r95",
            a.effective_boundary_r95,
            b.effective_boundary_r95,
        ),
    ] {
        assert!(
            (va - vb).abs() < tol,
            "{ctx}: {name} diff={} >= {tol} (a={va}, b={vb})",
            (va - vb).abs()
        );
    }
}
