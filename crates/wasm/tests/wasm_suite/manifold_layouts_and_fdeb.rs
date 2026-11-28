use crate::util;

use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::edge_polylines::JsEdgePolylines;
use egraph_wasm::fdeb::JsFdebSolver;
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::manifold::JsManifoldLayout2d;
use egraph_wasm::manifold_geodesic::{
    js_lorentz_distance, js_lorentz_exp_map, js_lorentz_log_map, js_poincare_distance,
    js_poincare_mobius_add, js_spherical_distance, js_torus_distance, js_wrap_torus_point,
};
use egraph_wasm::sgd::JsSgdSolver;
use util::fixtures::make_opts;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn manifold_layout_and_sgd_manifold_all_three_geometries() {
    // 1. ManifoldLayout2d constructors, aliases, projections, geodesic distances, mutators
    let hyp_z = JsManifoldLayout2d::zeros("poincare", 3.0).expect("zeros poincare");
    assert_eq!(hyp_z.kind(), "hyperbolicPoincare");
    assert_eq!(hyp_z.len(), 3);
    assert!(!hyp_z.is_empty());

    let mut hyp = JsManifoldLayout2d::from_coords("hyperbolicPoincare", &[3.0, 4.0, 0.0, 0.0])
        .expect("fromCoords hyp");
    assert_eq!(hyp.kind(), "hyperbolicPoincare");
    let r0 = hyp.x(0.0).unwrap().hypot(hyp.y(0.0).unwrap());
    assert!(r0 < 1.0 && r0 > 0.99);
    assert!(hyp.geodesic_distance(0.0, 1.0).unwrap() > 1.0);
    hyp.set_point(1.0, 0.25, -0.25).unwrap();
    assert!((hyp.x(1.0).unwrap() - 0.25).abs() < 1e-6);
    let euc = hyp.to_layout();
    assert_eq!(euc.len(), 2);
    let hyp_from_l = JsManifoldLayout2d::from_layout("hyperbolic", &euc).unwrap();
    assert_eq!(hyp_from_l.coords(), hyp.coords());

    let sph = JsManifoldLayout2d::from_coords("sphere", &[-0.5, -0.25, 1.0, 1.5]).unwrap();
    assert_eq!(sph.kind(), "spherical");
    assert!((0.0..std::f32::consts::TAU).contains(&sph.x(0.0).unwrap()));
    assert!((0.0..=std::f32::consts::PI).contains(&sph.y(0.0).unwrap()));

    let tor = JsManifoldLayout2d::from_coords("torus2d", &[1.2, -0.1, 0.9, 0.2]).unwrap();
    let mut tor_mut = tor.clone();
    assert_eq!(tor_mut.kind(), "torus");
    assert!((tor_mut.x(0.0).unwrap() - 0.2).abs() < 1e-6);
    assert!((tor_mut.y(0.0).unwrap() - 0.9).abs() < 1e-6);
    tor_mut.project_all().unwrap();
    assert!(tor_mut.is_all_finite());

    // Poincare Mobius centroid centering & safe initialization
    let m_rand = JsManifoldLayout2d::random_init(
        "hyperbolic",
        8.0,
        &make_opts(&[
            ("scale", JsValue::from_f64(0.25)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(m_rand.len(), 8);
    assert!(m_rand.is_all_finite());
    let big_euc = JsLayout2d::from_coords(&[10.0, -5.0, 20.0, 5.0, 30.0, 15.0]).unwrap();
    let mut m_scaled =
        JsManifoldLayout2d::from_layout_scaled("hyperbolic", &big_euc, Some(0.7)).unwrap();
    let d_before = m_scaled.geodesic_distance(0.0, 1.0).unwrap();
    m_scaled.center_poincare_centroid().unwrap();
    let c = m_scaled.poincare_centroid().unwrap();
    assert!(c[0].hypot(c[1]) < 1e-4);
    assert!((m_scaled.geodesic_distance(0.0, 1.0).unwrap() - d_before).abs() < 1e-3);
    m_scaled.recenter_poincare_at(2.0).unwrap();
    assert!(m_scaled.x(2.0).unwrap().hypot(m_scaled.y(2.0).unwrap()) < 1e-4);
    let m_add = js_poincare_mobius_add(-0.3, 0.4, 0.3, -0.4);
    assert!(m_add[0].hypot(m_add[1]) < 1e-12);
    assert!(tor_mut.poincare_centroid().is_err());
    assert!(tor_mut.recenter_poincare(0.1, 0.1).is_err());
    assert!(tor_mut.recenter_poincare_at(0.0).is_err());
    assert!(tor_mut.center_poincare_centroid().is_err());

    // Invalid manifold inputs + snake_case rejection
    assert!(JsManifoldLayout2d::zeros("kleinBottle", 2.0).is_err());
    assert!(JsManifoldLayout2d::zeros("hyperbolic_poincare", 2.0).is_err());
    assert!(JsManifoldLayout2d::from_coords("torus", &[0.0, 1.0, 2.0]).is_err());
    assert!(JsManifoldLayout2d::from_coords("spherical", &[f32::NAN, 0.0]).is_err());
    assert!(tor_mut.geodesic_distance(0.0, 9.0).is_err());
    assert!(tor_mut.set_point(0.0, f32::INFINITY, 0.0).is_err());
    assert!(tor_mut.x(5.0).is_err());

    // 2. SgdSolver stepManifold == runManifold across all 3 manifolds
    let cyc = JsCsrGraph::from_edge_list(
        6.0,
        &[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        None,
        &JsValue::UNDEFINED,
        None,
    )
    .unwrap();
    let sched_opts = make_opts(&[
        ("iterations", JsValue::from_f64(20.0)),
        ("eps", JsValue::from_f64(0.02)),
        ("kind", JsValue::from_str("exponential")),
        ("seed", JsValue::from_f64(19.0)),
    ]);

    let man_opts = make_opts(&[
        ("target", JsValue::from_str("manifold2d")),
        ("iterations", JsValue::from_f64(20.0)),
        ("eps", JsValue::from_f64(0.02)),
        ("kind", JsValue::from_str("exponential")),
        ("seed", JsValue::from_f64(19.0)),
    ]);
    let _ = &sched_opts;

    for (kind, init_coords) in [
        (
            "hyperbolic",
            vec![
                0.1f32, 0.0, 0.0, 0.1, -0.1, 0.0, 0.0, -0.1, 0.15, 0.15, -0.15, -0.15,
            ],
        ),
        (
            "spherical",
            vec![
                0.1f32, 1.2, 0.2, 1.3, 0.3, 1.1, 0.4, 1.4, 0.5, 1.25, 0.6, 1.35,
            ],
        ),
        (
            "torus",
            vec![
                0.05f32, 0.92, 0.85, 0.10, 0.45, 0.55, 0.12, 0.48, 0.3, 0.7, 0.6, 0.2,
            ],
        ),
    ] {
        let mut s_run = JsSgdSolver::full(&cyc, &man_opts).unwrap();
        let mut s_step = JsSgdSolver::full(&cyc, &man_opts).unwrap();
        let mut m_run = JsManifoldLayout2d::from_coords(kind, &init_coords).unwrap();
        let mut m_step = JsManifoldLayout2d::from_coords(kind, &init_coords).unwrap();

        assert_eq!(s_run.run_manifold(&mut m_run).unwrap(), "completed");
        assert_eq!(
            s_step.run_steps_manifold(&mut m_step, 5.0).unwrap(),
            "running"
        );
        while !s_step.is_finished() {
            s_step.step_manifold(&mut m_step).unwrap();
        }
        assert_eq!(s_step.status(), "completed");
        assert_eq!(m_run.coords(), m_step.coords());
    }
}

#[wasm_bindgen_test]
fn manifold_layout_and_distances() {
    let mut hyp = JsManifoldLayout2d::hyperbolic_poincare(3.0).unwrap();
    assert_eq!(hyp.kind(), "hyperbolicPoincare");
    assert_eq!(hyp.num_nodes(), 3);
    hyp.set_point(0.0, 0.1, 0.2).unwrap();
    hyp.set_point(1.0, -0.1, 0.05).unwrap();
    let delta = hyp.tangent_delta(0.0, 1.0).unwrap();
    assert_eq!(delta.len(), 2);
    hyp.apply_tangent_step(0.0, delta[0] * 0.1, delta[1] * 0.1)
        .unwrap();
    assert!(js_poincare_distance(0.0, 0.0, 0.3, 0.4) > 0.5);
    let m_add = js_poincare_mobius_add(-0.3, 0.4, 0.3, -0.4);
    assert!(m_add[0].hypot(m_add[1]) < 1e-12);
    let mut m_rand = JsManifoldLayout2d::random_init(
        "hyperbolic",
        4.0,
        &make_opts(&[
            ("scale", JsValue::from_f64(0.25)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    m_rand.center_poincare_centroid().unwrap();
    let mc = m_rand.poincare_centroid().unwrap();
    assert!(mc[0].hypot(mc[1]) < 1e-4);

    let sph = JsManifoldLayout2d::spherical(2.0).unwrap();
    assert_eq!(sph.kind(), "spherical");
    assert!(
        (js_spherical_distance(0.0, 0.0, 0.0, std::f64::consts::FRAC_PI_2)
            - std::f64::consts::FRAC_PI_2)
            .abs()
            < 1e-9
    );

    let tor = JsManifoldLayout2d::torus_2d(2.0).unwrap();
    assert_eq!(tor.kind(), "torus");
    assert!((js_torus_distance(0.05, 0.05, 0.95, 0.05) - 0.1).abs() < 1e-9);
    let wrapped = js_wrap_torus_point(1.25, -0.25);
    assert!((wrapped[0] - 0.25).abs() < 1e-9 && (wrapped[1] - 0.75).abs() < 1e-9);
}

#[wasm_bindgen_test]
fn fdeb_solver_reference_passing_and_step_eq_run() {
    let graph =
        JsCsrGraph::from_edge_list(4.0, &[0, 1, 2, 3], None, &JsValue::from_bool(false), None)
            .unwrap();
    // Two parallel horizontal edges separated by y=1.0 -> high compatibility
    let layout = JsLayout2d::from_coords(&[0.0, 0.0, 10.0, 0.0, 0.0, 1.0, 10.0, 1.0]).unwrap();
    let mut poly_run = JsEdgePolylines::straight_line_from_graph_2d(&graph, &layout).unwrap();
    let mut poly_step = poly_run.copy();
    let opts = make_opts(&[
        ("cycles", JsValue::from_f64(3.0)),
        ("stepsPerCycle", JsValue::from_f64(12.0)),
        ("s0", JsValue::from_f64(0.1)),
    ]);

    let mut f_run = JsFdebSolver::new(&poly_run, &opts).unwrap();
    let mut f_step = JsFdebSolver::new(&poly_step, &opts).unwrap();
    assert_eq!(f_run.num_edges(), 2);

    assert_eq!(f_run.run(&mut poly_run).unwrap(), "completed");
    assert!(f_run.is_finished());

    assert_eq!(f_step.run_steps(&mut poly_step, 5.0).unwrap(), "running");
    while !f_step.is_finished() {
        f_step.step(&mut poly_step).unwrap();
    }
    assert_eq!(f_step.status(), "completed");
    assert_eq!(poly_run.copy_edge_offsets(), poly_step.copy_edge_offsets());
    let pts_run = poly_run.copy_points();
    let pts_step = poly_step.copy_points();
    assert_eq!(pts_run.len(), pts_step.len());
    for (a, b) in pts_run.iter().zip(pts_step.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let offsets = poly_run.copy_edge_offsets();
    assert_eq!(offsets.len(), 3);
    assert_eq!(pts_run.len(), (offsets[2] as usize) * 2);
    assert!(pts_run.iter().all(|v| v.is_finite()));

    // Midpoint y-coordinates of the two parallel edges should attract toward 0.5
    let mid0 = (offsets[0] as usize + offsets[1] as usize) / 2;
    let mid1 = (offsets[1] as usize + offsets[2] as usize) / 2;
    let mid_gap = (pts_run[mid1 * 2 + 1] - pts_run[mid0 * 2 + 1]).abs();
    assert!(mid_gap < 1.0);

    // Adversarial checks: cycles=30 error & mismatched EdgePolylines target
    assert!(
        JsFdebSolver::new(
            &poly_run,
            &make_opts(&[("cycles", JsValue::from_f64(30.0))])
        )
        .is_err()
    );
    let layout_nd = egraph_wasm::layout_buffers::JsLayoutNd::zeros(4.0, 3.0).unwrap();
    let mut poly_nd = JsEdgePolylines::straight_line_from_graph_nd(&graph, &layout_nd).unwrap();
    let mismatch_err = f_run.step(&mut poly_nd).unwrap_err();
    assert_eq!(
        js_sys::Reflect::get(&mismatch_err, &"code".into())
            .unwrap()
            .as_string(),
        Some("UnsupportedOperation".to_owned())
    );
}

#[wasm_bindgen_test]
fn lorentz_extreme_precision_and_exp_map_wasm_parity() {
    let ln2 = std::f64::consts::LN_2;

    // Rotated collinear `p2 = 2 * p1` across `1e130..8e307` verifies software `f64::mul_add` on wasm32.
    for r in [1e130_f64, 1e139, 1e140, 1e200, 8e307] {
        let x1 = r * 0.7_f64.cos();
        let y1 = r * 0.7_f64.sin();
        let d = js_lorentz_distance(x1, y1, 2.0 * x1, 2.0 * y1);
        assert!((d - ln2).abs() < 1e-12, "wasm 2*p1 r={r:e}: d={d}");
        let v = js_lorentz_log_map(x1, y1, 2.0 * x1, 2.0 * y1);
        assert!((v[0].hypot(v[1]) - ln2).abs() < 1e-12 && v[0] > 0.0 && v[1] > 0.0);
    }

    // High-ratio `m_max < 1e140` pairs (700-digit oracle references) and subnormal continuity.
    let d_hr = js_lorentz_distance(
        1e50_f64 * 0.7_f64.cos(),
        1e50_f64 * 0.7_f64.sin(),
        1e139_f64 * 0.7_f64.cos(),
        1e139_f64 * 0.7_f64.sin(),
    );
    assert!((d_hr - 359.938_571_185_530_1).abs() < 1e-10);
    assert!((js_lorentz_distance(f64::from_bits(1), 0.0, 1e-200, 0.0) - 1e-200).abs() <= 1e-215);

    // Near-`f64::MAX` `cos_phi < 0` multiplication order (`[1, 2) * f64::MAX` stays finite).
    let sin_phi = (1.0 - 0.25_f64).sqrt();
    let p_fin = js_lorentz_exp_map(1.5e308, 0.0, 1.3 * -0.5, 1.3 * sin_phi);
    assert!(p_fin[0].is_finite() && p_fin[1].is_finite() && p_fin[0] != 1.5e308);
    assert!((js_lorentz_distance(1.5e308, 0.0, p_fin[0], p_fin[1]) - 1.3).abs() < 1e-11);

    // Rotated exact inward step and overflowing tangent `hypot(vx, vy)` preserve direction.
    let (xr, yr) = (1e130_f64 * 0.7_f64.cos(), 1e130_f64 * 0.7_f64.sin());
    let (vx, vy) = (-xr * 2.0_f64.powi(-424), -yr * 2.0_f64.powi(-424));
    let p_in = js_lorentz_exp_map(xr, yr, vx, vy);
    let exp_r = 1e130_f64 * (-vx.hypot(vy)).exp();
    assert!(((p_in[0].hypot(p_in[1]) - exp_r) / exp_r).abs() < 1e-12);

    let p_huge = js_lorentz_exp_map(1e200, 1e200, -1.5e308, -1.5e308);
    assert!(p_huge[0] < -1e100 && p_huge[1] < -1e100);
}
