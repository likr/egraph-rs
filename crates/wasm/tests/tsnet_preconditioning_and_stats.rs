mod util;

use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::tsnet::JsTsNetSolver;
use util::fixtures::{make_grid, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn tsnet_jacobi_precondition_and_option_validation() {
    let grid = make_grid(4, 4);

    // Invalid backend or extra positional arg with options rejected
    assert!(
        JsTsNetSolver::pure_2hop(
            &grid,
            &JsValue::from_str("bogus"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
        )
        .is_err()
    );
    let opts = make_opts(&[("backend", JsValue::from_str("exact"))]);
    assert!(
        JsTsNetSolver::pure_2hop(
            &grid,
            &opts,
            Some(10.0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
        )
        .is_err()
    );

    // jacobiPrecondition option on star graph (true vs false diverge; non-bool rejected)
    let mut star_edges = Vec::new();
    for v in 1..20u32 {
        star_edges.push(0);
        star_edges.push(v);
    }
    let star = JsCsrGraph::from_edge_list(20.0, &star_edges, None, &JsValue::UNDEFINED, None)
        .expect("star graph");
    let mut star_coords = Vec::with_capacity(40);
    for i in 0..20usize {
        if i == 0 {
            star_coords.push(1.0f32);
            star_coords.push(0.0f32);
        } else {
            let ang = std::f32::consts::TAU * (i as f32) / 19.0;
            star_coords.push(0.5 * ang.cos());
            star_coords.push(0.5 * ang.sin());
        }
    }
    let star_init = JsLayout2d::from_coords(&star_coords).expect("star layout");

    for (ctor_is_rn, label) in [(false, "pure2Hop"), (true, "fromGraphRowNormalized")] {
        let mut l_on = star_init.clone();
        let mut l_off = star_init.clone();
        if !ctor_is_rn {
            let opts_on = make_opts(&[
                ("backend", JsValue::from_str("exact")),
                ("iterationsStage1", JsValue::from_f64(4.0)),
                ("iterationsStage2", JsValue::from_f64(2.0)),
                ("iterationsStage3", JsValue::from_f64(2.0)),
                ("jacobiPrecondition", JsValue::from_bool(true)),
                ("seed", JsValue::from_f64(42.0)),
            ]);
            let opts_off = make_opts(&[
                ("backend", JsValue::from_str("exact")),
                ("iterationsStage1", JsValue::from_f64(4.0)),
                ("iterationsStage2", JsValue::from_f64(2.0)),
                ("iterationsStage3", JsValue::from_f64(2.0)),
                ("jacobiPrecondition", JsValue::from_bool(false)),
                ("seed", JsValue::from_f64(42.0)),
            ]);
            let mut s_on = JsTsNetSolver::pure_2hop(
                &star,
                &opts_on,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                &JsValue::UNDEFINED,
                None,
                None,
            )
            .expect(label);
            let mut s_off = JsTsNetSolver::pure_2hop(
                &star,
                &opts_off,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                &JsValue::UNDEFINED,
                None,
                None,
            )
            .expect(label);
            assert_eq!(s_on.run(&mut l_on).unwrap(), "completed");
            assert_eq!(s_off.run(&mut l_off).unwrap(), "completed");
        } else {
            let opts_on = make_opts(&[
                ("w2", JsValue::from_f64(0.0)),
                ("backend", JsValue::from_str("exact")),
                ("iterationsStage1", JsValue::from_f64(4.0)),
                ("iterationsStage2", JsValue::from_f64(2.0)),
                ("iterationsStage3", JsValue::from_f64(2.0)),
                ("jacobiPrecondition", JsValue::from_bool(true)),
                ("seed", JsValue::from_f64(42.0)),
            ]);
            let opts_off = make_opts(&[
                ("w2", JsValue::from_f64(0.0)),
                ("backend", JsValue::from_str("exact")),
                ("iterationsStage1", JsValue::from_f64(4.0)),
                ("iterationsStage2", JsValue::from_f64(2.0)),
                ("iterationsStage3", JsValue::from_f64(2.0)),
                ("jacobiPrecondition", JsValue::from_bool(false)),
                ("seed", JsValue::from_f64(42.0)),
            ]);
            let mut s_on = JsTsNetSolver::from_graph_row_normalized(
                &star,
                &opts_on,
                None,
                &JsValue::UNDEFINED,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                &JsValue::UNDEFINED,
                None,
                None,
                None,
            )
            .expect(label);
            let mut s_off = JsTsNetSolver::from_graph_row_normalized(
                &star,
                &opts_off,
                None,
                &JsValue::UNDEFINED,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                &JsValue::UNDEFINED,
                None,
                None,
                None,
            )
            .expect(label);
            assert_eq!(s_on.run(&mut l_on).unwrap(), "completed");
            assert_eq!(s_off.run(&mut l_off).unwrap(), "completed");
        }
        assert!(l_on.is_all_finite());
        assert!(l_off.is_all_finite());
        let mut max_diff = 0.0f32;
        for i in 0..20 {
            let idx = i as f64;
            max_diff = max_diff
                .max((l_on.x(idx).unwrap() - l_off.x(idx).unwrap()).abs())
                .max((l_on.y(idx).unwrap() - l_off.y(idx).unwrap()).abs());
        }
        assert!(
            max_diff > 1e-4,
            "{label}: expected jacobiPrecondition true vs false to diverge on star hub, diff={max_diff}"
        );
    }

    let bad_jacobi = make_opts(&[
        ("backend", JsValue::from_str("exact")),
        ("jacobiPrecondition", JsValue::from_str("nope")),
    ]);
    assert!(
        JsTsNetSolver::pure_2hop(
            &star,
            &bad_jacobi,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
        )
        .is_err()
    );
}
