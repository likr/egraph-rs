mod util;

use egraph_wasm::layout_buffers::JsLayout2d;
use egraph_wasm::screened_poisson::JsScreenedPoissonField2d;
use egraph_wasm::screened_poisson_refiner::JsScreenedPoissonRefiner;
use util::fixtures::{make_opts, make_path_graph};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn screened_poisson_density_field_and_refiner() {
    // 1. Finite-difference gradient oracle & outward sign check.
    let mut coords = Vec::new();
    let h = 12.0_f64 / 128.0_f64;
    for i in 0..10 {
        let ang = 0.31 + (i as f64) * std::f64::consts::TAU / 10.0;
        let mut x = 0.17 + 0.25 * ang.cos();
        let mut y = -0.13 + 0.25 * ang.sin();
        let ux = ((x + 6.0) / h).fract();
        let uy = ((y + 6.0) / h).fract();
        if ux.min(1.0 - ux) < 0.1 {
            x += 0.25 * h;
        }
        if uy.min(1.0 - uy) < 0.1 {
            y += 0.25 * h;
        }
        coords.push(x as f32);
        coords.push(y as f32);
    }
    for i in 0..4 {
        let ang = 0.43 + (i as f64) * std::f64::consts::TAU / 4.0;
        coords.push((0.17 + 0.70 * ang.cos()) as f32);
        coords.push((-0.13 + 0.70 * ang.sin()) as f32);
    }
    coords.extend_from_slice(&[3.23_f32, -2.91, -3.13, 2.83]);

    let bounds = [-5.0_f64, 5.0, -5.0, 5.0];
    for (exact_adjoint, rho_max) in [(true, 0.05_f64), (false, 0.0_f64)] {
        let mut field = JsScreenedPoissonField2d::new(&make_opts(&[
            ("kernel", JsValue::from_str("screenedPoisson")),
            ("gridSize", JsValue::from_f64(128.0)),
            ("paddingFactor", JsValue::from_f64(2.0)),
            ("sigma", JsValue::from_f64(0.35)),
            ("kappa", JsValue::from_f64(0.5)),
            ("rhoMax", JsValue::from_f64(rho_max)),
            ("exactAdjoint", JsValue::from_bool(exact_adjoint)),
        ]))
        .unwrap();

        let l_base = JsLayout2d::from_coords(&coords).unwrap();
        assert!(field.density_grid().unwrap().is_empty());
        let res = field
            .evaluate_in_bounds(&l_base, bounds[0], bounds[2], bounds[1], bounds[3])
            .unwrap();
        assert!(!field.density_grid().unwrap().is_empty());
        assert_eq!(field.density_grid_dim(), 128);
        assert!(!res.early_exit_no_excess());
        assert_eq!(res.fft_2d_calls() as u64, if exact_adjoint { 6 } else { 4 });

        let fx = res.force_x();
        let fy = res.force_y();
        let eps = 2e-3_f32;
        let mut max_err = 0.0_f64;
        let mut max_grad = 0.0_f64;
        for i in 0..16 {
            for (axis, anal_f) in [(0usize, fx[i] as f64), (1usize, fy[i] as f64)] {
                let mut cp = coords.clone();
                let mut cm = coords.clone();
                cp[2 * i + axis] += eps;
                cm[2 * i + axis] -= eps;
                let lp = field
                    .evaluate_in_bounds(
                        &JsLayout2d::from_coords(&cp).unwrap(),
                        bounds[0],
                        bounds[2],
                        bounds[1],
                        bounds[3],
                    )
                    .unwrap()
                    .energy();
                let lm = field
                    .evaluate_in_bounds(
                        &JsLayout2d::from_coords(&cm).unwrap(),
                        bounds[0],
                        bounds[2],
                        bounds[1],
                        bounds[3],
                    )
                    .unwrap()
                    .energy();
                let fd = (lp - lm) / (2.0 * (eps as f64));
                max_grad = max_grad.max(anal_f.abs());
                max_err = max_err.max(((-anal_f) - fd).abs());
            }
        }
        assert!(max_grad > 1e-4);
        assert!(max_err / max_grad < 1e-2 && max_err < 2e-4);
    }

    // 2. Unilateral early-exit zero force.
    let mut grid_coords = Vec::with_capacity(128);
    for r in 0..8 {
        for c in 0..8 {
            grid_coords.push(c as f32);
            grid_coords.push(r as f32);
        }
    }
    let mut field_ee = JsScreenedPoissonField2d::new(&make_opts(&[
        ("gridSize", JsValue::from_f64(128.0)),
        ("rhoMax", JsValue::from_f64(10.0)),
    ]))
    .unwrap();
    let res_ee = field_ee
        .evaluate(&JsLayout2d::from_coords(&grid_coords).unwrap())
        .unwrap();
    assert!(res_ee.early_exit_no_excess());
    assert_eq!(res_ee.energy(), 0.0);
    assert_eq!(res_ee.fft_2d_calls() as u64, 2);

    // 3. Refiner step == run, fromGraph, setCoarseTether, applyEpochStep.
    let mut disk_coords = Vec::new();
    for r in 0..16 {
        for c in 0..16 {
            let x = -0.8_f32 + 1.6_f32 * (c as f32) / 15.0;
            let y = -0.8_f32 + 1.6_f32 * (r as f32) / 15.0;
            if x * x + y * y <= 0.64 {
                disk_coords.push(x);
                disk_coords.push(y);
            }
        }
    }
    let n_disk = disk_coords.len() / 2;
    let ref_opts = make_opts(&[
        ("gridSize", JsValue::from_f64(64.0)),
        ("sigma", JsValue::from_f64(0.10)),
        ("kappa", JsValue::from_f64(1.0)),
        ("rhoMax", JsValue::from_f64(0.15)),
        ("iterations", JsValue::from_f64(4.0)),
        ("stepSize", JsValue::from_f64(0.3)),
        ("momentum", JsValue::from_f64(0.4)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut r_run = JsScreenedPoissonRefiner::new(n_disk as f64, &ref_opts).unwrap();
    let mut r_step = JsScreenedPoissonRefiner::new(n_disk as f64, &ref_opts).unwrap();
    let mut l_run = JsLayout2d::from_coords(&disk_coords).unwrap();
    let mut l_step = JsLayout2d::from_coords(&disk_coords).unwrap();
    r_run.run(&mut l_run).unwrap();
    r_step.run_steps(&mut l_step, 2.0).unwrap();
    assert_eq!(r_step.iteration(), 2);
    r_step.run(&mut l_step).unwrap();
    assert_eq!(l_run.copy_coords(), l_step.copy_coords());

    let g4 = make_path_graph(4);
    let mut r_g = JsScreenedPoissonRefiner::from_graph(
        &g4,
        &make_opts(&[
            ("gridSize", JsValue::from_f64(64.0)),
            ("rhoMax", JsValue::from_f64(0.1)),
            ("anchorWeight", JsValue::from_f64(0.2)),
            ("gammaTether", JsValue::from_f64(1.0)),
            ("recenter", JsValue::from_bool(false)),
        ]),
    )
    .unwrap();
    let coarse = JsLayout2d::from_coords(&[5.0_f32, 2.0, -5.0, -2.0]).unwrap();
    r_g.set_coarse_tether(&coarse, &[0, 0, 1, 1]).unwrap();
    let mut l4 = JsLayout2d::from_coords(&[0.0_f32, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1]).unwrap();
    let ep_st = r_g.apply_epoch_step(&mut l4).unwrap();
    assert!(!ep_st.early_exit_no_excess());
    let c4 = l4.copy_coords();
    assert!((0.5 * (c4[0] + c4[2]) - 5.0).abs() < 1e-4);

    // 4. Adversarial error checks.
    assert!(
        JsScreenedPoissonField2d::new(&make_opts(&[("kernel", JsValue::from_str("badKernel"))]))
            .is_err()
    );
    assert!(
        JsScreenedPoissonField2d::new(&make_opts(&[("gridSize", JsValue::from_f64(100.0))]))
            .is_err()
    );
}
