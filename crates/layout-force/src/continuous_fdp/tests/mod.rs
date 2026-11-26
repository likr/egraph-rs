mod force_atlas2;
mod yifan_hu;

use egraph_core::{CsrGraph, DeterministicRng, Layout2d, Scalar};

use super::*;

pub(super) fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push((u, u + 1));
            }
            if r + 1 < rows {
                edges.push((u, u + cols as u32));
            }
        }
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

pub(super) fn barbell_graph(k: usize) -> CsrGraph<f64> {
    let n = 2 * k;
    let mut edges = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            edges.push((i as u32, j as u32));
            edges.push(((k + i) as u32, (k + j) as u32));
        }
    }
    edges.push(((k - 1) as u32, k as u32));
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

pub(super) fn seeded_layout<S: Scalar>(n: usize, seed: u64) -> Layout2d<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(S::from_f64(2.0 * (rng.next_f64() - 0.5)));
        y.push(S::from_f64(2.0 * (rng.next_f64() - 0.5)));
    }
    Layout2d::from_xy(x, y).unwrap()
}

#[test]
fn test_configs_defaults_and_validation() {
    let fa2 = ForceAtlas2Config::default();
    assert_eq!(fa2.iterations, 200);
    assert_eq!(fa2.theta, 1.2);
    assert!(fa2.use_barnes_hut);
    assert!(!fa2.linlog_mode);
    assert!(!fa2.strong_gravity);
    assert_eq!(fa2.gravity, 1.0);
    assert_eq!(fa2.scaling_ratio, 2.0);
    assert_eq!(fa2.edge_weight_influence, 1.0);
    assert!(!fa2.outbound_attraction_distribution);
    assert_eq!(fa2.jitter_tolerance, 1.0);
    assert_eq!(fa2.max_displacement, None);
    assert_eq!(fa2.seed, 42);
    fa2.validate().unwrap();

    for bad in [
        ForceAtlas2Config { theta: -0.1, ..fa2 },
        ForceAtlas2Config {
            gravity: f64::NAN,
            ..fa2
        },
        ForceAtlas2Config {
            scaling_ratio: -1.0,
            ..fa2
        },
        ForceAtlas2Config {
            edge_weight_influence: -0.5,
            ..fa2
        },
        ForceAtlas2Config {
            jitter_tolerance: 0.0,
            ..fa2
        },
        ForceAtlas2Config {
            max_displacement: Some(0.0),
            ..fa2
        },
        ForceAtlas2Config {
            max_displacement: Some(-5.0),
            ..fa2
        },
        ForceAtlas2Config {
            max_displacement: Some(f64::NAN),
            ..fa2
        },
        ForceAtlas2Config {
            max_displacement: Some(f64::INFINITY),
            ..fa2
        },
        ForceAtlas2Config {
            max_displacement: Some(f64::NEG_INFINITY),
            ..fa2
        },
    ] {
        assert!(bad.validate().is_err());
    }

    let yh = YifanHuConfig::default();
    assert_eq!(yh.max_iterations_per_level, 200);
    assert_eq!(yh.step_size, 1.0);
    assert_eq!(yh.step_ratio, 0.9);
    assert_eq!(yh.relative_strength, 0.2);
    assert_eq!(yh.optimal_distance, None);
    assert_eq!(yh.theta, 1.2);
    assert_eq!(yh.min_coarsen_size, 30);
    assert_eq!(yh.min_coarsen_ratio, 0.75);
    assert_eq!(yh.convergence_tolerance, 1e-4);
    assert_eq!(yh.seed, 42);
    yh.validate().unwrap();

    for bad in [
        YifanHuConfig {
            step_size: 0.0,
            ..yh
        },
        YifanHuConfig {
            step_ratio: 1.0,
            ..yh
        },
        YifanHuConfig {
            step_ratio: 0.0,
            ..yh
        },
        YifanHuConfig {
            relative_strength: -0.1,
            ..yh
        },
        YifanHuConfig {
            optimal_distance: Some(0.0),
            ..yh
        },
        YifanHuConfig { theta: -0.1, ..yh },
        YifanHuConfig {
            min_coarsen_size: 0,
            ..yh
        },
        YifanHuConfig {
            min_coarsen_ratio: 0.0,
            ..yh
        },
        YifanHuConfig {
            min_coarsen_ratio: 1.1,
            ..yh
        },
        YifanHuConfig {
            convergence_tolerance: -1e-4,
            ..yh
        },
    ] {
        assert!(bad.validate().is_err());
    }
}
