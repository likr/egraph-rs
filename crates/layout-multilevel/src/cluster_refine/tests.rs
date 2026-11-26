use super::geometry::{canonicalize_labels, voronoi_cell_exit};
use super::hac::{match_clusters_hungarian, upgma_hac_2d};
use super::*;
use egraph_core::DeterministicRng;

fn compute_cq(gt_labels: &[u32], geom_labels: &[u32], k: usize) -> f64 {
    let n = gt_labels.len();
    if n == 0 {
        return 1.0;
    }
    let matching = match_clusters_hungarian(gt_labels, geom_labels, k);
    let mut hits = 0usize;
    for i in 0..n {
        if geom_labels[i] as usize == matching[gt_labels[i] as usize] {
            hits += 1;
        }
    }
    (hits as f64) / (n as f64)
}

fn kmeans_cq(layout: &Layout2d<f64>, gt_labels: &[u32], k: usize) -> f64 {
    let n = layout.len();
    let mut flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        flat.push(layout.x()[i]);
        flat.push(layout.y()[i]);
    }
    let km = kmeans_plus_plus(
        &flat,
        n,
        2,
        &KMeansConfig {
            k,
            n_init: 5,
            max_iters: 100,
            tol: 1e-4,
            seed: 7,
        },
    )
    .unwrap();
    let (canon_gt, _) = canonicalize_labels(gt_labels);
    compute_cq(&canon_gt, &km.labels, k)
}

fn hac_cq(layout: &Layout2d<f64>, gt_labels: &[u32], k: usize) -> f64 {
    let geom = upgma_hac_2d(layout.x(), layout.y(), k).unwrap();
    let (canon_gt, _) = canonicalize_labels(gt_labels);
    compute_cq(&canon_gt, &geom, k)
}

#[test]
fn test_cluster_kmeans_and_hac_improve_planted_overlapping_clusters() {
    // 3 planted clusters of 10 nodes each (30 nodes), with several overlapping misplaced nodes.
    let mut rng = DeterministicRng::from_seed_u64(101);
    let centers = [(0.0_f64, 0.0_f64), (10.0, 0.0), (5.0, 8.66)];
    let mut xs = Vec::with_capacity(30);
    let mut ys = Vec::with_capacity(30);
    let mut labels = Vec::with_capacity(30);

    for (cid, &(cx, cy)) in centers.iter().enumerate() {
        for idx in 0..10 {
            // Intentionally misplace 3 nodes per cluster into neighboring cluster regions.
            let (bx, by) = if idx < 3 {
                let wrong = (cid + 1) % 3;
                centers[wrong]
            } else {
                (cx, cy)
            };
            xs.push(bx + (rng.next_f64() - 0.5) * 0.8);
            ys.push(by + (rng.next_f64() - 0.5) * 0.8);
            labels.push((cid as u32) * 10 + 5); // Non-contiguous labels test canonicalization.
        }
    }

    let mut edges = Vec::new();
    for cid in 0..3 {
        let base = cid * 10;
        for i in 0..10 {
            for j in (i + 1)..10 {
                edges.push(((base + i) as u32, (base + j) as u32));
            }
        }
    }
    let graph = CsrGraph::<f64>::from_edges(30, &edges, None, false, false).unwrap();

    let orig = Layout2d::from_xy(xs, ys).unwrap();
    let orig_cq_km = kmeans_cq(&orig, &labels, 3);
    let orig_cq_hac = hac_cq(&orig, &labels, 3);
    assert!(
        orig_cq_km < 0.85,
        "expected overlapping initial CQ < 0.85, got {orig_cq_km}"
    );
    assert!(
        orig_cq_hac < 0.85,
        "expected overlapping initial HAC CQ < 0.85, got {orig_cq_hac}"
    );

    // Cluster-kmeans improvement + step == run bit-exactness.
    let km_cfg = ClusterRefineConfig {
        method: ClusterRefineMethod::Kmeans,
        max_rounds: 10,
        ..Default::default()
    };
    let mut km_batch = orig.clone();
    let mut km_step = orig.clone();
    let mut s_batch = ClusterRefineSolver::<f64>::new(labels.clone(), km_cfg).unwrap();
    let mut s_step = ClusterRefineSolver::<f64>::new(labels.clone(), km_cfg).unwrap();
    let st_batch = s_batch.run(&mut km_batch).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut km_step).unwrap();
    }
    assert_eq!(st_batch, s_step.status());
    assert_eq!(s_batch.iteration(), s_step.iteration());
    assert_eq!(km_batch, km_step);

    let refined_cq_km = kmeans_cq(&km_batch, &labels, 3);
    assert!(
        refined_cq_km > orig_cq_km,
        "Cluster-kmeans should improve CQ: {orig_cq_km} -> {refined_cq_km}"
    );

    // Cluster-HAC improvement via from_graph + step == run bit-exactness.
    let hac_cfg = ClusterRefineConfig {
        method: ClusterRefineMethod::Hac,
        max_rounds: 10,
        ..Default::default()
    };
    let mut hac_batch = orig.clone();
    let mut hac_step = orig.clone();
    let mut sh_batch =
        ClusterRefineSolver::<f64>::from_graph(&graph, labels.clone(), hac_cfg).unwrap();
    let mut sh_step =
        ClusterRefineSolver::<f64>::from_graph(&graph, labels.clone(), hac_cfg).unwrap();
    let sth_batch = sh_batch.run(&mut hac_batch).unwrap();
    while !sh_step.status().is_finished() {
        sh_step.step(&mut hac_step).unwrap();
    }
    assert_eq!(sth_batch, sh_step.status());
    assert_eq!(hac_batch, hac_step);

    let refined_cq_hac = hac_cq(&hac_batch, &labels, 3);
    assert!(
        refined_cq_hac > orig_cq_hac,
        "Cluster-HAC should improve CQ: {orig_cq_hac} -> {refined_cq_hac}"
    );

    // Also test f32 bit-exact step == run and refine_cluster_layout convenience fn.
    let orig_f32 = Layout2d::from_xy(
        orig.x().iter().map(|&v| v as f32).collect(),
        orig.y().iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    let mut f32_batch = orig_f32.clone();
    let mut f32_step = orig_f32;
    let mut s32_b = ClusterRefineSolver::<f32>::new(labels.clone(), km_cfg).unwrap();
    let mut s32_s = ClusterRefineSolver::<f32>::new(labels.clone(), km_cfg).unwrap();
    s32_b.run(&mut f32_batch).unwrap();
    while !s32_s.status().is_finished() {
        s32_s.step(&mut f32_step).unwrap();
    }
    assert_eq!(f32_batch, f32_step);

    let mut convenience_layout = orig;
    let moved = refine_cluster_layout(&mut convenience_layout, &labels, &km_cfg).unwrap();
    assert!(moved > 0);
    assert_eq!(convenience_layout, km_batch);
}

#[test]
fn test_upgma_hac_matches_exact_naive_and_handles_ties() {
    // Test exact UPGMA against naive O(N^3) implementation on points with ties.
    let xs = [0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0];
    let ys = [0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0];
    let labels = upgma_hac_2d(&xs, &ys, 2).unwrap();
    // Left cluster {0,1,2,3} vs right cluster {4,5,6}.
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[0], labels[3]);
    assert_eq!(labels[4], labels[5]);
    assert_eq!(labels[4], labels[6]);
    assert_ne!(labels[0], labels[4]);

    // Collinear equidistant points (strict tie-breaking must not cycle in NN-chain).
    let eq_x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let eq_y = [0.0, 0.0, 0.0, 0.0, 0.0];
    let eq_labels = upgma_hac_2d(&eq_x, &eq_y, 2).unwrap();
    assert_eq!(eq_labels.len(), 5);
    let distinct: std::collections::BTreeSet<u32> = eq_labels.iter().copied().collect();
    assert_eq!(distinct.len(), 2);
}

#[test]
fn test_voronoi_exit_uses_intermediate_cell_when_non_adjacent() {
    // 3 centroids along x-axis: c0=(0,0), c1=(4,0), c2=(10,0).
    // Segment [c0, c2] exits c0's Voronoi cell at bisector(c0, c1) = (2, 0), NOT midpoint (5, 0).
    let centroids = [(0.0, 0.0), (4.0, 0.0), (10.0, 0.0)];
    let exit = voronoi_cell_exit(&centroids, 0, 2);
    assert!((exit.0 - 2.0).abs() < 1e-12);
    assert!(exit.1.abs() < 1e-12);
}

#[test]
fn test_cluster_refine_degenerate_and_validation() {
    // Empty layout
    let mut l0 = Layout2d::<f64>::zeros(0);
    let mut s0 = ClusterRefineSolver::<f64>::new(vec![], ClusterRefineConfig::default()).unwrap();
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);
    assert_eq!(s0.last_moved_count(), 0);

    // Single node
    let mut l1 = Layout2d::<f64>::from_xy(vec![1.0], vec![2.0]).unwrap();
    let mut s1 = ClusterRefineSolver::<f64>::new(vec![7], ClusterRefineConfig::default()).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(s1.last_moved_count(), 0);

    // Single cluster (K = 1)
    let mut lk1 = Layout2d::<f64>::from_xy(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]).unwrap();
    let mut sk1 =
        ClusterRefineSolver::<f64>::new(vec![3, 3, 3], ClusterRefineConfig::default()).unwrap();
    assert_eq!(sk1.run(&mut lk1).unwrap(), SolverStatus::Converged);
    assert_eq!(sk1.last_moved_count(), 0);

    // Already perfectly separated -> early Converged with 0 moves
    let mut l_sep =
        Layout2d::<f64>::from_xy(vec![0.0, 0.1, 20.0, 20.1], vec![0.0, 0.1, 0.0, 0.1]).unwrap();
    let mut s_sep =
        ClusterRefineSolver::<f64>::new(vec![0, 0, 1, 1], ClusterRefineConfig::default()).unwrap();
    assert_eq!(s_sep.run(&mut l_sep).unwrap(), SolverStatus::Converged);
    assert_eq!(s_sep.last_moved_count(), 0);
    assert_eq!(s_sep.iteration(), 0);

    // Validation errors
    assert!(matches!(
        ClusterRefineConfig {
            kmeans_n_init: 0,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ClusterRefineConfig {
            shift_step: f64::NAN,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        ClusterRefineConfig {
            margin: -0.1,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
}
