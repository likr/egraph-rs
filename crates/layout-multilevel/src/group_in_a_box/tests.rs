use super::*;
use egraph_spatial_trees::{Rect2d, RectangleOverlapSweep};

fn build_planted_sbm_clusters() -> (CsrGraph<f64>, Vec<u32>) {
    // 5 clusters of varying sizes: [12, 8, 6, 4, 1] -> 31 nodes total,
    // where cluster 4 is a singleton and cluster 3 has 4 nodes with 0 intra edges.
    let sizes = [12usize, 8, 6, 4, 1];
    let label_ids = [10u32, 20, 30, 40, 50];
    let mut labels = Vec::new();
    let mut offsets = Vec::new();
    let mut cur = 0usize;
    for (&sz, &lid) in sizes.iter().zip(label_ids.iter()) {
        offsets.push(cur);
        for _ in 0..sz {
            labels.push(lid);
        }
        cur += sz;
    }
    let n = cur;
    let mut edges = Vec::new();
    // Dense intra-cluster edges for clusters 0, 1, 2; cluster 3 intentionally edgeless.
    for c in 0..3 {
        let start = offsets[c];
        let sz = sizes[c];
        for i in 0..sz {
            for j in (i + 1)..sz {
                edges.push(((start + i) as u32, (start + j) as u32));
            }
        }
    }
    // Sparse inter-cluster bridges.
    edges.push((0, offsets[1] as u32));
    edges.push((offsets[1] as u32, offsets[2] as u32));
    edges.push((offsets[2] as u32, offsets[3] as u32));
    edges.push((offsets[3] as u32, offsets[4] as u32));

    let graph = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
    (graph, labels)
}

fn verify_boxes_disjoint_and_nodes_inside<S: Scalar>(
    boxes: &[GibBox],
    layout: &Layout2d<S>,
    labels: &[u32],
) {
    let rects: Vec<Rect2d<f64>> = boxes
        .iter()
        .map(|b| Rect2d::new(b.x_min, b.x_max, b.y_min, b.y_max).unwrap())
        .collect();
    assert!(
        !RectangleOverlapSweep::has_overlaps(&rects).unwrap(),
        "GIB boxes must be strictly non-overlapping"
    );

    let xs = layout.x();
    let ys = layout.y();
    for i in 0..layout.len() {
        let cid = labels[i];
        let b = boxes.iter().find(|bx| bx.cluster_id == cid).unwrap();
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        assert!(
            x >= b.x_min - 1e-5 && x <= b.x_max + 1e-5,
            "node {i} x={x} outside box [{}, {}]",
            b.x_min,
            b.x_max
        );
        assert!(
            y >= b.y_min - 1e-5 && y <= b.y_max + 1e-5,
            "node {i} y={y} outside box [{}, {}]",
            b.y_min,
            b.y_max
        );
    }
}

fn compute_prop_vis(boxes: &[GibBox]) -> f64 {
    let area_sum: f64 = boxes
        .iter()
        .map(|b| (b.x_max - b.x_min) * (b.y_max - b.y_min))
        .sum();
    let weight_sum: f64 = boxes.iter().map(|b| b.weight).sum();
    let mut l1 = 0.0_f64;
    for b in boxes {
        let a_frac = ((b.x_max - b.x_min) * (b.y_max - b.y_min)) / area_sum;
        let w_frac = b.weight / weight_sum;
        l1 += (a_frac - w_frac).abs();
    }
    1.0 - 0.5 * l1
}

#[test]
fn test_squarified_treemap_exact_tiling_order_and_validation() {
    let weights = [6.0, 6.0, 4.0, 3.0, 2.0, 2.0, 1.0];
    let rects = squarified_treemap(&weights, 10.0, 20.0, 6.0, 4.0).unwrap();
    assert_eq!(rects.len(), weights.len());

    let total_w: f64 = weights.iter().sum();
    let mut area_sum = 0.0_f64;
    let mut rect2ds = Vec::new();
    for (i, r) in rects.iter().enumerate() {
        assert!(r[0] >= 10.0 - 1e-12 && r[1] <= 16.0 + 1e-12);
        assert!(r[2] >= 20.0 - 1e-12 && r[3] <= 24.0 + 1e-12);
        let w = r[1] - r[0];
        let h = r[3] - r[2];
        assert!(w > 0.0 && h > 0.0);
        let area = w * h;
        let expected = 24.0 * (weights[i] / total_w);
        assert!((area - expected).abs() < 1e-9);
        area_sum += area;
        rect2ds.push(Rect2d::new(r[0], r[1], r[2], r[3]).unwrap());
    }
    assert!((area_sum - 24.0).abs() < 1e-9);
    assert!(!RectangleOverlapSweep::has_overlaps(&rect2ds).unwrap());

    // Validation errors
    assert!(matches!(
        squarified_treemap(&[], 0.0, 0.0, 10.0, 10.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        squarified_treemap(&[1.0, 0.0], 0.0, 0.0, 10.0, 10.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        squarified_treemap(&[1.0, -2.0], 0.0, 0.0, 10.0, 10.0),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        squarified_treemap(&[1.0, f64::NAN], 0.0, 0.0, 10.0, 10.0),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        squarified_treemap(&[1.0], 0.0, 0.0, 0.0, 10.0),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_gib_treemap_and_fd_modes_and_step_equivalence() {
    let (graph, labels) = build_planted_sbm_clusters();
    let n = graph.num_nodes();

    // 1. TR-GIB with FR + step == run bit-exactness + prop_vis >= 0.90.
    let cfg_tr = GroupInABoxConfig {
        box_mode: GibBoxLayoutMode::SquarifiedTreemap,
        intra_mode: GibIntraLayoutMode::Fr,
        intra_iterations: 40,
        ..Default::default()
    };
    let mut l_batch = Layout2d::<f64>::zeros(n);
    let mut l_step = Layout2d::<f64>::zeros(n);
    let mut s_batch = GroupInABoxSolver::<f64>::new(&graph, labels.clone(), cfg_tr).unwrap();
    let mut s_step = GroupInABoxSolver::<f64>::new(&graph, labels.clone(), cfg_tr).unwrap();
    let st_b = s_batch.run(&mut l_batch).unwrap();
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(st_b, s_step.status());
    assert_eq!(l_batch, l_step);
    verify_boxes_disjoint_and_nodes_inside(s_batch.boxes(), &l_batch, &labels);
    let prop_vis_tr = compute_prop_vis(s_batch.boxes());
    assert!(
        prop_vis_tr >= 0.90,
        "TR-GIB prop_vis must be >= 0.90, got {prop_vis_tr}"
    );

    // Singleton placed at box center; edgeless cluster 3 spread out inside its box.
    let b_singleton = &s_batch.boxes()[4];
    let expected_sx = 0.5 * (b_singleton.x_min + b_singleton.x_max);
    let expected_sy = 0.5 * (b_singleton.y_min + b_singleton.y_max);
    assert!((l_batch.x()[n - 1] - expected_sx).abs() < 1e-12);
    assert!((l_batch.y()[n - 1] - expected_sy).abs() < 1e-12);
    let edgeless_start = 26;
    let dx_edgeless = (l_batch.x()[edgeless_start] - l_batch.x()[edgeless_start + 1]).abs();
    let dy_edgeless = (l_batch.y()[edgeless_start] - l_batch.y()[edgeless_start + 1]).abs();
    assert!(dx_edgeless + dy_edgeless > 1e-3);

    // 2. FD-GIB mode with SgdFull.
    let cfg_fd = GroupInABoxConfig {
        box_mode: GibBoxLayoutMode::ForceDirected,
        intra_mode: GibIntraLayoutMode::SgdFull,
        intra_iterations: 25,
        fd_iterations: 40,
        ..Default::default()
    };
    let mut l_fd = Layout2d::<f64>::zeros(n);
    let mut s_fd = GroupInABoxSolver::<f64>::new(&graph, labels.clone(), cfg_fd).unwrap();
    s_fd.run(&mut l_fd).unwrap();
    verify_boxes_disjoint_and_nodes_inside(s_fd.boxes(), &l_fd, &labels);
    assert!(compute_prop_vis(s_fd.boxes()) >= 0.90);

    // 3. YifanHu intra mode + with_cluster_weights + f32 precision.
    let custom_weights = vec![5.0, 2.0, 2.0, 1.0, 1.0];
    let cfg_yh = GroupInABoxConfig {
        box_mode: GibBoxLayoutMode::SquarifiedTreemap,
        intra_mode: GibIntraLayoutMode::YifanHu,
        intra_iterations: 20,
        ..Default::default()
    };
    let mut l_yh_b = Layout2d::<f32>::zeros(n);
    let mut l_yh_s = Layout2d::<f32>::zeros(n);
    let mut s_yh_b = GroupInABoxSolver::<f32>::with_cluster_weights(
        &graph,
        labels.clone(),
        custom_weights.clone(),
        cfg_yh,
    )
    .unwrap();
    let mut s_yh_s = GroupInABoxSolver::<f32>::with_cluster_weights(
        &graph,
        labels.clone(),
        custom_weights,
        cfg_yh,
    )
    .unwrap();
    s_yh_b.run(&mut l_yh_b).unwrap();
    while !s_yh_s.status().is_finished() {
        s_yh_s.step(&mut l_yh_s).unwrap();
    }
    assert_eq!(l_yh_b, l_yh_s);
    verify_boxes_disjoint_and_nodes_inside(s_yh_b.boxes(), &l_yh_b, &labels);
    assert!(compute_prop_vis(s_yh_b.boxes()) >= 0.90);
}

#[test]
fn test_gib_degenerate_and_validation() {
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let mut l0 = Layout2d::<f64>::zeros(0);
    let mut s0 = GroupInABoxSolver::<f64>::new(&g0, vec![], GroupInABoxConfig::default()).unwrap();
    assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);
    assert!(s0.boxes().is_empty());

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let mut l1 = Layout2d::<f64>::zeros(1);
    let mut s1 = GroupInABoxSolver::<f64>::new(&g1, vec![9], GroupInABoxConfig::default()).unwrap();
    assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
    assert_eq!(s1.boxes().len(), 1);
    assert!((l1.x()[0] - 50.0).abs() < 1e-12);
    assert!((l1.y()[0] - 50.0).abs() < 1e-12);

    // Validation errors
    let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    assert!(matches!(
        GroupInABoxSolver::<f64>::new(&g2, vec![0], GroupInABoxConfig::default()),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        GroupInABoxSolver::<f64>::with_cluster_weights(
            &g2,
            vec![0, 1],
            vec![1.0],
            GroupInABoxConfig::default()
        ),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        GroupInABoxSolver::<f64>::with_cluster_weights(
            &g2,
            vec![0, 1],
            vec![1.0, 0.0],
            GroupInABoxConfig::default()
        ),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        GroupInABoxConfig {
            box_padding: 0.5,
            ..Default::default()
        }
        .validate(),
        Err(CoreError::InvalidInput(_))
    ));
}
