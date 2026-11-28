use egraph_core::{CoreError, CsrGraph, DeterministicRng, Layout2d, NodeExtents2d};
use egraph_layout_constraints::{
    ConstraintProjector, LayeringMethod, SeparationConstraint, assign_layers, cycle_edges,
    generate_layered_constraints, project_clustered_rectangle_no_overlap_2d,
    project_rectangle_no_overlap_2d, remove_cycles,
};
use egraph_spatial_trees::RectangleOverlapSweep;

fn seeded_layout_f64(n: usize, scale: f64, seed: u64) -> Layout2d<f64> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        x.push((rng.next_f64() * 2.0 - 1.0) * scale);
        y.push((rng.next_f64() * 2.0 - 1.0) * scale);
    }
    Layout2d::from_xy(x, y).unwrap()
}

fn seeded_layout_f32(n: usize, scale: f32, seed: u64) -> Layout2d<f32> {
    let l64 = seeded_layout_f64(n, scale as f64, seed);
    Layout2d::from_xy(
        l64.x().iter().map(|&v| v as f32).collect(),
        l64.y().iter().map(|&v| v as f32).collect(),
    )
    .unwrap()
}

#[test]
fn test_constraint_projector_split_expand_merge_and_errors() {
    // 1. Simple merge check (matches QP optimal [-0.5, 1.5, 3.0]).
    let mut proj = ConstraintProjector::<f64>::new(3);
    let mut coords = [0.0_f64, 1.0, 3.0];
    let c = [
        SeparationConstraint::new(0, 1, 2.0).unwrap(),
        SeparationConstraint::new(1, 2, 1.0).unwrap(),
    ];
    proj.project_1d(&mut coords, &c).unwrap();
    assert!((coords[0] - (-0.5)).abs() < 1e-8);
    assert!((coords[1] - 1.5).abs() < 1e-8);
    assert!((coords[2] - 3.0).abs() < 1e-8);

    // 2. Regression test for legacy split_blocks early `break` and `blocks[s]` overwrite bugs:
    // Two independent blocks plus a 3-var block where splitting is needed across non-0 blocks.
    let mut proj5 = ConstraintProjector::<f64>::new(5);
    let mut x5 = [0.0_f64, 0.0, 10.0, 10.0, 9.5];
    let c5 = [
        SeparationConstraint::new(0, 1, 1.0).unwrap(), // block 0: no split needed
        SeparationConstraint::new(2, 3, 2.0).unwrap(),
        SeparationConstraint::new(3, 4, 2.0).unwrap(),
        SeparationConstraint::new(2, 4, 3.0).unwrap(),
    ];
    proj5.project_1d(&mut x5, &c5).unwrap();
    for sc in &c5 {
        assert!(x5[sc.left() as usize] + sc.gap() <= x5[sc.right() as usize] + 1e-8);
    }

    // 3. Layout axis helper + f32 support.
    let mut layout32 = Layout2d::from_xy(vec![1.0_f32, 1.2, 5.0], vec![0.0, 0.0, 0.0]).unwrap();
    let mut proj32 = ConstraintProjector::<f32>::new(3);
    proj32
        .project_layout_axis(
            &mut layout32,
            0,
            &[SeparationConstraint::new(0, 1, 1.0_f32).unwrap()],
        )
        .unwrap();
    assert!(layout32.x()[0] + 1.0 <= layout32.x()[1] + 1e-6);

    // 4. Infeasible cycle guard (must return error cleanly, never infinite loop).
    let cycle_c = [
        SeparationConstraint::new(0, 1, 2.0).unwrap(),
        SeparationConstraint::new(1, 0, 2.0).unwrap(),
    ];
    let mut cx = [0.0_f64, 0.5];
    let mut proj2 = ConstraintProjector::<f64>::new(2);
    assert!(matches!(
        proj2.project_1d(&mut cx, &cycle_c),
        Err(CoreError::InvalidInput(_))
    ));

    // Invalid axis / out of bounds.
    assert!(matches!(
        proj32.project_layout_axis(&mut layout32, 2, &[]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        proj2.project_1d(&mut cx, &[SeparationConstraint::new(0, 5, 1.0).unwrap()]),
        Err(CoreError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn test_project_rectangle_no_overlap_2d_and_clustered_f64_f32() {
    // Dense overlapping rectangles in f64 (tests exact 2.0 gap, no early-break misses).
    let n = 60usize;
    let mut l64 = seeded_layout_f64(n, 2.0, 999);
    let mut hw64 = Vec::with_capacity(n);
    let mut hh64 = Vec::with_capacity(n);
    for i in 0..n {
        hw64.push(0.4 + (i % 5) as f64 * 0.25);
        hh64.push(0.3 + (i % 3) as f64 * 0.2);
    }
    project_rectangle_no_overlap_2d(&mut l64, NodeExtents2d::new(&hw64, &hh64).unwrap()).unwrap();
    assert!(
        !RectangleOverlapSweep::has_layout_overlaps(&l64, &hw64, &hh64).unwrap(),
        "f64 rectangle overlap removal left overlaps"
    );

    // Dense overlapping rectangles in f32 (tests 1-ULP guard).
    let mut l32 = seeded_layout_f32(40, 1.5, 555);
    let hw32 = vec![0.5_f32; 40];
    let hh32 = vec![0.4_f32; 40];
    project_rectangle_no_overlap_2d(&mut l32, NodeExtents2d::new(&hw32, &hh32).unwrap()).unwrap();
    assert!(
        !RectangleOverlapSweep::has_layout_overlaps(&l32, &hw32, &hh32).unwrap(),
        "f32 rectangle overlap removal left overlaps"
    );

    // Clustered overlap removal preserves rigid intra-cluster distances.
    let mut cl =
        Layout2d::from_xy(vec![0.0_f64, 1.0, 0.8, 1.8], vec![0.0_f64, 0.0, 0.2, 0.2]).unwrap();
    let cluster_ids = [0u32, 0, 1, 1];
    let chw = [0.4_f64; 4];
    let chh = [0.4_f64; 4];
    let dx0_before = cl.x()[1] - cl.x()[0];
    let dy0_before = cl.y()[1] - cl.y()[0];
    let dx1_before = cl.x()[3] - cl.x()[2];
    let dy1_before = cl.y()[3] - cl.y()[2];

    project_clustered_rectangle_no_overlap_2d(
        &mut cl,
        &cluster_ids,
        NodeExtents2d::new(&chw, &chh).unwrap(),
    )
    .unwrap();
    assert!(((cl.x()[1] - cl.x()[0]) - dx0_before).abs() < 1e-12);
    assert!(((cl.y()[1] - cl.y()[0]) - dy0_before).abs() < 1e-12);
    assert!(((cl.x()[3] - cl.x()[2]) - dx1_before).abs() < 1e-12);
    assert!(((cl.y()[3] - cl.y()[2]) - dy1_before).abs() < 1e-12);

    // Cluster 0 bounding box [min_x0..max_x0] x [min_y0..max_y0] and cluster 1 bounding box must not overlap.
    let c0_xmin = (cl.x()[0] - chw[0]).min(cl.x()[1] - chw[1]);
    let c0_xmax = (cl.x()[0] + chw[0]).max(cl.x()[1] + chw[1]);
    let c0_ymin = (cl.y()[0] - chh[0]).min(cl.y()[1] - chh[1]);
    let c0_ymax = (cl.y()[0] + chh[0]).max(cl.y()[1] + chh[1]);
    let c1_xmin = (cl.x()[2] - chw[2]).min(cl.x()[3] - chw[3]);
    let c1_xmax = (cl.x()[2] + chw[2]).max(cl.x()[3] + chw[3]);
    let c1_ymin = (cl.y()[2] - chh[2]).min(cl.y()[3] - chh[3]);
    let c1_ymax = (cl.y()[2] + chh[2]).max(cl.y()[3] + chh[3]);
    let boxes_overlap =
        c0_xmin < c1_xmax && c1_xmin < c0_xmax && c0_ymin < c1_ymax && c1_ymin < c0_ymax;
    assert!(!boxes_overlap);
}

#[test]
fn test_generate_layered_constraints() {
    let dag =
        CsrGraph::<()>::from_unweighted_edges(4, &[(0, 1), (1, 2), (0, 3)], true, false).unwrap();
    let layers: Vec<u32> = assign_layers(&dag, LayeringMethod::LongestPath).unwrap();
    let constraints = generate_layered_constraints(&dag, &layers, 2.5_f64).unwrap();
    assert_eq!(constraints.len(), 3);

    let mut ys = vec![0.0_f64; 4];
    let mut proj = ConstraintProjector::new(4);
    proj.project_1d(&mut ys, &constraints).unwrap();
    for u in 0..4 {
        for &v in dag.neighbors(u) {
            assert!(
                ys[v as usize] - ys[u] >= 2.5 * ((layers[v as usize] - layers[u]) as f64) - 1e-9
            );
        }
    }
}

#[test]
fn test_layering_longest_path_network_simplex_and_cycles() {
    // 1. Graph where NetworkSimplex strictly beats LongestPath total edge span:
    // Long chain 0->1->2->3->4, plus node 5 with 1 incoming (0->5) and 2 outgoing (5->3, 5->4).
    // LongestPath: r[5]=1 (spans: 1+2+3=6). NetworkSimplex: r[5]=2 (spans: 2+1+2=5).
    let g = CsrGraph::<()>::from_unweighted_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 3), (5, 4)],
        true,
        false,
    )
    .unwrap();
    let lp = assign_layers(&g, LayeringMethod::LongestPath).unwrap();
    let ns = assign_layers(&g, LayeringMethod::NetworkSimplex).unwrap();

    let span = |layers: &[u32]| -> u64 {
        (0..g.num_nodes())
            .flat_map(|u| {
                g.neighbors(u)
                    .iter()
                    .map(move |&v| u64::from(layers[v as usize] - layers[u]))
            })
            .sum()
    };
    assert_eq!(*lp.iter().min().unwrap(), 0);
    assert_eq!(*ns.iter().min().unwrap(), 0);
    for u in 0..g.num_nodes() {
        for &v in g.neighbors(u) {
            assert!(lp[v as usize] > lp[u]);
            assert!(ns[v as usize] > ns[u]);
        }
    }
    assert!(
        span(&ns) < span(&lp),
        "ns_span={} lp_span={}",
        span(&ns),
        span(&lp)
    );

    // 2. Random DAGs: NetworkSimplex span <= LongestPath span.
    let mut rng = DeterministicRng::from_seed_u64(4242);
    let n_rand = 40usize;
    let mut dag_edges = Vec::new();
    for u in 0..n_rand {
        for v in (u + 1)..n_rand {
            if rng.next_f64() < 0.2 {
                dag_edges.push((u as u32, v as u32));
            }
        }
    }
    let rand_dag = CsrGraph::<()>::from_unweighted_edges(n_rand, &dag_edges, true, false).unwrap();
    let rand_lp = assign_layers(&rand_dag, LayeringMethod::LongestPath).unwrap();
    let rand_ns = assign_layers(&rand_dag, LayeringMethod::NetworkSimplex).unwrap();
    let rand_span = |layers: &[u32]| -> u64 {
        (0..rand_dag.num_nodes())
            .flat_map(|u| {
                rand_dag
                    .neighbors(u)
                    .iter()
                    .map(move |&v| u64::from(layers[v as usize] - layers[u]))
            })
            .sum()
    };
    assert!(rand_span(&rand_ns) <= rand_span(&rand_lp));

    // 3. Cycle removal + sourceless / disconnected components (returns unweighted CsrGraph<()>).
    let cyclic = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 3)], // 2 disconnected pure cycles + isolated node 5
        Some(&[1.0, 2.0, 3.0, 4.0, 5.0]),
        true,
        false,
    )
    .unwrap();
    let back = cycle_edges(&cyclic).unwrap();
    assert_eq!(back.len(), 2);
    let dag: CsrGraph<()> = remove_cycles(&cyclic).unwrap();
    assert!(cycle_edges(&dag).unwrap().is_empty());

    let cyc_lp = assign_layers(&cyclic, LayeringMethod::LongestPath).unwrap();
    let cyc_ns = assign_layers(&cyclic, LayeringMethod::NetworkSimplex).unwrap();
    assert_eq!(cyc_lp.len(), 6);
    assert_eq!(cyc_ns.len(), 6);
    for u in 0..dag.num_nodes() {
        for &v in dag.neighbors(u) {
            assert!(cyc_lp[v as usize] > cyc_lp[u]);
            assert!(cyc_ns[v as usize] > cyc_ns[u]);
        }
    }

    // 4. Deep chain (20,000 nodes) verifies zero recursion stack overflow.
    let deep_n = 20_000usize;
    let deep_edges: Vec<(u32, u32)> = (0..(deep_n - 1))
        .map(|i| (i as u32, (i + 1) as u32))
        .collect();
    let deep = CsrGraph::<()>::from_unweighted_edges(deep_n, &deep_edges, true, false).unwrap();
    assert!(cycle_edges(&deep).unwrap().is_empty());
    let deep_layers = assign_layers(&deep, LayeringMethod::LongestPath).unwrap();
    assert_eq!(deep_layers[0], 0);
    assert_eq!(deep_layers[deep_n - 1], (deep_n - 1) as u32);

    // 5. Undirected graph rejected cleanly across layering and layered constraint generation.
    let undir = CsrGraph::<()>::from_unweighted_edges(2, &[(0, 1)], false, false).unwrap();
    assert!(matches!(
        cycle_edges(&undir),
        Err(CoreError::UnsupportedOperation(_))
    ));
    assert!(matches!(
        generate_layered_constraints(&undir, &[0u32, 1], 1.0_f64),
        Err(CoreError::UnsupportedOperation(_))
    ));
}
