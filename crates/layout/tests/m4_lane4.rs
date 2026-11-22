use std::time::Instant;

use egraph_core::{CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, SolverStatus};
use egraph_layout::constraints::{
    ConstraintProjector, SeparationConstraint, generate_layered_constraints,
    project_clustered_rectangle_no_overlap_2d, project_rectangle_no_overlap_2d,
};
use egraph_layout::fdeb::{FdebConfig, FdebSolver};
use egraph_layout::layering::{LayeringMethod, assign_layers, cycle_edges, remove_cycles};
use egraph_layout::prism::{PrismConfig, PrismOverlapRemoval, SparsePcgWorkspace};
use egraph_spatial::RectangleOverlapSweep;

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
fn test_sparse_pcg_workspace_basic_and_errors() {
    let mut pcg = SparsePcgWorkspace::new(3);
    // 2x2 SPD block + isolated singular row:
    // [3, 1, 0; 1, 2, 0; 0, 0, 0] x = [5, 5, 0] => x = [1, 2, x2]
    let indptr = [0u32, 2, 4, 5];
    let indices = [0u32, 1, 0, 1, 2];
    let values = [3.0, 1.0, 1.0, 2.0, 0.0];
    let b = [5.0, 5.0, 0.0];
    let mut x = [0.0, 0.0, 7.0];

    let iters = pcg
        .solve_csr(&indptr, &indices, &values, &b, &mut x, 1e-10, 50)
        .unwrap();
    assert!(iters > 0);
    assert!((x[0] - 1.0).abs() < 1e-8);
    assert!((x[1] - 2.0).abs() < 1e-8);
    assert_eq!(x[2], 7.0);

    // Zero residual exits immediately.
    assert_eq!(
        pcg.solve_csr(&indptr, &indices, &values, &b, &mut x, 1e-10, 50)
            .unwrap(),
        0
    );

    // Dimension mismatch and invalid tolerance guards.
    assert!(matches!(
        pcg.solve_csr(&[0u32, 1], &indices, &values, &b, &mut x, 1e-10, 50),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        pcg.solve_csr(&indptr, &indices, &values, &b, &mut x, -1.0, 50),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_prism_step_vs_run_bit_exact_f64_and_f32() {
    let cfg = PrismConfig::default();

    // f64 bit-exact check
    let hw64 = vec![0.6_f64; 24];
    let hh64 = vec![0.4_f64; 24];
    let mut l64_run = seeded_layout_f64(24, 1.0, 101);
    let mut l64_step = l64_run.clone();

    let mut s64_run = PrismOverlapRemoval::new(hw64.clone(), hh64.clone(), cfg).unwrap();
    let mut s64_step = PrismOverlapRemoval::new(hw64.clone(), hh64.clone(), cfg).unwrap();

    let st_run = s64_run.run(&mut l64_run).unwrap();
    assert_eq!(st_run, SolverStatus::Converged);
    assert_eq!(s64_run.last_overlap_count(), 0);
    assert!(!RectangleOverlapSweep::has_layout_overlaps(&l64_run, &hw64, &hh64).unwrap());

    assert_eq!(s64_step.status(), SolverStatus::Initialized);
    let _ = s64_step.run_steps(&mut l64_step, 2).unwrap();
    while !s64_step.status().is_finished() {
        s64_step.step(&mut l64_step).unwrap();
    }
    assert_eq!(s64_run.status(), s64_step.status());
    assert_eq!(s64_run.iteration(), s64_step.iteration());
    assert_eq!(s64_run.last_overlap_count(), s64_step.last_overlap_count());
    for i in 0..24 {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }

    // f32 bit-exact check
    let radii32 = vec![0.5_f32; 20];
    let mut l32_run = seeded_layout_f32(20, 0.8, 202);
    let mut l32_step = l32_run.clone();
    let mut s32_run = PrismOverlapRemoval::from_radii(radii32.clone(), cfg).unwrap();
    let mut s32_step = PrismOverlapRemoval::from_radii(radii32.clone(), cfg).unwrap();

    let st32 = s32_run.run(&mut l32_run).unwrap();
    assert_eq!(st32, SolverStatus::Converged);
    assert_eq!(s32_run.last_overlap_count(), 0);
    assert!(!RectangleOverlapSweep::has_layout_overlaps(&l32_run, &radii32, &radii32).unwrap());

    while !s32_step.status().is_finished() {
        s32_step.step(&mut l32_step).unwrap();
    }
    for i in 0..20 {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }
}

#[test]
fn test_prism_coincident_collinear_padding_and_errors() {
    // Coincident points at origin + padding.
    let cfg = PrismConfig {
        padding: 0.2,
        ..Default::default()
    };
    let radii = vec![0.5_f64; 12];
    let mut coincident = Layout2d::zeros(12);
    let mut solver = PrismOverlapRemoval::from_radii(radii.clone(), cfg).unwrap();
    let st = solver.run(&mut coincident).unwrap();
    assert_eq!(st, SolverStatus::Converged);
    assert_eq!(solver.last_overlap_count(), 0);
    // Effective half-extents include 0.5 * padding = 0.1 => 0.6.
    let eff = vec![0.6_f64; 12];
    assert!(!RectangleOverlapSweep::has_layout_overlaps(&coincident, &eff, &eff).unwrap());

    // Collinear points overlapping along X.
    let mut collinear = Layout2d::from_xy(
        vec![0.0_f64, 0.2, 0.4, 0.6, 0.8],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let hw = vec![0.5_f64; 5];
    let hh = vec![0.5_f64; 5];
    let mut s_col =
        PrismOverlapRemoval::new(hw.clone(), hh.clone(), PrismConfig::default()).unwrap();
    assert_eq!(s_col.run(&mut collinear).unwrap(), SolverStatus::Converged);
    assert_eq!(s_col.last_overlap_count(), 0);
    assert!(!RectangleOverlapSweep::has_layout_overlaps(&collinear, &hw, &hh).unwrap());

    // Already non-overlapping layout converges without altering coordinates.
    let clean = Layout2d::from_xy(vec![0.0_f64, 5.0], vec![0.0, 5.0]).unwrap();
    let mut clean_work = clean.clone();
    let mut s_clean =
        PrismOverlapRemoval::from_radii(vec![0.5, 0.5], PrismConfig::default()).unwrap();
    assert_eq!(
        s_clean.run(&mut clean_work).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(clean, clean_work);

    // Invalid inputs.
    assert!(matches!(
        PrismOverlapRemoval::new(vec![1.0], vec![1.0, 2.0], PrismConfig::default()),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        PrismOverlapRemoval::from_radii(vec![-0.1], PrismConfig::default()),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        PrismOverlapRemoval::from_radii(
            vec![1.0],
            PrismConfig {
                max_scale_factor: 1.0,
                ..Default::default()
            }
        ),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_prism_large_scale_1000_to_2000_zero_overlaps_fast() {
    for &n in &[1000usize, 1500usize, 2000usize] {
        let mut layout = seeded_layout_f64(n, (n as f64).sqrt() * 0.6, 777 + n as u64);
        let hw = vec![0.5_f64; n];
        let hh = vec![0.35_f64; n];
        let mut solver = PrismOverlapRemoval::new(
            hw.clone(),
            hh.clone(),
            PrismConfig {
                max_iterations: 50,
                ..Default::default()
            },
        )
        .unwrap();

        let t0 = Instant::now();
        let status = solver.run(&mut layout).unwrap();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "PRISM n={n}: status={status:?} iters={} overlaps={} elapsed={elapsed_ms:.2} ms",
            solver.iteration(),
            solver.last_overlap_count()
        );

        assert_eq!(status, SolverStatus::Converged, "n={n} did not converge");
        assert_eq!(solver.last_overlap_count(), 0);
        assert!(!RectangleOverlapSweep::has_layout_overlaps(&layout, &hw, &hh).unwrap());
        let time_budget_ms = if cfg!(debug_assertions) {
            15_000.0
        } else {
            500.0
        };
        assert!(
            elapsed_ms < time_budget_ms,
            "n={n} took {elapsed_ms:.1} ms (budget {time_budget_ms:.1} ms)"
        );
    }
}

#[test]
fn test_constraint_projector_split_expand_merge_and_errors() {
    // 1. Simple merge check (matches QP optimal [-0.5, 1.5, 3.0]).
    let mut proj = ConstraintProjector::<f64>::new(3);
    let mut coords = [0.0_f64, 1.0, 3.0];
    let c = [
        SeparationConstraint::new(0, 1, 2.0),
        SeparationConstraint::new(1, 2, 1.0),
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
        SeparationConstraint::new(0, 1, 1.0), // block 0: no split needed
        SeparationConstraint::new(2, 3, 2.0),
        SeparationConstraint::new(3, 4, 2.0),
        SeparationConstraint::new(2, 4, 3.0),
    ];
    proj5.project_1d(&mut x5, &c5).unwrap();
    for sc in &c5 {
        assert!(x5[sc.left as usize] + sc.gap <= x5[sc.right as usize] + 1e-8);
    }

    // 3. Layout axis helper + f32 support.
    let mut layout32 = Layout2d::from_xy(vec![1.0_f32, 1.2, 5.0], vec![0.0, 0.0, 0.0]).unwrap();
    let mut proj32 = ConstraintProjector::<f32>::new(3);
    proj32
        .project_layout_axis(
            &mut layout32,
            0,
            &[SeparationConstraint::new(0, 1, 1.0_f32)],
        )
        .unwrap();
    assert!(layout32.x()[0] + 1.0 <= layout32.x()[1] + 1e-6);

    // 4. Infeasible cycle guard (must return error cleanly, never infinite loop).
    let cycle_c = [
        SeparationConstraint::new(0, 1, 2.0),
        SeparationConstraint::new(1, 0, 2.0),
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
        proj2.project_1d(&mut cx, &[SeparationConstraint::new(0, 5, 1.0)]),
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
    project_rectangle_no_overlap_2d(&mut l64, &hw64, &hh64).unwrap();
    assert!(
        !RectangleOverlapSweep::has_layout_overlaps(&l64, &hw64, &hh64).unwrap(),
        "f64 rectangle overlap removal left overlaps"
    );

    // Dense overlapping rectangles in f32 (tests 1-ULP guard).
    let mut l32 = seeded_layout_f32(40, 1.5, 555);
    let hw32 = vec![0.5_f32; 40];
    let hh32 = vec![0.4_f32; 40];
    project_rectangle_no_overlap_2d(&mut l32, &hw32, &hh32).unwrap();
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

    project_clustered_rectangle_no_overlap_2d(&mut cl, &cluster_ids, &chw, &chh).unwrap();
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
    let dag = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (0, 3)], None, true, false).unwrap();
    let layers = assign_layers(&dag, LayeringMethod::LongestPath).unwrap();
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
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 3), (5, 4)],
        None,
        true,
        false,
    )
    .unwrap();
    let lp = assign_layers(&g, LayeringMethod::LongestPath).unwrap();
    let ns = assign_layers(&g, LayeringMethod::NetworkSimplex).unwrap();

    let span = |layers: &[usize]| -> usize {
        (0..g.num_nodes())
            .flat_map(|u| {
                g.neighbors(u)
                    .iter()
                    .map(move |&v| layers[v as usize] - layers[u])
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
    let rand_dag = CsrGraph::<f64>::from_edges(n_rand, &dag_edges, None, true, false).unwrap();
    let rand_lp = assign_layers(&rand_dag, LayeringMethod::LongestPath).unwrap();
    let rand_ns = assign_layers(&rand_dag, LayeringMethod::NetworkSimplex).unwrap();
    let rand_span = |layers: &[usize]| -> usize {
        (0..rand_dag.num_nodes())
            .flat_map(|u| {
                rand_dag
                    .neighbors(u)
                    .iter()
                    .map(move |&v| layers[v as usize] - layers[u])
            })
            .sum()
    };
    assert!(rand_span(&rand_ns) <= rand_span(&rand_lp));

    // 3. Cycle removal + sourceless / disconnected components.
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
    let dag = remove_cycles(&cyclic).unwrap();
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
    let deep = CsrGraph::<f64>::from_edges(deep_n, &deep_edges, None, true, false).unwrap();
    assert!(cycle_edges(&deep).unwrap().is_empty());
    let deep_layers = assign_layers(&deep, LayeringMethod::LongestPath).unwrap();
    assert_eq!(deep_layers[0], 0);
    assert_eq!(deep_layers[deep_n - 1], deep_n - 1);

    // 5. Undirected graph rejected cleanly.
    let undir = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    assert!(matches!(
        cycle_edges(&undir),
        Err(CoreError::UnsupportedOperation(_))
    ));
}

#[test]
fn test_fdeb_step_vs_run_collinear_zero_length_and_bundling() {
    // 1. Two parallel close edges bundle closer together, plus collinear, antiparallel, and zero-length edges.
    let graph = CsrGraph::<f64>::from_edges(
        8,
        &[
            (0, 1), // horizontal y=0
            (2, 3), // horizontal y=0.3 (parallel, same direction -> bundles!)
            (4, 5), // collinear antiparallel on y=0.3
            (6, 7), // zero-length edge (same coords!)
        ],
        None,
        true,
        false,
    )
    .unwrap();
    let layout64 = Layout2d::from_xy(
        vec![0.0_f64, 10.0, 0.0, 10.0, 10.0, 0.0, 5.0, 5.0],
        vec![0.0_f64, 0.0, 0.3, 0.3, 0.3, 0.3, 2.0, 2.0],
    )
    .unwrap();

    let cfg = FdebConfig {
        cycles: 4,
        i0: 30,
        ..Default::default()
    };

    // f64 bit-exact step == run
    let mut s64_run = FdebSolver::new(&graph, &layout64, cfg).unwrap();
    let mut s64_step = FdebSolver::new(&graph, &layout64, cfg).unwrap();
    assert_eq!(s64_run.status(), SolverStatus::Initialized);
    assert_eq!(s64_run.num_edges(), 4);
    assert_eq!(s64_run.edge_offsets(), &[0, 2, 4, 6, 8]);

    assert_eq!(s64_run.run().unwrap(), SolverStatus::Completed);
    assert!(s64_run.is_finished());
    assert_eq!(s64_run.iteration(), s64_run.total_iterations());

    let _ = s64_step.run_steps(7).unwrap();
    while !s64_step.is_finished() {
        s64_step.step().unwrap();
    }
    assert_eq!(s64_run.edge_offsets(), s64_step.edge_offsets());
    for i in 0..s64_run.points_x().len() {
        assert_eq!(
            s64_run.points_x()[i].to_bits(),
            s64_step.points_x()[i].to_bits()
        );
        assert_eq!(
            s64_run.points_y()[i].to_bits(),
            s64_step.points_y()[i].to_bits()
        );
        assert!(s64_run.points_x()[i].is_finite());
        assert!(s64_run.points_y()[i].is_finite());
    }

    // Verify bundling: midpoint of edge 0 (y=0) and edge 1 (y=0.3) moved closer than initial 0.3!
    let off = s64_run.edge_offsets();
    let mid0 = (off[0] as usize + off[1] as usize) / 2;
    let mid1 = (off[1] as usize + off[2] as usize) / 2;
    let y_gap = (s64_run.points_y()[mid1] - s64_run.points_y()[mid0]).abs();
    assert!(
        y_gap < 0.25,
        "expected edges to bundle closer, y_gap={y_gap}"
    );

    // Interleaved length matches 2 * points_x.len().
    assert_eq!(
        s64_run.to_interleaved_points().len(),
        s64_run.points_x().len() * 2
    );

    // f32 bit-exact step == run
    let layout32 = Layout2d::from_xy(
        layout64.x().iter().map(|&v| v as f32).collect(),
        layout64.y().iter().map(|&v| v as f32).collect(),
    )
    .unwrap();
    let mut s32_run = FdebSolver::new(&graph, &layout32, cfg).unwrap();
    let mut s32_step = FdebSolver::new(&graph, &layout32, cfg).unwrap();
    s32_run.run().unwrap();
    while !s32_step.is_finished() {
        s32_step.step().unwrap();
    }
    for i in 0..s32_run.points_x().len() {
        assert_eq!(
            s32_run.points_x()[i].to_bits(),
            s32_step.points_x()[i].to_bits()
        );
        assert_eq!(
            s32_run.points_y()[i].to_bits(),
            s32_step.points_y()[i].to_bits()
        );
        assert!(s32_run.points_x()[i].is_finite());
        assert!(s32_run.points_y()[i].is_finite());
    }

    // Input layout was passed by reference and remains untouched!
    assert_eq!(layout64.x()[0], 0.0);

    // Invalid config and dimension mismatch checks.
    assert!(matches!(
        FdebSolver::new(
            &graph,
            &layout64,
            FdebConfig {
                minimum_edge_compatibility: 1.5,
                ..Default::default()
            }
        ),
        Err(CoreError::InvalidInput(_))
    ));
    let wrong_layout = Layout2d::<f64>::zeros(2);
    assert!(matches!(
        FdebSolver::new(&graph, &wrong_layout, cfg),
        Err(CoreError::DimensionMismatch { .. })
    ));
}

fn naive_overlap_count_f32(layout: &Layout2d<f32>, hw: &[f32], hh: &[f32]) -> usize {
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut count = 0usize;
    for i in 0..n {
        let ax_min = xs[i] - hw[i];
        let ax_max = xs[i] + hw[i];
        let ay_min = ys[i] - hh[i];
        let ay_max = ys[i] + hh[i];
        for j in (i + 1)..n {
            let bx_min = xs[j] - hw[j];
            let bx_max = xs[j] + hw[j];
            let by_min = ys[j] - hh[j];
            let by_max = ys[j] + hh[j];
            if ax_min < bx_max && bx_min < ax_max && ay_min < by_max && by_min < ay_max {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn test_prism_f32_large_coordinate_offsets_converge_zero_overlaps() {
    // Regression test for Bug #2: f32 layouts at coordinate offsets 5000.0 and 1_000_000.0
    // must converge with 0 overlaps across 20 random seeds (both Delaunay and StaticKdTree fallback).
    let n = 50usize;
    let hw = vec![0.5_f32; n];
    let hh = vec![0.5_f32; n];

    for &offset in &[5_000.0_f32, 1_000_000.0_f32] {
        for seed in 0..20u64 {
            let mut rng = DeterministicRng::from_seed_u64(5_000 + seed);
            let mut x = Vec::with_capacity(n);
            let mut y = Vec::with_capacity(n);
            for _ in 0..n {
                x.push(offset + (rng.next_f64() as f32 * 4.0 - 2.0));
                y.push(-offset + (rng.next_f64() as f32 * 4.0 - 2.0));
            }
            let mut layout = Layout2d::from_xy(x, y).unwrap();
            let mut solver =
                PrismOverlapRemoval::new(hw.clone(), hh.clone(), PrismConfig::default()).unwrap();
            let status = solver.run(&mut layout).unwrap();
            assert_eq!(
                status,
                SolverStatus::Converged,
                "PRISM f32 did not converge at offset={offset} seed={seed}: iters={} overlaps={}",
                solver.iteration(),
                solver.last_overlap_count()
            );
            assert_eq!(solver.last_overlap_count(), 0);
            assert_eq!(
                naive_overlap_count_f32(&layout, &hw, &hh),
                0,
                "naive O(N^2) overlap check failed at offset={offset} seed={seed}"
            );
            assert!(!RectangleOverlapSweep::has_layout_overlaps(&layout, &hw, &hh).unwrap());
        }
    }

    // Also verify project_rectangle_no_overlap_2d with non-power-of-two f32 extents at 1_000_000.0.
    let hw_nonpow = vec![0.3_f32; n];
    let hh_nonpow = vec![0.3_f32; n];
    for &offset in &[5_000.0_f32, 1_000_000.0_f32] {
        let mut rng = DeterministicRng::from_seed_u64(777);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(offset + (rng.next_f64() as f32 * 4.0 - 2.0));
            y.push(-offset + (rng.next_f64() as f32 * 4.0 - 2.0));
        }
        let mut layout = Layout2d::from_xy(x, y).unwrap();
        project_rectangle_no_overlap_2d(&mut layout, &hw_nonpow, &hh_nonpow).unwrap();
        assert_eq!(naive_overlap_count_f32(&layout, &hw_nonpow, &hh_nonpow), 0);
        assert!(
            !RectangleOverlapSweep::has_layout_overlaps(&layout, &hw_nonpow, &hh_nonpow).unwrap()
        );
    }
}
