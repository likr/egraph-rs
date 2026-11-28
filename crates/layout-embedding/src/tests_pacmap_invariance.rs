use egraph_core::{CsrGraph, DeterministicRng, IterativeSolver, Layout2d};
use egraph_neighborhood::extract_1hop_2hop_csr;

use crate::test_support::circle_layout;
use crate::{PacmapGraphConfig, PacmapGraphSolver};

#[test]
fn test_pacmap_graph_1hop_and_2hop_vertex_permutation_invariance() {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut next_id = 1u32;

    let add_k6_cap = |anchor: u32, edges: &mut Vec<(u32, u32)>, next_id: &mut u32| -> u32 {
        let base = *next_id;
        *next_id += 6;
        for i in 0..6u32 {
            for j in (i + 1)..6u32 {
                edges.push((base + i, base + j));
            }
        }
        edges.push((anchor, base));
        base
    };

    let add_k6_twin_pair =
        |anchor_a: u32, anchor_b: u32, edges: &mut Vec<(u32, u32)>, next_id: &mut u32| {
            let a = add_k6_cap(anchor_a, edges, next_id);
            let b = add_k6_cap(anchor_b, edges, next_id);
            edges.push((a, b));
        };

    for _ in 0..5 {
        edges.push((0, next_id));
        next_id += 1;
    }
    for _ in 0..75 {
        let w = next_id;
        let p = next_id + 1;
        let q1 = next_id + 2;
        let q2 = next_id + 3;
        next_id += 4;
        edges.extend([(0, w), (w, p), (p, q1), (p, q2)]);
    }

    let c0 = next_id;
    let m1 = next_id + 1;
    let m2 = next_id + 2;
    let l1 = next_id + 3;
    let l2 = next_id + 4;
    let t_mid = next_id + 5;
    let t_fast = next_id + 6;
    next_id += 7;
    edges.extend([
        (c0, m1),
        (c0, m2),
        (c0, l1),
        (c0, l2),
        (m1, t_mid),
        (m2, t_mid),
        (l1, l2),
        (l1, t_fast),
        (l2, t_fast),
    ]);
    for _ in 0..3 {
        add_k6_cap(t_mid, &mut edges, &mut next_id);
        add_k6_cap(t_fast, &mut edges, &mut next_id);
    }
    add_k6_cap(l1, &mut edges, &mut next_id);
    add_k6_cap(l1, &mut edges, &mut next_id);
    add_k6_cap(l2, &mut edges, &mut next_id);
    add_k6_cap(l2, &mut edges, &mut next_id);
    for m in [m1, m2] {
        for _ in 0..3 {
            add_k6_cap(m, &mut edges, &mut next_id);
        }
        for _ in 0..10 {
            add_k6_twin_pair(m, m, &mut edges, &mut next_id);
        }
    }
    let anchor_l2 = next_id;
    next_id += 1;
    add_k6_twin_pair(l2, anchor_l2, &mut edges, &mut next_id);
    for _ in 0..4 {
        add_k6_cap(anchor_l2, &mut edges, &mut next_id);
    }

    let z = next_id;
    next_id += 1;
    for _ in 0..5 {
        add_k6_cap(z, &mut edges, &mut next_id);
    }
    for _ in 0..3 {
        let h = next_id;
        next_id += 1;
        edges.push((z, h));
        for _ in 0..5 {
            add_k6_cap(h, &mut edges, &mut next_id);
        }
        for _ in 0..7 {
            add_k6_twin_pair(h, h, &mut edges, &mut next_id);
        }
    }

    let n = next_id as usize;
    let g_base = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
    let cfg = PacmapGraphConfig::new()
        .with_n_neighbors(5)
        .with_mn_ratio(0.0)
        .with_fp_ratio(0.0)
        .with_phase_iterations([1, 1, 1])
        .with_seed(42);
    let s_base = PacmapGraphSolver::<Layout2d<f64>>::new(&g_base, cfg.clone()).unwrap();
    assert!(
        s_base
            .nb_pairs()
            .contains(&(c0.min(t_fast), c0.max(t_fast))),
        "expected c0 to select unique Adamic-Adar top 2-hop neighbor t_fast"
    );

    let perm_reverse: Vec<u32> = (0..n as u32).map(|u| (n as u32 - 1) - u).collect();
    let perm_shift: Vec<u32> = (0..n as u32).map(|u| (u + 37) % (n as u32)).collect();
    let mut perm_shuffle: Vec<u32> = (0..n as u32).collect();
    let mut perm_rng = DeterministicRng::from_seed_u64(12345);
    perm_rng.shuffle(&mut perm_shuffle);

    for perm in [&perm_reverse, &perm_shift, &perm_shuffle] {
        let perm_edges: Vec<(u32, u32)> = edges
            .iter()
            .map(|&(u, v)| (perm[u as usize], perm[v as usize]))
            .collect();
        let g_perm = CsrGraph::<f64>::from_edges(n, &perm_edges, None, false, false).unwrap();
        let s_perm = PacmapGraphSolver::<Layout2d<f64>>::new(&g_perm, cfg.clone()).unwrap();

        let mut expected: Vec<(u32, u32)> = s_base
            .nb_pairs()
            .iter()
            .map(|&(u, v)| {
                let pu = perm[u as usize];
                let pv = perm[v as usize];
                if pu < pv { (pu, pv) } else { (pv, pu) }
            })
            .collect();
        expected.sort_unstable();
        assert_eq!(
            s_perm.nb_pairs(),
            expected.as_slice(),
            "nb_pairs failed exact vertex-permutation invariance"
        );
    }
}

#[test]
fn test_pacmap_graph_non_hub_2hop_and_fp_fallback_on_hub_and_small_graphs() {
    let n = 256usize;
    let mut edges: Vec<(u32, u32)> = (1..=100u32).map(|v| (0, v)).collect();
    for v in 101..=180u32 {
        edges.push((2, v));
    }
    edges.push((100, 181));
    for v in 181..255u32 {
        edges.push((v, v + 1));
    }
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
    let (adj1, adj2) = extract_1hop_2hop_csr(&g).unwrap();
    let cfg = PacmapGraphConfig::new()
        .with_n_neighbors(4)
        .with_mn_ratio(0.5)
        .with_fp_ratio(2.0)
        .with_phase_iterations([4, 4, 4])
        .with_seed(31);
    let mut solver = PacmapGraphSolver::<Layout2d<f64>>::new(&g, cfg).unwrap();

    let mut leaf_mn_counts = vec![0usize; n];
    let mut leaf_fp_counts = vec![0usize; n];
    for &(u, _) in solver.mn_pairs() {
        leaf_mn_counts[u as usize] += 1;
    }
    for &(u, _) in solver.fp_pairs() {
        leaf_fp_counts[u as usize] += 1;
    }
    for u in 1..=99usize {
        if u == 2 {
            continue;
        }
        assert_eq!(leaf_mn_counts[u], 2);
        assert_eq!(leaf_fp_counts[u], 8);
    }
    for &(u, v) in solver.mn_pairs().iter().chain(solver.fp_pairs().iter()) {
        let ui = u as usize;
        let vi = v as usize;
        if (200..=250).contains(&ui) && (200..=250).contains(&vi) {
            assert!(!adj1.neighbors(ui).contains(&v));
            assert!(!adj2.neighbors(ui).contains(&v));
        }
    }
    let mut layout = circle_layout::<f64>(n, 1.0);
    solver.run(&mut layout).unwrap();
    assert!(layout.is_all_finite());

    let small_cfg = PacmapGraphConfig::new()
        .with_n_neighbors(4)
        .with_mn_ratio(0.5)
        .with_fp_ratio(2.0)
        .with_phase_iterations([2, 2, 2])
        .with_seed(42);

    let mut k10_edges = Vec::new();
    for u in 0..10u32 {
        for v in (u + 1)..10u32 {
            k10_edges.push((u, v));
        }
    }
    let k10 = CsrGraph::<f64>::from_edges(10, &k10_edges, None, false, false).unwrap();
    let s_k10 = PacmapGraphSolver::<Layout2d<f64>>::new(&k10, small_cfg.clone()).unwrap();
    assert!(s_k10.fp_pairs().is_empty());

    let s20_edges: Vec<(u32, u32)> = (1..20u32).map(|v| (0, v)).collect();
    let s20 = CsrGraph::<f64>::from_edges(20, &s20_edges, None, false, false).unwrap();
    let s_s20 = PacmapGraphSolver::<Layout2d<f64>>::new(&s20, small_cfg.clone()).unwrap();
    let mut s20_fp_per_u: Vec<Vec<u32>> = vec![Vec::new(); 20];
    for &(u, v) in s_s20.fp_pairs() {
        assert_ne!(u, v);
        assert!(!s20.neighbors(u as usize).contains(&v));
        s20_fp_per_u[u as usize].push(v);
    }
    assert!(s20_fp_per_u[0].is_empty());
    for row in &mut s20_fp_per_u[1..20] {
        assert_eq!(row.len(), 8);
        row.sort_unstable();
        row.dedup();
        assert_eq!(row.len(), 8);
    }

    let c10_edges: Vec<(u32, u32)> = (0..10u32).map(|u| (u, (u + 1) % 10)).collect();
    let c10 = CsrGraph::<f64>::from_edges(10, &c10_edges, None, false, false).unwrap();
    let s_c10 = PacmapGraphSolver::<Layout2d<f64>>::new(&c10, small_cfg).unwrap();
    let mut c10_fp_per_u: Vec<Vec<u32>> = vec![Vec::new(); 10];
    for &(u, v) in s_c10.fp_pairs() {
        assert_ne!(u, v);
        assert!(!c10.neighbors(u as usize).contains(&v));
        let pair = if u < v { (u, v) } else { (v, u) };
        assert!(s_c10.nb_pairs().binary_search(&pair).is_err());
        c10_fp_per_u[u as usize].push(v);
    }
    for row in &mut c10_fp_per_u {
        assert_eq!(row.len(), 5);
        row.sort_unstable();
        row.dedup();
        assert_eq!(row.len(), 5);
    }
}
