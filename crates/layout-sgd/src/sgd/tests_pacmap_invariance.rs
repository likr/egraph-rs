use egraph_core::{CsrGraph, DeterministicRng, IterativeSolver};
use egraph_neighborhood::extract_1hop_2hop_csr;

use super::super::sgd::{PacmapGraphConfig, SgdSolver};
use super::test_support::circle_layout;

#[test]
fn test_pacmap_graph_1hop_and_2hop_vertex_permutation_invariance() {
    // Build an asymmetric graph where every vertex's 1-hop cutoff and 2-hop top-k boundary
    // is strictly tie-free (zero RNG consumption), combining:
    // (a) super-hub 0 (deg = 80 > 64) with 5 deg=1 leaves + 75 deg=2->deg=3 bridges,
    // (b) medium-degree intermediates (deg = 25, previously RNG-subsampled at per_w_cap = 20),
    // (c) local deg = 5..6 bridges with a unique top Adamic-Adar 2-hop target for c0, and
    // (d) a truncated 1-hop vertex z (deg(z) = 8 > n_neighbors = 5) whose 5 lowest-degree
    //     neighbors (deg = 6) are strictly separated from its 3 high-degree neighbors (deg = 20).
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut next_id = 1u32;

    // Helper: attach a K_6 gadget whose entry vertex gets deg = 6 (5 K_6 peers of deg=5 + anchor).
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

    // Helper: attach two K_6 gadgets linked at their entry vertices so each entry gets deg = 7
    // (5 K_6 peers of deg=5 + twin entry of deg=7 + anchor).
    let add_k6_twin_pair =
        |anchor_a: u32, anchor_b: u32, edges: &mut Vec<(u32, u32)>, next_id: &mut u32| {
            let a = add_k6_cap(anchor_a, edges, next_id);
            let b = add_k6_cap(anchor_b, edges, next_id);
            edges.push((a, b));
        };

    // (a) Super-hub 0: 5 deg=1 leaves + 75 deg=2 bridges (0 - w - p - {q1, q2}).
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

    // (b) & (c) Asymmetric local cluster around c0 (deg = 4 < n_neighbors = 5, rem = 1):
    //     c0 -> {m1(25), m2(25), l1(5), l2(6)}.
    //     t_fast has AA = 1/ln(5) + 1/ln(6) > 2/ln(25) (t_mid), and no other node enters
    //     an ambiguous tie.
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
    // Pad t_mid and t_fast to deg = 5 == n_neighbors.
    for _ in 0..3 {
        add_k6_cap(t_mid, &mut edges, &mut next_id);
        add_k6_cap(t_fast, &mut edges, &mut next_id);
    }
    // Pad l1 to deg = 5 (currently 3) and l2 to deg = 6 (currently 3; two deg=6 + one deg=7).
    add_k6_cap(l1, &mut edges, &mut next_id);
    add_k6_cap(l1, &mut edges, &mut next_id);
    add_k6_cap(l2, &mut edges, &mut next_id);
    add_k6_cap(l2, &mut edges, &mut next_id);
    // Pad m1 and m2 to deg = 25 (currently 2: c0(4) + t_mid(5)):
    // give each 3 deg=6 caps (so {c0, t_mid, g1, g2, g3} have deg <= 6 and tie_end == 5)
    // plus 20 deg=7 twin-K6 attachments (10 pairs on m1, 10 pairs on m2).
    for m in [m1, m2] {
        for _ in 0..3 {
            add_k6_cap(m, &mut edges, &mut next_id);
        }
        for _ in 0..10 {
            add_k6_twin_pair(m, m, &mut edges, &mut next_id);
        }
    }
    // Give l2 one side of a twin pair whose other side anchors to a fresh deg=5 hub.
    let anchor_l2 = next_id;
    next_id += 1;
    add_k6_twin_pair(l2, anchor_l2, &mut edges, &mut next_id);
    for _ in 0..4 {
        add_k6_cap(anchor_l2, &mut edges, &mut next_id);
    }

    // (d) Truncated 1-hop vertex z (deg(z) = 8 > n_neighbors = 5):
    //     5 neighbors with deg = 6 (K_6 caps) and 3 neighbors h_1..h_3 with deg = 20
    //     (each h_k has z(8), 5 K_6 caps of deg=6, and 14 deg=7 twin links).
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
    let cfg = PacmapGraphConfig {
        n_neighbors: 5,
        mn_ratio: 0.0,
        fp_ratio: 0.0,
        phase1_iters: 1,
        phase2_iters: 1,
        phase3_iters: 1,
        seed: 42,
        ..Default::default()
    };
    let s_base = SgdSolver::<f64>::pacmap_graph(&g_base, cfg).unwrap();
    let st_base = s_base.pacmap.as_ref().unwrap();
    assert!(
        st_base.nb_pairs.contains(&(c0.min(t_fast), c0.max(t_fast))),
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
        let s_perm = SgdSolver::<f64>::pacmap_graph(&g_perm, cfg).unwrap();
        let st_perm = s_perm.pacmap.as_ref().unwrap();

        let mut expected: Vec<(u32, u32)> = st_base
            .nb_pairs
            .iter()
            .map(|&(u, v)| {
                let pu = perm[u as usize];
                let pv = perm[v as usize];
                if pu < pv { (pu, pv) } else { (pv, pu) }
            })
            .collect();
        expected.sort_unstable();
        assert_eq!(
            st_perm.nb_pairs, expected,
            "nb_pairs failed exact vertex-permutation invariance"
        );
    }
}

#[test]
fn test_pacmap_graph_non_hub_2hop_and_fp_fallback_on_hub_and_small_graphs() {
    // 1. Hub-bypass in both stamp and binary_search branches of has_common_neighbor_non_hub:
    //    N = 256 -> non_hub_thresh = 16.
    //    - Hub 0 connects to 1..=100 (deg(0) = 100 > 16).
    //    - Vertex 2 connects to 101..=180 (deg(2) = 81 > 64) so comparing leaf u = 1
    //      (deg = 1) with c = 2 exercises the binary_search branch, while comparing u = 1
    //      with leaf c = 3 (deg = 1) exercises the stamp branch.
    //    - Path 100-181-...-255 gives non-hub 2-hop vertices (deg <= 2 <= 16).
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
    let cfg = PacmapGraphConfig {
        n_neighbors: 4,
        mn_ratio: 0.5,
        fp_ratio: 2.0,
        phase1_iters: 4,
        phase2_iters: 4,
        phase3_iters: 4,
        seed: 31,
        ..Default::default()
    };
    let mut solver = SgdSolver::<f64>::pacmap_graph(&g, cfg).unwrap();
    let st = solver.pacmap.as_ref().unwrap();

    let mut leaf_mn_counts = vec![0usize; n];
    let mut leaf_fp_counts = vec![0usize; n];
    for &(u, _) in &st.mn_pairs {
        leaf_mn_counts[u as usize] += 1;
    }
    for &(u, _) in &st.fp_pairs {
        leaf_fp_counts[u as usize] += 1;
    }
    for u in 1..=99usize {
        if u == 2 {
            continue;
        }
        assert_eq!(
            leaf_mn_counts[u], 2,
            "hub leaf {u} must receive full MN quota despite hub 0"
        );
        assert_eq!(
            leaf_fp_counts[u], 8,
            "hub leaf {u} must receive full FP quota despite hub 0"
        );
    }
    // True non-hub 2-hop neighbors on path 200..250 (deg <= 2 <= 16) stay excluded from MN & FP.
    for &(u, v) in st.mn_pairs.iter().chain(st.fp_pairs.iter()) {
        let ui = u as usize;
        let vi = v as usize;
        if (200..=250).contains(&ui) && (200..=250).contains(&vi) {
            assert!(!adj1.neighbors(ui).contains(&v));
            assert!(
                !adj2.neighbors(ui).contains(&v),
                "non-hub 2-hop pair ({u}, {v}) must remain excluded from MN/FP"
            );
        }
    }
    let mut layout = circle_layout::<f64>(n, 1.0);
    solver.run(&mut layout).unwrap();
    assert!(layout.is_all_finite());

    // 2. Small-graph FP fallback & O(1) n_excluded >= n guard:
    let small_cfg = PacmapGraphConfig {
        n_neighbors: 4,
        mn_ratio: 0.5,
        fp_ratio: 2.0,
        phase1_iters: 2,
        phase2_iters: 2,
        phase3_iters: 2,
        seed: 42,
        ..Default::default()
    };

    // 2a. Clique K_10: n_excluded == n for every vertex -> zero FP pairs, no hang.
    let mut k10_edges = Vec::new();
    for u in 0..10u32 {
        for v in (u + 1)..10u32 {
            k10_edges.push((u, v));
        }
    }
    let k10 = CsrGraph::<f64>::from_edges(10, &k10_edges, None, false, false).unwrap();
    let s_k10 = SgdSolver::<f64>::pacmap_graph(&k10, small_cfg).unwrap();
    assert!(s_k10.pacmap.as_ref().unwrap().fp_pairs.is_empty());

    // 2b. Star S_20: hub 0 (n_excluded = 20) gets 0 FPs, every leaf gets 8 FPs.
    let s20_edges: Vec<(u32, u32)> = (1..20u32).map(|v| (0, v)).collect();
    let s20 = CsrGraph::<f64>::from_edges(20, &s20_edges, None, false, false).unwrap();
    let s_s20 = SgdSolver::<f64>::pacmap_graph(&s20, small_cfg).unwrap();
    let st_s20 = s_s20.pacmap.as_ref().unwrap();
    let mut s20_fp_per_u: Vec<Vec<u32>> = vec![Vec::new(); 20];
    for &(u, v) in &st_s20.fp_pairs {
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

    // 2c. Cycle C_10: deg(u)=2, nb_u=4 (1-hop + 2-hop) -> 10 - 1 - 4 = 5 < 8 FPs per node,
    //     all 1-hop and NB pairs strictly excluded and zero duplicates per source vertex.
    let c10_edges: Vec<(u32, u32)> = (0..10u32).map(|u| (u, (u + 1) % 10)).collect();
    let c10 = CsrGraph::<f64>::from_edges(10, &c10_edges, None, false, false).unwrap();
    let s_c10 = SgdSolver::<f64>::pacmap_graph(&c10, small_cfg).unwrap();
    let st_c10 = s_c10.pacmap.as_ref().unwrap();
    let mut c10_fp_per_u: Vec<Vec<u32>> = vec![Vec::new(); 10];
    for &(u, v) in &st_c10.fp_pairs {
        assert_ne!(u, v);
        assert!(!c10.neighbors(u as usize).contains(&v));
        let pair = if u < v { (u, v) } else { (v, u) };
        assert!(st_c10.nb_pairs.binary_search(&pair).is_err());
        c10_fp_per_u[u as usize].push(v);
    }
    for row in &mut c10_fp_per_u {
        assert_eq!(row.len(), 5);
        row.sort_unstable();
        row.dedup();
        assert_eq!(row.len(), 5);
    }
}
