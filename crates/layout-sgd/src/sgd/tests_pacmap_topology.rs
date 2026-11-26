use egraph_core::{CsrGraph, IterativeSolver};
use egraph_neighborhood::{DogVaradhanConfig, compute_regularized_dog_varadhan_landmarks};

use super::super::sgd::{PacmapGraphConfig, SgdSolver};
use super::test_support::{circle_layout, grid_graph};

#[test]
fn test_pacmap_graph_max_1hop_degree_weight_and_external_landmarks() {
    // 1. Canonical dedup & max_1hop_per_node on 20-clique (all degrees 19 > n_neighbors=4).
    let n_clique = 20usize;
    let mut clique_edges = Vec::new();
    for u in 0..n_clique as u32 {
        for v in (u + 1)..n_clique as u32 {
            clique_edges.push((u, v));
        }
    }
    let clique = CsrGraph::<f64>::from_edges(n_clique, &clique_edges, None, false, false).unwrap();
    let cfg_all = PacmapGraphConfig {
        n_neighbors: 4,
        mn_ratio: 0.0,
        fp_ratio: 0.0,
        phase1_iters: 2,
        phase2_iters: 2,
        phase3_iters: 2,
        keep_all_1hop: true,
        max_1hop_per_node: None,
        ..Default::default()
    };
    let s_all = SgdSolver::<f64>::pacmap_graph(&clique, cfg_all).unwrap();
    // Without double registration, K_20 has at most 20*19/2 = 190 undirected pairs!
    assert_eq!(s_all.num_springs(), 190);

    let cfg_cap = PacmapGraphConfig {
        max_1hop_per_node: Some(3),
        ..cfg_all
    };
    let s_cap = SgdSolver::<f64>::pacmap_graph(&clique, cfg_cap).unwrap();
    // Each node picks 3 edges (at most 60 total, far below 190).
    assert!(s_cap.num_springs() <= 60);

    // 2. degree_weight_beta lowers hub NB attraction relative to FP repulsion.
    let mut hub_edges: Vec<(u32, u32)> = (1..=30).map(|v| (0, v)).collect();
    hub_edges.extend([(30, 31), (31, 32), (32, 33), (33, 34)]);
    let g_hub = CsrGraph::<f64>::from_edges(35, &hub_edges, None, false, false).unwrap();
    let cfg_unweighted = PacmapGraphConfig {
        n_neighbors: 6,
        keep_all_1hop: true,
        degree_weight_beta: 0.0,
        phase1_iters: 2,
        phase2_iters: 2,
        phase3_iters: 5,
        seed: 42,
        ..Default::default()
    };
    let cfg_beta = PacmapGraphConfig {
        degree_weight_beta: 0.5,
        ..cfg_unweighted
    };
    let s_unw = SgdSolver::<f64>::pacmap_graph(&g_hub, cfg_unweighted).unwrap();
    let s_beta = SgdSolver::<f64>::pacmap_graph(&g_hub, cfg_beta).unwrap();
    let st_unw = s_unw.pacmap.as_ref().unwrap();
    let st_beta = s_beta.pacmap.as_ref().unwrap();
    assert!(st_unw.nb_weights.is_empty());
    assert_eq!(st_beta.nb_weights.len(), st_beta.nb_pairs.len());
    let sum_unw: f64 = st_unw
        .nb_pairs
        .iter()
        .filter(|&&(u, v)| u == 0 || v == 0)
        .count() as f64;
    let sum_beta: f64 = st_beta
        .nb_pairs
        .iter()
        .zip(st_beta.nb_weights.iter())
        .filter(|&(&(u, v), _)| u == 0 || v == 0)
        .map(|(_, &w)| w)
        .sum();
    let fp_hub_unw = st_unw
        .fp_pairs
        .iter()
        .filter(|&&(u, v)| u == 0 || v == 0)
        .count()
        .max(1) as f64;
    let fp_hub_beta = st_beta
        .fp_pairs
        .iter()
        .filter(|&&(u, v)| u == 0 || v == 0)
        .count()
        .max(1) as f64;
    let hub_ratio_unw = sum_unw / fp_hub_unw;
    let hub_ratio_beta = sum_beta / fp_hub_beta;
    assert!(
        hub_ratio_beta < hub_ratio_unw * 0.25,
        "expected degree_weight_beta to lower hub nb/fp ratio: unw={hub_ratio_unw}, beta={hub_ratio_beta}"
    );

    // 3. External DogVaradhanLandmarks & raw landmark_distances + step == run parity.
    let grid = grid_graph(5, 5);
    let n_grid = grid.num_nodes();
    let (dog_lm, _) =
        compute_regularized_dog_varadhan_landmarks(&grid, &DogVaradhanConfig::default()).unwrap();
    let cfg_ext = PacmapGraphConfig {
        n_neighbors: 6,
        mn_ratio: 0.5,
        fp_ratio: 2.0,
        phase1_iters: 4,
        phase2_iters: 4,
        phase3_iters: 6,
        degree_weight_beta: 0.25,
        seed: 99,
        ..Default::default()
    };
    let mut s_lm = SgdSolver::<f64>::pacmap_graph_with_landmarks(&grid, &dog_lm, cfg_ext).unwrap();
    let mut s_raw = SgdSolver::<f64>::pacmap_graph_with_landmark_distances(
        &grid,
        dog_lm.d_dog(),
        dog_lm.num_landmarks(),
        cfg_ext,
    )
    .unwrap();
    assert_eq!(
        s_lm.pacmap.as_ref().unwrap().mn_pairs,
        s_raw.pacmap.as_ref().unwrap().mn_pairs
    );
    let mut l_lm = circle_layout::<f64>(n_grid, 1.0);
    let mut l_raw = l_lm.clone();
    s_lm.run(&mut l_lm).unwrap();
    while !s_raw.status().is_finished() {
        s_raw.step(&mut l_raw).unwrap();
    }
    for i in 0..n_grid {
        assert_eq!(l_lm.x()[i].to_bits(), l_raw.x()[i].to_bits());
        assert_eq!(l_lm.y()[i].to_bits(), l_raw.y()[i].to_bits());
    }

    // Validation errors on external landmark distances.
    assert!(
        SgdSolver::<f64>::pacmap_graph_with_landmark_distances(&grid, &[0.0; 10], 2, cfg_ext)
            .is_err()
    );
    assert!(
        SgdSolver::<f64>::pacmap_graph_with_landmark_distances(
            &grid,
            &vec![-1.0; n_grid * 2],
            2,
            cfg_ext,
        )
        .is_err()
    );
    assert!(
        SgdSolver::<f64>::pacmap_graph_with_landmark_distances(
            &grid,
            &vec![f64::NAN; n_grid * 2],
            2,
            cfg_ext,
        )
        .is_err()
    );
    assert!(
        SgdSolver::<f64>::pacmap_graph_with_landmark_distances(&grid, &[], 0, cfg_ext).is_err()
    );
}

#[test]
fn test_pacmap_graph_2hop_prefers_low_degree_bridges_and_adamic_adar() {
    // 1. Adamic-Adar beats raw path count even when medium hubs have lower IDs:
    //    - node 0 has n_neighbors = 6 and 5 1-hop neighbors: {1, 2, 3, 50, 51} -> 1 slot left.
    //    - medium hubs {1, 2, 3} each have deg = 30 <= 64 and connect to candidate 99
    //      (raw = 3, AA = 3 / ln(30) ~ 0.882). Candidate 99 has deg = 6 == n_neighbors so it
    //      does not backfill 0 reciprocally.
    //    - low-degree bridges {50, 51} each have deg = 3 and connect to candidate 100
    //      (raw = 2, AA = 2 / ln(3) ~ 1.820).
    let mut edges: Vec<(u32, u32)> = vec![(0, 1), (0, 2), (0, 3), (0, 50), (0, 51)];
    let t_hub = 99u32;
    let t_bridge = 100u32;
    let mut next_leaf = 200u32;
    for h in [1u32, 2, 3] {
        edges.push((h, t_hub));
        for _ in 0..28 {
            edges.push((h, next_leaf));
            next_leaf += 1;
        }
    }
    edges.extend([(t_hub, 101), (t_hub, 102), (t_hub, 103)]);
    edges.extend([(50, t_bridge), (51, t_bridge), (50, 104), (51, 105)]);
    let n1 = (next_leaf as usize).max(106);
    let g1 = CsrGraph::<f64>::from_edges(n1, &edges, None, false, false).unwrap();
    let cfg1 = PacmapGraphConfig {
        n_neighbors: 6,
        mn_ratio: 0.0,
        fp_ratio: 0.0,
        phase1_iters: 1,
        phase2_iters: 1,
        phase3_iters: 1,
        seed: 42,
        ..Default::default()
    };
    let s1 = SgdSolver::<f64>::pacmap_graph(&g1, cfg1).unwrap();
    let st1 = s1.pacmap.as_ref().unwrap();
    assert!(
        st1.nb_pairs.contains(&(0, t_bridge)),
        "expected Adamic-Adar top candidate (0, {t_bridge}) to be selected"
    );
    assert!(
        !st1.nb_pairs.contains(&(0, t_hub)),
        "expected lower-AA raw-count candidate (0, {t_hub}) to be rejected"
    );

    // 2. Degree-sorted cand_cap visits high-ID low-degree bridge before exhausting cand_cap:
    //    - node 0 has n_neighbors = 10 (cand_cap = 80), 4 low-ID deg=25 neighbors (1..=4)
    //      generating 4 * 24 = 96 >= 80 candidates, and 1 high-ID deg=2 bridge 150 -> 151.
    //    - 151 has deg = 10 == n_neighbors so only 0's 2-hop scan can insert (0, 151).
    let mut edges2: Vec<(u32, u32)> = vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 150), (150, 151)];
    let mut next_v = 10u32;
    for h in 1..=4u32 {
        for _ in 0..24 {
            edges2.push((h, next_v));
            next_v += 1;
        }
    }
    for k in 152..=160u32 {
        edges2.push((151, k));
    }
    let n2 = 165usize;
    let g2 = CsrGraph::<f64>::from_edges(n2, &edges2, None, false, false).unwrap();
    let cfg2 = PacmapGraphConfig {
        n_neighbors: 10,
        mn_ratio: 0.0,
        fp_ratio: 0.0,
        phase1_iters: 1,
        phase2_iters: 1,
        phase3_iters: 1,
        seed: 42,
        ..Default::default()
    };
    let s2 = SgdSolver::<f64>::pacmap_graph(&g2, cfg2).unwrap();
    let st2 = s2.pacmap.as_ref().unwrap();
    assert!(
        st2.nb_pairs.contains(&(0, 151)),
        "expected high-ID deg=2 bridge 150->151 to be scanned before cand_cap exhaustion"
    );
}

#[test]
fn test_pacmap_graph_2hop_suppresses_super_hubs_and_star_safety() {
    // 1. N = 120 (hub_thresh = 64): super-hub 1 (deg = 100 > 64) + local bridge 102 (deg = 3).
    let n = 120usize;
    let mut edges: Vec<(u32, u32)> = vec![(0, 1), (0, 102), (102, 103), (102, 104)];
    for leaf in 2..=100u32 {
        edges.push((1, leaf));
    }
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
    let cfg = PacmapGraphConfig {
        n_neighbors: 4,
        mn_ratio: 0.0,
        fp_ratio: 0.0,
        phase1_iters: 2,
        phase2_iters: 2,
        phase3_iters: 2,
        seed: 42,
        ..Default::default()
    };
    let s = SgdSolver::<f64>::pacmap_graph(&g, cfg).unwrap();
    let st = s.pacmap.as_ref().unwrap();

    for expected in [(0, 1), (0, 102), (0, 103), (0, 104)] {
        assert!(
            st.nb_pairs.contains(&expected),
            "missing expected local/1-hop pair {expected:?}"
        );
    }
    for leaf in 2..=100u32 {
        assert!(
            !st.nb_pairs.contains(&(0, leaf)),
            "super-hub 1 leaked barrel 2-hop pair (0, {leaf})"
        );
    }
    let leaf2_pairs: Vec<(u32, u32)> = st
        .nb_pairs
        .iter()
        .copied()
        .filter(|&(u, v)| u == 2 || v == 2)
        .collect();
    assert_eq!(leaf2_pairs, vec![(1, 2)]);

    // 2. Pure star S_100 (deg(0) = 99 > 64): keeps exact 99 1-hop edges, zero panic, finite layout.
    let n_star = 100usize;
    let star_edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
    let mut star_solver =
        SgdSolver::<f64>::pacmap_graph(&star, PacmapGraphConfig::default()).unwrap();
    let star_st = star_solver.pacmap.as_ref().unwrap();
    assert_eq!(star_st.nb_pairs.len(), 99);
    let mut star_layout = circle_layout::<f64>(n_star, 1.0);
    star_solver.run(&mut star_layout).unwrap();
    assert!(star_layout.is_all_finite());
}
