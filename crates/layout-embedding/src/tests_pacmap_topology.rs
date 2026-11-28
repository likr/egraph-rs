use egraph_core::{CsrGraph, IterativeSolver, Layout2d};
use egraph_neighborhood::{DogVaradhanConfig, compute_regularized_dog_varadhan_landmarks};

use crate::test_support::{circle_layout, grid_graph};
use crate::{PacmapGraphConfig, PacmapGraphSolver};

#[test]
fn test_pacmap_graph_max_1hop_degree_weight_and_external_landmarks() {
    let n_clique = 20usize;
    let mut clique_edges = Vec::new();
    for u in 0..n_clique as u32 {
        for v in (u + 1)..n_clique as u32 {
            clique_edges.push((u, v));
        }
    }
    let clique = CsrGraph::<f64>::from_edges(n_clique, &clique_edges, None, false, false).unwrap();
    let cfg_all = PacmapGraphConfig::new()
        .with_n_neighbors(4)
        .with_mn_ratio(0.0)
        .with_fp_ratio(0.0)
        .with_phase_iterations([2, 2, 2])
        .with_keep_all_1hop(true)
        .with_max_1hop_per_node(None);
    let s_all = PacmapGraphSolver::<Layout2d<f64>>::new(&clique, cfg_all.clone()).unwrap();
    assert_eq!(s_all.num_springs(), 190);

    let cfg_cap = cfg_all.with_max_1hop_per_node(Some(3));
    let s_cap = PacmapGraphSolver::<Layout2d<f64>>::new(&clique, cfg_cap).unwrap();
    assert!(s_cap.num_springs() <= 60);

    let mut hub_edges: Vec<(u32, u32)> = (1..=30).map(|v| (0, v)).collect();
    hub_edges.extend([(30, 31), (31, 32), (32, 33), (33, 34)]);
    let g_hub = CsrGraph::<f64>::from_edges(35, &hub_edges, None, false, false).unwrap();
    let cfg_unweighted = PacmapGraphConfig::new()
        .with_n_neighbors(6)
        .with_keep_all_1hop(true)
        .with_degree_weight_beta(0.0)
        .with_phase_iterations([2, 2, 5])
        .with_seed(42);
    let cfg_beta = cfg_unweighted.clone().with_degree_weight_beta(0.5);
    let s_unw = PacmapGraphSolver::<Layout2d<f64>>::new(&g_hub, cfg_unweighted).unwrap();
    let s_beta = PacmapGraphSolver::<Layout2d<f64>>::new(&g_hub, cfg_beta).unwrap();
    assert!(s_unw.nb_weights().is_empty());
    assert_eq!(s_beta.nb_weights().len(), s_beta.nb_pairs().len());
    let sum_unw: f64 = s_unw
        .nb_pairs()
        .iter()
        .filter(|&&(u, v)| u == 0 || v == 0)
        .count() as f64;
    let sum_beta: f64 = s_beta
        .nb_pairs()
        .iter()
        .zip(s_beta.nb_weights().iter())
        .filter(|&(&(u, v), _)| u == 0 || v == 0)
        .map(|(_, &w)| w)
        .sum();
    let fp_hub_unw = s_unw
        .fp_pairs()
        .iter()
        .filter(|&&(u, v)| u == 0 || v == 0)
        .count()
        .max(1) as f64;
    let fp_hub_beta = s_beta
        .fp_pairs()
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

    let grid = grid_graph(5, 5);
    let n_grid = grid.num_nodes();
    let (dog_lm, _) =
        compute_regularized_dog_varadhan_landmarks(&grid, &DogVaradhanConfig::default()).unwrap();
    let cfg_ext = PacmapGraphConfig::new()
        .with_n_neighbors(6)
        .with_mn_ratio(0.5)
        .with_fp_ratio(2.0)
        .with_phase_iterations([4, 4, 6])
        .with_degree_weight_beta(0.25)
        .with_seed(99);
    let mut s_lm =
        PacmapGraphSolver::<Layout2d<f64>>::with_landmarks(&grid, &dog_lm, cfg_ext.clone())
            .unwrap();
    let mut s_raw = PacmapGraphSolver::<Layout2d<f64>>::with_landmark_distances(
        &grid,
        dog_lm.d_dog(),
        dog_lm.num_landmarks(),
        cfg_ext.clone(),
    )
    .unwrap();
    assert_eq!(s_lm.mn_pairs(), s_raw.mn_pairs());
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

    assert!(
        PacmapGraphSolver::<Layout2d<f64>>::with_landmark_distances(
            &grid,
            &[0.0; 10],
            2,
            cfg_ext.clone(),
        )
        .is_err()
    );
    assert!(
        PacmapGraphSolver::<Layout2d<f64>>::with_landmark_distances(
            &grid,
            &vec![-1.0; n_grid * 2],
            2,
            cfg_ext.clone(),
        )
        .is_err()
    );
    assert!(
        PacmapGraphSolver::<Layout2d<f64>>::with_landmark_distances(
            &grid,
            &vec![f64::NAN; n_grid * 2],
            2,
            cfg_ext.clone(),
        )
        .is_err()
    );
    assert!(
        PacmapGraphSolver::<Layout2d<f64>>::with_landmark_distances(&grid, &[], 0, cfg_ext)
            .is_err()
    );
}

#[test]
fn test_pacmap_graph_2hop_prefers_low_degree_bridges_and_adamic_adar() {
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
    let cfg1 = PacmapGraphConfig::new()
        .with_n_neighbors(6)
        .with_mn_ratio(0.0)
        .with_fp_ratio(0.0)
        .with_phase_iterations([1, 1, 1])
        .with_seed(42);
    let s1 = PacmapGraphSolver::<Layout2d<f64>>::new(&g1, cfg1).unwrap();
    assert!(
        s1.nb_pairs().contains(&(0, t_bridge)),
        "expected Adamic-Adar top candidate (0, {t_bridge}) to be selected"
    );
    assert!(
        !s1.nb_pairs().contains(&(0, t_hub)),
        "expected lower-AA raw-count candidate (0, {t_hub}) to be rejected"
    );

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
    let cfg2 = PacmapGraphConfig::new()
        .with_n_neighbors(10)
        .with_mn_ratio(0.0)
        .with_fp_ratio(0.0)
        .with_phase_iterations([1, 1, 1])
        .with_seed(42);
    let s2 = PacmapGraphSolver::<Layout2d<f64>>::new(&g2, cfg2).unwrap();
    assert!(
        s2.nb_pairs().contains(&(0, 151)),
        "expected high-ID deg=2 bridge 150->151 to be scanned before cand_cap exhaustion"
    );
}

#[test]
fn test_pacmap_graph_2hop_suppresses_super_hubs_and_star_safety() {
    let n = 120usize;
    let mut edges: Vec<(u32, u32)> = vec![(0, 1), (0, 102), (102, 103), (102, 104)];
    for leaf in 2..=100u32 {
        edges.push((1, leaf));
    }
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
    let cfg = PacmapGraphConfig::new()
        .with_n_neighbors(4)
        .with_mn_ratio(0.0)
        .with_fp_ratio(0.0)
        .with_phase_iterations([2, 2, 2])
        .with_seed(42);
    let s = PacmapGraphSolver::<Layout2d<f64>>::new(&g, cfg).unwrap();

    for expected in [(0, 1), (0, 102), (0, 103), (0, 104)] {
        assert!(
            s.nb_pairs().contains(&expected),
            "missing expected local/1-hop pair {expected:?}"
        );
    }
    for leaf in 2..=100u32 {
        assert!(
            !s.nb_pairs().contains(&(0, leaf)),
            "super-hub 1 leaked barrel 2-hop pair (0, {leaf})"
        );
    }
    let leaf2_pairs: Vec<(u32, u32)> = s
        .nb_pairs()
        .iter()
        .copied()
        .filter(|&(u, v)| u == 2 || v == 2)
        .collect();
    assert_eq!(leaf2_pairs, vec![(1, 2)]);

    let n_star = 100usize;
    let star_edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
    let mut star_solver =
        PacmapGraphSolver::<Layout2d<f64>>::new(&star, PacmapGraphConfig::default()).unwrap();
    assert_eq!(star_solver.nb_pairs().len(), 99);
    let mut star_layout = circle_layout::<f64>(n_star, 1.0);
    star_solver.run(&mut star_layout).unwrap();
    assert!(star_layout.is_all_finite());
}
