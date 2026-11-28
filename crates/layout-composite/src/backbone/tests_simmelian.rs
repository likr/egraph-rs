use super::simmelian::{
    SimpleUndirectedGraph, WEIGHT_TIE_EPS, build_simple_undirected_graph,
    compute_simmelian_prefix_jaccard, count_quadrangles,
};
use super::*;
use egraph_core::DeterministicRng;

fn brute_force_quadrangles(n: usize, edges: &[(u32, u32)]) -> Vec<u64> {
    let mut adj = vec![vec![false; n]; n];
    let mut edge_id = vec![vec![usize::MAX; n]; n];
    for (idx, &(u, v)) in edges.iter().enumerate() {
        adj[u as usize][v as usize] = true;
        adj[v as usize][u as usize] = true;
        edge_id[u as usize][v as usize] = idx;
        edge_id[v as usize][u as usize] = idx;
    }
    let mut counts = vec![0u64; edges.len()];
    // Enumerate all 4-subsets {a, b, c, d} and each of the 3 possible 4-cycle orders.
    for a in 0..n {
        for b in (a + 1)..n {
            for c in (b + 1)..n {
                for d in (c + 1)..n {
                    let cycles = [[a, b, c, d], [a, b, d, c], [a, c, b, d]];
                    for cyc in cycles {
                        if adj[cyc[0]][cyc[1]]
                            && adj[cyc[1]][cyc[2]]
                            && adj[cyc[2]][cyc[3]]
                            && adj[cyc[3]][cyc[0]]
                        {
                            counts[edge_id[cyc[0]][cyc[1]]] += 1;
                            counts[edge_id[cyc[1]][cyc[2]]] += 1;
                            counts[edge_id[cyc[2]][cyc[3]]] += 1;
                            counts[edge_id[cyc[3]][cyc[0]]] += 1;
                        }
                    }
                }
            }
        }
    }
    counts
}

fn brute_force_simmelian_prefix_jaccard(g: &SimpleUndirectedGraph, q_edge: &[u64]) -> Vec<f64> {
    let n = g.num_nodes;
    let m = g.edges.len();
    let mut q_node = vec![0u128; n];
    for (idx, &(u, v)) in g.edges.iter().enumerate() {
        q_node[u as usize] += q_edge[idx] as u128;
        q_node[v as usize] += q_edge[idx] as u128;
    }
    let mut q_norm = vec![0.0_f64; m];
    for (idx, &(u, v)) in g.edges.iter().enumerate() {
        if q_edge[idx] > 0 {
            let denom = (q_node[u as usize] as f64).sqrt() * (q_node[v as usize] as f64).sqrt();
            if denom > 0.0 {
                q_norm[idx] = (q_edge[idx] as f64) / denom;
            }
        }
    }
    let mut lists: Vec<Vec<(u32, f64)>> = Vec::with_capacity(n);
    for u in 0..n {
        let s = g.adj_indptr[u] as usize;
        let e = g.adj_indptr[u + 1] as usize;
        let mut items: Vec<(u32, f64)> = (s..e)
            .map(|p| (g.adj_indices[p], q_norm[g.adj_edge_id[p] as usize]))
            .collect();
        items.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        lists.push(items);
    }
    let tie_prefix = |list: &[(u32, f64)], k: usize| -> Vec<u32> {
        let mut grp_start = 0usize;
        while grp_start < list.len() {
            let anchor = list[grp_start].1;
            let mut grp_end = grp_start + 1;
            while grp_end < list.len() && (anchor - list[grp_end].1).abs() <= WEIGHT_TIE_EPS {
                grp_end += 1;
            }
            if k <= grp_end {
                return list[..grp_end].iter().map(|x| x.0).collect();
            }
            grp_start = grp_end;
        }
        list.iter().map(|x| x.0).collect()
    };
    let mut out = vec![0.0_f64; m];
    for (idx, &(u, v)) in g.edges.iter().enumerate() {
        let lu = &lists[u as usize];
        let lv = &lists[v as usize];
        let limit = lu.len().min(lv.len());
        let mut best = 0.0_f64;
        for k in 1..=limit {
            let au = tie_prefix(lu, k);
            let bv = tie_prefix(lv, k);
            let inter = au.iter().filter(|x| bv.contains(x)).count();
            let union = au.len() + bv.len() - inter;
            if union > 0 {
                best = best.max((inter as f64) / (union as f64));
            }
        }
        out[idx] = best;
    }
    out
}

#[test]
fn test_quadrangle_counts_known_graphs_and_brute_force_oracle() {
    // 1. C_4: each edge belongs to 1 quadrangle.
    let c4 = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
        .unwrap();
    let s_c4 = build_simple_undirected_graph(&c4).unwrap();
    assert_eq!(count_quadrangles(&s_c4), vec![1, 1, 1, 1]);

    // 2. K_4: each of the 6 edges belongs to 2 quadrangles.
    let k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let k4 = CsrGraph::<f64>::from_edges(4, &k4_edges, None, false, false).unwrap();
    let s_k4 = build_simple_undirected_graph(&k4).unwrap();
    assert_eq!(count_quadrangles(&s_k4), vec![2; 6]);

    // 3. K_{2, 4}: each of the 8 edges belongs to 3 quadrangles.
    let mut k24_edges = Vec::new();
    for b in 2..6u32 {
        k24_edges.push((0, b));
        k24_edges.push((1, b));
    }
    let k24 = CsrGraph::<f64>::from_edges(6, &k24_edges, None, false, false).unwrap();
    let s_k24 = build_simple_undirected_graph(&k24).unwrap();
    assert_eq!(count_quadrangles(&s_k24), vec![3; 8]);

    // 4. Randomized dense & sparse graphs vs brute-force C_4 oracle.
    let mut rng = DeterministicRng::from_seed_u64(1337);
    for n in [6usize, 9, 12] {
        for p in [0.25, 0.55, 0.8] {
            let mut edges = Vec::new();
            for u in 0..n as u32 {
                for v in (u + 1)..n as u32 {
                    if rng.next_f64() < p {
                        edges.push((u, v));
                    }
                }
            }
            let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
            let sg = build_simple_undirected_graph(&g).unwrap();
            let fast = count_quadrangles(&sg);
            let oracle = brute_force_quadrangles(n, &sg.edges);
            assert_eq!(fast, oracle, "C_4 mismatch for n={n}, p={p}");
            let fast_w = compute_simmelian_prefix_jaccard(&sg, &fast);
            let oracle_w = brute_force_simmelian_prefix_jaccard(&sg, &oracle);
            for (fw, ow) in fast_w.iter().zip(oracle_w.iter()) {
                assert!(
                    (fw - ow).abs() < 1e-12,
                    "prefix-Jaccard mismatch: fast={fw}, oracle={ow}"
                );
            }
        }
    }
}

#[test]
fn test_simmelian_weights_and_tie_invariant_umst_symmetry() {
    // K_4: all 6 edges are structurally identical -> identical Simmelian weights (0.5) and all 6 in UMST.
    let k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let k4 = CsrGraph::<f64>::from_edges(4, &k4_edges, None, false, false).unwrap();
    let res = extract_quadrilateral_backbone(
        &k4,
        &BackboneConfig {
            retain_ratio: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(res.simmelian_weights.len(), 6);
    for &w in &res.simmelian_weights {
        assert!((w - 0.5).abs() < 1e-12, "expected 0.5, got {w}");
    }
    assert_eq!(res.umst_edges.len(), 6);
    assert_eq!(res.backbone_edges.len(), 6);
    assert_eq!(res.edge_weights, vec![1.0; 6]);
    assert!(!res.backbone_csr.is_weighted());
    assert_eq!(res.backbone_csr.num_edges(), 6);

    // Two K_4 cliques bridged by (3, 4): UMST must retain connectivity via (3, 4) and all symmetric clique edges.
    let mut barbell = Vec::new();
    for u in 0..4u32 {
        for v in (u + 1)..4u32 {
            barbell.push((u, v));
        }
    }
    for u in 4..8u32 {
        for v in (u + 1)..8u32 {
            barbell.push((u, v));
        }
    }
    barbell.push((3, 4));
    let g = CsrGraph::<f64>::from_edges(8, &barbell, None, false, false).unwrap();
    let b_res = extract_quadrilateral_backbone(
        &g,
        &BackboneConfig {
            retain_ratio: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(b_res.umst_edges.contains(&(3, 4)));
    // Backbone CSR is connected (1 component).
    assert_eq!(
        b_res.backbone_csr.largest_connected_component_nodes().len(),
        8
    );
}

#[test]
fn test_retain_ratio_and_custom_edge_weights() {
    // Planted 2-clique + 2 cross edges (bridge-like noisy hairball).
    let mut edges = Vec::new();
    for u in 0..6u32 {
        for v in (u + 1)..6u32 {
            edges.push((u, v));
        }
    }
    for u in 6..12u32 {
        for v in (u + 1)..12u32 {
            edges.push((u, v));
        }
    }
    edges.push((0, 6));
    edges.push((1, 7));
    let g = CsrGraph::<f64>::from_edges(12, &edges, None, false, false).unwrap();
    let m = g.num_edges();

    // retain_ratio = 0.2, umst_weight_scale = 2.0, non_backbone_weight_scale = 0.1
    let cfg = BackboneConfig {
        retain_ratio: 0.2,
        umst_weight_scale: 2.0,
        non_backbone_weight_scale: 0.1,
        ..Default::default()
    };
    let res = extract_quadrilateral_backbone(&g, &cfg).unwrap();
    assert_eq!(res.simmelian_weights.len(), m);
    assert_eq!(res.edge_weights.len(), m);
    assert!(res.backbone_edges.len() >= res.umst_edges.len());
    assert!(res.backbone_csr.is_weighted());
    assert_eq!(res.backbone_csr.num_edges(), m);
    assert_eq!(
        res.backbone_csr.largest_connected_component_nodes().len(),
        12
    );

    // Intra-cluster edges must strictly dominate inter-cluster edges in Simmelian weight.
    let simple = build_simple_undirected_graph(&g).unwrap();
    let mut min_intra = f64::INFINITY;
    let mut max_inter = 0.0_f64;
    for (idx, &(u, v)) in simple.edges.iter().enumerate() {
        let same = (u < 6) == (v < 6);
        let w = res.simmelian_weights[idx];
        if same {
            min_intra = min_intra.min(w);
        } else {
            max_inter = max_inter.max(w);
        }
    }
    assert!(
        min_intra > max_inter,
        "expected intra ({min_intra}) > inter ({max_inter})"
    );

    // retain_ratio = 1.0 keeps all edges in backbone_edges.
    let full_res = extract_quadrilateral_backbone(
        &g,
        &BackboneConfig {
            retain_ratio: 1.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(full_res.backbone_edges.len(), m);
}
