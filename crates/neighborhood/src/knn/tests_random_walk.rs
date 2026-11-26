use egraph_core::CsrGraph;

use super::*;

#[test]
fn test_two_step_random_walk_exact_when_pot_leq_cap_and_leaf_filter() {
    // Star+branch hub: vertex 0 connects to 20 dead-end leaves (1..=20) and 1 branch 21-22-23.
    // With leaf filter deg(w) > 1, non_leaf(0) = {21}, pot(0) = deg(21) - 1 = 1 <= cap = 4.
    // Even with 1000 leaves, TwoStepRandomWalk MUST deterministically extract (0, 22) without
    // wasting draws on dead-end leaves.
    let mut edges: Vec<(u32, u32)> = (1..=20u32).map(|v| (0, v)).collect();
    edges.extend([(0, 21), (21, 22), (22, 23)]);
    let g = CsrGraph::<f64>::from_edges(24, &edges, None, false, false).unwrap();

    let p_rw = extract_sparse_1hop_2hop_pairs_with_mode(
        &g,
        4,
        TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        42,
    )
    .unwrap();
    let pairs_2: Vec<(u32, u32)> = p_rw
        .u2
        .iter()
        .copied()
        .zip(p_rw.v2.iter().copied())
        .collect();
    assert!(
        pairs_2.contains(&(0, 22)),
        "leaf-filtered pot<=cap must extract (0, 22), got {pairs_2:?}"
    );
    assert!(
        pairs_2.contains(&(21, 23)),
        "exact pot<=cap must extract (21, 23), got {pairs_2:?}"
    );
    // Hub 0 has pot(0)=1, leaf i has pot(i)=20 > cap=4 so leaves sample up to 4 other leaves/21.
    for &(a, b) in &pairs_2 {
        assert!(a < b);
        assert!(!g.has_edge(a as usize, b as usize));
    }

    // With cap = 25 (> max pot = 20), TwoStepRandomWalk enumerates ALL 2-hop pairs deterministically
    // and matches exact extract_1hop_2hop_csr regardless of seed.
    let p_exact_1 = extract_sparse_1hop_2hop_pairs_with_mode(
        &g,
        25,
        TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        1,
    )
    .unwrap();
    let p_exact_2 = extract_sparse_1hop_2hop_pairs_with_mode(
        &g,
        25,
        TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        999,
    )
    .unwrap();
    assert_eq!(p_exact_1, p_exact_2);
    let (_, adj2) = extract_1hop_2hop_csr(&g).unwrap();
    assert_eq!(p_exact_1.u2.len(), adj2.num_edges());
}

#[test]
fn test_two_step_random_walk_pot_gt_cap_and_max_bridge_degree() {
    // Hub 0 (deg = 30) connects to 1..=30; bridge 30 connects to 31..=34 (deg(30) = 5).
    // Local ring 31-32-33-34-31 ensures deg >= 2.
    let n = 35usize;
    let mut edges: Vec<(u32, u32)> = (1..=30u32).map(|v| (0, v)).collect();
    edges.extend([(30, 31), (30, 32), (30, 33), (30, 34)]);
    edges.extend([(31, 32), (32, 33), (33, 34), (34, 31)]);
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

    // Without max_bridge_degree: vertex 31 can traverse bridge 30 (deg=5) or ring neighbors.
    let p_rw1 = extract_sparse_1hop_2hop_pairs_with_mode(
        &g,
        4,
        TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        42,
    )
    .unwrap();
    let p_rw2 = extract_sparse_1hop_2hop_pairs_with_mode(
        &g,
        4,
        TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        42,
    )
    .unwrap();
    assert_eq!(p_rw1, p_rw2);
    assert!(!p_rw1.u2.is_empty());

    // With max_bridge_degree = Some(4): hub 0 (deg=30) and bridge 30 (deg=5) are excluded as
    // intermediates! So leaves 1..=29 get zero 2-hop pairs through hub 0, and 31..=34 cannot
    // reach 0 via bridge 30 (they only reach each other via degree-3 ring vertices).
    let p_capped = extract_sparse_1hop_2hop_pairs_with_mode(
        &g,
        8,
        TwoHopSamplingMode::TwoStepRandomWalk,
        Some(4),
        42,
    )
    .unwrap();
    for (&a, &b) in p_capped.u2.iter().zip(p_capped.v2.iter()) {
        assert!(
            !(1..=29).contains(&a) && !(1..=29).contains(&b),
            "leaf leaked through bridge with deg=30 > max_bridge_degree=4: ({a}, {b})"
        );
        assert!(
            a != 0 && b != 0,
            "node 0 leaked through bridge 30 (deg=5 > 4): ({a}, {b})"
        );
    }

    // DegreeOrderedHubSuppressed with max_bridge_degree = Some(4) also blocks bridge 30 (deg=5).
    let p_deg_capped = extract_sparse_1hop_2hop_pairs_with_mode(
        &g,
        8,
        TwoHopSamplingMode::DegreeOrderedHubSuppressed,
        Some(4),
        42,
    )
    .unwrap();
    for (&a, &b) in p_deg_capped.u2.iter().zip(p_deg_capped.v2.iter()) {
        assert!(a != 0 && b != 0);
        assert!(!(1..=29).contains(&a) && !(1..=29).contains(&b));
    }

    // DegreeOrderedHubSuppressed with None matches extract_sparse_1hop_2hop_pairs identically.
    let p_default = extract_sparse_1hop_2hop_pairs(&g, 8, 42).unwrap();
    let p_mode_default = extract_sparse_1hop_2hop_pairs_with_mode(
        &g,
        8,
        TwoHopSamplingMode::DegreeOrderedHubSuppressed,
        None,
        42,
    )
    .unwrap();
    assert_eq!(p_default, p_mode_default);
}

#[test]
fn test_two_step_random_walk_boundary_graphs() {
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let pe = extract_sparse_1hop_2hop_pairs_with_mode(
        &empty,
        8,
        TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        42,
    )
    .unwrap();
    assert!(pe.u1.is_empty() && pe.u2.is_empty());

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let ps = extract_sparse_1hop_2hop_pairs_with_mode(
        &single,
        8,
        TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        42,
    )
    .unwrap();
    assert!(ps.u1.is_empty() && ps.u2.is_empty());

    let path =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let p0 = extract_sparse_1hop_2hop_pairs_with_mode(
        &path,
        0,
        TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        42,
    )
    .unwrap();
    assert_eq!(p0.u1.len(), 3);
    assert!(p0.u2.is_empty());
}
