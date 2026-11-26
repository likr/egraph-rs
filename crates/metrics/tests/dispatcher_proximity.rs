use egraph_core::{CsrGraph, Layout2d, Scalar};
use egraph_metrics::{
    MetricConfig, MetricTarget, evaluate_metrics, gabriel_graph_preservation,
    gabriel_graph_property, node_resolution, relative_neighborhood_preservation,
};

fn assert_dispatcher_proximity_parity<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) {
    let gg_prop = gabriel_graph_property(graph, layout).unwrap();
    let nr = node_resolution(layout).unwrap();
    let s_rng = relative_neighborhood_preservation(graph, layout).unwrap();
    let s_gg = gabriel_graph_preservation(graph, layout).unwrap();

    let rep = evaluate_metrics(
        graph,
        layout,
        &[
            MetricTarget::GabrielGraphProperty,
            MetricTarget::NodeResolution,
            MetricTarget::RelativeNeighborhoodPreservation,
            MetricTarget::GabrielGraphPreservation,
        ],
        &MetricConfig::default(),
    )
    .unwrap();
    assert_eq!(
        rep.gabriel_graph_property.unwrap().to_bits(),
        gg_prop.to_bits()
    );
    assert_eq!(rep.node_resolution.unwrap().to_bits(), nr.to_bits());
    assert_eq!(
        rep.relative_neighborhood_preservation.unwrap().to_bits(),
        s_rng.to_bits()
    );
    assert_eq!(
        rep.gabriel_graph_preservation.unwrap().to_bits(),
        s_gg.to_bits()
    );
}

#[test]
fn test_dispatcher_proximity_boundary_and_directed_parity() {
    // 1. N = 0 empty graph & empty layout
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let l0 = Layout2d::<f64>::zeros(0);
    assert_dispatcher_proximity_parity(&g0, &l0);

    // 2. N = 1 f64 singleton
    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let l1 = Layout2d::from_xy(vec![3.5], vec![-2.25]).unwrap();
    assert_dispatcher_proximity_parity(&g1, &l1);

    // 3. N = 1 f32 singleton
    let g1_32 = CsrGraph::<f32>::from_edges(1, &[], None, false, false).unwrap();
    let l1_32 = Layout2d::from_xy(vec![1.25_f32], vec![-0.5_f32]).unwrap();
    assert_dispatcher_proximity_parity(&g1_32, &l1_32);

    // 4. Edgeless N = 4 square
    let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l_square = Layout2d::from_xy(vec![0.0, 2.0, 2.0, 0.0], vec![0.0, 0.0, 2.0, 2.0]).unwrap();
    assert_dispatcher_proximity_parity(&g_edgeless, &l_square);

    // 5. Two disjoint triangles
    let g_disjoint = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let l_disjoint = Layout2d::from_xy(
        vec![0.0, 4.0, 2.0, 2.0, 7.0, 7.0],
        vec![0.0, 0.0, 3.0, 0.5, 0.0, 2.0],
    )
    .unwrap();
    assert_dispatcher_proximity_parity(&g_disjoint, &l_disjoint);

    // 6. Connected triangle + co-located isolated vertex
    let g_iso =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
    let l_iso_coloc = Layout2d::from_xy(
        vec![0.0, 2.0, 1.0, 0.0],
        vec![0.0, 0.0, 3.0_f64.sqrt(), 0.0],
    )
    .unwrap();
    assert_dispatcher_proximity_parity(&g_iso, &l_iso_coloc);

    // 7. Directed reciprocal graph
    let base_edges = [(0_u32, 1_u32), (1, 2), (2, 3), (3, 0), (0, 2)];
    let mut recip_edges = Vec::new();
    for &(u, v) in &base_edges {
        recip_edges.push((u, v));
        recip_edges.push((v, u));
    }
    let g_dir_recip = CsrGraph::<f64>::from_edges(4, &recip_edges, None, true, false).unwrap();
    let layout = Layout2d::from_xy(vec![0.0, 4.0, 2.0, 1.9], vec![0.0, 0.0, 1.0, 0.2]).unwrap();
    assert_dispatcher_proximity_parity(&g_dir_recip, &layout);
}
