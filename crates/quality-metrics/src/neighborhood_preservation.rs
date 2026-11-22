use egraph_spatial::StaticKdTree;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNeighbors, NodeIndexable};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};
use std::collections::HashSet;

fn build_kdtree<N, S>(drawing: &DrawingEuclidean2d<N, S>) -> Option<StaticKdTree<2, f64>>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    let n = drawing.len();
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let entry = drawing.raw_entry(i);
        let (x, y) = (entry.0, entry.1);
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        points.push([x.into(), y.into()]);
    }
    StaticKdTree::<2, f64>::build(&points).ok()
}

/// Calculates the neighborhood preservation metric for a graph layout.
///
/// This metric assesses how well the layout preserves local neighborhoods from
/// the original graph structure. It calculates the ratio of nodes that are both
/// graph-theoretical neighbors and spatial neighbors in the layout.
///
/// The implementation works by:
/// 1. Identifying all edges in the graph
/// 2. For each node, finding its k nearest neighbors in the layout (where k is its degree in the graph)
/// 3. Calculating the ratio of neighbors that are preserved (i.e., are both connected in the
///    graph and close in the layout)
///
/// A higher value indicates better preservation of the graph's neighborhood structure.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The 2D Euclidean layout of the graph
///
/// # Returns
///
/// An `S` value in the range [0, 1] representing the neighborhood preservation metric.
/// A value of 1 indicates perfect preservation of neighborhoods, while 0 indicates
/// no preservation.
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
pub fn neighborhood_preservation<G, S>(graph: G, drawing: &DrawingEuclidean2d<G::NodeId, S>) -> S
where
    G: IntoEdgeReferences + IntoNeighbors + NodeIndexable,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    let mut graph_edges = HashSet::new();
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        graph_edges.insert((graph.to_index(u), graph.to_index(v)));
        graph_edges.insert((graph.to_index(v), graph.to_index(u)));
    }

    let n = drawing.len();
    if n <= 1 || graph_edges.is_empty() {
        return S::zero();
    }
    let Some(nn) = build_kdtree(drawing) else {
        return S::zero();
    };

    let mut cap = 0;
    let mut cup = graph_edges.len();
    let mut neighbors = Vec::new();
    for i in 0..n {
        let u = *drawing.node_id(i);
        let d = graph.neighbors(u).count();
        if d == 0 {
            continue;
        }
        if nn
            .k_nearest_excluding_self_into(i, d, &mut neighbors)
            .is_err()
        {
            return S::zero();
        }
        for &(j, _) in &neighbors {
            let v = *drawing.node_id(j);
            if graph_edges.contains(&(graph.to_index(u), graph.to_index(v))) {
                cap += 1;
            } else {
                cup += 1;
            }
        }
    }

    if cup == 0 {
        return S::zero();
    }
    S::from_usize(cap).unwrap() / S::from_usize(cup).unwrap()
}

/// Calculates the 2-hop neighborhood preservation metric for a graph layout.
///
/// This metric evaluates how well the layout preserves 2-hop local neighborhoods from
/// the graph topological structure. For each node v, it considers all nodes within
/// shortest path distance <= 2 in the graph as N_G(v, 2), and finds the |N_G(v, 2)|
/// nearest spatial neighbors in the layout as N_D(v, 2). It then averages the node-wise
/// Jaccard similarity coefficients |N_G(v, 2) \cap N_D(v, 2)| / |N_G(v, 2) \cup N_D(v, 2)|.
///
/// # Parameters
///
/// * `graph`: The graph structure to evaluate
/// * `drawing`: The 2D Euclidean layout of the graph
///
/// # Returns
///
/// An `S` value in the range [0, 1] representing the 2-hop neighborhood preservation metric.
/// A value of 1 indicates perfect preservation of 2-hop neighborhoods, while 0 indicates
/// no preservation.
///
/// # Type Parameters
///
/// * `G`: A graph type that implements the required traits
pub fn neighborhood_preservation_2hop<G, S>(
    graph: G,
    drawing: &DrawingEuclidean2d<G::NodeId, S>,
) -> S
where
    G: IntoEdgeReferences + IntoNeighbors + NodeIndexable,
    G::NodeId: DrawingIndex,
    S: DrawingValue,
{
    let n = drawing.len();
    if n == 0 {
        return S::zero();
    }

    let mut adj = vec![HashSet::new(); n];
    for e in graph.edge_references() {
        let u = graph.to_index(e.source());
        let v = graph.to_index(e.target());
        if u != v {
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }

    let Some(nn) = build_kdtree(drawing) else {
        return S::zero();
    };

    let mut sum_jaccard = S::zero();
    let mut spatial_neighbors = Vec::new();

    for i in 0..n {
        let mut ng_2 = HashSet::new();
        for &v in &adj[i] {
            ng_2.insert(v);
            for &w in &adj[v] {
                if w != i {
                    ng_2.insert(w);
                }
            }
        }

        let k = ng_2.len();
        if k == 0 {
            sum_jaccard += S::one();
            continue;
        }

        if nn
            .k_nearest_excluding_self_into(i, k, &mut spatial_neighbors)
            .is_err()
        {
            return S::zero();
        }

        let mut intersection_count = 0;
        for &(j, _) in &spatial_neighbors {
            if ng_2.contains(&j) {
                intersection_count += 1;
            }
        }

        let union_count = 2 * k - intersection_count;
        let jaccard =
            S::from_usize(intersection_count).unwrap() / S::from_usize(union_count).unwrap();
        sum_jaccard += jaccard;
    }

    sum_jaccard / S::from_usize(n).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;
    use petgraph_drawing::MetricEuclidean2d;

    #[test]
    fn test_neighborhood_preservation_2hop_empty() {
        let graph = Graph::<(), ()>::new();
        let drawing = DrawingEuclidean2d::initial_placement(&graph);
        let score: f32 = neighborhood_preservation_2hop(&graph, &drawing);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_neighborhood_preservation_2hop_perfect() {
        let mut graph = Graph::<(), ()>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        let mut drawing = DrawingEuclidean2d::initial_placement(&graph);
        *drawing.position_mut(a).unwrap() = MetricEuclidean2d(0.0, 0.0);
        *drawing.position_mut(b).unwrap() = MetricEuclidean2d(1.0, 0.0);
        *drawing.position_mut(c).unwrap() = MetricEuclidean2d(2.0, 0.0);

        let score: f32 = neighborhood_preservation_2hop(&graph, &drawing);
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_neighborhood_preservation_edgeless_and_empty_finite() {
        let empty = petgraph::graph::UnGraph::<(), ()>::new_undirected();
        let d_empty = DrawingEuclidean2d::<_, f64>::initial_placement(&empty);
        assert_eq!(neighborhood_preservation(&empty, &d_empty), 0.0);

        let mut g_edgeless = petgraph::graph::UnGraph::<(), ()>::new_undirected();
        for _ in 0..4 {
            g_edgeless.add_node(());
        }
        let d_edgeless = DrawingEuclidean2d::<_, f64>::initial_placement(&g_edgeless);
        let score = neighborhood_preservation(&g_edgeless, &d_edgeless);
        assert!(score.is_finite());
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_neighborhood_preservation_exact_duplicates_isolated_and_nonfinite() {
        // Star graph of 5 nodes: optimal placement yields 1.0, suboptimal yields < 1.0.
        let mut star = petgraph::graph::UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..5).map(|_| star.add_node(())).collect();
        for &leaf in &nodes[1..] {
            star.add_edge(nodes[0], leaf, ());
        }
        let mut d_opt = DrawingEuclidean2d::<_, f32>::initial_placement(&star);
        *d_opt.position_mut(nodes[0]).unwrap() = MetricEuclidean2d(0.0, 0.0);
        *d_opt.position_mut(nodes[1]).unwrap() = MetricEuclidean2d(1.0, 0.0);
        *d_opt.position_mut(nodes[2]).unwrap() = MetricEuclidean2d(0.0, 1.0);
        *d_opt.position_mut(nodes[3]).unwrap() = MetricEuclidean2d(-1.0, 0.0);
        *d_opt.position_mut(nodes[4]).unwrap() = MetricEuclidean2d(0.0, -1.0);
        assert_eq!(neighborhood_preservation(&star, &d_opt), 1.0);
        assert_eq!(neighborhood_preservation_2hop(&star, &d_opt), 1.0);

        // Collapsed duplicates with an isolated node (degree 0): self must always be excluded
        // even when > d + 1 points share the same coordinates.
        let mut g = petgraph::graph::UnGraph::<(), ()>::new_undirected();
        let g_nodes: Vec<_> = (0..5).map(|_| g.add_node(())).collect();
        g.add_edge(g_nodes[0], g_nodes[1], ());
        g.add_edge(g_nodes[2], g_nodes[3], ());
        // g_nodes[4] is isolated (degree 0).
        let mut d_dup = DrawingEuclidean2d::<_, f64>::initial_placement(&g);
        for &u in &g_nodes {
            *d_dup.position_mut(u).unwrap() = MetricEuclidean2d(0.0, 0.0);
        }
        // 1-hop: i=0 -> neighbor 1 (match); i=1 -> neighbor 0 (match);
        // i=2 -> neighbor 0 (miss); i=3 -> neighbor 0 (miss); i=4 -> d=0 (skipped).
        // graph_edges.len() = 4, cap = 2, cup = 4 + 2 = 6 -> 2/6.
        let np1 = neighborhood_preservation(&g, &d_dup);
        assert!((np1 - (2.0 / 6.0)).abs() < 1e-12, "np1={}", np1);

        // 2-hop with collapsed coordinates: deterministic index tie-breaking, Jaccard in [0, 1].
        // i=0 -> {1} (1.0), i=1 -> {0} (1.0), i=2 -> {0} vs {3} (0.0), i=3 -> {0} vs {2} (0.0), i=4 -> isolated (1.0).
        let np2 = neighborhood_preservation_2hop(&g, &d_dup);
        assert!((np2 - (3.0 / 5.0)).abs() < 1e-12, "np2={}", np2);

        // Non-finite coordinates return 0.0 cleanly for both metrics.
        let mut d_nan = d_dup.clone();
        *d_nan.position_mut(g_nodes[0]).unwrap() = MetricEuclidean2d(f64::NAN, 0.0);
        assert_eq!(neighborhood_preservation(&g, &d_nan), 0.0);
        assert_eq!(neighborhood_preservation_2hop(&g, &d_nan), 0.0);
    }
}
