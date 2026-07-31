use linfa::Float;
use linfa_nn::{distance::L2Dist, BallTree, NearestNeighbour};
use ndarray::prelude::*;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNeighbors, NodeIndexable};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};
use std::collections::HashSet;

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
    S: DrawingValue + Float,
{
    let mut graph_edges = HashSet::new();
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        graph_edges.insert((graph.to_index(u), graph.to_index(v)));
        graph_edges.insert((graph.to_index(v), graph.to_index(u)));
    }

    let n = drawing.len();
    let mut points = Array2::zeros((n, 2));
    for i in 0..n {
        points[[i, 0]] = drawing.raw_entry(i).0;
        points[[i, 1]] = drawing.raw_entry(i).1;
    }
    let nn = BallTree::new().from_batch(&points, L2Dist).unwrap();

    let mut cap = 0;
    let mut cup = graph_edges.len();
    for i in 0..n {
        let u = *drawing.node_id(i);
        let x = drawing.raw_entry(i).0;
        let y = drawing.raw_entry(i).1;
        let d = graph.neighbors(u).count();
        let query = arr1(&[x, y]);
        let neighbors = nn.k_nearest(query.view(), d + 1).unwrap();
        for &(_, j) in neighbors.iter() {
            if i == j {
                continue;
            }
            let v = *drawing.node_id(j);
            if graph_edges.contains(&(graph.to_index(u), graph.to_index(v))) {
                cap += 1;
            } else {
                cup += 1;
            }
        }
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
    S: DrawingValue + Float,
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

    let mut points = Array2::zeros((n, 2));
    for i in 0..n {
        points[[i, 0]] = drawing.raw_entry(i).0;
        points[[i, 1]] = drawing.raw_entry(i).1;
    }
    let nn = BallTree::new().from_batch(&points, L2Dist).unwrap();

    let mut sum_jaccard = S::zero();

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

        let x = drawing.raw_entry(i).0;
        let y = drawing.raw_entry(i).1;
        let query = arr1(&[x, y]);
        let spatial_neighbors = nn.k_nearest(query.view(), k + 1).unwrap();

        let mut intersection_count = 0;
        for &(_, j) in spatial_neighbors.iter() {
            if i == j {
                continue;
            }
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
}
