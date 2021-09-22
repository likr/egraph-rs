use petgraph::graph::{EdgeIndex, Graph, IndexType, NodeIndex};
use petgraph::visit::{EdgeCount, IntoNeighbors, IntoNodeIdentifiers};
use petgraph::EdgeType;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

pub fn louvain_step<G>(graph: &G) -> Option<HashMap<G::NodeId, G::NodeId>>
where
    G: EdgeCount + IntoNeighbors + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash,
{
    let m = graph.edge_count() as f32;
    let k = graph
        .node_identifiers()
        .map(|u| (u, graph.neighbors(u).count() as f32))
        .collect::<HashMap<_, _>>();
    let mut sigma_total = k.clone();
    let mut communities = graph
        .node_identifiers()
        .map(|u| (u, u))
        .collect::<HashMap<_, _>>();
    let mut community_nodes = graph
        .node_identifiers()
        .map(|u| (u, HashSet::new()))
        .collect::<HashMap<_, _>>();
    let mut improve = false;

    for u in graph.node_identifiers() {
        let mut neighboring_communities = HashSet::new();
        for v in graph.neighbors(u) {
            neighboring_communities.insert(communities[&v]);
        }
        neighboring_communities.remove(&communities[&u]);
        for &c in neighboring_communities.iter() {
            let prev_c = communities[&u];
            community_nodes.get_mut(&prev_c).unwrap().remove(&u);

            let mut k_in = 0.;
            for v in graph.neighbors(u) {
                if communities[&v] == c {
                    k_in += 1.;
                }
            }
            let delta_q = 0.5 * (k_in - k[&u] * sigma_total[&c] / m) / m;
            if delta_q > 0. {
                *sigma_total.get_mut(&c).unwrap() += k[&u];
                *sigma_total.get_mut(&prev_c).unwrap() -= k[&u];
                *communities.get_mut(&u).unwrap() = c;
                community_nodes.get_mut(&c).unwrap().insert(u);
                improve = true;
            } else {
                community_nodes.get_mut(&prev_c).unwrap().insert(u);
            }
        }
    }
    if improve {
        Some(communities)
    } else {
        None
    }
}

pub fn coarsen<
    N1,
    N2,
    E1,
    E2,
    Ty: EdgeType,
    Ix: IndexType,
    NF: FnMut(&Graph<N1, E1, Ty, Ix>, &Vec<NodeIndex<Ix>>) -> N2,
    EF: FnMut(&Graph<N1, E1, Ty, Ix>, &Vec<EdgeIndex<Ix>>) -> E2,
>(
    graph: &Graph<N1, E1, Ty, Ix>,
    node_groups: &HashMap<NodeIndex<Ix>, NodeIndex<Ix>>,
    shrink_node: &mut NF,
    shrink_edge: &mut EF,
) -> Graph<N2, E2, Ty, Ix> {
    let mut groups = HashMap::new();
    for u in graph.node_indices() {
        let g = node_groups[&u];
        groups.entry(g).or_insert(vec![]).push(u);
    }
    let mut group_edges = HashMap::new();
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        let key = {
            let source_group = node_groups[&u];
            let target_group = node_groups[&v];
            if source_group == target_group {
                continue;
            }
            if source_group < target_group {
                (source_group, target_group)
            } else {
                (target_group, source_group)
            }
        };
        group_edges.entry(key).or_insert(vec![]).push(e);
    }

    let mut coarsened_graph = Graph::with_capacity(0, 0);
    for node_ids in groups.values() {
        coarsened_graph.add_node(shrink_node(graph, &node_ids));
    }
    for (&(u, v), edge_ids) in group_edges.iter() {
        coarsened_graph.add_edge(u, v, shrink_edge(graph, &edge_ids));
    }
    coarsened_graph
}
