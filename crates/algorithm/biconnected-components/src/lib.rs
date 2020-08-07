use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::EdgeType;
use std::cmp::min;
use std::collections::{HashMap, HashSet, VecDeque};

fn dfs<N, E, Ty: EdgeType, Ix: IndexType>(
    u: NodeIndex<Ix>,
    d: usize,
    graph: &Graph<N, E, Ty, Ix>,
    visited: &mut HashSet<NodeIndex<Ix>>,
    depth: &mut HashMap<NodeIndex<Ix>, usize>,
    low: &mut HashMap<NodeIndex<Ix>, usize>,
    parent: &mut HashMap<NodeIndex<Ix>, Option<NodeIndex<Ix>>>,
    articulation_nodes: &mut HashSet<NodeIndex<Ix>>,
) {
    visited.insert(u);
    depth.insert(u, d);
    low.insert(u, d);
    let mut child_count = 0;
    let mut is_articulation = false;

    for v in graph.neighbors(u) {
        if !visited.contains(&v) {
            parent.insert(v, Some(u));
            dfs(
                v,
                d + 1,
                graph,
                visited,
                depth,
                low,
                parent,
                articulation_nodes,
            );
            child_count += 1;
            if low[&v] >= depth[&u] {
                is_articulation = true;
            }
            low.insert(u, min(low[&u], low[&v]));
        } else if parent[&u] != Some(v) {
            low.insert(u, min(low[&u], depth[&v]));
        }
    }
    if (parent[&u].is_some() && is_articulation) || (parent[&u].is_none() && child_count > 1) {
        articulation_nodes.insert(u);
    }
}

pub fn articulation_nodes<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> HashSet<NodeIndex<Ix>> {
    let mut visited = HashSet::new();
    let mut depth = HashMap::new();
    let mut low = HashMap::new();
    let mut parent = HashMap::new();
    let mut articulation_nodes = HashSet::new();
    for u in graph.node_indices() {
        if !visited.contains(&u) {
            parent.insert(u, None);
            dfs(
                u,
                0,
                graph,
                &mut visited,
                &mut depth,
                &mut low,
                &mut parent,
                &mut articulation_nodes,
            );
        }
    }
    articulation_nodes
}

pub fn bridges<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> HashSet<(NodeIndex<Ix>, NodeIndex<Ix>)> {
    let mut visited = HashSet::new();
    let mut depth = HashMap::new();
    let mut low = HashMap::new();
    let mut parent = HashMap::new();
    let mut articulation_nodes = HashSet::new();
    for u in graph.node_indices() {
        if !visited.contains(&u) {
            parent.insert(u, None);
            dfs(
                u,
                0,
                graph,
                &mut visited,
                &mut depth,
                &mut low,
                &mut parent,
                &mut articulation_nodes,
            );
        }
    }
    let mut bridges = HashSet::new();
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        let (u, v) = if depth[&u] < depth[&v] {
            (u, v)
        } else {
            (v, u)
        };
        if low[&v] == depth[&v] {
            bridges.insert((u, v));
        }
    }
    bridges
}

pub fn biconnected_components<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> Vec<Vec<NodeIndex<Ix>>> {
    let bridges = bridges(graph);
    let mut component_nodes = vec![];
    let mut visited_global = HashSet::new();
    for u in graph.node_indices() {
        if visited_global.contains(&u) {
            continue;
        }
        let mut count = 0;
        for v in graph.neighbors(u) {
            if visited_global.contains(&v) {
                continue;
            }
            if bridges.contains(&(u, v)) || bridges.contains(&(v, u)) {
                continue;
            }

            let mut visited = HashSet::new();
            visited.insert(u);
            let mut queue = VecDeque::new();
            queue.push_back(v);
            while let Some(u) = queue.pop_front() {
                if visited.contains(&u) {
                    continue;
                }
                visited.insert(u);
                for v in graph.neighbors(u) {
                    if bridges.contains(&(u, v)) || bridges.contains(&(v, u)) {
                        continue;
                    }
                    queue.push_back(v);
                }
            }
            for &u in &visited {
                visited_global.insert(u);
            }
            let mut nodes = visited.into_iter().collect::<Vec<_>>();
            nodes.sort();
            component_nodes.push(nodes);
            count += 1;
        }
        if count == 0 {
            component_nodes.push(vec![u]);
        }
    }
    component_nodes.sort();
    component_nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::{node_index, UnGraph};

    fn create_graph() -> UnGraph<(), ()> {
        let mut graph = petgraph::Graph::new_undirected();
        let nodes = (0..23).map(|_| graph.add_node(())).collect::<Vec<_>>();
        graph.add_edge(nodes[0], nodes[1], ());
        graph.add_edge(nodes[0], nodes[2], ());
        graph.add_edge(nodes[0], nodes[3], ());
        graph.add_edge(nodes[1], nodes[2], ());
        graph.add_edge(nodes[1], nodes[3], ());
        graph.add_edge(nodes[2], nodes[3], ());
        graph.add_edge(nodes[2], nodes[4], ());
        graph.add_edge(nodes[4], nodes[5], ());
        graph.add_edge(nodes[4], nodes[6], ());
        graph.add_edge(nodes[4], nodes[7], ());
        graph.add_edge(nodes[4], nodes[8], ());
        graph.add_edge(nodes[4], nodes[10], ());
        graph.add_edge(nodes[5], nodes[6], ());
        graph.add_edge(nodes[7], nodes[8], ());
        graph.add_edge(nodes[7], nodes[9], ());
        graph.add_edge(nodes[10], nodes[11], ());
        graph.add_edge(nodes[10], nodes[12], ());
        graph.add_edge(nodes[11], nodes[13], ());
        graph.add_edge(nodes[12], nodes[13], ());
        graph.add_edge(nodes[13], nodes[14], ());
        graph.add_edge(nodes[14], nodes[15], ());
        graph.add_edge(nodes[14], nodes[16], ());
        graph.add_edge(nodes[15], nodes[16], ());
        graph.add_edge(nodes[15], nodes[17], ());
        graph.add_edge(nodes[15], nodes[18], ());
        graph.add_edge(nodes[16], nodes[17], ());
        graph.add_edge(nodes[16], nodes[19], ());
        graph.add_edge(nodes[16], nodes[20], ());
        graph.add_edge(nodes[16], nodes[18], ());
        graph.add_edge(nodes[20], nodes[21], ());
        graph.add_edge(nodes[20], nodes[22], ());
        graph.add_edge(nodes[21], nodes[22], ());
        graph
    }

    fn create_path_graph() -> UnGraph<(), ()> {
        let mut graph = petgraph::Graph::new_undirected();
        let nodes = (0..5).map(|_| graph.add_node(())).collect::<Vec<_>>();
        graph.add_edge(nodes[0], nodes[1], ());
        graph.add_edge(nodes[1], nodes[2], ());
        graph.add_edge(nodes[2], nodes[3], ());
        graph.add_edge(nodes[3], nodes[4], ());
        graph
    }

    #[test]
    fn find_articulation_nodes_of_connected_graph() {
        let graph = create_graph();
        let result = articulation_nodes(&graph);
        let expected = [2, 4, 7, 10, 13, 14, 16, 20]
            .iter()
            .map(|&u| node_index(u))
            .collect::<HashSet<NodeIndex>>();
        assert_eq!(result, expected);
    }

    #[test]
    fn find_articulation_nodes_of_path_graph() {
        let graph = create_path_graph();
        let result = articulation_nodes(&graph);
        let expected = [1, 2, 3]
            .iter()
            .map(|&u| node_index(u))
            .collect::<HashSet<NodeIndex>>();
        assert_eq!(result, expected);
    }

    #[test]
    fn find_bridges_of_connected_graph() {
        let graph = create_graph();
        let result = bridges(&graph);
        let expected = [(2, 4), (4, 10), (7, 9), (13, 14), (16, 19), (16, 20)]
            .iter()
            .map(|&(u, v)| (node_index(u), node_index(v)))
            .collect::<HashSet<(NodeIndex, NodeIndex)>>();
        assert_eq!(result, expected);
    }

    #[test]
    fn find_bridges_of_path_graph() {
        let graph = create_path_graph();
        let result = bridges(&graph);
        let expected = [(0, 1), (1, 2), (2, 3), (3, 4)]
            .iter()
            .map(|&(u, v)| (node_index(u), node_index(v)))
            .collect::<HashSet<(NodeIndex, NodeIndex)>>();
        assert_eq!(result, expected);
    }

    #[test]
    fn find_biconnected_components_of_connected_graph() {
        let graph = create_graph();
        let result = biconnected_components(&graph);
        let expected = vec![
            vec![0, 1, 2, 3],
            vec![4, 5, 6],
            vec![4, 7, 8],
            vec![9],
            vec![10, 11, 12, 13],
            vec![14, 15, 16, 17, 18],
            vec![19],
            vec![20, 21, 22],
        ]
        .into_iter()
        .map(|nodes| nodes.into_iter().map(|u| node_index(u)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        assert_eq!(result, expected);
    }
}
