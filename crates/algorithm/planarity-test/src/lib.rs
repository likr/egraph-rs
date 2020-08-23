use petgraph::algo::is_bipartite_undirected;
use petgraph::graph::{IndexType, NodeIndex};
use petgraph::prelude::*;
use petgraph::unionfind::UnionFind;
use petgraph::visit::{depth_first_search, Control, DfsEvent};
use petgraph::EdgeType;
use std::collections::{HashMap, HashSet};

fn is_path<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    segment: &Vec<NodeIndex<Ix>>,
) -> bool {
    let nodes = segment
        .iter()
        .map(|&u| u)
        .collect::<HashSet<NodeIndex<Ix>>>();
    for &u in segment.iter() {
        let mut degree = 0;
        for v in graph.neighbors_undirected(u) {
            if nodes.contains(&v) {
                degree += 1;
            }
        }
        if degree > 2 {
            return false;
        }
    }
    true
}

fn find_cycle<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> Option<Vec<NodeIndex<Ix>>> {
    let mut stack = vec![];
    let mut visited_edge = HashSet::new();
    let u = graph.node_indices().nth(0).unwrap();
    let result = depth_first_search(graph, Some(u), |event| {
        match event {
            DfsEvent::Discover(u, _) => {
                stack.push(u);
            }
            DfsEvent::Finish(_, _) => {
                stack.pop();
            }
            DfsEvent::TreeEdge(u, v) => {
                visited_edge.insert((u, v));
            }
            DfsEvent::BackEdge(u, v) => {
                if !visited_edge.contains(&(v, u)) {
                    eprintln!("{:?}", stack);
                    stack.reverse();
                    while stack[stack.len() - 1] != v {
                        stack.pop();
                    }
                    return Control::Break(());
                }
            }
            _ => {}
        }
        Control::Continue
    });
    if let Control::Break(_) = result {
        Some(stack)
    } else {
        None
    }
}

fn find_separating_cycle<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> Option<(Vec<NodeIndex<Ix>>, Vec<Vec<NodeIndex<Ix>>>)> {
    if let Some(cycle) = find_cycle(graph) {
        let cycle_nodes = cycle.iter().map(|&u| u).collect::<HashSet<NodeIndex<Ix>>>();
        let mut components = UnionFind::new(graph.node_count());
        for e in graph.edge_indices() {
            let (u, v) = graph.edge_endpoints(e).unwrap();
            if !cycle_nodes.contains(&u) && !cycle_nodes.contains(&v) {
                components.union(u.index(), v.index());
            }
        }
        let mut segments = HashMap::new();
        for u in graph.node_indices() {
            if cycle_nodes.contains(&u) {
                continue;
            }
            let segment = segments
                .entry(components.find(u.index()))
                .or_insert(HashSet::new());
            segment.insert(u);
            for v in graph.neighbors_undirected(u) {
                if cycle_nodes.contains(&v) {
                    segment.insert(v);
                }
            }
        }
        let segments = segments
            .values()
            .filter(|nodes| !nodes.is_empty())
            .map(|nodes| nodes.iter().map(|&u| u).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        eprintln!("{:?}", segments);
        if segments.is_empty() || segments.len() == 1 {
            None
        } else {
            Some((cycle, segments))
        }
    } else {
        None
    }
}

fn separated_subgraph<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    cycle: &Vec<NodeIndex<Ix>>,
    segment: &Vec<NodeIndex<Ix>>,
) -> Graph<(), (), Ty, Ix> {
    let mut nodes = HashSet::new();
    for &u in cycle {
        nodes.insert(u);
    }
    for &u in segment {
        nodes.insert(u);
    }
    graph.filter_map(
        |u, _| if nodes.contains(&u) { Some(()) } else { None },
        |_, _| Some(()),
    )
}

fn interlacement_graph<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    cycle: &Vec<NodeIndex<Ix>>,
    segments: &Vec<Vec<NodeIndex<Ix>>>,
) -> UnGraph<(), ()> {
    let mut h = Graph::new_undirected();
    for segment in segments {
        h.add_node(());
    }
    h
}

fn auslander_parter<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> bool {
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        eprintln!("{:?} {:?}", u, v);
    }
    if let Some((cycle, segments)) = find_separating_cycle(graph) {
        let h = interlacement_graph(graph, &cycle, &segments);
        if !is_bipartite_undirected(&h, h.node_indices().nth(0).unwrap()) {
            return false;
        }
        for segment in segments.iter() {
            if !auslander_parter(&separated_subgraph(graph, &cycle, segment)) {
                return false;
            }
        }
        true
    } else {
        false
    }
}

pub fn is_planar<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> bool {
    auslander_parter(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planar_graph() -> UnGraph<(), ()> {
        let mut graph = Graph::new_undirected();
        let u1 = graph.add_node(());
        let u2 = graph.add_node(());
        let u3 = graph.add_node(());
        let u4 = graph.add_node(());
        let u5 = graph.add_node(());
        let u6 = graph.add_node(());
        let u7 = graph.add_node(());
        let u8 = graph.add_node(());
        let u9 = graph.add_node(());
        graph.add_edge(u1, u2, ());
        graph.add_edge(u1, u6, ());
        graph.add_edge(u1, u7, ());
        graph.add_edge(u2, u3, ());
        graph.add_edge(u2, u6, ());
        graph.add_edge(u2, u8, ());
        graph.add_edge(u2, u9, ());
        graph.add_edge(u3, u4, ());
        graph.add_edge(u3, u8, ());
        graph.add_edge(u4, u5, ());
        graph.add_edge(u4, u8, ());
        graph.add_edge(u4, u9, ());
        graph.add_edge(u5, u6, ());
        graph.add_edge(u5, u7, ());
        graph.add_edge(u6, u9, ());
        graph
    }

    #[test]
    fn test_find_cycle() {
        let mut graph = Graph::new();
        let u1 = graph.add_node(());
        let u2 = graph.add_node(());
        let u3 = graph.add_node(());
        let u4 = graph.add_node(());
        let u5 = graph.add_node(());
        graph.add_edge(u1, u2, ());
        graph.add_edge(u2, u3, ());
        graph.add_edge(u3, u4, ());
        graph.add_edge(u4, u1, ());
        graph.add_edge(u1, u5, ());
        let cycle = find_cycle(&graph).unwrap();
        assert_eq!(cycle.len(), 4);
    }

    #[test]
    fn test_find_separating_cycle() {
        let graph = planar_graph();
        let (cycle, segments) = find_separating_cycle(&graph).unwrap();
        eprintln!("{:?}", cycle);
        eprintln!("{:?}", segments);
    }

    #[test]
    fn test_is_planar() {
        let graph = planar_graph();
        assert_eq!(is_planar(&graph), true);
    }
}
