use crate::Graph;
use std::collections::{HashMap, HashSet, VecDeque};

pub fn connected_components<D, G: Graph<D>>(graph: &G) -> HashMap<usize, usize> {
    let mut components = HashMap::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    for u in graph.nodes() {
        if visited.contains(&u) {
            continue;
        }
        queue.push_back(u);
        while queue.len() > 0 {
            let v = queue.pop_front().unwrap();
            if visited.contains(&v) {
                continue;
            }
            visited.insert(v);
            components.insert(v, u);
            for w in graph.neighbors(v) {
                queue.push_back(w);
            }
        }
    }
    components
}

#[cfg(test)]
mod test {
    use super::*;
    use egraph_petgraph_adapter::PetgraphWrapper;
    use petgraph::prelude::Graph;

    #[test]
    fn test_connected_components() {
        let mut graph = Graph::new_undirected();
        let u1 = graph.add_node(());
        let u2 = graph.add_node(());
        let u3 = graph.add_node(());
        let u4 = graph.add_node(());
        let u5 = graph.add_node(());
        graph.add_edge(u1, u2, ());
        graph.add_edge(u1, u3, ());
        graph.add_edge(u2, u3, ());
        graph.add_edge(u4, u5, ());
        let graph = PetgraphWrapper::new(graph);
        let components = connected_components(&graph);
        assert_eq!(components[&0], 0);
        assert_eq!(components[&1], 0);
        assert_eq!(components[&2], 0);
        assert_eq!(components[&3], 3);
        assert_eq!(components[&4], 3);
    }
}
