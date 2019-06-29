use crate::graph::neighbors;
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
            for w in neighbors(graph, v) {
                queue.push_back(w);
            }
        }
    }
    components
}
