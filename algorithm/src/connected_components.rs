use egraph_graph::neighbors;
use egraph_interface::Graph;
use std::collections::VecDeque;

pub fn connected_components(graph: &Graph) -> Vec<usize> {
    let mut components = graph.nodes().map(|u| u).collect::<Vec<_>>();
    let mut visited = graph.nodes().map(|_| false).collect::<Vec<_>>();
    let mut queue = VecDeque::new();
    for u in graph.nodes() {
        if visited[u] {
            continue;
        }
        queue.push_back(u);
        let c = components[u];
        while queue.len() > 0 {
            let v = queue.pop_front().unwrap();
            if visited[v] {
                continue;
            }
            visited[v] = true;
            components[v] = c;
            for w in neighbors(graph, v) {
                queue.push_back(w);
            }
        }
    }
    components
}

#[test]
fn test_connected_components() {
    // let mut graph = Graph::new_undirected();
    // let u1 = graph.add_node(());
    // let u2 = graph.add_node(());
    // let u3 = graph.add_node(());
    // let u4 = graph.add_node(());
    // let u5 = graph.add_node(());
    // graph.add_edge(u1, u2, ());
    // graph.add_edge(u1, u3, ());
    // graph.add_edge(u2, u3, ());
    // graph.add_edge(u4, u5, ());
    // let components = connected_components(&graph);
    // assert_eq!(components[0], 0);
    // assert_eq!(components[1], 0);
    // assert_eq!(components[2], 0);
    // assert_eq!(components[3], 3);
    // assert_eq!(components[4], 3);
}
