use fixedbitset::FixedBitSet;
use petgraph::graph::NodeIndex;
use petgraph::visit::GetAdjacencyMatrix;
use petgraph::{Directed, Graph};
use std::collections::HashMap;

fn bary_center<N, E>(
    graph: &Graph<N, E, Directed>,
    matrix: &FixedBitSet,
    h1: &Vec<NodeIndex>,
    h2: &Vec<NodeIndex>,
) -> HashMap<NodeIndex, f64> {
    let mut result = HashMap::new();
    for v in h2 {
        let mut sum = 0;
        let mut count = 0;
        for (i, u) in h1.iter().enumerate() {
            if graph.is_adjacent(&matrix, u.clone(), v.clone()) {
                sum += i;
                count += 1;
            }
        }
        result.insert(v.clone(), sum as f64 / count as f64);
    }
    result
}

pub fn crossing_reduction<N, E>(
    graph: &Graph<N, E, Directed>,
    matrix: &FixedBitSet,
    h1: &Vec<NodeIndex>,
    h2: &mut Vec<NodeIndex>,
) {
    let values = bary_center(graph, matrix, h1, h2);
    h2.sort_by(|u, v| {
        let cu = values.get(u).unwrap();
        let cv = values.get(v).unwrap();
        cu.partial_cmp(cv).unwrap()
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::visit::GetAdjacencyMatrix;
    use petgraph::Graph;

    #[test]
    fn it_works() {
        let mut graph = Graph::<&str, &str>::new();
        let u1 = graph.add_node("u1");
        let u2 = graph.add_node("u2");
        let u3 = graph.add_node("u3");
        let u4 = graph.add_node("u4");
        let v1 = graph.add_node("v1");
        let v2 = graph.add_node("v2");
        let v3 = graph.add_node("v3");
        graph.add_edge(u1, v2, "");
        graph.add_edge(u2, v2, "");
        graph.add_edge(u2, v3, "");
        graph.add_edge(u3, v1, "");
        graph.add_edge(u3, v3, "");
        graph.add_edge(u4, v2, "");
        let h1 = vec![u1, u2, u3, u4];
        let mut h2 = vec![v1, v2, v3];
        let matrix = graph.adjacency_matrix();
        crossing_reduction(&graph, &matrix, &h1, &mut h2);
        assert_eq!(h2, vec![v2, v3, v1]);
    }
}
