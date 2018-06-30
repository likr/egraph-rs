use fixedbitset::FixedBitSet;
use petgraph::graph::NodeIndex;
use petgraph::visit::GetAdjacencyMatrix;
use petgraph::{Directed, Graph};

pub fn cross<N, E>(
    graph: &Graph<N, E, Directed>,
    matrix: &FixedBitSet,
    h1: &Vec<NodeIndex>,
    h2: &Vec<NodeIndex>,
) -> u32 {
    let mut result = 0;
    let n = h1.len();
    let m = h2.len();
    for j2 in 0..m - 1 {
        let mut count = 0;
        for i2 in (1..n).rev() {
            let i1 = i2 - 1;
            if graph.is_adjacent(&matrix, h1[i2].clone(), h2[j2].clone()) {
                count += 1
            }
            if count > 0 {
                for j1 in j2 + 1..m {
                    if graph.is_adjacent(&matrix, h1[i1].clone(), h2[j1].clone()) {
                        result += count
                    }
                }
            }
        }
    }
    result
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
        let h2 = vec![v1, v2, v3];
        let matrix = graph.adjacency_matrix();
        assert_eq!(cross(&graph, &matrix, &h1, &h2), 5);
    }
}
