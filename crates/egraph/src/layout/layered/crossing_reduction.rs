use crate::{Graph, NodeIndex};
use std::collections::HashMap;

fn bary_center<D, G: Graph<D>>(
    graph: &G,
    h1: &Vec<NodeIndex>,
    h2: &Vec<NodeIndex>,
) -> HashMap<NodeIndex, f64> {
    let mut result = HashMap::new();
    for &v in h2 {
        let mut sum = 0;
        let mut count = 0;
        for (i, &u) in h1.iter().enumerate() {
            if graph.has_edge(u, v) {
                sum += i;
                count += 1;
            }
        }
        result.insert(v.clone(), sum as f64 / count as f64);
    }
    result
}

pub fn crossing_reduction<D, G: Graph<D>>(graph: &G, h1: &Vec<NodeIndex>, h2: &mut Vec<NodeIndex>) {
    let values = bary_center(graph, h1, h2);
    h2.sort_by(|u, v| {
        let cu = values.get(u).unwrap();
        let cv = values.get(v).unwrap();
        cu.partial_cmp(cv).unwrap()
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_petgraph_adapter::PetgraphWrapper;
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
        let h1 = vec![u1.index(), u2.index(), u3.index(), u4.index()];
        let mut h2 = vec![v1.index(), v2.index(), v3.index()];
        let graph = PetgraphWrapper::new(graph);
        crossing_reduction(&graph, &h1, &mut h2);
        assert_eq!(h2, vec![v2.index(), v3.index(), v1.index()]);
    }
}
