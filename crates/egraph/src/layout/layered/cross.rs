use crate::{Graph, NodeIndex};

pub fn cross<D, G: Graph<D>>(graph: &G, h1: &Vec<NodeIndex>, h2: &Vec<NodeIndex>) -> u32 {
    let mut result = 0;
    let n = h1.len();
    let m = h2.len();
    for j2 in 0..m - 1 {
        let mut count = 0;
        for i2 in (1..n).rev() {
            let i1 = i2 - 1;
            if graph.has_edge(h1[i2], h2[j2]) {
                count += 1
            }
            if count > 0 {
                for j1 in j2 + 1..m {
                    if graph.has_edge(h1[i1], h2[j1]) {
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
    use egraph_petgraph_adapter::PetgraphWrapper;
    use petgraph::Graph;

    #[test]
    fn it_works() {
        let mut graph = Graph::new();
        let u1 = graph.add_node("u1");
        let u2 = graph.add_node("u2");
        let u3 = graph.add_node("u3");
        let u4 = graph.add_node("u4");
        let v1 = graph.add_node("v1");
        let v2 = graph.add_node("v2");
        let v3 = graph.add_node("v3");
        graph.add_edge(u1, v2, ());
        graph.add_edge(u2, v2, ());
        graph.add_edge(u2, v3, ());
        graph.add_edge(u3, v1, ());
        graph.add_edge(u3, v3, ());
        graph.add_edge(u4, v2, ());
        let h1 = vec![u1.index(), u2.index(), u3.index(), u4.index()];
        let h2 = vec![v1.index(), v2.index(), v3.index()];
        let graph = PetgraphWrapper::new(graph);
        assert_eq!(cross(&graph, &h1, &h2), 5);
    }
}
