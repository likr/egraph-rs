use super::crossing_reduction::crossing_reduction;
use super::cycle_removal::remove_cycle;
use super::graph::{Edge, Node};
use super::normalize::normalize;
use super::position_assignment::brandes::brandes;
use super::ranking::{LongetPathRanking, RankingModule};
use petgraph::visit::GetAdjacencyMatrix;
use petgraph::{Directed, Graph};
use std::cmp;

pub trait Setting {
    fn node_width<N>(&self, node: N) -> usize;

    fn node_height<N>(&self, node: N) -> usize;
}

pub struct SugiyamaLayout<R = LongetPathRanking>
where
    R: RankingModule,
{
    ranking_module: R,
}

impl SugiyamaLayout {
    pub fn new() -> SugiyamaLayout<LongetPathRanking> {
        SugiyamaLayout {
            ranking_module: LongetPathRanking::new(),
        }
    }
}

impl<R: RankingModule> SugiyamaLayout<R> {
    pub fn call<N, E>(&self, input: &Graph<N, E, Directed>) -> Graph<Node, Edge> {
        let mut graph = input.map(|_, _| Node::new(), |_, _| Edge::new());
        remove_cycle(&mut graph);
        let mut layers_map = self.ranking_module.call(&graph);
        normalize(&mut graph, &mut layers_map);
        let height = 1 + graph
            .node_indices()
            .fold(0, |max, u| cmp::max(max, *layers_map.get(&u).unwrap()));
        let mut layers: Vec<_> = (0..height).map(|_| vec![]).collect();
        for u in graph.node_indices() {
            let layer = layers_map.get(&u).unwrap();
            layers[*layer].push(u);
        }
        let matrix = graph.adjacency_matrix();
        for i in 1..height {
            let h1 = layers.get_mut(i - 1).unwrap().clone();
            let mut h2 = layers.get_mut(i).unwrap();
            crossing_reduction(&graph, &matrix, &h1, &mut h2);
        }
        for (i, layer) in layers.iter().enumerate() {
            for (j, &u) in layer.iter().enumerate() {
                graph[u].width = 100;
                graph[u].height = 100;
                graph[u].orig_width = 100;
                graph[u].orig_height = 100;
                graph[u].layer = i;
                graph[u].order = j;
            }
        }
        brandes(&mut graph, &layers);
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;

    struct MySetting;

    impl Setting for MySetting {
        fn node_width<N>(&self, node: N) -> usize {
            10
        }

        fn node_height<N>(&self, node: N) -> usize {
            10
        }
    }

    #[test]
    fn test_sugiyama_layout() {
        let mut graph = Graph::new();
        let a1 = graph.add_node("a1");
        let a2 = graph.add_node("a2");
        let a3 = graph.add_node("a3");
        let b1 = graph.add_node("b1");
        let b2 = graph.add_node("b2");
        let b3 = graph.add_node("b3");
        let c1 = graph.add_node("c1");
        let c2 = graph.add_node("c2");
        let c3 = graph.add_node("c3");
        let d1 = graph.add_node("d1");
        let d2 = graph.add_node("d2");
        let d3 = graph.add_node("d3");
        graph.add_edge(a1, b2, "");
        graph.add_edge(a2, b1, "");
        graph.add_edge(a3, b1, "");
        graph.add_edge(b1, c1, "");
        graph.add_edge(b2, c1, "");
        graph.add_edge(b2, c2, "");
        graph.add_edge(b2, c3, "");
        graph.add_edge(b3, c2, "");
        graph.add_edge(c1, d3, "");
        graph.add_edge(c2, d1, "");
        graph.add_edge(c2, d2, "");
        let sugiyama_layout = SugiyamaLayout::new();
        let result = sugiyama_layout.call(&graph);
        for u in result.node_indices() {
            println!("{} {}", result[u].x, result[u].y);
        }
    }
}
