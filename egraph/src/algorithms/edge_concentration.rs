use petgraph::{Graph, EdgeType};
use petgraph::graph::{IndexType, NodeIndex};
use ::algorithms::biclustering::{Bicluster};

pub fn edge_concentration<N, E, Ty: EdgeType, Ix: IndexType, NF: Fn(usize) -> N, EF: Fn(usize) -> E, DF: Fn(&Bicluster) -> N>(
    graph: &Graph<N, E, Ty, Ix>,
    biclusters: &Vec<Bicluster>,
    new_node: NF,
    new_edge: EF,
    dummy_node: DF,
) -> Graph<N, E, Ty, Ix> {
    let mut transformed = Graph::<N, E, Ty, Ix>::with_capacity(0, 0);
    for u in graph.node_indices() {
        transformed.add_node(new_node(u.index()));
    }
    for bicluster in biclusters {
        let d = transformed.add_node(dummy_node(&bicluster));
        for &u in bicluster.source.iter() {
            transformed.add_edge(NodeIndex::new(u), d, new_edge(u));
        }
        for &v in bicluster.target.iter() {
            transformed.add_edge(d, NodeIndex::new(v), new_edge(v));
        }
    }
    transformed
}
