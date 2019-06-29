use egraph::algorithm::connected_components;
use egraph::Graph;
use egraph_petgraph_adapter::PetgraphWrapper;
use petgraph::prelude::Graph as PetGraph;

#[test]
fn test_connected_components() {
    let mut graph = PetGraph::new_undirected();
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
    println!("{}", graph.node_count());
    let components = connected_components(&graph);
    assert_eq!(components[&0], 0);
    assert_eq!(components[&1], 0);
    assert_eq!(components[&2], 0);
    assert_eq!(components[&3], 3);
    assert_eq!(components[&4], 3);
}
