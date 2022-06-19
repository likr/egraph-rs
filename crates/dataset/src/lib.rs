use petgraph::{graph::IndexType, prelude::*, EdgeType};
use std::collections::HashMap;

#[allow(dead_code)]
fn parse<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>(input: &str) -> Graph<N, E, Ty, Ix> {
    let rows = input
        .split_ascii_whitespace()
        .map(|row| {
            row.split('/')
                .map(|v| v.parse::<usize>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let n = rows[0][0];
    let m = rows[0][1];
    let mut graph = Graph::with_capacity(n, m);
    let mut indices = HashMap::new();
    for u in 0..n {
        indices.insert(u, graph.add_node(N::default()));
    }
    for row in &rows[1..] {
        let u = row[0];
        let v = row[1];
        graph.add_edge(indices[&u], indices[&v], E::default());
    }
    graph
}

#[cfg(feature = "1138_bus")]
pub fn dataset_1138_bus<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix>
{
    parse(include_str!("data/1138_bus.csv"))
}

#[cfg(feature = "3_elt")]
pub fn dataset_3_elt<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/3_elt.csv"))
}

#[cfg(feature = "dwt_1005")]
pub fn dataset_dwt_1005<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix>
{
    parse(include_str!("data/dwt_1005.csv"))
}

#[cfg(feature = "dwt_2680")]
pub fn dataset_dwt_2680<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix>
{
    parse(include_str!("data/dwt_2680.csv"))
}

#[cfg(feature = "poli")]
pub fn dataset_poli<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/poli.csv"))
}

#[cfg(feature = "qh882")]
pub fn dataset_qh882<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/qh882.csv"))
}

#[cfg(feature = "USpowerGrid")]
pub fn dataset_USpowerGrid<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>(
) -> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/USpowerGrid.csv"))
}
