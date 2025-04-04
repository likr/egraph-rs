use petgraph::{graph::IndexType, prelude::*, EdgeType};
use std::collections::HashMap;

#[allow(dead_code)]
fn parse<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>(input: &str) -> Graph<N, E, Ty, Ix> {
    let rows = input
        .split_ascii_whitespace()
        .map(|row| {
            row.split(',')
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

/// Loads the "1138_bus" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `1138_bus` feature flag to be enabled.
#[cfg(feature = "1138_bus")]
pub fn dataset_1138_bus<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix>
{
    parse(include_str!("data/1138_bus.csv"))
}

/// Loads the "3elt" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `3_elt` feature flag to be enabled.
#[cfg(feature = "3_elt")]
pub fn dataset_3_elt<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/3elt.csv"))
}

/// Loads the "dwt_1005" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `dwt_1005` feature flag to be enabled.
#[cfg(feature = "dwt_1005")]
pub fn dataset_dwt_1005<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix>
{
    parse(include_str!("data/dwt_1005.csv"))
}

/// Loads the "dwt_2680" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `dwt_2680` feature flag to be enabled.
#[cfg(feature = "dwt_2680")]
pub fn dataset_dwt_2680<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix>
{
    parse(include_str!("data/dwt_2680.csv"))
}

/// Loads the "poli" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `poli` feature flag to be enabled.
#[cfg(feature = "poli")]
pub fn dataset_poli<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/poli.csv"))
}

/// Loads the "qh882" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `qh882` feature flag to be enabled.
#[cfg(feature = "qh882")]
pub fn dataset_qh882<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/qh882.csv"))
}

/// Loads the "USpowerGrid" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `USpowerGrid` feature flag to be enabled.
#[cfg(feature = "USpowerGrid")]
pub fn dataset_uspower_grid<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>(
) -> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/USpowerGrid.csv"))
}
