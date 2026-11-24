use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};
#[cfg(feature = "petgraph")]
use petgraph::{EdgeType, graph::IndexType, prelude::*};
#[cfg(feature = "petgraph")]
use std::collections::HashMap;

/// Parses an edge-list CSV string into a [`CsrGraph<W>`].
///
/// Expected format is a header line `num_nodes,num_edges` followed by `num_edges` lines of `u,v`.
pub fn parse_csr_csv<W: Scalar>(
    input: &str,
    directed: bool,
    sanitize: bool,
) -> CoreResult<CsrGraph<W>> {
    let mut tokens = input.split_ascii_whitespace();
    let header = tokens
        .next()
        .ok_or_else(|| CoreError::invalid_input("dataset CSV is empty"))?;
    let (num_nodes, expected_edges) = parse_usize_pair(header, "CSV header")?;

    let mut edges = Vec::with_capacity(expected_edges);
    for (row_idx, row) in tokens.enumerate() {
        let (u_usize, v_usize) = parse_usize_pair(row, "CSV edge row")?;
        let u = u32::try_from(u_usize).map_err(|_| {
            CoreError::invalid_input(format!(
                "vertex index {u_usize} overflows u32 at row {row_idx}"
            ))
        })?;
        let v = u32::try_from(v_usize).map_err(|_| {
            CoreError::invalid_input(format!(
                "vertex index {v_usize} overflows u32 at row {row_idx}"
            ))
        })?;
        edges.push((u, v));
    }

    if edges.len() != expected_edges {
        return Err(CoreError::dimension_mismatch(
            expected_edges,
            edges.len(),
            "egraph_dataset::parse_csr_csv",
        ));
    }

    CsrGraph::from_edges(num_nodes, &edges, None, directed, sanitize)
}

fn parse_usize_pair(token: &str, context: &str) -> CoreResult<(usize, usize)> {
    let mut parts = token.split(',');
    let a = parts
        .next()
        .ok_or_else(|| CoreError::invalid_input(format!("missing first integer in {context}")))?;
    let b = parts
        .next()
        .ok_or_else(|| CoreError::invalid_input(format!("missing second integer in {context}")))?;
    if parts.next().is_some() {
        return Err(CoreError::invalid_input(format!(
            "expected exactly two comma-separated integers in {context}: '{token}'"
        )));
    }
    let first = a.parse::<usize>().map_err(|err| {
        CoreError::invalid_input(format!("invalid integer '{a}' in {context}: {err}"))
    })?;
    let second = b.parse::<usize>().map_err(|err| {
        CoreError::invalid_input(format!("invalid integer '{b}' in {context}: {err}"))
    })?;
    Ok((first, second))
}

#[cfg(feature = "petgraph")]
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

#[allow(dead_code)]
fn parse_embedded_csr(input: &str) -> CsrGraph<f64> {
    parse_csr_csv(input, false, false).expect("embedded dataset CSV must be valid")
}

/// Loads the "1138_bus" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `1138_bus` and `petgraph` feature flags to be enabled.
#[cfg(all(feature = "petgraph", feature = "1138_bus"))]
pub fn dataset_1138_bus<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix>
{
    parse(include_str!("data/1138_bus.csv"))
}

/// Loads the "1138_bus" graph dataset as an undirected [`CsrGraph<f64>`].
///
/// Requires the `1138_bus` feature flag to be enabled.
#[cfg(feature = "1138_bus")]
pub fn dataset_1138_bus_csr() -> CsrGraph<f64> {
    parse_embedded_csr(include_str!("data/1138_bus.csv"))
}

/// Loads the "3elt" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `3elt` and `petgraph` feature flags to be enabled.
#[cfg(all(feature = "petgraph", feature = "3elt"))]
pub fn dataset_3_elt<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/3elt.csv"))
}

/// Loads the "3elt" graph dataset as an undirected [`CsrGraph<f64>`].
///
/// Requires the `3elt` feature flag to be enabled.
#[cfg(feature = "3elt")]
pub fn dataset_3_elt_csr() -> CsrGraph<f64> {
    parse_embedded_csr(include_str!("data/3elt.csv"))
}

/// Loads the "dwt_1005" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `dwt_1005` and `petgraph` feature flags to be enabled.
#[cfg(all(feature = "petgraph", feature = "dwt_1005"))]
pub fn dataset_dwt_1005<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix>
{
    parse(include_str!("data/dwt_1005.csv"))
}

/// Loads the "dwt_1005" graph dataset as an undirected [`CsrGraph<f64>`].
///
/// Requires the `dwt_1005` feature flag to be enabled.
#[cfg(feature = "dwt_1005")]
pub fn dataset_dwt_1005_csr() -> CsrGraph<f64> {
    parse_embedded_csr(include_str!("data/dwt_1005.csv"))
}

/// Loads the "dwt_2680" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `dwt_2680` and `petgraph` feature flags to be enabled.
#[cfg(all(feature = "petgraph", feature = "dwt_2680"))]
pub fn dataset_dwt_2680<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix>
{
    parse(include_str!("data/dwt_2680.csv"))
}

/// Loads the "dwt_2680" graph dataset as an undirected [`CsrGraph<f64>`].
///
/// Requires the `dwt_2680` feature flag to be enabled.
#[cfg(feature = "dwt_2680")]
pub fn dataset_dwt_2680_csr() -> CsrGraph<f64> {
    parse_embedded_csr(include_str!("data/dwt_2680.csv"))
}

/// Loads the "poli" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `poli` and `petgraph` feature flags to be enabled.
#[cfg(all(feature = "petgraph", feature = "poli"))]
pub fn dataset_poli<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/poli.csv"))
}

/// Loads the "poli" graph dataset as an undirected [`CsrGraph<f64>`].
///
/// Requires the `poli` feature flag to be enabled.
#[cfg(feature = "poli")]
pub fn dataset_poli_csr() -> CsrGraph<f64> {
    parse_embedded_csr(include_str!("data/poli.csv"))
}

/// Loads the "qh882" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `qh882` and `petgraph` feature flags to be enabled.
#[cfg(all(feature = "petgraph", feature = "qh882"))]
pub fn dataset_qh882<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/qh882.csv"))
}

/// Loads the "qh882" graph dataset as an undirected [`CsrGraph<f64>`].
///
/// Requires the `qh882` feature flag to be enabled.
#[cfg(feature = "qh882")]
pub fn dataset_qh882_csr() -> CsrGraph<f64> {
    parse_embedded_csr(include_str!("data/qh882.csv"))
}

/// Loads the "USpowerGrid" graph dataset from the SuiteSparse Matrix Collection.
///
/// Requires the `USpowerGrid` and `petgraph` feature flags to be enabled.
#[cfg(all(feature = "petgraph", feature = "USpowerGrid"))]
pub fn dataset_uspower_grid<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>()
-> Graph<N, E, Ty, Ix> {
    parse(include_str!("data/USpowerGrid.csv"))
}

/// Loads the "USpowerGrid" graph dataset as an undirected [`CsrGraph<f64>`].
///
/// Requires the `USpowerGrid` feature flag to be enabled.
#[cfg(feature = "USpowerGrid")]
pub fn dataset_uspower_grid_csr() -> CsrGraph<f64> {
    parse_embedded_csr(include_str!("data/USpowerGrid.csv"))
}

/// Loads the "karate" graph dataset (Zachary's Karate Club).
///
/// Requires the `karate` and `petgraph` feature flags to be enabled.
#[cfg(all(feature = "petgraph", feature = "karate"))]
pub fn dataset_karate<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix>
{
    parse(include_str!("data/karate.csv"))
}

/// Loads the "karate" graph dataset (Zachary's Karate Club) as an undirected [`CsrGraph<f64>`].
///
/// Requires the `karate` feature flag to be enabled.
#[cfg(feature = "karate")]
pub fn dataset_karate_csr() -> CsrGraph<f64> {
    parse_embedded_csr(include_str!("data/karate.csv"))
}

/// Loads the "lesmis" graph dataset (Les Misérables character co-occurrence network).
///
/// Requires the `lesmis` and `petgraph` feature flags to be enabled.
#[cfg(all(feature = "petgraph", feature = "lesmis"))]
pub fn dataset_lesmis<N: Default, E: Default, Ty: EdgeType, Ix: IndexType>() -> Graph<N, E, Ty, Ix>
{
    parse(include_str!("data/lesmis.csv"))
}

/// Loads the "lesmis" graph dataset (Les Misérables) as an undirected [`CsrGraph<f64>`].
///
/// Requires the `lesmis` feature flag to be enabled.
#[cfg(feature = "lesmis")]
pub fn dataset_lesmis_csr() -> CsrGraph<f64> {
    parse_embedded_csr(include_str!("data/lesmis.csv"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csr_csv_valid_and_errors() {
        let csv = "4,3\n0,1\r\n1,2\n2,3\n";
        let g = parse_csr_csv::<f64>(csv, false, false).unwrap();
        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.num_edges(), 3);
        assert!(!g.is_directed());
        assert_eq!(g.neighbors(1), &[0, 2]);

        let dg = parse_csr_csv::<f64>("3,2\n0,1\n1,0\n", true, false).unwrap();
        assert!(dg.is_directed());
        assert_eq!(dg.num_edges(), 2);

        assert!(matches!(
            parse_csr_csv::<f64>("", false, false),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            parse_csr_csv::<f64>("3,2\n0,1\n", false, false),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            parse_csr_csv::<f64>("3,1\n0,1,2\n", false, false),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            parse_csr_csv::<f64>("3,1\n0,5\n", false, false),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            parse_csr_csv::<f64>("3,1\n0,0\n", false, false),
            Err(CoreError::InvalidCsr(_))
        ));
    }

    #[test]
    fn test_parse_csr_csv_f32() {
        let csv = "4,3\n0,1\n1,2\n2,3\n";
        let g: CsrGraph<f32> = parse_csr_csv::<f32>(csv, false, false).unwrap();
        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.num_edges(), 3);
        assert!(!g.is_directed());
        assert!(g.weights().is_none());
        assert_eq!(g.neighbors(1), &[0, 2]);

        // Sanitize deduplicates undirected edges and removes self-loops.
        let dirty = "3,3\n0,1\n1,0\n0,0\n";
        let clean = parse_csr_csv::<f32>(dirty, false, true).unwrap();
        assert_eq!(clean.num_nodes(), 3);
        assert_eq!(clean.num_edges(), 1);
        assert_eq!(clean.neighbors(0), &[1]);

        let dg = parse_csr_csv::<f32>("3,2\n0,1\n1,0\n", true, false).unwrap();
        assert!(dg.is_directed());
        assert_eq!(dg.num_edges(), 2);

        assert!(matches!(
            parse_csr_csv::<f32>("3,2\n0,1\n", false, false),
            Err(CoreError::DimensionMismatch { .. })
        ));
    }

    #[cfg(all(
        feature = "petgraph",
        feature = "1138_bus",
        feature = "3elt",
        feature = "dwt_1005",
        feature = "dwt_2680",
        feature = "poli",
        feature = "qh882",
        feature = "USpowerGrid",
        feature = "karate",
        feature = "lesmis"
    ))]
    #[test]
    fn test_all_embedded_datasets_match_petgraph_counts() {
        let cases: Vec<(UnGraph<(), ()>, CsrGraph<f64>)> = vec![
            (dataset_1138_bus(), dataset_1138_bus_csr()),
            (dataset_3_elt(), dataset_3_elt_csr()),
            (dataset_dwt_1005(), dataset_dwt_1005_csr()),
            (dataset_dwt_2680(), dataset_dwt_2680_csr()),
            (dataset_poli(), dataset_poli_csr()),
            (dataset_qh882(), dataset_qh882_csr()),
            (dataset_uspower_grid(), dataset_uspower_grid_csr()),
            (dataset_karate(), dataset_karate_csr()),
            (dataset_lesmis(), dataset_lesmis_csr()),
        ];

        for (pg, csr) in cases {
            assert_eq!(csr.num_nodes(), pg.node_count());
            assert_eq!(csr.num_edges(), pg.edge_count());
            assert!(!csr.is_directed());
            for u in 0..csr.num_nodes() {
                assert_eq!(csr.degree(u), pg.neighbors(NodeIndex::new(u)).count());
            }
        }
    }
}
