//! Bidirectional adapters between `petgraph` graph types and `egraph-core`'s
//! compressed sparse row (`CsrGraph`) representation.
//!
//! This crate is an integration boundary for callers that construct graphs using
//! `petgraph` while keeping solver internals strictly decoupled from `petgraph`.

use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};
use petgraph::graph::{DiGraph, Graph, IndexType, NodeIndex, UnGraph};
use petgraph::visit::{EdgeRef, NodeIndexable};
use petgraph::{Directed, EdgeType, Undirected};

/// Extracts and validates edge weights from any `petgraph::Graph` using `weight_fn`.
///
/// Edge weights are returned in `graph.edge_references()` order and validated to be finite.
pub fn extract_edge_weights<W, N, E, Ty, Ix, F>(
    graph: &Graph<N, E, Ty, Ix>,
    mut weight_fn: F,
) -> CoreResult<Vec<W>>
where
    W: Scalar,
    Ty: EdgeType,
    Ix: IndexType,
    F: FnMut(&E) -> W,
{
    let mut weights = Vec::with_capacity(graph.edge_count());
    for edge in graph.edge_references() {
        let w = weight_fn(edge.weight());
        if !w.is_finite() {
            return Err(CoreError::non_finite(
                "egraph_petgraph::extract_edge_weights",
            ));
        }
        weights.push(w);
    }
    Ok(weights)
}

/// Converts an undirected `petgraph::UnGraph` into an unweighted `CsrGraph<f64>`.
///
/// Rejects self-loops and parallel edges with [`CoreError::InvalidCsr`].
pub fn from_petgraph_ungraph<N, E, Ix: IndexType>(
    graph: &UnGraph<N, E, Ix>,
) -> CoreResult<CsrGraph<f64>> {
    from_petgraph_impl::<f64, N, E, Undirected, Ix, fn(&E) -> f64>(graph, false, false, None)
}

/// Converts an undirected `petgraph::UnGraph` into an unweighted `CsrGraph<W>`.
///
/// Rejects self-loops and parallel edges with [`CoreError::InvalidCsr`].
pub fn from_petgraph_ungraph_as<W, N, E, Ix>(graph: &UnGraph<N, E, Ix>) -> CoreResult<CsrGraph<W>>
where
    W: Scalar,
    Ix: IndexType,
{
    from_petgraph_impl::<W, N, E, Undirected, Ix, fn(&E) -> W>(graph, false, false, None)
}

/// Converts an undirected `petgraph::UnGraph` into a weighted `CsrGraph<W>`.
///
/// Rejects self-loops, parallel edges, and non-finite edge weights.
pub fn from_petgraph_ungraph_with_weights<W, N, E, Ix, F>(
    graph: &UnGraph<N, E, Ix>,
    weight_fn: F,
) -> CoreResult<CsrGraph<W>>
where
    W: Scalar,
    Ix: IndexType,
    F: FnMut(&E) -> W,
{
    from_petgraph_impl(graph, false, false, Some(weight_fn))
}

/// Converts an undirected `petgraph::UnGraph` into an unweighted `CsrGraph<f64>`,
/// dropping self-loops and deduplicating parallel edges.
pub fn from_petgraph_ungraph_sanitized<N, E, Ix: IndexType>(
    graph: &UnGraph<N, E, Ix>,
) -> CoreResult<CsrGraph<f64>> {
    from_petgraph_impl::<f64, N, E, Undirected, Ix, fn(&E) -> f64>(graph, false, true, None)
}

/// Converts an undirected `petgraph::UnGraph` into a weighted `CsrGraph<W>`,
/// dropping self-loops and retaining the maximum weight across parallel edges.
pub fn from_petgraph_ungraph_sanitized_with_weights<W, N, E, Ix, F>(
    graph: &UnGraph<N, E, Ix>,
    weight_fn: F,
) -> CoreResult<CsrGraph<W>>
where
    W: Scalar,
    Ix: IndexType,
    F: FnMut(&E) -> W,
{
    from_petgraph_impl(graph, false, true, Some(weight_fn))
}

/// Converts a directed `petgraph::DiGraph` into an unweighted directed `CsrGraph<f64>`.
///
/// Rejects self-loops and duplicate directed edges with [`CoreError::InvalidCsr`].
pub fn from_petgraph_digraph<N, E, Ix: IndexType>(
    graph: &DiGraph<N, E, Ix>,
) -> CoreResult<CsrGraph<f64>> {
    from_petgraph_impl::<f64, N, E, Directed, Ix, fn(&E) -> f64>(graph, true, false, None)
}

/// Converts a directed `petgraph::DiGraph` into an unweighted directed `CsrGraph<W>`.
///
/// Rejects self-loops and duplicate directed edges with [`CoreError::InvalidCsr`].
pub fn from_petgraph_digraph_as<W, N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> CoreResult<CsrGraph<W>>
where
    W: Scalar,
    Ix: IndexType,
{
    from_petgraph_impl::<W, N, E, Directed, Ix, fn(&E) -> W>(graph, true, false, None)
}

/// Converts a directed `petgraph::DiGraph` into a weighted directed `CsrGraph<W>`.
///
/// Rejects self-loops, duplicate directed edges, and non-finite edge weights.
pub fn from_petgraph_digraph_with_weights<W, N, E, Ix, F>(
    graph: &DiGraph<N, E, Ix>,
    weight_fn: F,
) -> CoreResult<CsrGraph<W>>
where
    W: Scalar,
    Ix: IndexType,
    F: FnMut(&E) -> W,
{
    from_petgraph_impl(graph, true, false, Some(weight_fn))
}

/// Converts a directed `petgraph::DiGraph` into an unweighted directed `CsrGraph<f64>`,
/// dropping self-loops and deduplicating parallel directed edges.
pub fn from_petgraph_digraph_sanitized<N, E, Ix: IndexType>(
    graph: &DiGraph<N, E, Ix>,
) -> CoreResult<CsrGraph<f64>> {
    from_petgraph_impl::<f64, N, E, Directed, Ix, fn(&E) -> f64>(graph, true, true, None)
}

/// Converts a directed `petgraph::DiGraph` into a weighted directed `CsrGraph<W>`,
/// dropping self-loops and retaining the maximum weight across parallel directed edges.
pub fn from_petgraph_digraph_sanitized_with_weights<W, N, E, Ix, F>(
    graph: &DiGraph<N, E, Ix>,
    weight_fn: F,
) -> CoreResult<CsrGraph<W>>
where
    W: Scalar,
    Ix: IndexType,
    F: FnMut(&E) -> W,
{
    from_petgraph_impl(graph, true, true, Some(weight_fn))
}

/// Converts an undirected [`CsrGraph<W>`] into a `petgraph::UnGraph<(), Option<W>>`.
///
/// Returns [`CoreError::UnsupportedOperation`] if `csr.is_directed()` is `true`.
pub fn to_petgraph_ungraph<W: Scalar>(csr: &CsrGraph<W>) -> CoreResult<UnGraph<(), Option<W>>> {
    to_petgraph_ungraph_with(csr, |_| (), |_, _, w| w)
}

/// Converts an undirected [`CsrGraph<W>`] into a `petgraph::UnGraph<N, E>` using custom node
/// and edge payload constructors.
///
/// Returns [`CoreError::UnsupportedOperation`] if `csr.is_directed()` is `true`.
pub fn to_petgraph_ungraph_with<W, N, E, FN, FE>(
    csr: &CsrGraph<W>,
    node_fn: FN,
    edge_fn: FE,
) -> CoreResult<UnGraph<N, E>>
where
    W: Scalar,
    FN: FnMut(usize) -> N,
    FE: FnMut(usize, usize, Option<W>) -> E,
{
    if csr.is_directed() {
        return Err(CoreError::unsupported(
            "cannot convert directed CsrGraph to petgraph::UnGraph; use to_petgraph_digraph",
        ));
    }
    to_petgraph_impl(csr, false, node_fn, edge_fn)
}

/// Converts a directed [`CsrGraph<W>`] into a `petgraph::DiGraph<(), Option<W>>`.
///
/// Returns [`CoreError::UnsupportedOperation`] if `csr.is_directed()` is `false`.
pub fn to_petgraph_digraph<W: Scalar>(csr: &CsrGraph<W>) -> CoreResult<DiGraph<(), Option<W>>> {
    to_petgraph_digraph_with(csr, |_| (), |_, _, w| w)
}

/// Converts a directed [`CsrGraph<W>`] into a `petgraph::DiGraph<N, E>` using custom node
/// and edge payload constructors.
///
/// Returns [`CoreError::UnsupportedOperation`] if `csr.is_directed()` is `false`.
pub fn to_petgraph_digraph_with<W, N, E, FN, FE>(
    csr: &CsrGraph<W>,
    node_fn: FN,
    edge_fn: FE,
) -> CoreResult<DiGraph<N, E>>
where
    W: Scalar,
    FN: FnMut(usize) -> N,
    FE: FnMut(usize, usize, Option<W>) -> E,
{
    if !csr.is_directed() {
        return Err(CoreError::unsupported(
            "cannot convert undirected CsrGraph to petgraph::DiGraph; use to_petgraph_ungraph",
        ));
    }
    to_petgraph_impl(csr, true, node_fn, edge_fn)
}

fn from_petgraph_impl<W, N, E, Ty, Ix, F>(
    graph: &Graph<N, E, Ty, Ix>,
    directed: bool,
    sanitize: bool,
    mut weight_fn: Option<F>,
) -> CoreResult<CsrGraph<W>>
where
    W: Scalar,
    Ty: EdgeType,
    Ix: IndexType,
    F: FnMut(&E) -> W,
{
    let num_nodes = graph.node_count();
    if u32::try_from(num_nodes).is_err() {
        return Err(CoreError::invalid_csr(
            "petgraph node count exceeds u32::MAX",
        ));
    }
    if graph.node_bound() != num_nodes {
        return Err(CoreError::invalid_csr(
            "petgraph node indices must be contiguous 0..node_count",
        ));
    }

    let mut edges = Vec::with_capacity(graph.edge_count());
    let mut weights = weight_fn
        .as_ref()
        .map(|_| Vec::with_capacity(graph.edge_count()));

    for edge in graph.edge_references() {
        let u = u32::try_from(edge.source().index())
            .map_err(|_| CoreError::invalid_csr("source vertex index exceeds u32::MAX"))?;
        let v = u32::try_from(edge.target().index())
            .map_err(|_| CoreError::invalid_csr("target vertex index exceeds u32::MAX"))?;
        edges.push((u, v));
        if let (Some(f), Some(ws)) = (&mut weight_fn, &mut weights) {
            ws.push(f(edge.weight()));
        }
    }

    CsrGraph::from_edges(num_nodes, &edges, weights.as_deref(), directed, sanitize)
}

fn to_petgraph_impl<W, N, E, Ty, FN, FE>(
    csr: &CsrGraph<W>,
    directed: bool,
    mut node_fn: FN,
    mut edge_fn: FE,
) -> CoreResult<Graph<N, E, Ty>>
where
    W: Scalar,
    Ty: EdgeType,
    FN: FnMut(usize) -> N,
    FE: FnMut(usize, usize, Option<W>) -> E,
{
    let num_nodes = csr.num_nodes();
    let mut pg = Graph::<N, E, Ty>::with_capacity(num_nodes, csr.num_edges());
    let nodes: Vec<NodeIndex> = (0..num_nodes).map(|u| pg.add_node(node_fn(u))).collect();

    for (u, &nu) in nodes.iter().enumerate() {
        let neighbors = csr.neighbors(u);
        let n_weights = csr.neighbor_weights(u);
        for (pos, &v_u32) in neighbors.iter().enumerate() {
            let v = v_u32 as usize;
            // Undirected CsrGraph stores symmetric pairs (u, v) and (v, u); emit each edge once.
            if !directed && u >= v {
                continue;
            }
            let w = n_weights.map(|ws| ws[pos]);
            if let Some(val) = w
                && !val.is_finite()
            {
                return Err(CoreError::non_finite("egraph_petgraph::to_petgraph"));
            }
            pg.add_edge(nu, nodes[v], edge_fn(u, v, w));
        }
    }

    Ok(pg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ungraph_roundtrip_unweighted_and_weighted() {
        let mut pg = UnGraph::<&str, f64>::new_undirected();
        let n0 = pg.add_node("a");
        let n1 = pg.add_node("b");
        let n2 = pg.add_node("c");
        let n3 = pg.add_node("isolated");
        let _ = n3;
        pg.add_edge(n0, n1, 1.5);
        pg.add_edge(n1, n2, 2.75);
        pg.add_edge(n0, n2, 0.5);

        let csr_unweighted = from_petgraph_ungraph(&pg).unwrap();
        assert_eq!(csr_unweighted.num_nodes(), 4);
        assert_eq!(csr_unweighted.num_edges(), 3);
        assert!(!csr_unweighted.is_directed());
        assert!(!csr_unweighted.is_weighted());
        assert_eq!(csr_unweighted.neighbors(3), &[]);

        let pg_back = to_petgraph_ungraph(&csr_unweighted).unwrap();
        assert_eq!(pg_back.node_count(), 4);
        assert_eq!(pg_back.edge_count(), 3);
        assert!(pg_back.edge_references().all(|e| e.weight().is_none()));
        let csr_back = from_petgraph_ungraph(&pg_back).unwrap();
        assert_eq!(csr_back, csr_unweighted);

        let extracted = extract_edge_weights(&pg, |w| *w).unwrap();
        assert_eq!(extracted, vec![1.5, 2.75, 0.5]);

        let csr_weighted = from_petgraph_ungraph_with_weights(&pg, |w| *w).unwrap();
        assert!(csr_weighted.is_weighted());
        assert_eq!(csr_weighted.neighbor_weights(0), Some(&[1.5, 0.5][..]));

        let pg_weighted = to_petgraph_ungraph(&csr_weighted).unwrap();
        let csr_weighted_back =
            from_petgraph_ungraph_with_weights(&pg_weighted, |w| w.unwrap()).unwrap();
        assert_eq!(csr_weighted_back, csr_weighted);

        let csr_f32: CsrGraph<f32> = from_petgraph_ungraph_as(&pg).unwrap();
        assert_eq!(csr_f32.num_edges(), 3);
        let pg_custom =
            to_petgraph_ungraph_with(&csr_weighted, |i| format!("v{i}"), |u, v, w| (u, v, w))
                .unwrap();
        assert_eq!(pg_custom[NodeIndex::new(2)], "v2");
        assert_eq!(pg_custom.edge_count(), 3);
    }

    #[test]
    fn test_digraph_roundtrip_and_mutual_edges() {
        let mut dg = DiGraph::<(), f32>::new();
        let a = dg.add_node(());
        let b = dg.add_node(());
        let c = dg.add_node(());
        // Mutual edges (0 -> 1) and (1 -> 0) are distinct in a directed graph.
        dg.add_edge(a, b, 1.25);
        dg.add_edge(b, a, 3.5);
        dg.add_edge(b, c, 2.0);

        let csr_unweighted = from_petgraph_digraph(&dg).unwrap();
        assert!(csr_unweighted.is_directed());
        assert_eq!(csr_unweighted.num_nodes(), 3);
        assert_eq!(csr_unweighted.num_edges(), 3);
        assert_eq!(csr_unweighted.neighbors(0), &[1]);
        assert_eq!(csr_unweighted.neighbors(1), &[0, 2]);
        assert_eq!(csr_unweighted.neighbors(2), &[]);

        let dg_back = to_petgraph_digraph(&csr_unweighted).unwrap();
        assert_eq!(dg_back.node_count(), 3);
        assert_eq!(dg_back.edge_count(), 3);
        let csr_back = from_petgraph_digraph(&dg_back).unwrap();
        assert_eq!(csr_back, csr_unweighted);

        let csr_w = from_petgraph_digraph_with_weights(&dg, |w| *w).unwrap();
        assert_eq!(csr_w.neighbor_weights(1), Some(&[3.5_f32, 2.0_f32][..]));
        let dg_w = to_petgraph_digraph(&csr_w).unwrap();
        let csr_w_back = from_petgraph_digraph_with_weights(&dg_w, |w| w.unwrap()).unwrap();
        assert_eq!(csr_w_back, csr_w);
    }

    #[test]
    fn test_empty_graphs_and_alternate_index_type() {
        let empty_un = UnGraph::<(), ()>::new_undirected();
        let csr_empty_un = from_petgraph_ungraph(&empty_un).unwrap();
        assert_eq!(csr_empty_un.num_nodes(), 0);
        assert_eq!(csr_empty_un.num_edges(), 0);
        let pg_empty_un = to_petgraph_ungraph(&csr_empty_un).unwrap();
        assert_eq!(pg_empty_un.node_count(), 0);
        assert_eq!(pg_empty_un.edge_count(), 0);

        let empty_di = DiGraph::<(), ()>::new();
        let csr_empty_di = from_petgraph_digraph(&empty_di).unwrap();
        assert_eq!(csr_empty_di.num_nodes(), 0);
        assert!(csr_empty_di.is_directed());
        let pg_empty_di = to_petgraph_digraph(&csr_empty_di).unwrap();
        assert_eq!(pg_empty_di.node_count(), 0);

        let mut u16_graph = UnGraph::<(), f64, u16>::default();
        let x = u16_graph.add_node(());
        let y = u16_graph.add_node(());
        u16_graph.add_edge(x, y, 4.0);
        let csr_u16 = from_petgraph_ungraph_with_weights(&u16_graph, |w| *w).unwrap();
        assert_eq!(csr_u16.num_nodes(), 2);
        assert_eq!(csr_u16.num_edges(), 1);
    }

    #[test]
    fn test_validation_errors_and_sanitization() {
        // 1. Self-loop rejected unless sanitized.
        let mut loop_un = UnGraph::<(), f64>::new_undirected();
        let n0 = loop_un.add_node(());
        let n1 = loop_un.add_node(());
        loop_un.add_edge(n0, n0, 5.0);
        loop_un.add_edge(n0, n1, 2.0);
        assert!(matches!(
            from_petgraph_ungraph(&loop_un),
            Err(CoreError::InvalidCsr(_))
        ));
        let sanitized_un = from_petgraph_ungraph_sanitized_with_weights(&loop_un, |w| *w).unwrap();
        assert_eq!(sanitized_un.num_edges(), 1);
        assert_eq!(sanitized_un.neighbor_weights(0), Some(&[2.0][..]));

        // 2. Parallel edges rejected unless sanitized.
        let mut parallel_un = UnGraph::<(), f64>::new_undirected();
        let a = parallel_un.add_node(());
        let b = parallel_un.add_node(());
        parallel_un.add_edge(a, b, 1.0);
        parallel_un.add_edge(b, a, 3.0);
        assert!(matches!(
            from_petgraph_ungraph(&parallel_un),
            Err(CoreError::InvalidCsr(_))
        ));
        let dedup_un = from_petgraph_ungraph_sanitized_with_weights(&parallel_un, |w| *w).unwrap();
        assert_eq!(dedup_un.num_edges(), 1);
        assert_eq!(dedup_un.neighbor_weights(0), Some(&[3.0][..]));

        let dedup_un_unweighted = from_petgraph_ungraph_sanitized(&parallel_un).unwrap();
        assert_eq!(dedup_un_unweighted.num_edges(), 1);

        // 3. Parallel directed edges rejected unless sanitized.
        let mut parallel_di = DiGraph::<(), f64>::new();
        let d0 = parallel_di.add_node(());
        let d1 = parallel_di.add_node(());
        parallel_di.add_edge(d0, d1, 1.0);
        parallel_di.add_edge(d0, d1, 4.0);
        assert!(matches!(
            from_petgraph_digraph(&parallel_di),
            Err(CoreError::InvalidCsr(_))
        ));
        let dedup_di = from_petgraph_digraph_sanitized_with_weights(&parallel_di, |w| *w).unwrap();
        assert_eq!(dedup_di.num_edges(), 1);
        assert_eq!(dedup_di.neighbor_weights(0), Some(&[4.0][..]));
        assert_eq!(
            from_petgraph_digraph_sanitized(&parallel_di)
                .unwrap()
                .num_edges(),
            1
        );

        // 4. Non-finite edge weights rejected.
        let mut nan_un = UnGraph::<(), f64>::new_undirected();
        let u0 = nan_un.add_node(());
        let u1 = nan_un.add_node(());
        nan_un.add_edge(u0, u1, f64::NAN);
        assert!(matches!(
            extract_edge_weights(&nan_un, |w| *w),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            from_petgraph_ungraph_with_weights(&nan_un, |w| *w),
            Err(CoreError::NonFiniteValue { .. })
        ));

        // 5. Directedness mismatch rejected on to_petgraph_*.
        let csr_un = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        assert!(matches!(
            to_petgraph_digraph(&csr_un),
            Err(CoreError::UnsupportedOperation(_))
        ));
        let csr_di = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, true, false).unwrap();
        assert!(matches!(
            to_petgraph_ungraph(&csr_di),
            Err(CoreError::UnsupportedOperation(_))
        ));
    }
}
