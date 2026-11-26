//! Edge-length evaluation and NumPy edge-list extraction helpers for Python graphs.

use super::{Edge, IndexType, Node};
use crate::FloatType;
use numpy::PyReadonlyArray2;
use petgraph::graph::node_index;
use petgraph::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

/// Pre-resolved edge lengths avoiding GIL acquisition inside graph traversal loops.
pub(crate) enum ResolvedEdgeLengths {
    Constant(FloatType),
    PerEdge(Vec<FloatType>),
}

impl ResolvedEdgeLengths {
    #[inline]
    pub(crate) fn get(&self, edge_idx: usize) -> FloatType {
        match self {
            Self::Constant(c) => *c,
            Self::PerEdge(v) => v[edge_idx],
        }
    }
}

/// Edge-length evaluator supporting `None` (`1.0`), a Python numeric constant,
/// or a Python callable accepting an edge index.
pub(crate) enum EdgeLength {
    Constant(FloatType),
    Callable(Py<PyAny>),
}

impl EdgeLength {
    pub(crate) fn from_py(f: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        match f {
            None => Ok(Self::Constant(1.0)),
            Some(obj) if obj.is_none() => Ok(Self::Constant(1.0)),
            Some(obj) => {
                if obj.is_instance_of::<PyBool>() {
                    return Err(PyTypeError::new_err(
                        "f must be None, a number, or a callable, not bool",
                    ));
                }
                if !obj.is_callable()
                    && let Ok(val) = obj.extract::<FloatType>()
                {
                    return Ok(Self::Constant(val));
                }
                if obj.is_callable() {
                    return Ok(Self::Callable(obj.clone().unbind()));
                }
                Err(PyTypeError::new_err(
                    "f must be None, a number, or a callable",
                ))
            }
        }
    }

    pub(crate) fn resolve_for_graph<Ty: petgraph::EdgeType>(
        &self,
        graph: &Graph<Node, Edge, Ty, IndexType>,
        require_non_negative: bool,
    ) -> PyResult<ResolvedEdgeLengths> {
        use petgraph::visit::{EdgeIndexable, EdgeRef};
        let validate = |val: FloatType| -> PyResult<FloatType> {
            if !val.is_finite() || (require_non_negative && val < 0.0) {
                return Err(PyValueError::new_err(if require_non_negative {
                    "edge length must be finite and non-negative"
                } else {
                    "edge length must be finite"
                }));
            }
            Ok(val)
        };
        match self {
            Self::Constant(c) => Ok(ResolvedEdgeLengths::Constant(validate(*c)?)),
            Self::Callable(func) => Python::attach(|py| {
                let mut lengths = vec![0.0 as FloatType; graph.edge_bound()];
                for e in graph.edge_references() {
                    let val: FloatType = func.call1(py, (e.id().index(),))?.extract(py)?;
                    lengths[e.id().index()] = validate(val)?;
                }
                Ok(ResolvedEdgeLengths::PerEdge(lengths))
            }),
        }
    }
}

fn convert_edges_2col<T>(arr: PyReadonlyArray2<'_, T>) -> PyResult<Vec<(usize, usize)>>
where
    T: Copy + numpy::Element + TryInto<usize>,
{
    let view = arr.as_array();
    let [m, cols] = [view.nrows(), view.ncols()];
    if cols != 2 {
        return Err(PyValueError::new_err(format!(
            "edges must have shape (M, 2), got ({}, {})",
            m, cols
        )));
    }
    let mut out = Vec::with_capacity(m);
    for row in view.rows() {
        let u = row[0].try_into().map_err(|_| {
            PyValueError::new_err("edge node indices must be non-negative and fit in usize")
        })?;
        let v = row[1].try_into().map_err(|_| {
            PyValueError::new_err("edge node indices must be non-negative and fit in usize")
        })?;
        out.push((u, v));
    }
    Ok(out)
}

pub(crate) fn extract_edges_2col(edges: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, usize)>> {
    if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, usize>>() {
        return convert_edges_2col(arr);
    }
    if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, i64>>() {
        return convert_edges_2col(arr);
    }
    if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, u64>>() {
        return convert_edges_2col(arr);
    }
    if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, i32>>() {
        return convert_edges_2col(arr);
    }
    if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, u32>>() {
        return convert_edges_2col(arr);
    }
    Err(PyTypeError::new_err(
        "edges must be a 2D integer numpy array of shape (M, 2) with dtype uintp/uint64/int64/int32/uint32",
    ))
}

pub(crate) fn build_graph_from_edges_numpy<Ty: petgraph::EdgeType>(
    py: Python<'_>,
    num_nodes: usize,
    edges: &Bound<'_, PyAny>,
) -> PyResult<Graph<Node, Edge, Ty, IndexType>> {
    if num_nodes > IndexType::MAX as usize {
        return Err(PyValueError::new_err(format!(
            "num_nodes ({}) exceeds maximum index capacity ({})",
            num_nodes,
            IndexType::MAX
        )));
    }
    let pairs = extract_edges_2col(edges)?;
    for &(u, v) in &pairs {
        if u >= num_nodes || v >= num_nodes {
            return Err(PyValueError::new_err(format!(
                "edge ({}, {}) out of bounds for graph with {} nodes",
                u, v, num_nodes
            )));
        }
    }
    let mut graph = Graph::<Node, Edge, Ty, IndexType>::with_capacity(num_nodes, pairs.len());
    for _ in 0..num_nodes {
        graph.add_node(py.None());
    }
    for (u, v) in pairs {
        graph.add_edge(
            node_index::<IndexType>(u),
            node_index::<IndexType>(v),
            py.None(),
        );
    }
    Ok(graph)
}

pub(super) fn add_edges_to_graph<Ty: petgraph::EdgeType>(
    py: Python<'_>,
    graph: &mut Graph<Node, Edge, Ty, IndexType>,
    pairs: &[(usize, usize)],
) -> PyResult<()> {
    for &(u, v) in pairs {
        if u > IndexType::MAX as usize
            || v > IndexType::MAX as usize
            || graph.node_weight(node_index::<IndexType>(u)).is_none()
            || graph.node_weight(node_index::<IndexType>(v)).is_none()
        {
            return Err(PyValueError::new_err(format!(
                "edge ({}, {}) references non-existent node",
                u, v
            )));
        }
    }
    graph.reserve_edges(pairs.len());
    for &(u, v) in pairs {
        graph.add_edge(
            node_index::<IndexType>(u),
            node_index::<IndexType>(v),
            py.None(),
        );
    }
    Ok(())
}
