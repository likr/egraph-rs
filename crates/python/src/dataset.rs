//! Built-in dataset catalog, metadata lookup, and synthetic graph generator bindings.

use egraph::core::{CoreError, CoreResult, CsrGraph};
use egraph::dataset as egraph_dataset;
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::array_extract::extract_usize_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, core_error_to_py};

/// Static metadata describing a built-in embedded graph dataset.
#[pyclass(name = "DatasetInfo", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyDatasetInfo {
    name: &'static str,
    num_nodes: usize,
    num_edges: usize,
    source: &'static str,
    description: &'static str,
}

impl From<&egraph_dataset::DatasetInfo> for PyDatasetInfo {
    fn from(info: &egraph_dataset::DatasetInfo) -> Self {
        Self {
            name: info.name,
            num_nodes: info.num_nodes,
            num_edges: info.num_edges,
            source: info.source,
            description: info.description,
        }
    }
}

#[pymethods]
impl PyDatasetInfo {
    /// Canonical dataset identifier accepted by `load_dataset`.
    #[getter]
    fn name(&self) -> &'static str {
        self.name
    }

    /// Number of vertices in the dataset.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Number of undirected edges in the dataset.
    #[getter]
    fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Source collection or benchmark origin.
    #[getter]
    fn source(&self) -> &'static str {
        self.source
    }

    /// Concise topological summary of the dataset.
    #[getter]
    fn description(&self) -> &'static str {
        self.description
    }

    /// Return a concise string representation of the `DatasetInfo`.
    fn __repr__(&self) -> String {
        format!(
            "DatasetInfo(name='{}', num_nodes={}, num_edges={})",
            self.name, self.num_nodes, self.num_edges
        )
    }
}

/// Return the canonical names of all built-in embedded datasets.
#[pyfunction]
#[pyo3(signature = (), text_signature = "()")]
fn list_datasets() -> Vec<&'static str> {
    egraph_dataset::list_datasets().to_vec()
}

/// Return static metadata records for all built-in embedded datasets.
#[pyfunction]
#[pyo3(signature = (), text_signature = "()")]
fn dataset_catalog() -> Vec<PyDatasetInfo> {
    egraph_dataset::dataset_catalog()
        .iter()
        .map(PyDatasetInfo::from)
        .collect()
}

/// Look up static metadata for a built-in dataset by canonical name.
#[pyfunction]
#[pyo3(signature = (name), text_signature = "(name)")]
fn dataset_info(name: &str) -> PyResult<PyDatasetInfo> {
    egraph_dataset::dataset_info(name)
        .map(PyDatasetInfo::from)
        .ok_or_else(|| {
            core_error_to_py(CoreError::invalid_input(format!(
                "unknown dataset {name:?}; expected one of {:?}",
                egraph_dataset::list_datasets()
            )))
        })
}

/// Load a built-in SuiteSparse / classic benchmark dataset as an undirected `CsrGraph`.
#[pyfunction]
#[pyo3(signature = (name), text_signature = "(name)")]
fn load_dataset(py: Python<'_>, name: &str) -> PyResult<PyCsrGraph> {
    let key: String = name
        .trim()
        .chars()
        .filter(|c| *c != '_' && *c != '-' && *c != ' ')
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let csr = py
        .detach(|| -> CoreResult<CsrGraph<f64>> {
            match key.as_str() {
                "1138bus" => Ok(egraph_dataset::dataset_1138_bus_csr()),
                "3elt" => Ok(egraph_dataset::dataset_3_elt_csr()),
                "dwt1005" => Ok(egraph_dataset::dataset_dwt_1005_csr()),
                "dwt2680" => Ok(egraph_dataset::dataset_dwt_2680_csr()),
                "poli" => Ok(egraph_dataset::dataset_poli_csr()),
                "qh882" => Ok(egraph_dataset::dataset_qh882_csr()),
                "uspowergrid" => Ok(egraph_dataset::dataset_uspower_grid_csr()),
                "karate" => Ok(egraph_dataset::dataset_karate_csr()),
                "lesmis" => Ok(egraph_dataset::dataset_lesmis_csr()),
                _ => Err(CoreError::invalid_input(format!(
                    "unknown dataset {name:?}; expected one of '1138_bus', '3elt', 'dwt_1005', 'dwt_2680', 'poli', 'qh882', 'USpowerGrid', 'karate', 'lesmis'"
                ))),
            }
        })
        .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Parse an edge-list CSV string (`num_nodes,num_edges` header followed by `u,v` or `u,v,w` lines) into a `CsrGraph`.
#[pyfunction]
#[pyo3(
    signature = (csv, *, directed = false, sanitize = true),
    text_signature = "(csv, *, directed=False, sanitize=True)"
)]
fn parse_csr_csv(
    py: Python<'_>,
    csv: &str,
    directed: bool,
    sanitize: bool,
) -> PyResult<PyCsrGraph> {
    let csr = py
        .detach(|| egraph_dataset::parse_csr_csv(csv, directed, sanitize))
        .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Generate a deterministic undirected weighted graph on `num_nodes` vertices.
#[pyfunction]
#[pyo3(
    signature = (num_nodes, avg_degree = StrictUsize(4), min_weight = StrictF64(0.1), max_weight = StrictF64(2.0), seed = StrictU64(42)),
    text_signature = "(num_nodes, avg_degree=4, min_weight=0.1, max_weight=2.0, seed=42)"
)]
fn generate_weighted_graph(
    py: Python<'_>,
    num_nodes: StrictUsize,
    avg_degree: StrictUsize,
    min_weight: StrictF64,
    max_weight: StrictF64,
    seed: StrictU64,
) -> PyResult<PyCsrGraph> {
    let csr = py
        .detach(|| {
            egraph_dataset::generate_weighted_graph(
                num_nodes.0,
                avg_degree.0,
                min_weight.0,
                max_weight.0,
                seed.0,
            )
        })
        .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Generate a deterministic layered directed acyclic graph (DAG).
#[pyfunction]
#[pyo3(
    signature = (num_layers, nodes_per_layer, edge_prob = StrictF64(0.5), seed = StrictU64(42)),
    text_signature = "(num_layers, nodes_per_layer, edge_prob=0.5, seed=42)"
)]
fn generate_layered_dag(
    py: Python<'_>,
    num_layers: StrictUsize,
    nodes_per_layer: StrictUsize,
    edge_prob: StrictF64,
    seed: StrictU64,
) -> PyResult<PyCsrGraph> {
    let csr = py
        .detach(|| {
            egraph_dataset::generate_layered_dag(
                num_layers.0,
                nodes_per_layer.0,
                edge_prob.0,
                seed.0,
            )
        })
        .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Generate a deterministic Stochastic Block Model graph and ground-truth community labels.
#[pyfunction]
#[pyo3(
    signature = (block_sizes, p_intra = StrictF64(0.3), p_inter = StrictF64(0.02), seed = StrictU64(42)),
    text_signature = "(block_sizes, p_intra=0.3, p_inter=0.02, seed=42)"
)]
fn generate_sbm_graph<'py>(
    py: Python<'py>,
    block_sizes: &Bound<'_, PyAny>,
    p_intra: StrictF64,
    p_inter: StrictF64,
    seed: StrictU64,
) -> PyResult<(PyCsrGraph, Bound<'py, PyArray1<u32>>)> {
    let sizes = extract_usize_1d(block_sizes, "block_sizes")?;
    let (csr, labels) = py
        .detach(|| egraph_dataset::generate_sbm_graph(&sizes, p_intra.0, p_inter.0, seed.0))
        .map_err(core_error_to_py)?;
    Ok((
        PyCsrGraph::from_csr_graph(csr),
        PyArray1::from_vec(py, labels),
    ))
}

/// Generate a periodic 2D torus grid graph of size `rows x cols`.
#[pyfunction]
#[pyo3(
    signature = (rows, cols, *, triangulated = false),
    text_signature = "(rows, cols, *, triangulated=False)"
)]
fn generate_torus_grid(
    py: Python<'_>,
    rows: StrictUsize,
    cols: StrictUsize,
    triangulated: bool,
) -> PyResult<PyCsrGraph> {
    let csr = py
        .detach(|| egraph_dataset::generate_torus_grid(rows.0, cols.0, triangulated))
        .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Generate a closed genus-0 triangulated spherical icosphere with `subdivisions` refinements.
#[pyfunction]
#[pyo3(signature = (subdivisions = StrictUsize(2)), text_signature = "(subdivisions=2)")]
fn generate_spherical_icosphere(py: Python<'_>, subdivisions: StrictUsize) -> PyResult<PyCsrGraph> {
    let csr = py
        .detach(|| egraph_dataset::generate_spherical_icosphere(subdivisions.0))
        .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Generate a closed UV spherical grid graph with `latitudes` rings of `longitudes` vertices.
#[pyfunction]
#[pyo3(signature = (latitudes, longitudes), text_signature = "(latitudes, longitudes)")]
fn generate_spherical_grid(
    py: Python<'_>,
    latitudes: StrictUsize,
    longitudes: StrictUsize,
) -> PyResult<PyCsrGraph> {
    let csr = py
        .detach(|| egraph_dataset::generate_spherical_grid(latitudes.0, longitudes.0))
        .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Generate a regular hyperbolic branching tree with `branching_factor` and `depth`.
#[pyfunction]
#[pyo3(
    signature = (branching_factor = StrictUsize(2), depth = StrictUsize(4)),
    text_signature = "(branching_factor=2, depth=4)"
)]
fn generate_hyperbolic_tree(
    py: Python<'_>,
    branching_factor: StrictUsize,
    depth: StrictUsize,
) -> PyResult<PyCsrGraph> {
    let csr = py
        .detach(|| egraph_dataset::generate_hyperbolic_tree(branching_factor.0, depth.0))
        .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Register dataset catalog and synthetic graph generator bindings on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatasetInfo>()?;
    m.add_function(wrap_pyfunction!(list_datasets, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_info, m)?)?;
    m.add_function(wrap_pyfunction!(load_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(parse_csr_csv, m)?)?;
    m.add_function(wrap_pyfunction!(generate_weighted_graph, m)?)?;
    m.add_function(wrap_pyfunction!(generate_layered_dag, m)?)?;
    m.add_function(wrap_pyfunction!(generate_sbm_graph, m)?)?;
    m.add_function(wrap_pyfunction!(generate_torus_grid, m)?)?;
    m.add_function(wrap_pyfunction!(generate_spherical_icosphere, m)?)?;
    m.add_function(wrap_pyfunction!(generate_spherical_grid, m)?)?;
    m.add_function(wrap_pyfunction!(generate_hyperbolic_tree, m)?)?;
    Ok(())
}
