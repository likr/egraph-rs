//! Frozen result types for `CoarseBackbone` layout generation.

use egraph::clustering::CoarseLouvainConfig;
use egraph::layout::{
    CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats, CoarseDistanceBackend,
    CoarseEdgeLengthMode,
};
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, unsupported_operation_err};
use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, normalize_enum_key};
use crate::layout_buffers::PyLayout2d;

/// Diagnostic telemetry produced by `CoarseBackbone` initialization.
#[pyclass(
    name = "CoarseBackboneStats",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyCoarseBackboneStats {
    inner: CoarseBackboneStats,
}

impl PyCoarseBackboneStats {
    pub(crate) fn from_core(inner: CoarseBackboneStats) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCoarseBackboneStats {
    #[getter]
    fn num_fine_nodes(&self) -> usize {
        self.inner.num_fine_nodes
    }
    #[getter]
    fn num_coarse_nodes(&self) -> usize {
        self.inner.num_coarse_nodes
    }
    #[getter]
    fn num_coarse_edges(&self) -> usize {
        self.inner.num_coarse_edges
    }
    #[getter]
    fn louvain_levels(&self) -> usize {
        self.inner.louvain_levels
    }
    #[getter]
    fn mean_cluster_size(&self) -> f64 {
        self.inner.mean_cluster_size
    }
    #[getter]
    fn max_cluster_size(&self) -> usize {
        self.inner.max_cluster_size
    }
    #[getter]
    fn median_l_edge(&self) -> f64 {
        self.inner.median_l_edge
    }
    #[getter]
    fn max_radius(&self) -> f64 {
        self.inner.max_radius
    }
    #[getter]
    fn coarse_overlap_pairs_initial(&self) -> usize {
        self.inner.coarse_overlap_pairs_initial
    }
    #[getter]
    fn coarse_overlap_pairs_final(&self) -> usize {
        self.inner.coarse_overlap_pairs_final
    }
    #[getter]
    fn scale_factor_applied(&self) -> f64 {
        self.inner.scale_factor_applied
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("num_fine_nodes", self.inner.num_fine_nodes)?;
        dict.set_item("num_coarse_nodes", self.inner.num_coarse_nodes)?;
        dict.set_item("num_coarse_edges", self.inner.num_coarse_edges)?;
        dict.set_item("louvain_levels", self.inner.louvain_levels)?;
        dict.set_item("mean_cluster_size", self.inner.mean_cluster_size)?;
        dict.set_item("max_cluster_size", self.inner.max_cluster_size)?;
        dict.set_item("median_l_edge", self.inner.median_l_edge)?;
        dict.set_item("max_radius", self.inner.max_radius)?;
        dict.set_item(
            "coarse_overlap_pairs_initial",
            self.inner.coarse_overlap_pairs_initial,
        )?;
        dict.set_item(
            "coarse_overlap_pairs_final",
            self.inner.coarse_overlap_pairs_final,
        )?;
        dict.set_item("scale_factor_applied", self.inner.scale_factor_applied)?;
        Ok(dict.unbind())
    }

    fn keys(&self) -> Vec<&'static str> {
        vec![
            "num_fine_nodes",
            "num_coarse_nodes",
            "num_coarse_edges",
            "louvain_levels",
            "mean_cluster_size",
            "max_cluster_size",
            "median_l_edge",
            "max_radius",
            "coarse_overlap_pairs_initial",
            "coarse_overlap_pairs_final",
            "scale_factor_applied",
        ]
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "num_fine_nodes" => self.inner.num_fine_nodes.into_py_any(py),
            "num_coarse_nodes" => self.inner.num_coarse_nodes.into_py_any(py),
            "num_coarse_edges" => self.inner.num_coarse_edges.into_py_any(py),
            "louvain_levels" => self.inner.louvain_levels.into_py_any(py),
            "mean_cluster_size" => self.inner.mean_cluster_size.into_py_any(py),
            "max_cluster_size" => self.inner.max_cluster_size.into_py_any(py),
            "median_l_edge" => self.inner.median_l_edge.into_py_any(py),
            "max_radius" => self.inner.max_radius.into_py_any(py),
            "coarse_overlap_pairs_initial" => {
                self.inner.coarse_overlap_pairs_initial.into_py_any(py)
            }
            "coarse_overlap_pairs_final" => self.inner.coarse_overlap_pairs_final.into_py_any(py),
            "scale_factor_applied" => self.inner.scale_factor_applied.into_py_any(py),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "num_fine_nodes"
                | "num_coarse_nodes"
                | "num_coarse_edges"
                | "louvain_levels"
                | "mean_cluster_size"
                | "max_cluster_size"
                | "median_l_edge"
                | "max_radius"
                | "coarse_overlap_pairs_initial"
                | "coarse_overlap_pairs_final"
                | "scale_factor_applied"
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "CoarseBackboneStats(fine={}, coarse={}, edges={}, levels={})",
            self.inner.num_fine_nodes,
            self.inner.num_coarse_nodes,
            self.inner.num_coarse_edges,
            self.inner.louvain_levels
        )
    }
}

/// Output produced by `CoarseBackbone.generate()` and `coarse_backbone_init()`.
#[pyclass(name = "CoarseBackboneResult", module = "egraph._core", frozen)]
pub struct PyCoarseBackboneResult {
    fine_layout: Py<PyLayout2d>,
    coarse_layout: Py<PyLayout2d>,
    coarse_graph: Py<PyCsrGraph>,
    node_to_coarse: Py<PyArray1<u32>>,
    coarse_sizes: Py<PyArray1<u32>>,
    radii: Py<PyArray1<f64>>,
    stats: Py<PyCoarseBackboneStats>,
}

impl PyCoarseBackboneResult {
    pub(crate) fn from_f64(py: Python<'_>, res: CoarseBackboneResult<f64>) -> PyResult<Self> {
        Ok(Self {
            fine_layout: Py::new(py, PyLayout2d::from_f64_layout(&res.fine_layout))?,
            coarse_layout: Py::new(py, PyLayout2d::from_f64_layout(&res.coarse_layout))?,
            coarse_graph: Py::new(py, PyCsrGraph::from_csr_graph(res.hierarchy.coarsest.graph))?,
            node_to_coarse: PyArray1::from_vec(py, res.hierarchy.coarsest.node_to_coarse).unbind(),
            coarse_sizes: PyArray1::from_vec(py, res.hierarchy.coarsest.coarse_sizes).unbind(),
            radii: PyArray1::from_vec(py, res.radii).unbind(),
            stats: Py::new(py, PyCoarseBackboneStats::from_core(res.stats))?,
        })
    }

    pub(crate) fn from_f32(py: Python<'_>, res: CoarseBackboneResult<f32>) -> PyResult<Self> {
        Ok(Self {
            fine_layout: Py::new(py, PyLayout2d::from_f32_layout(&res.fine_layout))?,
            coarse_layout: Py::new(py, PyLayout2d::from_f32_layout(&res.coarse_layout))?,
            coarse_graph: Py::new(py, PyCsrGraph::from_csr_graph(res.hierarchy.coarsest.graph))?,
            node_to_coarse: PyArray1::from_vec(py, res.hierarchy.coarsest.node_to_coarse).unbind(),
            coarse_sizes: PyArray1::from_vec(py, res.hierarchy.coarsest.coarse_sizes).unbind(),
            radii: PyArray1::from_vec(py, res.radii).unbind(),
            stats: Py::new(py, PyCoarseBackboneStats::from_core(res.stats))?,
        })
    }
}

#[pymethods]
impl PyCoarseBackboneResult {
    #[getter]
    fn fine_layout(&self, py: Python<'_>) -> Py<PyLayout2d> {
        self.fine_layout.clone_ref(py)
    }
    #[getter]
    fn coarse_layout(&self, py: Python<'_>) -> Py<PyLayout2d> {
        self.coarse_layout.clone_ref(py)
    }
    #[getter]
    fn coarse_graph(&self, py: Python<'_>) -> Py<PyCsrGraph> {
        self.coarse_graph.clone_ref(py)
    }
    #[getter]
    fn node_to_coarse(&self, py: Python<'_>) -> Py<PyArray1<u32>> {
        self.node_to_coarse.clone_ref(py)
    }
    #[getter]
    fn coarse_sizes(&self, py: Python<'_>) -> Py<PyArray1<u32>> {
        self.coarse_sizes.clone_ref(py)
    }
    #[getter]
    fn radii(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.radii.clone_ref(py)
    }
    #[getter]
    fn stats(&self, py: Python<'_>) -> Py<PyCoarseBackboneStats> {
        self.stats.clone_ref(py)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("fine_layout", self.fine_layout.clone_ref(py))?;
        dict.set_item("coarse_layout", self.coarse_layout.clone_ref(py))?;
        dict.set_item("coarse_graph", self.coarse_graph.clone_ref(py))?;
        dict.set_item("node_to_coarse", self.node_to_coarse.clone_ref(py))?;
        dict.set_item("coarse_sizes", self.coarse_sizes.clone_ref(py))?;
        dict.set_item("radii", self.radii.clone_ref(py))?;
        dict.set_item("stats", self.stats.bind(py).borrow().to_dict(py)?)?;
        Ok(dict.unbind())
    }

    fn keys(&self) -> Vec<&'static str> {
        vec![
            "fine_layout",
            "coarse_layout",
            "coarse_graph",
            "node_to_coarse",
            "coarse_sizes",
            "radii",
            "stats",
        ]
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "fine_layout" => Ok(self.fine_layout.clone_ref(py).into_any()),
            "coarse_layout" => Ok(self.coarse_layout.clone_ref(py).into_any()),
            "coarse_graph" => Ok(self.coarse_graph.clone_ref(py).into_any()),
            "node_to_coarse" => Ok(self.node_to_coarse.clone_ref(py).into_any()),
            "coarse_sizes" => Ok(self.coarse_sizes.clone_ref(py).into_any()),
            "radii" => Ok(self.radii.clone_ref(py).into_any()),
            "stats" => Ok(self.stats.clone_ref(py).into_any()),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "fine_layout"
                | "coarse_layout"
                | "coarse_graph"
                | "node_to_coarse"
                | "coarse_sizes"
                | "radii"
                | "stats"
        )
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let st = self.stats.bind(py).borrow();
        format!(
            "CoarseBackboneResult(fine_nodes={}, coarse_nodes={})",
            st.inner.num_fine_nodes, st.inner.num_coarse_nodes
        )
    }
}

pub(super) fn parse_coarse_edge_length_mode(raw: &str) -> Result<CoarseEdgeLengthMode, String> {
    match normalize_enum_key(raw).as_str() {
        "conductanceinverse" | "conductance" | "inverseconductance" => {
            Ok(CoarseEdgeLengthMode::ConductanceInverse)
        }
        "resistancecurvatureflow" | "resistancecurvature" | "rcf" => {
            Ok(CoarseEdgeLengthMode::ResistanceCurvatureFlow)
        }
        _ => Err(format!(
            "unknown edge_length_mode {raw:?}; expected 'conductance_inverse' or 'resistance_curvature_flow'"
        )),
    }
}

pub(super) fn parse_coarse_distance_backend(raw: &str) -> Result<CoarseDistanceBackend, String> {
    match normalize_enum_key(raw).as_str() {
        "rhgh" => Ok(CoarseDistanceBackend::Rhgh),
        "chebyshevdog" | "dog" => Ok(CoarseDistanceBackend::ChebyshevDog),
        "shortestpath" | "sp" => Ok(CoarseDistanceBackend::ShortestPath),
        _ => Err(format!(
            "unknown distance_backend {raw:?}; expected 'rhgh', 'chebyshev_dog', or 'shortest_path'"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_coarse_backbone_config(
    max_coarse_nodes: StrictUsize,
    resolution: StrictF64,
    resolution_decay: StrictF64,
    max_passes: StrictUsize,
    min_modularity_gain: StrictF64,
    max_outer_rounds: StrictUsize,
    edge_length_mode: &str,
    distance_backend: &str,
    alpha: StrictF64,
    c0: StrictF64,
    kappa: StrictF64,
    gamma_w: StrictF64,
    gamma_m: StrictF64,
    mu_ov: StrictF64,
    m_gap: StrictF64,
    coarse_pivots: StrictUsize,
    coarse_neg_per_node: StrictUsize,
    coarse_t_max: StrictUsize,
    rhgh_gamma: Option<StrictF64>,
    rhgh_p: Option<StrictF64>,
    rcf_steps: StrictUsize,
    rcf_eta: StrictF64,
    jitter_scale: StrictF64,
    gamma_tether: StrictF64,
    target_std: Option<StrictF64>,
    robust_quantile: StrictF64,
    seed: StrictU64,
) -> PyResult<CoarseBackboneConfig> {
    let mode =
        parse_coarse_edge_length_mode(edge_length_mode).map_err(unsupported_operation_err)?;
    let backend =
        parse_coarse_distance_backend(distance_backend).map_err(unsupported_operation_err)?;
    let louvain = CoarseLouvainConfig::new()
        .with_max_coarse_nodes(max_coarse_nodes.0)
        .with_resolution(resolution.0)
        .with_resolution_decay(resolution_decay.0)
        .with_max_passes(max_passes.0)
        .with_min_modularity_gain(min_modularity_gain.0)
        .with_max_outer_rounds(max_outer_rounds.0)
        .with_seed(seed.0);
    let cfg = CoarseBackboneConfig::new()
        .with_louvain(louvain)
        .with_edge_length_mode(mode)
        .with_distance_backend(backend)
        .with_alpha(alpha.0)
        .with_c0(c0.0)
        .with_kappa(kappa.0)
        .with_gamma_w(gamma_w.0)
        .with_gamma_m(gamma_m.0)
        .with_mu_ov(mu_ov.0)
        .with_m_gap(m_gap.0)
        .with_coarse_pivots(coarse_pivots.0)
        .with_coarse_neg_per_node(coarse_neg_per_node.0)
        .with_coarse_t_max(coarse_t_max.0)
        .with_rhgh_gamma(rhgh_gamma.map(|v| v.0))
        .with_rhgh_p(rhgh_p.map(|v| v.0))
        .with_rcf_steps(rcf_steps.0)
        .with_rcf_eta(rcf_eta.0)
        .with_jitter_scale(jitter_scale.0)
        .with_gamma_tether(gamma_tether.0)
        .with_target_std(target_std.map(|v| v.0))
        .with_robust_quantile(robust_quantile.0)
        .with_seed(seed.0);
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCoarseBackboneStats>()?;
    m.add_class::<PyCoarseBackboneResult>()?;
    Ok(())
}
