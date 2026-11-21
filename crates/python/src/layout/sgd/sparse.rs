use crate::{
    distance_matrix::{
        with_distance, DistanceMatrixType, EdgeLengthFn, PyDistanceMatrix, PyPivotedNegLogDistance,
        PyPivotedNegLogSimDistance,
    },
    graph::{GraphType, PyGraphAdapter},
    layout::{
        knn::{extract_f64_array_2d, extract_index_array_1d_with_isize},
        sgd::PySgd,
    },
    rng::PyRng,
    FloatType,
};
use petgraph::visit::{EdgeRef, IntoNodeIdentifiers};
use petgraph_layout_sgd::{BarrierSymmetry, SparseSgd};
use pyo3::{prelude::*, types::PyDict};
use rand::SeedableRng;

fn parse_barrier_sym(value: &str) -> PyResult<BarrierSymmetry> {
    match value {
        "sym_max" => Ok(BarrierSymmetry::SymMax),
        "sym_avg" => Ok(BarrierSymmetry::SymAvg),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "barrier_sym must be 'sym_max' or 'sym_avg', got {value:?}"
        ))),
    }
}

fn validate_nonneg_finite(name: &str, value: f64) -> PyResult<f64> {
    if !value.is_finite() || value < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be finite and >= 0, got {value}"
        )));
    }
    Ok(value)
}

fn validate_pos_finite(name: &str, value: f64) -> PyResult<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be finite and > 0, got {value}"
        )));
    }
    Ok(value)
}

fn merge_edge_length_arg<'py>(
    f: Option<&Bound<'py, PyAny>>,
    length_fn: Option<&Bound<'py, PyAny>>,
) -> PyResult<EdgeLengthFn> {
    if f.is_some() && length_fn.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot specify both f and length_fn",
        ));
    }
    EdgeLengthFn::from_bound(length_fn.or(f))
}

fn collect_pivot_nodes(
    native_graph: &petgraph::graph::UnGraph<Py<PyAny>, Py<PyAny>, crate::graph::IndexType>,
    pivot: &[usize],
) -> PyResult<Vec<petgraph::graph::NodeIndex<crate::graph::IndexType>>> {
    let nodes = native_graph.node_identifiers().collect::<Vec<_>>();
    pivot
        .iter()
        .map(|&i| {
            nodes.get(i).copied().ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "Pivot index {i} out of bounds for graph with {} nodes",
                    nodes.len()
                ))
            })
        })
        .collect()
}

/// Sparse Stochastic Gradient Descent (SGD) layout builder.
#[pyclass]
#[pyo3(name = "SparseSgd")]
pub struct PySparseSgd {
    builder: SparseSgd,
}

#[pymethods]
impl PySparseSgd {
    /// Creates a new `SparseSgd` builder.
    #[new]
    fn new() -> Self {
        Self {
            builder: SparseSgd::new(),
        }
    }

    /// Creates a `SparseSgd` preset configured with 1-hop calibration, hub-safe 2-hop
    /// registration, relative-margin barrier spacing, and one-sided hinge repulsion.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        h = 64,
        calibrate_one_hop = true,
        max_2hop_per_node = 32,
        hinge_repulsion = true,
        delta_rel = 0.25,
        barrier_sym = "sym_max",
        omega_1 = 1.0,
        omega_2 = 1.0,
        gamma_non2 = 1.0,
        min_ratio = 0.5,
        eps_dist = 1e-6,
        seed = 42
    ))]
    pub fn barrier(
        h: usize,
        calibrate_one_hop: bool,
        max_2hop_per_node: usize,
        hinge_repulsion: bool,
        delta_rel: f64,
        barrier_sym: &str,
        omega_1: f64,
        omega_2: f64,
        gamma_non2: f64,
        min_ratio: f64,
        eps_dist: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let sym = parse_barrier_sym(barrier_sym)?;
        let mut builder = SparseSgd::barrier();
        builder
            .h(h)
            .calibrate_one_hop(calibrate_one_hop)
            .max_2hop_per_node(max_2hop_per_node)
            .hinge_repulsion(hinge_repulsion)
            .delta_rel(validate_nonneg_finite("delta_rel", delta_rel)?)
            .barrier_sym(sym)
            .omega_1(validate_nonneg_finite("omega_1", omega_1)?)
            .omega_2(validate_nonneg_finite("omega_2", omega_2)?)
            .gamma_non2(validate_nonneg_finite("gamma_non2", gamma_non2)?)
            .min_ratio(validate_nonneg_finite("min_ratio", min_ratio)?)
            .eps_dist(validate_pos_finite("eps_dist", eps_dist)?)
            .seed(seed);
        Ok(Self { builder })
    }

    /// Sets the number of pivot nodes `h`.
    fn h(mut slf: PyRefMut<Self>, h: usize) -> Py<Self> {
        slf.builder.h(h);
        slf.into()
    }

    /// Enables or disables 1-hop anchor calibration against the distance oracle.
    pub fn calibrate_one_hop(mut slf: PyRefMut<Self>, enabled: bool) -> Py<Self> {
        slf.builder.calibrate_one_hop(enabled);
        slf.into()
    }

    /// Sets the hub-safe maximum 2-hop pairs registered per node (`0` disables 2-hop pairs).
    pub fn max_2hop_per_node(mut slf: PyRefMut<Self>, cap: usize) -> Py<Self> {
        slf.builder.max_2hop_per_node(cap);
        slf.into()
    }

    /// Enables or disables one-sided hinge repulsion on Tier 3 barrier pairs.
    pub fn hinge_repulsion(mut slf: PyRefMut<Self>, enabled: bool) -> Py<Self> {
        slf.builder.hinge_repulsion(enabled);
        slf.into()
    }

    /// Sets the relative margin barrier multiplier `delta_rel >= 0`.
    pub fn delta_rel(mut slf: PyRefMut<Self>, delta_rel: f64) -> PyResult<Py<Self>> {
        slf.builder
            .delta_rel(validate_nonneg_finite("delta_rel", delta_rel)?);
        Ok(slf.into())
    }

    /// Sets the symmetrization rule (`"sym_max"` or `"sym_avg"`) for `R_pair(u, v)`.
    pub fn barrier_sym(mut slf: PyRefMut<Self>, barrier_sym: &str) -> PyResult<Py<Self>> {
        slf.builder.barrier_sym(parse_barrier_sym(barrier_sym)?);
        Ok(slf.into())
    }

    /// Sets the Tier 1 (1-hop) spring weight multiplier `omega_1 >= 0`.
    pub fn omega_1(mut slf: PyRefMut<Self>, omega_1: f64) -> PyResult<Py<Self>> {
        slf.builder
            .omega_1(validate_nonneg_finite("omega_1", omega_1)?);
        Ok(slf.into())
    }

    /// Sets the Tier 2 (2-hop) spring weight multiplier `omega_2 >= 0`.
    pub fn omega_2(mut slf: PyRefMut<Self>, omega_2: f64) -> PyResult<Py<Self>> {
        slf.builder
            .omega_2(validate_nonneg_finite("omega_2", omega_2)?);
        Ok(slf.into())
    }

    /// Sets the Tier 3 (non-2-hop) weight multiplier `gamma_non2 >= 0`.
    pub fn gamma_non2(mut slf: PyRefMut<Self>, gamma_non2: f64) -> PyResult<Py<Self>> {
        slf.builder
            .gamma_non2(validate_nonneg_finite("gamma_non2", gamma_non2)?);
        Ok(slf.into())
    }

    /// Sets the relative distance floor multiplier `min_ratio >= 0`.
    pub fn min_ratio(mut slf: PyRefMut<Self>, min_ratio: f64) -> PyResult<Py<Self>> {
        slf.builder
            .min_ratio(validate_nonneg_finite("min_ratio", min_ratio)?);
        Ok(slf.into())
    }

    /// Sets the minimum pairwise target distance clamp `eps_dist > 0`.
    pub fn eps_dist(mut slf: PyRefMut<Self>, eps_dist: f64) -> PyResult<Py<Self>> {
        slf.builder
            .eps_dist(validate_pos_finite("eps_dist", eps_dist)?);
        Ok(slf.into())
    }

    /// Sets the deterministic RNG seed used for 2-hop subsampling and non-2-hop sampling.
    pub fn seed(mut slf: PyRefMut<Self>, seed: u64) -> Py<Self> {
        slf.builder.seed(seed);
        slf.into()
    }

    /// Builds an `Sgd` schedule using random max-min pivots.
    ///
    /// Edge length is optional: passing `None` (or omitting `f`/`length_fn`) uses uniform `1.0`
    /// edge lengths without acquiring the GIL.
    #[pyo3(signature = (graph, f = None, rng = None, *, length_fn = None))]
    fn build(
        &self,
        graph: &PyGraphAdapter,
        f: Option<&Bound<PyAny>>,
        rng: Option<&Bound<PyRng>>,
        length_fn: Option<&Bound<PyAny>>,
    ) -> PyResult<PySgd> {
        let (edge_length, rng_opt) = if let Some(rng_bound) = rng {
            (
                merge_edge_length_arg(f, length_fn)?,
                Some(rng_bound.borrow_mut()),
            )
        } else if let (Some(first), None) = (f, length_fn) {
            if let Ok(rng_bound) = first.cast::<PyRng>() {
                (EdgeLengthFn::Constant(1.0), Some(rng_bound.borrow_mut()))
            } else {
                (EdgeLengthFn::from_bound(Some(first))?, None)
            }
        } else {
            (merge_edge_length_arg(f, length_fn)?, None)
        };

        let sgd = match graph.graph() {
            GraphType::Graph(native_graph) => {
                let len_closure =
                    |e: petgraph::graph::EdgeReference<Py<PyAny>>| edge_length.eval(e.id().index());
                match rng_opt {
                    Some(mut r) => self.builder.build(native_graph, len_closure, r.get_mut()),
                    None => {
                        let mut local = rand::rngs::StdRng::from_entropy();
                        self.builder.build(native_graph, len_closure, &mut local)
                    }
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unsupported graph type",
                ))
            }
        };
        Ok(PySgd::new_with_sgd(sgd))
    }

    /// Builds an `Sgd` schedule with user-specified pivot node indices.
    #[pyo3(signature = (graph, f = None, pivot = None, *, length_fn = None))]
    pub fn new_with_pivot(
        &self,
        graph: &PyGraphAdapter,
        f: Option<&Bound<PyAny>>,
        pivot: Option<Vec<usize>>,
        length_fn: Option<&Bound<PyAny>>,
    ) -> PyResult<PySgd> {
        let (edge_length, pivot_vec) = match (pivot, f, length_fn) {
            (Some(p), f_arg, lf_arg) => (merge_edge_length_arg(f_arg, lf_arg)?, p),
            (None, Some(first), lf_arg) => {
                let p = first.extract::<Vec<usize>>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("pivot argument is required")
                })?;
                (EdgeLengthFn::from_bound(lf_arg)?, p)
            }
            (None, None, _) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "pivot argument is required",
                ))
            }
        };

        let sgd = match graph.graph() {
            GraphType::Graph(native_graph) => {
                let pivot_nodes = collect_pivot_nodes(native_graph, &pivot_vec)?;
                self.builder.build_with_pivot(
                    native_graph,
                    |e| edge_length.eval(e.id().index()),
                    &pivot_nodes,
                )
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unsupported graph type",
                ))
            }
        };
        Ok(PySgd::new_with_sgd(sgd))
    }

    /// Builds an `Sgd` schedule using given pivot indices and a precomputed distance matrix.
    #[pyo3(signature = (graph, f = None, pivot = None, d = None, *, length_fn = None))]
    pub fn build_with_pivot_and_distance_matrix(
        &self,
        graph: &PyGraphAdapter,
        f: Option<&Bound<PyAny>>,
        pivot: Option<&Bound<PyAny>>,
        d: Option<&Bound<PyAny>>,
        length_fn: Option<&Bound<PyAny>>,
    ) -> PyResult<PySgd> {
        let (edge_length, pivot_vec, d_bound) = if let Some(d_arg) = d {
            let p = pivot
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("pivot is required"))?
                .extract::<Vec<usize>>()?;
            (merge_edge_length_arg(f, length_fn)?, p, d_arg)
        } else if let (Some(p_arg), Some(d_arg)) = (f, pivot) {
            if length_fn.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err("d is required"));
            }
            let p = p_arg.extract::<Vec<usize>>()?;
            (EdgeLengthFn::Constant(1.0), p, d_arg)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pivot and distance matrix d are required",
            ));
        };

        match graph.graph() {
            GraphType::Graph(native_graph) => {
                let pivot_nodes = collect_pivot_nodes(native_graph, &pivot_vec)?;
                let sgd = with_distance(d_bound, |distance| {
                    self.builder.build_with_pivot_and_distance_matrix(
                        native_graph,
                        |e| edge_length.eval(e.id().index()),
                        &pivot_nodes,
                        distance,
                    )
                })?;
                Ok(PySgd::new_with_sgd(sgd))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "unsupported graph type",
            )),
        }
    }

    /// Builds an `Sgd` schedule using a pivoted distance matrix (`PivotedNegLogSimDistance`,
    /// `PivotedNegLogDistance`, or pivoted `DistanceMatrix`).
    #[pyo3(signature = (graph, f = None, d = None, *, length_fn = None))]
    pub fn build_with_pivoted_distance(
        &self,
        graph: &PyGraphAdapter,
        f: Option<&Bound<PyAny>>,
        d: Option<&Bound<PyAny>>,
        length_fn: Option<&Bound<PyAny>>,
    ) -> PyResult<PySgd> {
        let (edge_length, d_bound) = if let Some(d_arg) = d {
            (merge_edge_length_arg(f, length_fn)?, d_arg)
        } else if let Some(d_arg) = f {
            (EdgeLengthFn::from_bound(length_fn)?, d_arg)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pivoted distance matrix d is required",
            ));
        };

        let GraphType::Graph(native_graph) = graph.graph() else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "unsupported graph type",
            ));
        };
        let len_closure = |e: petgraph::graph::EdgeReference<Py<PyAny>>| -> FloatType {
            edge_length.eval(e.id().index())
        };

        if let Ok(dm) = d_bound.extract::<PyRef<PyPivotedNegLogSimDistance>>() {
            let sgd = self.builder.build_with_pivoted_neg_log_sim_distance(
                native_graph,
                len_closure,
                &dm.matrix,
            );
            Ok(PySgd::new_with_sgd(sgd))
        } else if let Ok(dm) = d_bound.extract::<PyRef<PyPivotedNegLogDistance>>() {
            let sgd = self.builder.build_with_pivoted_neg_log_distance(
                native_graph,
                len_closure,
                &dm.matrix,
            );
            Ok(PySgd::new_with_sgd(sgd))
        } else if let Ok(dm) = d_bound.extract::<PyRef<PyDistanceMatrix>>() {
            match dm.distance_matrix() {
                DistanceMatrixType::Pivoted(pivoted_dm) => {
                    let sgd = self.builder.build_with_pivoted_distance_matrix(
                        native_graph,
                        len_closure,
                        pivoted_dm,
                    );
                    Ok(PySgd::new_with_sgd(sgd))
                }
                _ => Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected a pivoted distance matrix",
                )),
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected PivotedNegLogSimDistance, PivotedNegLogDistance, or PivotedDistanceMatrix",
            ))
        }
    }

    /// Builds a 3-tier Sparse Barrier `Sgd` schedule directly from `(N, k_lm)` Varadhan landmark tables.
    ///
    /// :param graph: Undirected input graph
    /// :type graph: Graph
    /// :param d_ms: 2D float64 array of shape `(n_nodes, k_lm)` with multi-scale landmark distances
    /// :type d_ms: numpy.ndarray
    /// :param d_coarse: Optional 2D float64 array of shape `(n_nodes, k_lm)` with coarse scaffold distances
    /// :type d_coarse: numpy.ndarray or None
    /// :param pivots: Optional 1D integer array of pivot node indices for global repulsion coverage
    /// :type pivots: numpy.ndarray or None
    /// :param neg_per_node: Optional explicit non-2-hop pairs per node (`None` uses adaptive 2-hop perplexity in `[8, 32]`)
    /// :type neg_per_node: int or None
    /// :return: Tuple `(sgd, stats)` where `stats` is a dict of pair counts and calibrated parameters
    /// :rtype: tuple[Sgd, dict]
    #[pyo3(signature = (graph, d_ms, d_coarse=None, pivots=None, neg_per_node=None))]
    pub fn build_barrier_from_landmark_tables(
        &self,
        py: Python<'_>,
        graph: &PyGraphAdapter,
        d_ms: &Bound<'_, PyAny>,
        d_coarse: Option<&Bound<'_, PyAny>>,
        pivots: Option<&Bound<'_, PyAny>>,
        neg_per_node: Option<usize>,
    ) -> PyResult<(PySgd, Py<PyDict>)> {
        let GraphType::Graph(native_graph) = graph.graph() else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "unsupported graph type",
            ));
        };
        let d_ms_arr = extract_f64_array_2d(d_ms, "d_ms")?;
        let d_coarse_arr = match d_coarse {
            Some(dc) => Some(extract_f64_array_2d(dc, "d_coarse")?),
            None => None,
        };
        let pivots_vec = match pivots {
            Some(p) => extract_index_array_1d_with_isize(p, "pivots")?,
            None => Vec::new(),
        };

        let (sgd, stats) = self
            .builder
            .build_barrier_from_landmark_tables(
                native_graph,
                d_ms_arr.as_array(),
                d_coarse_arr.as_ref().map(|a| a.as_array()),
                &pivots_vec,
                neg_per_node,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let dict = PyDict::new(py);
        dict.set_item("n_pairs_1hop", stats.n_pairs_1hop)?;
        dict.set_item("n_pairs_2hop", stats.n_pairs_2hop)?;
        dict.set_item("n_pairs_non2", stats.n_pairs_non2)?;
        dict.set_item("n_pairs_hinge", stats.n_pairs_hinge)?;
        dict.set_item("n_pairs_two_sided", stats.n_pairs_two_sided)?;
        dict.set_item("perplexity_val", stats.perplexity_val)?;
        dict.set_item("per_node_neg_used", stats.per_node_neg_used)?;
        dict.set_item("perplexity_source", stats.perplexity_source)?;
        dict.set_item("d_floor", stats.d_floor)?;
        Ok((PySgd::new_with_sgd(sgd), dict.unbind()))
    }
}
