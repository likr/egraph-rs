//! Frozen structured result classes for `egraph` quality metrics.

use egraph::metrics::{
    CrossingSummary, LayoutDegeneracy, MinimumAngleSummary, RtaEvaluation, StressEvaluation,
};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Structured layout degeneracy diagnostics returned by `layout_degeneracy`.
#[pyclass(
    name = "LayoutDegeneracy",
    module = "egraph._core",
    frozen,
    eq,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyLayoutDegeneracy {
    pub(crate) inner: LayoutDegeneracy,
}

impl PartialEq for PyLayoutDegeneracy {
    fn eq(&self, other: &Self) -> bool {
        self.inner.num_nodes == other.inner.num_nodes
            && self.inner.num_edges == other.inner.num_edges
            && self.inner.duplicate_nodes_f32 == other.inner.duplicate_nodes_f32
            && self.inner.duplicate_ratio_f32 == other.inner.duplicate_ratio_f32
            && self.inner.duplicate_nodes_f64 == other.inner.duplicate_nodes_f64
            && self.inner.duplicate_ratio_f64 == other.inner.duplicate_ratio_f64
            && self.inner.excess_duplicate_ratio == other.inner.excess_duplicate_ratio
            && self.inner.near_collision_ratio == other.inner.near_collision_ratio
            && self.inner.zero_length_edges == other.inner.zero_length_edges
            && self.inner.zero_length_edge_ratio == other.inner.zero_length_edge_ratio
            && self.inner.bbox_width == other.inner.bbox_width
            && self.inner.bbox_height == other.inner.bbox_height
            && self.inner.is_bbox_degenerate == other.inner.is_bbox_degenerate
    }
}

#[pymethods]
impl PyLayoutDegeneracy {
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges
    }
    #[getter]
    fn duplicate_nodes_f32(&self) -> usize {
        self.inner.duplicate_nodes_f32
    }
    #[getter]
    fn duplicate_ratio_f32(&self) -> f64 {
        self.inner.duplicate_ratio_f32
    }
    #[getter]
    fn duplicate_nodes_f64(&self) -> usize {
        self.inner.duplicate_nodes_f64
    }
    #[getter]
    fn duplicate_ratio_f64(&self) -> f64 {
        self.inner.duplicate_ratio_f64
    }
    #[getter]
    fn excess_duplicate_ratio(&self) -> f64 {
        self.inner.excess_duplicate_ratio
    }
    #[getter]
    fn near_collision_ratio(&self) -> f64 {
        self.inner.near_collision_ratio
    }
    #[getter]
    fn zero_length_edges(&self) -> usize {
        self.inner.zero_length_edges
    }
    #[getter]
    fn zero_length_edge_ratio(&self) -> f64 {
        self.inner.zero_length_edge_ratio
    }
    #[getter]
    fn bbox_width(&self) -> f64 {
        self.inner.bbox_width
    }
    #[getter]
    fn bbox_height(&self) -> f64 {
        self.inner.bbox_height
    }
    #[getter]
    fn is_bbox_degenerate(&self) -> bool {
        self.inner.is_bbox_degenerate
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        for &k in deg_keys() {
            if let Some(v) = deg_lookup(self, py, k)? {
                d.set_item(k, v)?;
            }
        }
        Ok(d.unbind())
    }
    fn keys(&self) -> Vec<&'static str> {
        deg_keys().to_vec()
    }
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        deg_lookup(self, py, key)?.ok_or_else(|| PyKeyError::new_err(key.to_owned()))
    }
    fn __contains__(&self, key: &str) -> bool {
        deg_keys().contains(&key)
    }
    fn __repr__(&self) -> String {
        format!(
            "LayoutDegeneracy(excess_duplicate_ratio={:.6}, near_collision_ratio={:.6}, is_bbox_degenerate={})",
            self.inner.excess_duplicate_ratio,
            self.inner.near_collision_ratio,
            self.inner.is_bbox_degenerate
        )
    }
}

fn deg_keys() -> &'static [&'static str] {
    &[
        "num_nodes",
        "num_edges",
        "duplicate_nodes_f32",
        "duplicate_ratio_f32",
        "duplicate_nodes_f64",
        "duplicate_ratio_f64",
        "excess_duplicate_ratio",
        "near_collision_ratio",
        "zero_length_edges",
        "zero_length_edge_ratio",
        "bbox_width",
        "bbox_height",
        "is_bbox_degenerate",
    ]
}

fn deg_lookup(r: &PyLayoutDegeneracy, py: Python<'_>, key: &str) -> PyResult<Option<Py<PyAny>>> {
    let inner = &r.inner;
    let obj = match key {
        "num_nodes" => inner.num_nodes.into_py_any(py)?,
        "num_edges" => inner.num_edges.into_py_any(py)?,
        "duplicate_nodes_f32" => inner.duplicate_nodes_f32.into_py_any(py)?,
        "duplicate_ratio_f32" => inner.duplicate_ratio_f32.into_py_any(py)?,
        "duplicate_nodes_f64" => inner.duplicate_nodes_f64.into_py_any(py)?,
        "duplicate_ratio_f64" => inner.duplicate_ratio_f64.into_py_any(py)?,
        "excess_duplicate_ratio" => inner.excess_duplicate_ratio.into_py_any(py)?,
        "near_collision_ratio" => inner.near_collision_ratio.into_py_any(py)?,
        "zero_length_edges" => inner.zero_length_edges.into_py_any(py)?,
        "zero_length_edge_ratio" => inner.zero_length_edge_ratio.into_py_any(py)?,
        "bbox_width" => inner.bbox_width.into_py_any(py)?,
        "bbox_height" => inner.bbox_height.into_py_any(py)?,
        "is_bbox_degenerate" => inner.is_bbox_degenerate.into_py_any(py)?,
        _ => return Ok(None),
    };
    Ok(Some(obj))
}

/// Structured minimum-angle resolution summary returned by `minimum_angle_summary`.
#[pyclass(
    name = "MinimumAngleSummary",
    module = "egraph._core",
    frozen,
    eq,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyMinimumAngleSummary {
    pub(crate) inner: MinimumAngleSummary,
}

impl PartialEq for PyMinimumAngleSummary {
    fn eq(&self, other: &Self) -> bool {
        self.inner.a1 == other.inner.a1 && self.inner.a2 == other.inner.a2
    }
}

#[pymethods]
impl PyMinimumAngleSummary {
    #[getter]
    fn a1(&self) -> f64 {
        self.inner.a1
    }
    #[getter]
    fn a2(&self) -> f64 {
        self.inner.a2
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("a1", self.inner.a1)?;
        d.set_item("a2", self.inner.a2)?;
        Ok(d.unbind())
    }
    fn keys(&self) -> Vec<&'static str> {
        vec!["a1", "a2"]
    }
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "a1" => self.inner.a1.into_py_any(py),
            "a2" => self.inner.a2.into_py_any(py),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }
    fn __contains__(&self, key: &str) -> bool {
        matches!(key, "a1" | "a2")
    }
    fn __repr__(&self) -> String {
        format!(
            "MinimumAngleSummary(a1={:.6}, a2={:.6})",
            self.inner.a1, self.inner.a2
        )
    }
}

/// Structured crossing summary returned by `crossing_summary`.
#[pyclass(
    name = "CrossingSummary",
    module = "egraph._core",
    frozen,
    eq,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyCrossingSummary {
    pub(crate) inner: CrossingSummary,
}

impl PartialEq for PyCrossingSummary {
    fn eq(&self, other: &Self) -> bool {
        self.inner.count == other.inner.count
            && self.inner.angle_cos_sq_sum == other.inner.angle_cos_sq_sum
            && self.inner.min_angle_rad == other.inner.min_angle_rad
    }
}

#[pymethods]
impl PyCrossingSummary {
    #[getter]
    fn count(&self) -> usize {
        self.inner.count
    }
    #[getter]
    fn angle_cos_sq_sum(&self) -> f64 {
        self.inner.angle_cos_sq_sum
    }
    #[getter]
    fn min_angle_rad(&self) -> Option<f64> {
        self.inner.min_angle_rad
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("count", self.inner.count)?;
        d.set_item("angle_cos_sq_sum", self.inner.angle_cos_sq_sum)?;
        d.set_item("min_angle_rad", self.inner.min_angle_rad)?;
        Ok(d.unbind())
    }
    fn keys(&self) -> Vec<&'static str> {
        vec!["count", "angle_cos_sq_sum", "min_angle_rad"]
    }
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "count" => self.inner.count.into_py_any(py),
            "angle_cos_sq_sum" => self.inner.angle_cos_sq_sum.into_py_any(py),
            "min_angle_rad" => self.inner.min_angle_rad.into_py_any(py),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }
    fn __contains__(&self, key: &str) -> bool {
        matches!(key, "count" | "angle_cos_sq_sum" | "min_angle_rad")
    }
    fn __repr__(&self) -> String {
        format!(
            "CrossingSummary(count={}, angle_cos_sq_sum={:.6})",
            self.inner.count, self.inner.angle_cos_sq_sum
        )
    }
}

/// Structured Scale-Normalized Stress evaluation diagnostics.
#[pyclass(
    name = "StressEvaluation",
    module = "egraph._core",
    frozen,
    eq,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyStressEvaluation {
    pub(crate) inner: StressEvaluation,
}

impl PartialEq for PyStressEvaluation {
    fn eq(&self, other: &Self) -> bool {
        self.inner.normalized_stress == other.inner.normalized_stress
            && self.inner.raw_stress == other.inner.raw_stress
            && self.inner.alpha_star == other.inner.alpha_star
            && self.inner.pair_count == other.inner.pair_count
            && self.inner.weight_distance_sq_sum == other.inner.weight_distance_sq_sum
    }
}

#[pymethods]
impl PyStressEvaluation {
    #[getter]
    fn normalized_stress(&self) -> f64 {
        self.inner.normalized_stress
    }
    #[getter]
    fn raw_stress(&self) -> f64 {
        self.inner.raw_stress
    }
    #[getter]
    fn alpha_star(&self) -> f64 {
        self.inner.alpha_star
    }
    #[getter]
    fn pair_count(&self) -> usize {
        self.inner.pair_count
    }
    #[getter]
    fn weight_distance_sq_sum(&self) -> f64 {
        self.inner.weight_distance_sq_sum
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("normalized_stress", self.inner.normalized_stress)?;
        d.set_item("raw_stress", self.inner.raw_stress)?;
        d.set_item("alpha_star", self.inner.alpha_star)?;
        d.set_item("pair_count", self.inner.pair_count)?;
        d.set_item("weight_distance_sq_sum", self.inner.weight_distance_sq_sum)?;
        Ok(d.unbind())
    }
    fn keys(&self) -> Vec<&'static str> {
        vec![
            "normalized_stress",
            "raw_stress",
            "alpha_star",
            "pair_count",
            "weight_distance_sq_sum",
        ]
    }
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "normalized_stress" => self.inner.normalized_stress.into_py_any(py),
            "raw_stress" => self.inner.raw_stress.into_py_any(py),
            "alpha_star" => self.inner.alpha_star.into_py_any(py),
            "pair_count" => self.inner.pair_count.into_py_any(py),
            "weight_distance_sq_sum" => self.inner.weight_distance_sq_sum.into_py_any(py),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }
    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "normalized_stress"
                | "raw_stress"
                | "alpha_star"
                | "pair_count"
                | "weight_distance_sq_sum"
        )
    }
    fn __repr__(&self) -> String {
        format!(
            "StressEvaluation(normalized_stress={:.6}, pair_count={})",
            self.inner.normalized_stress, self.inner.pair_count
        )
    }
}

/// Structured Landmark-Stratified RTA evaluation diagnostics.
#[pyclass(
    name = "RtaEvaluation",
    module = "egraph._core",
    frozen,
    eq,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyRtaEvaluation {
    pub(crate) inner: RtaEvaluation,
}

impl PartialEq for PyRtaEvaluation {
    fn eq(&self, other: &Self) -> bool {
        self.inner.rta == other.inner.rta
            && self.inner.num_landmarks == other.inner.num_landmarks
            && self.inner.num_triplets == other.inner.num_triplets
            && self.inner.num_strata == other.inner.num_strata
            && self.inner.is_exact == other.inner.is_exact
    }
}

#[pymethods]
impl PyRtaEvaluation {
    #[getter]
    fn rta(&self) -> f64 {
        self.inner.rta
    }
    #[getter]
    fn num_landmarks(&self) -> usize {
        self.inner.num_landmarks
    }
    #[getter]
    fn num_triplets(&self) -> usize {
        self.inner.num_triplets
    }
    #[getter]
    fn num_strata(&self) -> usize {
        self.inner.num_strata
    }
    #[getter]
    fn is_exact(&self) -> bool {
        self.inner.is_exact
    }
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("rta", self.inner.rta)?;
        d.set_item("num_landmarks", self.inner.num_landmarks)?;
        d.set_item("num_triplets", self.inner.num_triplets)?;
        d.set_item("num_strata", self.inner.num_strata)?;
        d.set_item("is_exact", self.inner.is_exact)?;
        Ok(d.unbind())
    }
    fn keys(&self) -> Vec<&'static str> {
        vec![
            "rta",
            "num_landmarks",
            "num_triplets",
            "num_strata",
            "is_exact",
        ]
    }
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "rta" => self.inner.rta.into_py_any(py),
            "num_landmarks" => self.inner.num_landmarks.into_py_any(py),
            "num_triplets" => self.inner.num_triplets.into_py_any(py),
            "num_strata" => self.inner.num_strata.into_py_any(py),
            "is_exact" => self.inner.is_exact.into_py_any(py),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }
    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "rta" | "num_landmarks" | "num_triplets" | "num_strata" | "is_exact"
        )
    }
    fn __repr__(&self) -> String {
        format!(
            "RtaEvaluation(rta={:.6}, num_triplets={}, is_exact={})",
            self.inner.rta, self.inner.num_triplets, self.inner.is_exact
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLayoutDegeneracy>()?;
    m.add_class::<PyMinimumAngleSummary>()?;
    m.add_class::<PyCrossingSummary>()?;
    m.add_class::<PyStressEvaluation>()?;
    m.add_class::<PyRtaEvaluation>()?;
    Ok(())
}
