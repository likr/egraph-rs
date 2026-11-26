//! Separation constraint, 1D/2D overlap projection, and directed layering bindings.

use egraph::{core as egraph_core, layout as egraph_layout};
use egraph_core::{CoreError, CoreResult};
use egraph_layout::{
    ConstraintProjector, LayeringMethod, SeparationConstraint,
    assign_layers as layout_assign_layers, cycle_edges as layout_cycle_edges,
    generate_layered_constraints as layout_generate_layered_constraints,
    project_clustered_rectangle_no_overlap_2d as layout_project_clustered_rectangle_no_overlap_2d,
    project_rectangle_no_overlap_2d as layout_project_rectangle_no_overlap_2d,
    remove_cycles as layout_remove_cycles,
};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array_extract::{extract_f64_1d, extract_usize_1d};
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictUsize, core_error_to_py, extract_u32_1d,
    is_non_sequence_container, normalize_enum_key,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::solver_common::{
    SolverLayout, extract_solver_layout, vec_f64_to_f32_checked, write_back_solver_layout,
};

fn parse_layering_method(raw: &str) -> Result<LayeringMethod, String> {
    match normalize_enum_key(raw).as_str() {
        "longestpath" | "lp" => Ok(LayeringMethod::LongestPath),
        "networksimplex" | "ns" | "simplex" => Ok(LayeringMethod::NetworkSimplex),
        _ => Err(format!(
            "unknown layering method {raw:?}; expected 'longest_path' or 'network_simplex'"
        )),
    }
}

/// One-dimensional separation constraint `coord[left] + gap <= coord[right]` for 2D layouts.
#[pyclass(name = "SeparationConstraint", module = "egraph._core", from_py_object)]
#[derive(Clone, Debug)]
pub struct PySeparationConstraint {
    inner: SeparationConstraint<f64>,
}

impl PySeparationConstraint {
    pub(crate) fn from_inner(inner: SeparationConstraint<f64>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySeparationConstraint {
    /// Construct a separation constraint enforcing `coord[left] + gap <= coord[right]`.
    #[new]
    #[pyo3(signature = (left, right, gap), text_signature = "(left, right, gap)")]
    fn new(left: StrictUsize, right: StrictUsize, gap: StrictF64) -> PyResult<Self> {
        let left_u32 = u32::try_from(left.0)
            .map_err(|_| PyValueError::new_err(format!("left index {} exceeds u32", left.0)))?;
        let right_u32 = u32::try_from(right.0)
            .map_err(|_| PyValueError::new_err(format!("right index {} exceeds u32", right.0)))?;
        if !gap.0.is_finite() {
            return Err(PyValueError::new_err(
                "SeparationConstraint gap must be finite",
            ));
        }
        Ok(Self {
            inner: SeparationConstraint::new(left_u32, right_u32, gap.0),
        })
    }

    /// Left (preceding) variable index.
    #[getter]
    fn left(&self) -> u32 {
        self.inner.left
    }

    /// Right (following) variable index.
    #[getter]
    fn right(&self) -> u32 {
        self.inner.right
    }

    /// Minimum required separation distance `gap`.
    #[getter]
    fn gap(&self) -> f64 {
        self.inner.gap
    }

    /// Return a concise string representation of the `SeparationConstraint`.
    fn __repr__(&self) -> String {
        format!(
            "SeparationConstraint(left={}, right={}, gap={})",
            self.inner.left, self.inner.right, self.inner.gap
        )
    }
}

fn extract_separation_constraints(
    constraints: &Bound<'_, PyAny>,
) -> PyResult<Vec<SeparationConstraint<f64>>> {
    if is_non_sequence_container(constraints) {
        return Err(PyTypeError::new_err(
            "constraints must be a sequence or iterable of SeparationConstraint",
        ));
    }
    let iter = constraints.try_iter().map_err(|_| {
        PyTypeError::new_err("constraints must be a sequence or iterable of SeparationConstraint")
    })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res?;
        let c = item.extract::<PySeparationConstraint>().map_err(|_| {
            PyTypeError::new_err("constraints entries must be SeparationConstraint instances")
        })?;
        out.push(c.inner);
    }
    Ok(out)
}

fn constraints_to_f32(
    constraints: &[SeparationConstraint<f64>],
) -> PyResult<Vec<SeparationConstraint<f32>>> {
    let mut out = Vec::with_capacity(constraints.len());
    for c in constraints {
        if !c.gap.is_finite() {
            return Err(PyValueError::new_err(
                "non-finite value in SeparationConstraint.gap",
            ));
        }
        let g32 = c.gap as f32;
        if !g32.is_finite() {
            return Err(PyValueError::new_err(format!(
                "SeparationConstraint.gap {} overflows float32",
                c.gap
            )));
        }
        out.push(SeparationConstraint::new(c.left, c.right, g32));
    }
    Ok(out)
}

/// Project 1D coordinates `coords` onto `constraints` while minimizing quadratic displacement.
///
/// If `coords` is a writable `float64` or `float32` NumPy 1D array, updates it in place and
/// returns the projected NumPy array with matching dtype. Otherwise returns a new `float64`
/// NumPy array. Releases the Python GIL during projection.
pub(crate) fn project_1d_csr_impl<'py>(
    py: Python<'py>,
    coords: &Bound<'py, PyAny>,
    constraints: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let c64 = extract_separation_constraints(constraints)?;

    if let Ok(untyped) = coords.cast::<PyUntypedArray>() {
        if untyped.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "coords must be a 1D array, got {}D",
                untyped.ndim()
            )));
        }
        if let Ok(arr) = coords.cast::<PyArray1<f64>>() {
            let mut vec: Vec<f64> = {
                let ro = arr
                    .try_readonly()
                    .map_err(|e| PyValueError::new_err(format!("coords borrow error: {e}")))?;
                ro.as_array().iter().copied().collect()
            };
            let n = vec.len();
            py.detach(|| ConstraintProjector::<f64>::new(n).project_1d(&mut vec, &c64))
                .map_err(core_error_to_py)?;
            if let Ok(mut rw) = arr.try_readwrite() {
                for (dst, &src) in rw.as_array_mut().iter_mut().zip(vec.iter()) {
                    *dst = src;
                }
            }
            return Ok(PyArray1::from_vec(py, vec).into_any());
        }
        if let Ok(arr) = coords.cast::<PyArray1<f32>>() {
            let c32 = constraints_to_f32(&c64)?;
            let mut vec: Vec<f32> = {
                let ro = arr
                    .try_readonly()
                    .map_err(|e| PyValueError::new_err(format!("coords borrow error: {e}")))?;
                ro.as_array().iter().copied().collect()
            };
            let n = vec.len();
            py.detach(|| ConstraintProjector::<f32>::new(n).project_1d(&mut vec, &c32))
                .map_err(core_error_to_py)?;
            if let Ok(mut rw) = arr.try_readwrite() {
                for (dst, &src) in rw.as_array_mut().iter_mut().zip(vec.iter()) {
                    *dst = src;
                }
            }
            return Ok(PyArray1::from_vec(py, vec).into_any());
        }
    }

    let mut vec = extract_f64_1d(coords, "coords")?;
    let n = vec.len();
    py.detach(|| ConstraintProjector::<f64>::new(n).project_1d(&mut vec, &c64))
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, vec).into_any())
}

/// Project one axis (`0` for X, `1` for Y) of a `Layout2d` onto `constraints` in place.
///
/// Releases the Python GIL during projection.
#[pyfunction]
#[pyo3(
    signature = (layout, axis, constraints),
    text_signature = "(layout, axis, constraints)"
)]
fn project_layout_axis(
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    axis: StrictUsize,
    constraints: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let c64 = extract_separation_constraints(constraints)?;
    let target_dtype = match layout.try_borrow()?.inner() {
        Layout2dInner::F64(_) => ScalarDtype::F64,
        Layout2dInner::F32(_) => ScalarDtype::F32,
    };
    let mut soa = extract_solver_layout(layout, target_dtype)?;
    let axis_val = axis.0;
    match &mut soa {
        SolverLayout::F64(l64) => {
            let n = l64.len();
            py.detach(|| {
                ConstraintProjector::<f64>::new(n).project_layout_axis(l64, axis_val, &c64)
            })
            .map_err(core_error_to_py)?;
        }
        SolverLayout::F32(l32) => {
            let c32 = constraints_to_f32(&c64)?;
            let n = l32.len();
            py.detach(|| {
                ConstraintProjector::<f32>::new(n).project_layout_axis(l32, axis_val, &c32)
            })
            .map_err(core_error_to_py)?;
        }
    }
    write_back_solver_layout(layout, &soa)
}

/// Generate hierarchical separation constraints for directed edges spanning `layers[u] < layers[v]`.
///
/// Releases the Python GIL during constraint generation.
pub(crate) fn layering_constraints_csr_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layers: &Bound<'_, PyAny>,
    min_layer_distance: StrictF64,
) -> PyResult<Vec<PySeparationConstraint>> {
    if !min_layer_distance.0.is_finite() {
        return Err(PyValueError::new_err("min_layer_distance must be finite"));
    }
    let layers_vec = extract_usize_1d(layers, "layers")?;
    let g = graph.inner();
    let dist = min_layer_distance.0;
    let raw = py
        .detach(|| layout_generate_layered_constraints(g, &layers_vec, dist))
        .map_err(core_error_to_py)?;
    Ok(raw
        .into_iter()
        .map(PySeparationConstraint::from_inner)
        .collect())
}

/// Project 2D rectangle positions in `layout` in place to remove all axis-aligned overlaps.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, half_widths, half_heights),
    text_signature = "(layout, half_widths, half_heights)"
)]
fn project_rectangle_no_overlap_2d(
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    half_widths: &Bound<'_, PyAny>,
    half_heights: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
    let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
    let target_dtype = match layout.try_borrow()?.inner() {
        Layout2dInner::F64(_) => ScalarDtype::F64,
        Layout2dInner::F32(_) => ScalarDtype::F32,
    };
    let mut soa = extract_solver_layout(layout, target_dtype)?;
    match &mut soa {
        SolverLayout::F64(l64) => py
            .detach(|| layout_project_rectangle_no_overlap_2d(l64, &hw_f64, &hh_f64))
            .map_err(core_error_to_py)?,
        SolverLayout::F32(l32) => {
            let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
            let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
            py.detach(|| layout_project_rectangle_no_overlap_2d(l32, &hw_f32, &hh_f32))
                .map_err(core_error_to_py)?;
        }
    }
    write_back_solver_layout(layout, &soa)
}

/// Remove overlaps between axis-aligned cluster bounding boxes in `layout` in place via VPSC
/// while rigidly translating each cluster's intra-cluster coordinates.
///
/// Note: This rigid AABB VPSC post-processing pass is distinct from Dwyer et al. (2006)
/// iterative constrained stress minimization (`IPSep-CoLa`). Releases the Python GIL during
/// computation.
#[pyfunction]
#[pyo3(
    signature = (layout, cluster_ids, half_widths, half_heights),
    text_signature = "(layout, cluster_ids, half_widths, half_heights)"
)]
fn project_clustered_rectangle_no_overlap_2d(
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    cluster_ids: &Bound<'_, PyAny>,
    half_widths: &Bound<'_, PyAny>,
    half_heights: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let cids = extract_u32_1d(cluster_ids, "cluster_ids")?;
    let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
    let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
    let target_dtype = match layout.try_borrow()?.inner() {
        Layout2dInner::F64(_) => ScalarDtype::F64,
        Layout2dInner::F32(_) => ScalarDtype::F32,
    };
    let mut soa = extract_solver_layout(layout, target_dtype)?;
    match &mut soa {
        SolverLayout::F64(l64) => py
            .detach(|| {
                layout_project_clustered_rectangle_no_overlap_2d(l64, &cids, &hw_f64, &hh_f64)
            })
            .map_err(core_error_to_py)?,
        SolverLayout::F32(l32) => {
            let hw_f32 = vec_f64_to_f32_checked(&hw_f64, "half_widths")?;
            let hh_f32 = vec_f64_to_f32_checked(&hh_f64, "half_heights")?;
            py.detach(|| {
                layout_project_clustered_rectangle_no_overlap_2d(l32, &cids, &hw_f32, &hh_f32)
            })
            .map_err(core_error_to_py)?;
        }
    }
    write_back_solver_layout(layout, &soa)
}

/// Identify back-edges `(u, v)` whose reversal eliminates all directed cycles in a `CsrGraph`.
///
/// Returns a `uint32` NumPy array of shape `(M, 2)` and releases the Python GIL.
pub(crate) fn cycle_edges_csr_impl<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let g = graph.inner();
    let pairs = py
        .detach(|| layout_cycle_edges(g))
        .map_err(core_error_to_py)?;
    let m = pairs.len();
    let mut flat = Vec::with_capacity(m * 2);
    for (u, v) in pairs {
        flat.push(u);
        flat.push(v);
    }
    let arr =
        Array2::from_shape_vec((m, 2), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Return a directed acyclic `CsrGraph` by reversing cycle-closing back-edges.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn remove_cycles(py: Python<'_>, graph: &PyCsrGraph) -> PyResult<PyCsrGraph> {
    let g = graph.inner();
    let dag = py
        .detach(|| layout_remove_cycles(g))
        .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(dag))
}

/// Assign hierarchical layer indices (`0, 1, ...`) to all vertices of a directed `CsrGraph`.
///
/// Supports `method="longest_path"` (default) and `method="network_simplex"`.
/// Returns a `uint32` NumPy array of length `graph.num_nodes` and releases the GIL.
#[pyfunction]
#[pyo3(
    signature = (graph, *, method = "longest_path"),
    text_signature = "(graph, *, method='longest_path')"
)]
fn assign_layers<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let parsed_method = parse_layering_method(method).map_err(PyValueError::new_err)?;
    let g = graph.inner();
    let layers = py
        .detach(|| layout_assign_layers(g, parsed_method))
        .map_err(core_error_to_py)?;
    let layers_u32: Vec<u32> = layers
        .into_iter()
        .map(|l| {
            u32::try_from(l).map_err(|_| CoreError::invalid_input("layer index exceeds u32::MAX"))
        })
        .collect::<CoreResult<Vec<u32>>>()
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, layers_u32))
}

/// Register separation constraint, overlap projection, and layering bindings on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySeparationConstraint>()?;
    m.add_function(wrap_pyfunction!(project_layout_axis, m)?)?;
    m.add_function(wrap_pyfunction!(project_rectangle_no_overlap_2d, m)?)?;
    m.add_function(wrap_pyfunction!(
        project_clustered_rectangle_no_overlap_2d,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(remove_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(assign_layers, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{LayeringMethod, parse_layering_method};

    #[test]
    fn test_layering_method_parser() {
        assert_eq!(
            parse_layering_method("longest_path").unwrap(),
            LayeringMethod::LongestPath
        );
        assert_eq!(
            parse_layering_method("LP").unwrap(),
            LayeringMethod::LongestPath
        );
        assert_eq!(
            parse_layering_method("network_simplex").unwrap(),
            LayeringMethod::NetworkSimplex
        );
        assert_eq!(
            parse_layering_method("Network-Simplex").unwrap(),
            LayeringMethod::NetworkSimplex
        );
        assert_eq!(
            parse_layering_method("ns").unwrap(),
            LayeringMethod::NetworkSimplex
        );
        assert!(parse_layering_method("coffman_graham").is_err());
    }
}
