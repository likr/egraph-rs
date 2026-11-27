//! Spatial geometry (`Delaunay2d`, rectangle overlap sweep) bindings.

use egraph::core::{CoreError, CoreResult};
use egraph::spatial::{Delaunay2d, RectangleOverlapSweep};
use numpy::PyArray2;
use numpy::ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::core_error_to_py;
use crate::layout_buffers::{Layout2dInner, PyLayout2d, dispatch_layout2d};

type DelaunayArrays<'py> = (Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<u32>>);

/// Compute the deterministic 2D Delaunay triangulation of `layout`.
///
/// Returns `(edges, triangles)` as `uint32` NumPy arrays with shapes `(E, 2)` and `(T, 3)`.
/// Deduplicates coincident vertices deterministically and handles collinear layouts without
/// panicking. Releases the Python GIL during computation.
#[pyfunction]
fn delaunay_triangulation<'py>(
    py: Python<'py>,
    layout: &PyLayout2d,
) -> PyResult<DelaunayArrays<'py>> {
    let (flat_edges, e_len, flat_tris, t_len) = py
        .detach(|| -> CoreResult<(Vec<u32>, usize, Vec<u32>, usize)> {
            let dt = match layout.inner() {
                Layout2dInner::F64(buf) => Delaunay2d::triangulate(&buf.to_layout2d())?,
                Layout2dInner::F32(buf) => Delaunay2d::triangulate(&buf.to_layout2d())?,
            };
            let edges = dt.edges();
            let mut flat_edges = Vec::with_capacity(edges.len() * 2);
            for &(u, v) in edges {
                flat_edges.push(u);
                flat_edges.push(v);
            }
            let tris = dt.triangles();
            let mut flat_tris = Vec::with_capacity(tris.len() * 3);
            for &[a, b, c] in tris {
                flat_tris.push(a);
                flat_tris.push(b);
                flat_tris.push(c);
            }
            Ok((flat_edges, edges.len(), flat_tris, tris.len()))
        })
        .map_err(core_error_to_py)?;

    let edges_arr = Array2::from_shape_vec((e_len, 2), flat_edges)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let tris_arr = Array2::from_shape_vec((t_len, 3), flat_tris)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        PyArray2::from_owned_array(py, edges_arr),
        PyArray2::from_owned_array(py, tris_arr),
    ))
}

/// Compute the deterministic 2D Delaunay triangulation of `layout` and return an undirected `CsrGraph`.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn delaunay_to_csr(py: Python<'_>, layout: &PyLayout2d) -> PyResult<PyCsrGraph> {
    let csr = dispatch_layout2d!(py, layout, |soa| Delaunay2d::triangulate(&soa)?.to_csr())?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Find all strictly overlapping axis-aligned rectangle pairs `(i, j)` (`i < j`) as a `(K, 2)` `uint32` array.
#[pyfunction]
#[pyo3(
    signature = (layout, half_widths, half_heights),
    text_signature = "(layout, half_widths, half_heights)"
)]
fn rectangle_overlaps<'py>(
    py: Python<'py>,
    layout: &PyLayout2d,
    half_widths: &Bound<'_, PyAny>,
    half_heights: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let hw_f64 = extract_f64_1d(half_widths, "half_widths")?;
    let hh_f64 = extract_f64_1d(half_heights, "half_heights")?;
    let pairs = py
        .detach(|| -> CoreResult<Vec<(usize, usize)>> {
            match layout.inner() {
                Layout2dInner::F64(buf) => RectangleOverlapSweep::find_layout_overlaps(
                    &buf.to_layout2d(),
                    &hw_f64,
                    &hh_f64,
                ),
                Layout2dInner::F32(buf) => {
                    let hw32: Vec<f32> = hw_f64.iter().map(|&v| v as f32).collect();
                    let hh32: Vec<f32> = hh_f64.iter().map(|&v| v as f32).collect();
                    if hw32.iter().any(|v| !v.is_finite()) {
                        return Err(CoreError::non_finite("rectangle_overlaps::half_widths"));
                    }
                    if hh32.iter().any(|v| !v.is_finite()) {
                        return Err(CoreError::non_finite("rectangle_overlaps::half_heights"));
                    }
                    RectangleOverlapSweep::find_layout_overlaps(&buf.to_layout2d(), &hw32, &hh32)
                }
            }
        })
        .map_err(core_error_to_py)?;
    let mut flat = Vec::with_capacity(pairs.len() * 2);
    for (i, j) in &pairs {
        let iu = u32::try_from(*i)
            .map_err(|_| PyValueError::new_err(format!("rectangle index {i} exceeds u32")))?;
        let ju = u32::try_from(*j)
            .map_err(|_| PyValueError::new_err(format!("rectangle index {j} exceeds u32")))?;
        flat.push(iu);
        flat.push(ju);
    }
    let arr = Array2::from_shape_vec((pairs.len(), 2), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Register spatial geometry functions on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(delaunay_triangulation, m)?)?;
    m.add_function(wrap_pyfunction!(delaunay_to_csr, m)?)?;
    m.add_function(wrap_pyfunction!(rectangle_overlaps, m)?)?;
    Ok(())
}
