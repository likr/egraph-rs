//! Non-Euclidean manifold crossing counters, edge segment clipping, and crossing pair finders.

use egraph::spatial::{
    CrossingPair, crossing_edges_hyperbolic as core_crossing_edges_hyperbolic,
    crossing_edges_spherical as core_crossing_edges_spherical,
    hyperbolic_crossing_number as core_hyperbolic_crossing_number,
    spherical_crossing_number as core_spherical_crossing_number,
    torus_edge_segments as core_torus_edge_segments,
    torus_layout_edge_segments as core_torus_layout_edge_segments,
};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictF64, core_error_to_py};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};

use super::geodesic::{build_core_f32, build_core_f64};
use super::layout::{ManifoldLayout2dInner, PyManifoldLayout2d};

#[derive(FromPyObject)]
pub(crate) enum TorusLayoutInput<'py> {
    Euclidean(Bound<'py, PyLayout2d>),
    Manifold(Bound<'py, PyManifoldLayout2d>),
}

fn crossing_pairs_to_py_dicts<'py>(
    py: Python<'py>,
    pairs: Vec<CrossingPair>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let mut out = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let d = PyDict::new(py);
        d.set_item("edge_a", pair.edge_a)?;
        d.set_item("edge_b", pair.edge_b)?;
        d.set_item("u1", pair.u1)?;
        d.set_item("v1", pair.v1)?;
        d.set_item("u2", pair.u2)?;
        d.set_item("v2", pair.v2)?;
        d.set_item("angle_rad", pair.angle_rad)?;
        d.set_item("cos_sq", pair.cos_sq)?;
        out.push(d);
    }
    Ok(out)
}

/// Split a periodic torus edge `(x1, y1) -> (x2, y2)` into Euclidean sub-segments in `[0, 1)^2`.
#[pyfunction]
#[pyo3(signature = (x1, y1, x2, y2), text_signature = "(x1, y1, x2, y2)")]
fn torus_edge_segments<'py>(
    py: Python<'py>,
    x1: StrictF64,
    y1: StrictF64,
    x2: StrictF64,
    y2: StrictF64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let segs = core_torus_edge_segments(x1.0, y1.0, x2.0, y2.0).map_err(core_error_to_py)?;
    let n_segs = segs.len();
    let mut flat = Vec::with_capacity(n_segs * 4);
    for seg in segs {
        flat.extend_from_slice(&seg);
    }
    let arr = Array2::from_shape_vec((n_segs, 4), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Return clipped `[0, 1)^2` Euclidean sub-segments `[x0, y0, x1, y1]` for all graph edges on a torus.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn torus_layout_edge_segments<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    layout: TorusLayoutInput<'py>,
) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    let g = graph.inner();
    let segs: Vec<[f64; 4]> = match layout {
        TorusLayoutInput::Euclidean(l2) => {
            let borrowed = l2.try_borrow()?;
            let l2_ref: &PyLayout2d = &borrowed;
            py.detach(|| match l2_ref.inner() {
                Layout2dInner::F64(buf) => core_torus_layout_edge_segments(&*g, &buf.to_layout2d()),
                Layout2dInner::F32(buf) => core_torus_layout_edge_segments(&*g, &buf.to_layout2d()),
            })
            .map_err(core_error_to_py)?
        }
        TorusLayoutInput::Manifold(ml) => {
            let borrowed = ml.try_borrow()?;
            let ml_ref: &PyManifoldLayout2d = &borrowed;
            py.detach(|| match ml_ref.inner() {
                ManifoldLayout2dInner::F64 { kind, metric, buf } => {
                    let core_m = build_core_f64(*kind, *metric, buf)?;
                    core_torus_layout_edge_segments(&*g, &core_m)
                }
                ManifoldLayout2dInner::F32 { kind, metric, buf } => {
                    let core_m = build_core_f32(*kind, *metric, buf)?;
                    core_torus_layout_edge_segments(&*g, &core_m)
                }
            })
            .map_err(core_error_to_py)?
        }
    };
    Ok(segs
        .into_iter()
        .map(|s| PyArray1::from_slice(py, &s))
        .collect())
}

/// Count exact shortest great-circle geodesic arc crossings on a spherical layout.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn spherical_crossing_number(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyManifoldLayout2d,
) -> PyResult<usize> {
    let g = graph.inner();
    py.detach(|| match layout.inner() {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let core_m = build_core_f64(*kind, *metric, buf)?;
            core_spherical_crossing_number(&*g, &core_m)
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let core_m = build_core_f32(*kind, *metric, buf)?;
            core_spherical_crossing_number(&*g, &core_m)
        }
    })
    .map_err(core_error_to_py)
}

/// Find all shortest great-circle geodesic arc crossing pairs on a spherical layout.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn crossing_edges_spherical<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    layout: &PyManifoldLayout2d,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let g = graph.inner();
    let pairs = py
        .detach(|| match layout.inner() {
            ManifoldLayout2dInner::F64 { kind, metric, buf } => {
                let core_m = build_core_f64(*kind, *metric, buf)?;
                core_crossing_edges_spherical(&*g, &core_m)
            }
            ManifoldLayout2dInner::F32 { kind, metric, buf } => {
                let core_m = build_core_f32(*kind, *metric, buf)?;
                core_crossing_edges_spherical(&*g, &core_m)
            }
        })
        .map_err(core_error_to_py)?;
    crossing_pairs_to_py_dicts(py, pairs)
}

/// Count exact hyperbolic geodesic crossings on a Poincaré or Lorentz hyperbolic layout.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn hyperbolic_crossing_number(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyManifoldLayout2d,
) -> PyResult<usize> {
    let g = graph.inner();
    py.detach(|| match layout.inner() {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            let core_m = build_core_f64(*kind, *metric, buf)?;
            core_hyperbolic_crossing_number(&*g, &core_m)
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            let core_m = build_core_f32(*kind, *metric, buf)?;
            core_hyperbolic_crossing_number(&*g, &core_m)
        }
    })
    .map_err(core_error_to_py)
}

/// Find all hyperbolic geodesic crossing pairs on a Poincaré or Lorentz hyperbolic layout.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn crossing_edges_hyperbolic<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    layout: &PyManifoldLayout2d,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let g = graph.inner();
    let pairs = py
        .detach(|| match layout.inner() {
            ManifoldLayout2dInner::F64 { kind, metric, buf } => {
                let core_m = build_core_f64(*kind, *metric, buf)?;
                core_crossing_edges_hyperbolic(&*g, &core_m)
            }
            ManifoldLayout2dInner::F32 { kind, metric, buf } => {
                let core_m = build_core_f32(*kind, *metric, buf)?;
                core_crossing_edges_hyperbolic(&*g, &core_m)
            }
        })
        .map_err(core_error_to_py)?;
    crossing_pairs_to_py_dicts(py, pairs)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(torus_edge_segments, m)?)?;
    m.add_function(wrap_pyfunction!(torus_layout_edge_segments, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(crossing_edges_spherical, m)?)?;
    m.add_function(wrap_pyfunction!(hyperbolic_crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(crossing_edges_hyperbolic, m)?)?;
    Ok(())
}
