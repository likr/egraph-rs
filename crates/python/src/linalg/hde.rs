//! High-Dimensional Embedding (`HDE`) linear-algebra projection bindings.

use egraph::linalg as egraph_linalg;
use egraph_linalg::{
    HdeConfig, hde as linalg_hde, hde_nd as linalg_hde_nd,
    hde_nd_with_pivots as linalg_hde_nd_with_pivots, hde_with_pivots as linalg_hde_with_pivots,
};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d,
};
use crate::layout_buffers::PyLayout2d;
use crate::layout_nd::PyLayoutNd;

/// Compute a 2D `Layout2d` via Harel-Koren High-Dimensional Embedding (`HDE`, 2004).
///
/// Embeds the graph into `dimensions` shortest-path pivot coordinates using MaxMin
/// pivot selection (or explicit `pivots`) and projects via iterative Jacobi/PCA to 2D.
/// Supports `"float64"` (default) and `"float32"` while releasing the Python GIL.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        graph,
        *,
        dimensions = StrictUsize(50),
        seed = StrictU64(42),
        tol = StrictF64(1e-3),
        max_sweeps = StrictUsize(100),
        normalize = true,
        pivots = None,
        dtype = ScalarDtype::F64
    ),
    text_signature = "(graph, *, dimensions=50, seed=42, tol=1e-3, max_sweeps=100, normalize=True, pivots=None, dtype='float64')"
)]
fn hde(
    py: Python<'_>,
    graph: &PyCsrGraph,
    dimensions: StrictUsize,
    seed: StrictU64,
    tol: StrictF64,
    max_sweeps: StrictUsize,
    normalize: bool,
    pivots: Option<&Bound<'_, PyAny>>,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let pivots_vec = match pivots {
        Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "pivots")?),
        _ => None,
    };
    let cfg = HdeConfig {
        dimensions: dimensions.0,
        output_dims: 2,
        seed: seed.0,
        tol: tol.0,
        max_sweeps: max_sweeps.0,
        normalize,
    };
    let g = graph.inner();
    py.detach(|| match (dtype, pivots_vec.as_deref()) {
        (ScalarDtype::F64, None) => {
            linalg_hde::<f64, f64>(&*g, &cfg).map(|l| PyLayout2d::from_f64_layout(&l))
        }
        (ScalarDtype::F32, None) => {
            linalg_hde::<f64, f32>(&*g, &cfg).map(|l| PyLayout2d::from_f32_layout(&l))
        }
        (ScalarDtype::F64, Some(p)) => linalg_hde_with_pivots::<f64, f64>(&*g, p, &cfg)
            .map(|l| PyLayout2d::from_f64_layout(&l)),
        (ScalarDtype::F32, Some(p)) => linalg_hde_with_pivots::<f64, f32>(&*g, p, &cfg)
            .map(|l| PyLayout2d::from_f32_layout(&l)),
    })
    .map_err(core_error_to_py)
}

/// Compute an `output_dims`-dimensional `LayoutNd` via Harel-Koren High-Dimensional Embedding (`HDE`, 2004).
///
/// Embeds the graph into `dimensions` shortest-path pivot coordinates using MaxMin
/// pivot selection (or explicit `pivots`) and projects via iterative Jacobi/PCA to `output_dims`.
/// Supports `"float64"` (default) and `"float32"` while releasing the Python GIL.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        graph,
        *,
        dimensions = StrictUsize(50),
        output_dims = StrictUsize(2),
        seed = StrictU64(42),
        tol = StrictF64(1e-3),
        max_sweeps = StrictUsize(100),
        normalize = true,
        pivots = None,
        dtype = ScalarDtype::F64
    ),
    text_signature = "(graph, *, dimensions=50, output_dims=2, seed=42, tol=1e-3, max_sweeps=100, normalize=True, pivots=None, dtype='float64')"
)]
fn hde_nd(
    py: Python<'_>,
    graph: &PyCsrGraph,
    dimensions: StrictUsize,
    output_dims: StrictUsize,
    seed: StrictU64,
    tol: StrictF64,
    max_sweeps: StrictUsize,
    normalize: bool,
    pivots: Option<&Bound<'_, PyAny>>,
    dtype: ScalarDtype,
) -> PyResult<PyLayoutNd> {
    let pivots_vec = match pivots {
        Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "pivots")?),
        _ => None,
    };
    let cfg = HdeConfig {
        dimensions: dimensions.0,
        output_dims: output_dims.0,
        seed: seed.0,
        tol: tol.0,
        max_sweeps: max_sweeps.0,
        normalize,
    };
    let g = graph.inner();
    py.detach(|| match (dtype, pivots_vec.as_deref()) {
        (ScalarDtype::F64, None) => {
            linalg_hde_nd::<f64, f64>(&*g, &cfg).map(PyLayoutNd::from_f64_nd)
        }
        (ScalarDtype::F32, None) => {
            linalg_hde_nd::<f64, f32>(&*g, &cfg).map(PyLayoutNd::from_f32_nd)
        }
        (ScalarDtype::F64, Some(p)) => {
            linalg_hde_nd_with_pivots::<f64, f64>(&*g, p, &cfg).map(PyLayoutNd::from_f64_nd)
        }
        (ScalarDtype::F32, Some(p)) => {
            linalg_hde_nd_with_pivots::<f64, f32>(&*g, p, &cfg).map(PyLayoutNd::from_f32_nd)
        }
    })
    .map_err(core_error_to_py)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hde, m)?)?;
    m.add_function(wrap_pyfunction!(hde_nd, m)?)?;
    Ok(())
}
