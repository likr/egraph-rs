//! Classical, Pivot, and Representative Multidimensional Scaling (MDS) Python bindings.

use egraph::linalg::{
    PivotMdsConfig, classical_mds as linalg_classical_mds,
    classical_mds_2d as linalg_classical_mds_2d,
    classical_mds_from_distances as linalg_classical_mds_from_distances,
    pivot_mds as linalg_pivot_mds, pivot_mds_from_distances as linalg_pivot_mds_from_distances,
    pivot_mds_with_pivots as linalg_pivot_mds_with_pivots,
    representative_mds as linalg_representative_mds,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_2d_raw;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d,
};
use crate::layout_buffers::PyLayout2d;
use crate::layout_nd::PyLayoutNd;

/// Compute a `dims`-dimensional `LayoutNd` via zero-guarded Classical MDS on `graph`.
#[pyfunction]
#[pyo3(
    signature = (graph, dims = StrictUsize(2), *, dtype = ScalarDtype::F64),
    text_signature = "(graph, dims=2, *, dtype='float64')"
)]
fn classical_mds(
    py: Python<'_>,
    graph: &PyCsrGraph,
    dims: StrictUsize,
    dtype: ScalarDtype,
) -> PyResult<PyLayoutNd> {
    let g = graph.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            linalg_classical_mds::<f64, f64>(&*g, dims.0).map(PyLayoutNd::from_f64_nd)
        }
        ScalarDtype::F32 => {
            linalg_classical_mds::<f64, f32>(&*g, dims.0).map(PyLayoutNd::from_f32_nd)
        }
    })
    .map_err(core_error_to_py)
}

/// Compute a `dims`-dimensional `LayoutNd` via Classical MDS from an `(N, N)` distance matrix.
#[pyfunction]
#[pyo3(
    signature = (distances, dims = StrictUsize(2), *, dtype = ScalarDtype::F64),
    text_signature = "(distances, dims=2, *, dtype='float64')"
)]
fn classical_mds_from_distances(
    py: Python<'_>,
    distances: &Bound<'_, PyAny>,
    dims: StrictUsize,
    dtype: ScalarDtype,
) -> PyResult<PyLayoutNd> {
    let (flat, nrows, ncols) = extract_f64_2d_raw(distances, "distances")?;
    if nrows != ncols {
        return Err(PyValueError::new_err(format!(
            "distances must be a square (N, N) matrix, got ({nrows}, {ncols})"
        )));
    }
    py.detach(|| match dtype {
        ScalarDtype::F64 => linalg_classical_mds_from_distances::<f64>(&flat, nrows, dims.0)
            .map(PyLayoutNd::from_f64_nd),
        ScalarDtype::F32 => linalg_classical_mds_from_distances::<f32>(&flat, nrows, dims.0)
            .map(PyLayoutNd::from_f32_nd),
    })
    .map_err(core_error_to_py)
}

/// Compute a 2D `Layout2d` via zero-guarded Classical MDS from all-pairs shortest paths.
#[pyfunction]
#[pyo3(
    signature = (graph, *, dtype = ScalarDtype::F64),
    text_signature = "(graph, *, dtype='float64')"
)]
fn classical_mds_2d(
    py: Python<'_>,
    graph: &PyCsrGraph,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let g = graph.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            linalg_classical_mds_2d::<f64, f64>(&*g).map(|l| PyLayout2d::from_f64_layout(&l))
        }
        ScalarDtype::F32 => {
            linalg_classical_mds_2d::<f64, f32>(&*g).map(|l| PyLayout2d::from_f32_layout(&l))
        }
    })
    .map_err(core_error_to_py)
}

/// Compute a 2D `Layout2d` via Core-First MaxMin PivotMDS (or explicit `pivots`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, *, num_pivots = StrictUsize(50), seed = StrictU64(42), min_degree = StrictUsize(2), target_std = Some(StrictF64(1.0)), pivots = None, dtype = ScalarDtype::F64),
    text_signature = "(graph, *, num_pivots=50, seed=42, min_degree=2, target_std=1.0, pivots=None, dtype='float64')"
)]
fn pivot_mds(
    py: Python<'_>,
    graph: &PyCsrGraph,
    num_pivots: StrictUsize,
    seed: StrictU64,
    min_degree: StrictUsize,
    target_std: Option<StrictF64>,
    pivots: Option<&Bound<'_, PyAny>>,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let explicit_pivots = pivots
        .filter(|p| !p.is_none())
        .map(|p| extract_u32_1d(p, "pivots"))
        .transpose()?;
    let g = graph.inner();
    let t_std = target_std.map(|v| v.0);
    if let Some(piv) = explicit_pivots {
        return py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    linalg_pivot_mds_with_pivots::<f64, f64>(&*g, &piv, t_std, seed.0)
                        .map(|l| PyLayout2d::from_f64_layout(&l))
                }
                ScalarDtype::F32 => {
                    linalg_pivot_mds_with_pivots::<f64, f32>(&*g, &piv, t_std, seed.0)
                        .map(|l| PyLayout2d::from_f32_layout(&l))
                }
            })
            .map_err(core_error_to_py);
    }
    let cfg = PivotMdsConfig {
        num_pivots: num_pivots.0,
        min_degree: min_degree.0,
        seed: seed.0,
        target_std: t_std,
    };
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            linalg_pivot_mds::<f64, f64>(&*g, &cfg).map(|l| PyLayout2d::from_f64_layout(&l))
        }
        ScalarDtype::F32 => {
            linalg_pivot_mds::<f64, f32>(&*g, &cfg).map(|l| PyLayout2d::from_f32_layout(&l))
        }
    })
    .map_err(core_error_to_py)
}

/// Compute a 2D `Layout2d` via Core-First MaxMin PivotMDS with optional isotropic std normalization.
#[pyfunction]
#[pyo3(
    signature = (graph, *, num_pivots = StrictUsize(50), seed = StrictU64(42), min_degree = StrictUsize(2), target_std = Some(StrictF64(1.0)), dtype = ScalarDtype::F64),
    text_signature = "(graph, *, num_pivots=50, seed=42, min_degree=2, target_std=1.0, dtype='float64')"
)]
fn pivot_mds_init(
    py: Python<'_>,
    graph: &PyCsrGraph,
    num_pivots: StrictUsize,
    seed: StrictU64,
    min_degree: StrictUsize,
    target_std: Option<StrictF64>,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    pivot_mds(
        py, graph, num_pivots, seed, min_degree, target_std, None, dtype,
    )
}

/// Compute a 2D `Layout2d` via PivotMDS from a precomputed `(N, K)` distance matrix.
#[pyfunction]
#[pyo3(
    signature = (distances_nk, *, target_std = Some(StrictF64(1.0)), seed = StrictU64(42), dtype = ScalarDtype::F64),
    text_signature = "(distances_nk, *, target_std=1.0, seed=42, dtype='float64')"
)]
fn pivot_mds_from_distances(
    py: Python<'_>,
    distances_nk: &Bound<'_, PyAny>,
    target_std: Option<StrictF64>,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let (flat, n_nodes, n_pivots) = extract_f64_2d_raw(distances_nk, "distances_nk")?;
    let t_std = target_std.map(|v| v.0);
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            linalg_pivot_mds_from_distances::<f64>(&flat, n_nodes, n_pivots, t_std, seed.0)
                .map(|l| PyLayout2d::from_f64_layout(&l))
        }
        ScalarDtype::F32 => {
            linalg_pivot_mds_from_distances::<f32>(&flat, n_nodes, n_pivots, t_std, seed.0)
                .map(|l| PyLayout2d::from_f32_layout(&l))
        }
    })
    .map_err(core_error_to_py)
}

/// Compute a `dims`-dimensional `LayoutNd` via Representative/Landmark MDS from an `(N, R)` distance matrix.
#[pyfunction]
#[pyo3(
    signature = (distances_to_reps, dims = StrictUsize(2), *, dtype = ScalarDtype::F64),
    text_signature = "(distances_to_reps, dims=2, *, dtype='float64')"
)]
fn representative_mds(
    py: Python<'_>,
    distances_to_reps: &Bound<'_, PyAny>,
    dims: StrictUsize,
    dtype: ScalarDtype,
) -> PyResult<PyLayoutNd> {
    let (flat, n_nodes, n_reps) = extract_f64_2d_raw(distances_to_reps, "distances_to_reps")?;
    py.detach(|| match dtype {
        ScalarDtype::F64 => linalg_representative_mds::<f64>(&flat, n_nodes, n_reps, dims.0)
            .map(PyLayoutNd::from_f64_nd),
        ScalarDtype::F32 => linalg_representative_mds::<f32>(&flat, n_nodes, n_reps, dims.0)
            .map(PyLayoutNd::from_f32_nd),
    })
    .map_err(core_error_to_py)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(classical_mds, m)?)?;
    m.add_function(wrap_pyfunction!(classical_mds_from_distances, m)?)?;
    m.add_function(wrap_pyfunction!(classical_mds_2d, m)?)?;
    m.add_function(wrap_pyfunction!(pivot_mds, m)?)?;
    m.add_function(wrap_pyfunction!(pivot_mds_init, m)?)?;
    m.add_function(wrap_pyfunction!(pivot_mds_from_distances, m)?)?;
    m.add_function(wrap_pyfunction!(representative_mds, m)?)?;
    Ok(())
}
