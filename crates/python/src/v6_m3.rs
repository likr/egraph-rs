//! M3 v6 clustering, non-Euclidean manifold layout, and cooperative layout solver
//! bindings backed by `egraph-clustering` and `egraph-layout`.

use egraph::{
    clustering as egraph_clustering, core as egraph_core, layout as egraph_layout,
    neighborhood as egraph_neighborhood, spatial as egraph_spatial,
};
use egraph_clustering::{
    KMeansConfig, LabelPropagationConfig, LouvainConfig, SpectralClusteringConfig,
    coarsen_graph as clu_coarsen_graph, kmeans_plus_plus as clu_kmeans_plus_plus,
    label_propagation as clu_label_propagation, louvain as clu_louvain,
    project_clustered_layout as clu_project_clustered_layout,
    spectral_clustering as clu_spectral_clustering,
};
use egraph_core::{
    CoreError, CoreResult, IterativeSolver, Layout2d, ManifoldKind, ManifoldLayout2d, SolverStatus,
    poincare_distance as core_poincare_distance, spherical_distance as core_spherical_distance,
    torus_distance as core_torus_distance, wrap_torus_point as core_wrap_torus_point,
};
use egraph_layout::{
    BarrierSymmetry, GuMapConfig, GuMapSolver, KamadaKawaiConfig, KamadaKawaiSolver,
    SgdScheduleConfig, SgdScheduleKind, SgdSolver, SparseBarrierConfig, SparseBarrierStats,
    SpatialIntruderRefiner, SpatialRefinerConfig, SpatialRefinerStats, StreamingNegativeConfig,
    StressMajorizationConfig, StressMajorizationSolver, TsNetBackend, TsNetConfig, TsNetSolver,
};
use egraph_neighborhood::{DogVaradhanConfig, DogVaradhanStats, SymmetrizationMode};
use egraph_spatial::{IbFftConfig, InterpolationOrder};
use numpy::ndarray::{Array2, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};

use super::v6_m2::{
    PyDogVaradhanLandmarks, PyKnnGraph, PySparseProbabilities, extract_f64_1d, extract_f64_2d,
    extract_perplexities,
};
use super::{
    Layout2dInner, PyCsrGraph, PyLayout2d, ScalarDtype, StrictF64, StrictU64, StrictUsize,
    core_error_to_py, extract_layout_inner, extract_u32_1d, ffi_error_to_py,
    is_non_sequence_container,
};

pub(super) fn normalize_enum_key(raw: &str) -> String {
    raw.trim()
        .chars()
        .filter(|c| *c != '_' && *c != '-' && *c != ' ')
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn parse_manifold_kind(raw: &str) -> Result<ManifoldKind, String> {
    match normalize_enum_key(raw).as_str() {
        "hyperbolic" | "hyperbolicpoincare" | "poincare" => Ok(ManifoldKind::HyperbolicPoincare),
        "spherical" | "sphere" => Ok(ManifoldKind::Spherical),
        "torus" | "torus2d" => Ok(ManifoldKind::Torus2d),
        _ => Err(format!(
            "unknown manifold kind {raw:?}; expected 'hyperbolic', 'spherical', or 'torus'"
        )),
    }
}

fn manifold_kind_str(kind: ManifoldKind) -> &'static str {
    match kind {
        ManifoldKind::HyperbolicPoincare => "hyperbolic",
        ManifoldKind::Spherical => "spherical",
        ManifoldKind::Torus2d => "torus",
    }
}

fn parse_tsnet_backend(raw: &str) -> Result<TsNetBackend, String> {
    match normalize_enum_key(raw).as_str() {
        "exact" => Ok(TsNetBackend::Exact),
        "barneshut" | "bh" => Ok(TsNetBackend::BarnesHut),
        "fft" | "fftinterpolation" | "fit" => Ok(TsNetBackend::FftInterpolation),
        "linearfft" | "l" => Ok(TsNetBackend::LinearFft),
        _ => Err(format!(
            "unknown tsNET backend {raw:?}; expected 'exact', 'barnes_hut', 'fft', or 'linear_fft'"
        )),
    }
}

fn parse_sgd_schedule_kind(raw: &str) -> Result<SgdScheduleKind, String> {
    match normalize_enum_key(raw).as_str() {
        "exponential" => Ok(SgdScheduleKind::Exponential),
        "linear" => Ok(SgdScheduleKind::Linear),
        "quadratic" => Ok(SgdScheduleKind::Quadratic),
        "reciprocal" => Ok(SgdScheduleKind::Reciprocal),
        "constant" => Ok(SgdScheduleKind::Constant),
        "squared" => Ok(SgdScheduleKind::Squared),
        _ => Err(format!(
            "unknown SGD schedule {raw:?}; expected 'exponential', 'linear', 'quadratic', 'reciprocal', 'constant', or 'squared'"
        )),
    }
}

fn parse_barrier_symmetry(raw: &str) -> Result<BarrierSymmetry, String> {
    match normalize_enum_key(raw).as_str() {
        "symmax" => Ok(BarrierSymmetry::SymMax),
        "symavg" => Ok(BarrierSymmetry::SymAvg),
        _ => Err(format!(
            "unknown barrier_sym {raw:?}; expected 'sym_max' or 'sym_avg'"
        )),
    }
}

pub(super) fn solver_status_str(status: SolverStatus) -> &'static str {
    match status {
        SolverStatus::Initialized => "initialized",
        SolverStatus::Running => "running",
        SolverStatus::Converged => "converged",
        SolverStatus::Completed => "completed",
    }
}

fn convert_distance_view2_to_f64<T, F>(
    view: ArrayView2<'_, T>,
    mut to_f64: F,
) -> PyResult<(Vec<f64>, usize)>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let nrows = view.nrows();
    let ncols = view.ncols();
    if nrows != ncols {
        return Err(PyValueError::new_err(format!(
            "distance_matrix must be square (N, N), got ({nrows}, {ncols})"
        )));
    }
    let mut flat = Vec::with_capacity(nrows.saturating_mul(ncols));
    for row in view.rows() {
        for &val in row.iter() {
            flat.push(to_f64(val));
        }
    }
    Ok((flat, nrows))
}

fn extract_distance_matrix_2d(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, usize)> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        if untyped.ndim() != 2 {
            return Err(PyValueError::new_err(format!(
                "distance_matrix must be a 2D square numeric array, got {}D",
                untyped.ndim()
            )));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
            return convert_distance_view2_to_f64(arr.as_array(), |v| v);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f32>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i64>>() {
            return convert_distance_view2_to_f64(arr.as_array(), |v| v as f64);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i32>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u64>>() {
            return convert_distance_view2_to_f64(arr.as_array(), |v| v as f64);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u32>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, isize>>() {
            return convert_distance_view2_to_f64(arr.as_array(), |v| v as f64);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, usize>>() {
            return convert_distance_view2_to_f64(arr.as_array(), |v| v as f64);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i16>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u16>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i8>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u8>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        return Err(PyTypeError::new_err(
            "distance_matrix NumPy array must have a numeric dtype",
        ));
    }

    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(
            "distance_matrix must be a 2D square numeric array or sequence of sequences",
        ));
    }
    let outer = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "distance_matrix must be a 2D square numeric array or sequence of sequences",
        )
    })?;
    let mut flat = Vec::new();
    let mut nrows = 0usize;
    let mut ncols: Option<usize> = None;

    for row_res in outer {
        let row = row_res?;
        if is_non_sequence_container(&row) {
            return Err(PyTypeError::new_err(
                "distance_matrix rows must be numeric sequences",
            ));
        }
        let inner = row
            .try_iter()
            .map_err(|_| PyTypeError::new_err("distance_matrix rows must be numeric sequences"))?;
        let mut row_len = 0usize;
        for elem_res in inner {
            let elem = elem_res?;
            if elem.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "distance_matrix entries must be numeric, not bool",
                ));
            }
            let val = elem
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("distance_matrix entries must be numeric"))?;
            flat.push(val);
            row_len += 1;
        }
        match ncols {
            None => ncols = Some(row_len),
            Some(expected) if row_len != expected => {
                return Err(PyValueError::new_err(format!(
                    "distance_matrix has ragged row {nrows}: expected {expected} columns, got {row_len}"
                )));
            }
            Some(_) => {}
        }
        nrows += 1;
    }

    let c = ncols.unwrap_or(0);
    if nrows != c {
        return Err(PyValueError::new_err(format!(
            "distance_matrix must be square (N, N), got ({nrows}, {c})"
        )));
    }
    Ok((flat, nrows))
}

pub(super) fn vec_f64_to_f32_checked(values: &[f64], ctx: &str) -> PyResult<Vec<f32>> {
    let mut out = Vec::with_capacity(values.len());
    for &v in values {
        if !v.is_finite() {
            return Err(PyValueError::new_err(format!("non-finite value in {ctx}")));
        }
        let vf = v as f32;
        if !vf.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{ctx} entry {v} overflows float32"
            )));
        }
        out.push(vf);
    }
    Ok(out)
}

pub(super) enum SolverLayout {
    F64(Layout2d<f64>),
    F32(Layout2d<f32>),
}

fn layout_f32_to_f64_checked(soa32: &Layout2d<f32>) -> CoreResult<Layout2d<f64>> {
    let x: Vec<f64> = soa32.x().iter().map(|&v| f64::from(v)).collect();
    let y: Vec<f64> = soa32.y().iter().map(|&v| f64::from(v)).collect();
    Layout2d::from_xy(x, y)
}

fn layout_f64_to_f32_checked(soa64: &Layout2d<f64>) -> CoreResult<Layout2d<f32>> {
    let x: Vec<f32> = soa64.x().iter().map(|&v| v as f32).collect();
    let y: Vec<f32> = soa64.y().iter().map(|&v| v as f32).collect();
    Layout2d::from_xy(x, y)
}

pub(super) fn extract_solver_layout(
    layout: &Bound<'_, PyLayout2d>,
    target_dtype: ScalarDtype,
) -> PyResult<SolverLayout> {
    let borrowed = layout.try_borrow()?;
    match (borrowed.inner(), target_dtype) {
        (Layout2dInner::F64(buf), ScalarDtype::F64) => Ok(SolverLayout::F64(buf.to_layout2d())),
        (Layout2dInner::F32(buf), ScalarDtype::F32) => Ok(SolverLayout::F32(buf.to_layout2d())),
        (Layout2dInner::F32(buf), ScalarDtype::F64) => {
            layout_f32_to_f64_checked(&buf.to_layout2d())
                .map(SolverLayout::F64)
                .map_err(core_error_to_py)
        }
        (Layout2dInner::F64(buf), ScalarDtype::F32) => {
            layout_f64_to_f32_checked(&buf.to_layout2d())
                .map(SolverLayout::F32)
                .map_err(core_error_to_py)
        }
    }
}

pub(super) fn write_back_solver_layout(
    layout: &Bound<'_, PyLayout2d>,
    soa: &SolverLayout,
) -> PyResult<()> {
    let mut borrowed = layout.try_borrow_mut()?;
    match (borrowed.inner_mut(), soa) {
        (Layout2dInner::F64(buf), SolverLayout::F64(l64)) => {
            buf.copy_from_layout2d(l64).map_err(ffi_error_to_py)
        }
        (Layout2dInner::F32(buf), SolverLayout::F32(l32)) => {
            buf.copy_from_layout2d(l32).map_err(ffi_error_to_py)
        }
        (Layout2dInner::F32(buf), SolverLayout::F64(l64)) => {
            let n = l64.len();
            if buf.len() != n {
                return Err(PyValueError::new_err(format!(
                    "layout length mismatch: expected {n}, got {}",
                    buf.len()
                )));
            }
            for (i, [px, py_coord]) in buf
                .as_mut_slice()
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .enumerate()
            {
                *px = l64.x()[i] as f32;
                *py_coord = l64.y()[i] as f32;
            }
            Ok(())
        }
        (Layout2dInner::F64(buf), SolverLayout::F32(l32)) => {
            let n = l32.len();
            if buf.len() != n {
                return Err(PyValueError::new_err(format!(
                    "layout length mismatch: expected {n}, got {}",
                    buf.len()
                )));
            }
            for (i, [px, py_coord]) in buf
                .as_mut_slice()
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .enumerate()
            {
                *px = f64::from(l32.x()[i]);
                *py_coord = f64::from(l32.y()[i]);
            }
            Ok(())
        }
    }
}

pub(super) trait DualIterativeSolver: Send {
    fn dtype(&self) -> ScalarDtype;
    fn step_layout(&mut self, layout: &mut SolverLayout) -> CoreResult<SolverStatus>;
    fn run_steps_layout(
        &mut self,
        layout: &mut SolverLayout,
        steps: usize,
    ) -> CoreResult<SolverStatus>;
    fn run_layout(&mut self, layout: &mut SolverLayout) -> CoreResult<SolverStatus>;
    fn solver_status(&self) -> SolverStatus;
    fn solver_iteration(&self) -> usize;
    fn solver_total_iterations(&self) -> usize;
}

macro_rules! impl_dual_solver {
    ($enum_name:ident, $solver_ty:ident) => {
        pub(super) enum $enum_name {
            F64($solver_ty<f64>),
            F32($solver_ty<f32>),
        }

        impl DualIterativeSolver for $enum_name {
            fn dtype(&self) -> ScalarDtype {
                match self {
                    Self::F64(_) => ScalarDtype::F64,
                    Self::F32(_) => ScalarDtype::F32,
                }
            }

            fn step_layout(&mut self, layout: &mut SolverLayout) -> CoreResult<SolverStatus> {
                match (self, layout) {
                    (Self::F64(s), SolverLayout::F64(l)) => s.step(l),
                    (Self::F32(s), SolverLayout::F32(l)) => s.step(l),
                    _ => Err(CoreError::invalid_input(
                        "internal solver layout dtype mismatch",
                    )),
                }
            }

            fn run_steps_layout(
                &mut self,
                layout: &mut SolverLayout,
                steps: usize,
            ) -> CoreResult<SolverStatus> {
                match (self, layout) {
                    (Self::F64(s), SolverLayout::F64(l)) => s.run_steps(l, steps),
                    (Self::F32(s), SolverLayout::F32(l)) => s.run_steps(l, steps),
                    _ => Err(CoreError::invalid_input(
                        "internal solver layout dtype mismatch",
                    )),
                }
            }

            fn run_layout(&mut self, layout: &mut SolverLayout) -> CoreResult<SolverStatus> {
                match (self, layout) {
                    (Self::F64(s), SolverLayout::F64(l)) => s.run(l),
                    (Self::F32(s), SolverLayout::F32(l)) => s.run(l),
                    _ => Err(CoreError::invalid_input(
                        "internal solver layout dtype mismatch",
                    )),
                }
            }

            fn solver_status(&self) -> SolverStatus {
                match self {
                    Self::F64(s) => s.status(),
                    Self::F32(s) => s.status(),
                }
            }

            fn solver_iteration(&self) -> usize {
                match self {
                    Self::F64(s) => s.iteration(),
                    Self::F32(s) => s.iteration(),
                }
            }

            fn solver_total_iterations(&self) -> usize {
                match self {
                    Self::F64(s) => s.total_iterations(),
                    Self::F32(s) => s.total_iterations(),
                }
            }
        }
    };
}

pub(super) use impl_dual_solver;

impl_dual_solver!(TsNetInner, TsNetSolver);
impl_dual_solver!(SgdInner, SgdSolver);
impl_dual_solver!(GuMapInner, GuMapSolver);
impl_dual_solver!(RefinerInner, SpatialIntruderRefiner);
impl_dual_solver!(StressMajorizationInner, StressMajorizationSolver);
impl_dual_solver!(KamadaKawaiInner, KamadaKawaiSolver);

pub(super) enum SolverAction {
    Step,
    RunSteps(usize),
    Run,
}

pub(super) fn execute_solver_action<I: DualIterativeSolver>(
    inner: &mut I,
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    action: SolverAction,
) -> PyResult<&'static str> {
    let target_dtype = inner.dtype();
    let mut soa = extract_solver_layout(layout, target_dtype)?;
    let status = py
        .detach(|| match action {
            SolverAction::Step => inner.step_layout(&mut soa),
            SolverAction::RunSteps(steps) => inner.run_steps_layout(&mut soa, steps),
            SolverAction::Run => inner.run_layout(&mut soa),
        })
        .map_err(core_error_to_py)?;
    write_back_solver_layout(layout, &soa)?;
    Ok(solver_status_str(status))
}

// ---------------------------------------------------------------------------
// Clustering functions (`egraph-clustering`)
// ---------------------------------------------------------------------------

/// Cluster `points` of shape `(N, D)` using deterministic K-Means++ initialization and Lloyd refinement.
///
/// Returns a dict containing `"labels"` (`uint32` array of shape `(N,)`), `"centroids"` (`float64`
/// array of shape `(k, D)`), `"inertia"`, `"iterations"`, and `"converged"`.
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (points, k = StrictUsize(2), max_iters = StrictUsize(100), tol = StrictF64(1e-4), n_init = StrictUsize(1), seed = StrictU64(42)),
    text_signature = "(points, k=2, max_iters=100, tol=1e-4, n_init=1, seed=42)"
)]
fn kmeans_plus_plus(
    py: Python<'_>,
    points: &Bound<'_, PyAny>,
    k: StrictUsize,
    max_iters: StrictUsize,
    tol: StrictF64,
    n_init: StrictUsize,
    seed: StrictU64,
) -> PyResult<Py<PyDict>> {
    let (flat, n, dims) = extract_f64_2d(points, "points")?;
    let cfg = KMeansConfig {
        k: k.0,
        max_iters: max_iters.0,
        tol: tol.0,
        n_init: n_init.0,
        seed: seed.0,
    };
    let res = py
        .detach(|| clu_kmeans_plus_plus(&flat, n, dims, &cfg))
        .map_err(core_error_to_py)?;
    let centroids_arr = Array2::from_shape_vec((cfg.k, dims), res.centroids)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("labels", PyArray1::from_vec(py, res.labels))?;
    dict.set_item("centroids", PyArray2::from_owned_array(py, centroids_arr))?;
    dict.set_item("inertia", res.inertia)?;
    dict.set_item("iterations", res.iterations)?;
    dict.set_item("converged", res.converged)?;
    Ok(dict.unbind())
}

/// Partition an undirected `CsrGraph` into `k` clusters using Ng-Jordan-Weiss spectral clustering.
///
/// Rejects directed graphs with `ValueError` and returns a `uint32` NumPy label array of length `N`.
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, k = StrictUsize(2), max_eigen_iters = StrictUsize(300), eigen_tol = StrictF64(1e-4), kmeans_max_iters = StrictUsize(100), kmeans_tol = StrictF64(1e-4), seed = StrictU64(42)),
    text_signature = "(graph, k=2, max_eigen_iters=300, eigen_tol=1e-4, kmeans_max_iters=100, kmeans_tol=1e-4, seed=42)"
)]
fn spectral_clustering<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    k: StrictUsize,
    max_eigen_iters: StrictUsize,
    eigen_tol: StrictF64,
    kmeans_max_iters: StrictUsize,
    kmeans_tol: StrictF64,
    seed: StrictU64,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let cfg = SpectralClusteringConfig {
        k: k.0,
        max_eigen_iters: max_eigen_iters.0,
        eigen_tol: eigen_tol.0,
        kmeans_max_iters: kmeans_max_iters.0,
        kmeans_tol: kmeans_tol.0,
        seed: seed.0,
    };
    let g = graph.inner();
    let labels = py
        .detach(|| clu_spectral_clustering(g, &cfg))
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, labels))
}

/// Detect communities on an undirected `CsrGraph` via multi-level Louvain modularity optimization.
///
/// Returns a dict with keys `"labels"`, `"num_communities"`, `"modularity"`, and `"passes"`.
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, max_passes = StrictUsize(100), resolution = StrictF64(1.0), min_modularity_gain = StrictF64(1e-7), seed = StrictU64(42)),
    text_signature = "(graph, max_passes=100, resolution=1.0, min_modularity_gain=1e-7, seed=42)"
)]
fn louvain_communities(
    py: Python<'_>,
    graph: &PyCsrGraph,
    max_passes: StrictUsize,
    resolution: StrictF64,
    min_modularity_gain: StrictF64,
    seed: StrictU64,
) -> PyResult<Py<PyDict>> {
    let cfg = LouvainConfig {
        max_passes: max_passes.0,
        resolution: resolution.0,
        min_modularity_gain: min_modularity_gain.0,
        seed: seed.0,
    };
    let g = graph.inner();
    let res = py
        .detach(|| clu_louvain(g, &cfg))
        .map_err(core_error_to_py)?;
    let dict = PyDict::new(py);
    dict.set_item("labels", PyArray1::from_vec(py, res.labels))?;
    dict.set_item("num_communities", res.num_communities)?;
    dict.set_item("modularity", res.modularity)?;
    dict.set_item("passes", res.passes)?;
    Ok(dict.unbind())
}

/// Detect communities on an undirected `CsrGraph` via asynchronous deterministic Label Propagation.
///
/// Returns a `uint32` NumPy label array of length `N`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, max_iters = StrictUsize(100), seed = StrictU64(42)),
    text_signature = "(graph, max_iters=100, seed=42)"
)]
fn label_propagation<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    max_iters: StrictUsize,
    seed: StrictU64,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let cfg = LabelPropagationConfig {
        max_iters: max_iters.0,
        seed: seed.0,
    };
    let g = graph.inner();
    let labels = py
        .detach(|| clu_label_propagation(g, &cfg))
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, labels))
}

type CoarsenReturn<'py> = (
    PyCsrGraph,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    usize,
);

/// Coarsen an undirected `CsrGraph` by contracting vertices sharing a cluster label.
///
/// Returns `(coarse_graph, node_to_coarse, coarse_sizes, num_clusters)`.
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, labels), text_signature = "(graph, labels)")]
fn coarsen_graph<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    labels: &Bound<'py, PyAny>,
) -> PyResult<CoarsenReturn<'py>> {
    let labels_vec = extract_u32_1d(labels, "labels")?;
    let g = graph.inner();
    let res = py
        .detach(|| clu_coarsen_graph(g, &labels_vec))
        .map_err(core_error_to_py)?;
    Ok((
        PyCsrGraph::from_csr_graph(res.graph),
        PyArray1::from_vec(py, res.node_to_coarse),
        PyArray1::from_vec(py, res.coarse_sizes),
        res.num_clusters,
    ))
}

/// Project a coarse cluster `Layout2d` onto fine vertices specified by `node_to_coarse`.
///
/// Preserves `coarse_layout.dtype` (`"float64"` or `"float32"`). Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (coarse_layout, node_to_coarse, spread_radius = StrictF64(0.1), seed = StrictU64(42)),
    text_signature = "(coarse_layout, node_to_coarse, spread_radius=0.1, seed=42)"
)]
fn project_clustered_layout(
    py: Python<'_>,
    coarse_layout: &PyLayout2d,
    node_to_coarse: &Bound<'_, PyAny>,
    spread_radius: StrictF64,
    seed: StrictU64,
) -> PyResult<PyLayout2d> {
    if !spread_radius.0.is_finite() {
        return Err(PyValueError::new_err(
            "non-finite floating-point value in spread_radius",
        ));
    }
    let n2c = extract_u32_1d(node_to_coarse, "node_to_coarse")?;
    let r_f64 = spread_radius.0;
    let seed_val = seed.0;
    py.detach(|| match coarse_layout.inner() {
        Layout2dInner::F64(buf) => {
            let soa = buf.to_layout2d();
            let fine = clu_project_clustered_layout(&soa, &n2c, r_f64, seed_val)?;
            Ok(PyLayout2d::from_f64_layout(&fine))
        }
        Layout2dInner::F32(buf) => {
            let r_f32 = r_f64 as f32;
            if !r_f32.is_finite() {
                return Err(CoreError::non_finite(
                    "project_clustered_layout::spread_radius",
                ));
            }
            let soa = buf.to_layout2d();
            let fine = clu_project_clustered_layout(&soa, &n2c, r_f32, seed_val)?;
            Ok(PyLayout2d::from_f32_layout(&fine))
        }
    })
    .map_err(core_error_to_py)
}

// ---------------------------------------------------------------------------
// Non-Euclidean `ManifoldLayout2d`
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum ManifoldLayout2dInner {
    F64(ManifoldLayout2d<f64>),
    F32(ManifoldLayout2d<f32>),
}

/// 2D node coordinate layout constrained to a non-Euclidean manifold (`"hyperbolic"`, `"spherical"`, or `"torus"`).
///
/// Supports both `"float64"` and `"float32"` precision.
#[pyclass(name = "ManifoldLayout2d", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyManifoldLayout2d {
    inner: ManifoldLayout2dInner,
}

impl PyManifoldLayout2d {
    fn dtype_enum(&self) -> ScalarDtype {
        match &self.inner {
            ManifoldLayout2dInner::F64(_) => ScalarDtype::F64,
            ManifoldLayout2dInner::F32(_) => ScalarDtype::F32,
        }
    }
}

#[pymethods]
impl PyManifoldLayout2d {
    /// Create a zero-initialized `ManifoldLayout2d` of `n` vertices on `kind`.
    #[staticmethod]
    #[pyo3(
        signature = (kind, n, dtype = ScalarDtype::F64),
        text_signature = "(kind, n, dtype='float64')"
    )]
    fn zeros(kind: &str, n: StrictUsize, dtype: ScalarDtype) -> PyResult<Self> {
        let mk = parse_manifold_kind(kind).map_err(PyValueError::new_err)?;
        let inner = match dtype {
            ScalarDtype::F64 => ManifoldLayout2dInner::F64(ManifoldLayout2d::zeros(mk, n.0)),
            ScalarDtype::F32 => ManifoldLayout2dInner::F32(ManifoldLayout2d::zeros(mk, n.0)),
        };
        Ok(Self { inner })
    }

    /// Create a zero-initialized Poincaré disk `ManifoldLayout2d` for `n` vertices.
    #[staticmethod]
    #[pyo3(signature = (n, dtype = ScalarDtype::F64), text_signature = "(n, dtype='float64')")]
    fn hyperbolic_poincare(n: StrictUsize, dtype: ScalarDtype) -> PyResult<Self> {
        let inner = match dtype {
            ScalarDtype::F64 => ManifoldLayout2dInner::F64(ManifoldLayout2d::zeros(
                ManifoldKind::HyperbolicPoincare,
                n.0,
            )),
            ScalarDtype::F32 => ManifoldLayout2dInner::F32(ManifoldLayout2d::zeros(
                ManifoldKind::HyperbolicPoincare,
                n.0,
            )),
        };
        Ok(Self { inner })
    }

    /// Create a zero-initialized spherical `ManifoldLayout2d` for `n` vertices.
    #[staticmethod]
    #[pyo3(signature = (n, dtype = ScalarDtype::F64), text_signature = "(n, dtype='float64')")]
    fn spherical(n: StrictUsize, dtype: ScalarDtype) -> PyResult<Self> {
        let inner = match dtype {
            ScalarDtype::F64 => {
                ManifoldLayout2dInner::F64(ManifoldLayout2d::zeros(ManifoldKind::Spherical, n.0))
            }
            ScalarDtype::F32 => {
                ManifoldLayout2dInner::F32(ManifoldLayout2d::zeros(ManifoldKind::Spherical, n.0))
            }
        };
        Ok(Self { inner })
    }

    /// Create a zero-initialized flat unit torus `ManifoldLayout2d` for `n` vertices.
    #[staticmethod]
    #[pyo3(signature = (n, dtype = ScalarDtype::F64), text_signature = "(n, dtype='float64')")]
    fn torus_2d(n: StrictUsize, dtype: ScalarDtype) -> PyResult<Self> {
        let inner = match dtype {
            ScalarDtype::F64 => {
                ManifoldLayout2dInner::F64(ManifoldLayout2d::zeros(ManifoldKind::Torus2d, n.0))
            }
            ScalarDtype::F32 => {
                ManifoldLayout2dInner::F32(ManifoldLayout2d::zeros(ManifoldKind::Torus2d, n.0))
            }
        };
        Ok(Self { inner })
    }

    /// Wrap an existing Euclidean `Layout2d` and project all points onto manifold `kind`.
    #[staticmethod]
    #[pyo3(signature = (kind, layout), text_signature = "(kind, layout)")]
    fn from_layout(py: Python<'_>, kind: &str, layout: &PyLayout2d) -> PyResult<Self> {
        let mk = parse_manifold_kind(kind).map_err(PyValueError::new_err)?;
        py.detach(|| match layout.inner() {
            Layout2dInner::F64(buf) => {
                ManifoldLayout2d::from_layout(mk, buf.to_layout2d()).map(ManifoldLayout2dInner::F64)
            }
            Layout2dInner::F32(buf) => {
                ManifoldLayout2d::from_layout(mk, buf.to_layout2d()).map(ManifoldLayout2dInner::F32)
            }
        })
        .map(|inner| Self { inner })
        .map_err(core_error_to_py)
    }

    /// Construct a `ManifoldLayout2d` from an `(N, 2)` array or sequence and project onto `kind`.
    #[staticmethod]
    #[pyo3(signature = (kind, coords), text_signature = "(kind, coords)")]
    fn from_coords(py: Python<'_>, kind: &str, coords: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mk = parse_manifold_kind(kind).map_err(PyValueError::new_err)?;
        let buf_inner = extract_layout_inner(coords)?;
        py.detach(|| match buf_inner {
            Layout2dInner::F64(buf) => {
                ManifoldLayout2d::from_layout(mk, buf.to_layout2d()).map(ManifoldLayout2dInner::F64)
            }
            Layout2dInner::F32(buf) => {
                ManifoldLayout2d::from_layout(mk, buf.to_layout2d()).map(ManifoldLayout2dInner::F32)
            }
        })
        .map(|inner| Self { inner })
        .map_err(core_error_to_py)
    }

    /// Canonical manifold kind name (`"hyperbolic"`, `"spherical"`, or `"torus"`).
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            ManifoldLayout2dInner::F64(m) => manifold_kind_str(m.kind()),
            ManifoldLayout2dInner::F32(m) => manifold_kind_str(m.kind()),
        }
    }

    /// Alias for `kind` matching WASM `ManifoldLayout2d.manifold`.
    #[getter]
    fn manifold(&self) -> &'static str {
        self.kind()
    }

    /// Scalar coordinate precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype_enum().as_str()
    }

    /// Number of vertices in the manifold layout.
    fn len(&self) -> usize {
        match &self.inner {
            ManifoldLayout2dInner::F64(m) => m.len(),
            ManifoldLayout2dInner::F32(m) => m.len(),
        }
    }

    /// Number of vertices in the manifold layout.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.len()
    }

    /// Return `True` if the manifold layout contains zero vertices.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of vertices in the manifold layout.
    fn __len__(&self) -> usize {
        self.len()
    }

    /// Return a newly allocated `(N, 2)` NumPy coordinate array matching `self.dtype`.
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            ManifoldLayout2dInner::F64(m) => {
                let interleaved = m.as_layout().to_interleaved();
                let arr = Array2::from_shape_vec((m.len(), 2), interleaved)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
            ManifoldLayout2dInner::F32(m) => {
                let interleaved = m.as_layout().to_interleaved();
                let arr = Array2::from_shape_vec((m.len(), 2), interleaved)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
        }
    }

    /// Convert to a Euclidean `Layout2d` copy with matching precision.
    fn to_layout(&self) -> PyLayout2d {
        match &self.inner {
            ManifoldLayout2dInner::F64(m) => PyLayout2d::from_f64_layout(m.as_layout()),
            ManifoldLayout2dInner::F32(m) => PyLayout2d::from_f32_layout(m.as_layout()),
        }
    }

    /// Return the first coordinate (`x` or longitude or toroidal `x`) of vertex `i`.
    #[pyo3(signature = (i), text_signature = "($self, i)")]
    fn x(&self, i: StrictUsize) -> PyResult<f64> {
        let n = self.len();
        if i.0 >= n {
            return Err(PyIndexError::new_err(format!(
                "ManifoldLayout2d::x index {} out of bounds for length {}",
                i.0, n
            )));
        }
        Ok(match &self.inner {
            ManifoldLayout2dInner::F64(m) => m.x()[i.0],
            ManifoldLayout2dInner::F32(m) => f64::from(m.x()[i.0]),
        })
    }

    /// Return the second coordinate (`y` or colatitude or toroidal `y`) of vertex `i`.
    #[pyo3(signature = (i), text_signature = "($self, i)")]
    fn y(&self, i: StrictUsize) -> PyResult<f64> {
        let n = self.len();
        if i.0 >= n {
            return Err(PyIndexError::new_err(format!(
                "ManifoldLayout2d::y index {} out of bounds for length {}",
                i.0, n
            )));
        }
        Ok(match &self.inner {
            ManifoldLayout2dInner::F64(m) => m.y()[i.0],
            ManifoldLayout2dInner::F32(m) => f64::from(m.y()[i.0]),
        })
    }

    /// Return `True` if every coordinate in the manifold layout is finite.
    fn is_all_finite(&self) -> bool {
        match &self.inner {
            ManifoldLayout2dInner::F64(m) => {
                m.x().iter().all(|v| v.is_finite()) && m.y().iter().all(|v| v.is_finite())
            }
            ManifoldLayout2dInner::F32(m) => {
                m.x().iter().all(|v| v.is_finite()) && m.y().iter().all(|v| v.is_finite())
            }
        }
    }

    /// Set vertex `index`'s coordinates, projecting onto the manifold domain.
    #[pyo3(signature = (index, x, y), text_signature = "($self, index, x, y)")]
    fn set_point(&mut self, index: StrictUsize, x: StrictF64, y: StrictF64) -> PyResult<()> {
        match &mut self.inner {
            ManifoldLayout2dInner::F64(m) => {
                m.set_point(index.0, x.0, y.0).map_err(core_error_to_py)
            }
            ManifoldLayout2dInner::F32(m) => {
                let x32 = x.0 as f32;
                let y32 = y.0 as f32;
                if x.0.is_finite() && !x32.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "ManifoldLayout2d.set_point x={} overflows float32",
                        x.0
                    )));
                }
                if y.0.is_finite() && !y32.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "ManifoldLayout2d.set_point y={} overflows float32",
                        y.0
                    )));
                }
                m.set_point(index.0, x32, y32).map_err(core_error_to_py)
            }
        }
    }

    /// Compute the exact geodesic distance between vertices `i` and `j` on the manifold.
    #[pyo3(signature = (i, j), text_signature = "($self, i, j)")]
    fn geodesic_distance(&self, i: StrictUsize, j: StrictUsize) -> PyResult<f64> {
        match &self.inner {
            ManifoldLayout2dInner::F64(m) => {
                m.geodesic_distance(i.0, j.0).map_err(core_error_to_py)
            }
            ManifoldLayout2dInner::F32(m) => m
                .geodesic_distance(i.0, j.0)
                .map(f64::from)
                .map_err(core_error_to_py),
        }
    }

    /// Compute the Riemannian logarithmic map (tangent vector at `i` pointing toward `j`).
    #[pyo3(signature = (i, j), text_signature = "($self, i, j)")]
    fn tangent_delta(&self, i: StrictUsize, j: StrictUsize) -> PyResult<(f64, f64)> {
        match &self.inner {
            ManifoldLayout2dInner::F64(m) => m.tangent_delta(i.0, j.0).map_err(core_error_to_py),
            ManifoldLayout2dInner::F32(m) => m
                .tangent_delta(i.0, j.0)
                .map(|(vx, vy)| (f64::from(vx), f64::from(vy)))
                .map_err(core_error_to_py),
        }
    }

    /// Advance vertex `i` along the geodesic defined by tangent vector `(vx, vy)`.
    #[pyo3(signature = (i, vx, vy), text_signature = "($self, i, vx, vy)")]
    fn apply_tangent_step(&mut self, i: StrictUsize, vx: StrictF64, vy: StrictF64) -> PyResult<()> {
        match &mut self.inner {
            ManifoldLayout2dInner::F64(m) => m
                .apply_tangent_step(i.0, vx.0, vy.0)
                .map_err(core_error_to_py),
            ManifoldLayout2dInner::F32(m) => {
                let vx32 = vx.0 as f32;
                let vy32 = vy.0 as f32;
                if vx.0.is_finite() && !vx32.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "ManifoldLayout2d.apply_tangent_step vx={} overflows float32",
                        vx.0
                    )));
                }
                if vy.0.is_finite() && !vy32.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "ManifoldLayout2d.apply_tangent_step vy={} overflows float32",
                        vy.0
                    )));
                }
                m.apply_tangent_step(i.0, vx32, vy32)
                    .map_err(core_error_to_py)
            }
        }
    }

    /// Project all coordinates in place onto the manifold's canonical domain.
    fn project_all(&mut self) -> PyResult<()> {
        match &mut self.inner {
            ManifoldLayout2dInner::F64(m) => m.project_all().map_err(core_error_to_py),
            ManifoldLayout2dInner::F32(m) => m.project_all().map_err(core_error_to_py),
        }
    }

    /// Return a concise string representation of the `ManifoldLayout2d`.
    fn __repr__(&self) -> String {
        format!(
            "ManifoldLayout2d(kind='{}', len={}, dtype='{}')",
            self.kind(),
            self.len(),
            self.dtype()
        )
    }
}

/// Compute exact Poincaré disk hyperbolic geodesic distance between `(x1, y1)` and `(x2, y2)`.
#[pyfunction]
#[pyo3(signature = (x1, y1, x2, y2), text_signature = "(x1, y1, x2, y2)")]
fn poincare_distance(x1: StrictF64, y1: StrictF64, x2: StrictF64, y2: StrictF64) -> f64 {
    core_poincare_distance(x1.0, y1.0, x2.0, y2.0)
}

/// Compute great-circle distance on the unit sphere given `(lon, colatitude)` in radians.
#[pyfunction]
#[pyo3(signature = (lon1, colat1, lon2, colat2), text_signature = "(lon1, colat1, lon2, colat2)")]
fn spherical_distance(
    lon1: StrictF64,
    colat1: StrictF64,
    lon2: StrictF64,
    colat2: StrictF64,
) -> f64 {
    core_spherical_distance(lon1.0, colat1.0, lon2.0, colat2.0)
}

/// Compute shortest periodic Euclidean distance on the flat unit torus `[0, 1)^2`.
#[pyfunction]
#[pyo3(signature = (x1, y1, x2, y2), text_signature = "(x1, y1, x2, y2)")]
fn torus_distance(x1: StrictF64, y1: StrictF64, x2: StrictF64, y2: StrictF64) -> f64 {
    core_torus_distance(x1.0, y1.0, x2.0, y2.0)
}

/// Wrap `(x, y)` periodically into the fundamental domain `[0, 1)^2` of the unit torus.
#[pyfunction]
#[pyo3(signature = (x, y), text_signature = "(x, y)")]
fn wrap_torus_point(x: StrictF64, y: StrictF64) -> (f64, f64) {
    core_wrap_torus_point(x.0, y.0)
}

// ---------------------------------------------------------------------------
// 1. TsNetSolver
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_tsnet_config(
    backend: &str,
    perplexity_c: f64,
    min_perplexity: f64,
    max_perplexity: f64,
    power: f64,
    sigma_iters: usize,
    sigma_tolerance: f64,
    row_normalize: bool,
    iterations_stage1: usize,
    iterations_stage2: usize,
    iterations_stage3: usize,
    exaggeration: f64,
    lambda_c_stage1: f64,
    lambda_c_stage2: f64,
    lambda_c_stage3: f64,
    lambda_r_stage3: f64,
    learning_rate: Option<f64>,
    momentum: f64,
    epsilon_r: f64,
    theta: f64,
    intervals: usize,
    interpolation_order: usize,
    min_grid_dim: usize,
    max_grid_dim: usize,
    seed: u64,
) -> PyResult<TsNetConfig> {
    let parsed_backend = parse_tsnet_backend(backend).map_err(PyValueError::new_err)?;
    let order = InterpolationOrder::from_order(interpolation_order).map_err(core_error_to_py)?;
    let cfg = TsNetConfig {
        backend: parsed_backend,
        perplexity_c,
        min_perplexity,
        max_perplexity,
        power,
        sigma_iters,
        sigma_tolerance,
        symmetrization: if row_normalize {
            SymmetrizationMode::RowNormalized
        } else {
            SymmetrizationMode::StandardTsne
        },
        iterations_stage1,
        iterations_stage2,
        iterations_stage3,
        exaggeration,
        lambda_c_stage1,
        lambda_c_stage2,
        lambda_c_stage3,
        lambda_r_stage3,
        learning_rate,
        momentum,
        epsilon_r,
        theta,
        ibfft: IbFftConfig {
            order,
            intervals,
            min_grid_dim,
            max_grid_dim,
        },
        seed,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

/// Cooperative unified tsNET solver supporting Exact, Barnes-Hut, FIt-tsNET (`"fft"`), and L-tsNET (`"linear_fft"`).
///
/// Supports both `"float64"` and `"float32"` `Layout2d` and releases the Python GIL during `step()`, `run_steps()`, and `run()`.
#[pyclass(name = "TsNetSolver", module = "egraph._core")]
pub struct PyTsNetSolver {
    inner: TsNetInner,
}

#[pymethods]
impl PyTsNetSolver {
    /// Construct a `TsNetSolver` directly from a `CsrGraph` using the Pure 1+2-Hop adaptive perplexity preset.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            backend = "fft",
            *,
            perplexity_c = StrictF64(0.9),
            min_perplexity = StrictF64(5.0),
            max_perplexity = StrictF64(1200.0),
            power = StrictF64(2.0),
            sigma_iters = StrictUsize(50),
            sigma_tolerance = StrictF64(1e-5),
            row_normalize = false,
            iterations_stage1 = StrictUsize(250),
            iterations_stage2 = StrictUsize(250),
            iterations_stage3 = StrictUsize(500),
            exaggeration = StrictF64(4.0),
            lambda_c_stage1 = StrictF64(0.1),
            lambda_c_stage2 = StrictF64(0.01),
            lambda_c_stage3 = StrictF64(0.0),
            lambda_r_stage3 = StrictF64(0.0),
            learning_rate = None,
            momentum = StrictF64(0.8),
            epsilon_r = StrictF64(0.05),
            theta = StrictF64(0.5),
            intervals = StrictUsize(25),
            interpolation_order = StrictUsize(1),
            min_grid_dim = StrictUsize(64),
            max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, backend='fft', *, perplexity_c=0.9, min_perplexity=5.0, max_perplexity=1200.0, power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False, iterations_stage1=250, iterations_stage2=250, iterations_stage3=500, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=1, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn pure_2hop(
        py: Python<'_>,
        graph: &PyCsrGraph,
        backend: &str,
        perplexity_c: StrictF64,
        min_perplexity: StrictF64,
        max_perplexity: StrictF64,
        power: StrictF64,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        row_normalize: bool,
        iterations_stage1: StrictUsize,
        iterations_stage2: StrictUsize,
        iterations_stage3: StrictUsize,
        exaggeration: StrictF64,
        lambda_c_stage1: StrictF64,
        lambda_c_stage2: StrictF64,
        lambda_c_stage3: StrictF64,
        lambda_r_stage3: StrictF64,
        learning_rate: Option<StrictF64>,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_tsnet_config(
            backend,
            perplexity_c.0,
            min_perplexity.0,
            max_perplexity.0,
            power.0,
            sigma_iters.0,
            sigma_tolerance.0,
            row_normalize,
            iterations_stage1.0,
            iterations_stage2.0,
            iterations_stage3.0,
            exaggeration.0,
            lambda_c_stage1.0,
            lambda_c_stage2.0,
            lambda_c_stage3.0,
            lambda_r_stage3.0,
            learning_rate.map(|v| v.0),
            momentum.0,
            epsilon_r.0,
            theta.0,
            intervals.0,
            interpolation_order.0,
            min_grid_dim.0,
            max_grid_dim.0,
            seed.0,
        )?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    TsNetSolver::<f64>::from_graph_pure_2hop(g, cfg).map(TsNetInner::F64)
                }
                ScalarDtype::F32 => {
                    TsNetSolver::<f32>::from_graph_pure_2hop(g, cfg).map(TsNetInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `TsNetSolver` from a precomputed `KnnGraph` and optional per-node or scalar perplexities.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            knn,
            perplexities = None,
            uniform_perplexity = StrictF64(15.0),
            *,
            backend = "fft",
            power = StrictF64(2.0),
            sigma_iters = StrictUsize(50),
            sigma_tolerance = StrictF64(1e-5),
            row_normalize = false,
            iterations_stage1 = StrictUsize(250),
            iterations_stage2 = StrictUsize(250),
            iterations_stage3 = StrictUsize(500),
            exaggeration = StrictF64(4.0),
            lambda_c_stage1 = StrictF64(0.1),
            lambda_c_stage2 = StrictF64(0.01),
            lambda_c_stage3 = StrictF64(0.0),
            lambda_r_stage3 = StrictF64(0.0),
            learning_rate = None,
            momentum = StrictF64(0.8),
            epsilon_r = StrictF64(0.05),
            theta = StrictF64(0.5),
            intervals = StrictUsize(25),
            interpolation_order = StrictUsize(1),
            min_grid_dim = StrictUsize(64),
            max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(knn, perplexities=None, uniform_perplexity=15.0, *, backend='fft', power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False, iterations_stage1=250, iterations_stage2=250, iterations_stage3=500, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=1, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn: &PyKnnGraph,
        perplexities: Option<&Bound<'_, PyAny>>,
        uniform_perplexity: StrictF64,
        backend: &str,
        power: StrictF64,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        row_normalize: bool,
        iterations_stage1: StrictUsize,
        iterations_stage2: StrictUsize,
        iterations_stage3: StrictUsize,
        exaggeration: StrictF64,
        lambda_c_stage1: StrictF64,
        lambda_c_stage2: StrictF64,
        lambda_c_stage3: StrictF64,
        lambda_r_stage3: StrictF64,
        learning_rate: Option<StrictF64>,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_tsnet_config(
            backend,
            0.9,
            5.0,
            1200.0,
            power.0,
            sigma_iters.0,
            sigma_tolerance.0,
            row_normalize,
            iterations_stage1.0,
            iterations_stage2.0,
            iterations_stage3.0,
            exaggeration.0,
            lambda_c_stage1.0,
            lambda_c_stage2.0,
            lambda_c_stage3.0,
            lambda_r_stage3.0,
            learning_rate.map(|v| v.0),
            momentum.0,
            epsilon_r.0,
            theta.0,
            intervals.0,
            interpolation_order.0,
            min_grid_dim.0,
            max_grid_dim.0,
            seed.0,
        )?;
        let knn_ref = knn.inner();
        let perps_opt = match perplexities {
            Some(obj) if !obj.is_none() => Some(extract_perplexities(obj, knn_ref.num_nodes())?),
            _ => None,
        };
        let u_perp = uniform_perplexity.0;
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    TsNetSolver::<f64>::from_knn(knn_ref, perps_opt.as_deref(), u_perp, cfg)
                        .map(TsNetInner::F64)
                }
                ScalarDtype::F32 => {
                    TsNetSolver::<f32>::from_knn(knn_ref, perps_opt.as_deref(), u_perp, cfg)
                        .map(TsNetInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `TsNetSolver` directly from precomputed symmetric joint `SparseProbabilities`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            probs,
            backend = "fft",
            *,
            iterations_stage1 = StrictUsize(250),
            iterations_stage2 = StrictUsize(250),
            iterations_stage3 = StrictUsize(500),
            exaggeration = StrictF64(4.0),
            lambda_c_stage1 = StrictF64(0.1),
            lambda_c_stage2 = StrictF64(0.01),
            lambda_c_stage3 = StrictF64(0.0),
            lambda_r_stage3 = StrictF64(0.0),
            learning_rate = None,
            momentum = StrictF64(0.8),
            epsilon_r = StrictF64(0.05),
            theta = StrictF64(0.5),
            intervals = StrictUsize(25),
            interpolation_order = StrictUsize(1),
            min_grid_dim = StrictUsize(64),
            max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(probs, backend='fft', *, iterations_stage1=250, iterations_stage2=250, iterations_stage3=500, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=1, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_probabilities(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        backend: &str,
        iterations_stage1: StrictUsize,
        iterations_stage2: StrictUsize,
        iterations_stage3: StrictUsize,
        exaggeration: StrictF64,
        lambda_c_stage1: StrictF64,
        lambda_c_stage2: StrictF64,
        lambda_c_stage3: StrictF64,
        lambda_r_stage3: StrictF64,
        learning_rate: Option<StrictF64>,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_tsnet_config(
            backend,
            0.9,
            5.0,
            1200.0,
            2.0,
            50,
            1e-5,
            false,
            iterations_stage1.0,
            iterations_stage2.0,
            iterations_stage3.0,
            exaggeration.0,
            lambda_c_stage1.0,
            lambda_c_stage2.0,
            lambda_c_stage3.0,
            lambda_r_stage3.0,
            learning_rate.map(|v| v.0),
            momentum.0,
            epsilon_r.0,
            theta.0,
            intervals.0,
            interpolation_order.0,
            min_grid_dim.0,
            max_grid_dim.0,
            seed.0,
        )?;
        let probs_owned = probs.inner().clone();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    TsNetSolver::<f64>::from_probabilities(probs_owned, cfg).map(TsNetInner::F64)
                }
                ScalarDtype::F32 => {
                    TsNetSolver::<f32>::from_probabilities(probs_owned, cfg).map(TsNetInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `TsNetSolver` from a `CsrGraph` using partial-BFS k-NN and uniform perplexity.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            k = StrictUsize(15),
            uniform_perplexity = StrictF64(15.0),
            *,
            backend = "fft",
            power = StrictF64(2.0),
            sigma_iters = StrictUsize(50),
            sigma_tolerance = StrictF64(1e-5),
            row_normalize = false,
            iterations_stage1 = StrictUsize(250),
            iterations_stage2 = StrictUsize(250),
            iterations_stage3 = StrictUsize(500),
            exaggeration = StrictF64(4.0),
            lambda_c_stage1 = StrictF64(0.1),
            lambda_c_stage2 = StrictF64(0.01),
            lambda_c_stage3 = StrictF64(0.0),
            lambda_r_stage3 = StrictF64(0.0),
            learning_rate = None,
            momentum = StrictF64(0.8),
            epsilon_r = StrictF64(0.05),
            theta = StrictF64(0.5),
            intervals = StrictUsize(25),
            interpolation_order = StrictUsize(1),
            min_grid_dim = StrictUsize(64),
            max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, k=15, uniform_perplexity=15.0, *, backend='fft', power=2.0, sigma_iters=50, sigma_tolerance=1e-5, row_normalize=False, iterations_stage1=250, iterations_stage2=250, iterations_stage3=500, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=1, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_graph_partial_bfs(
        py: Python<'_>,
        graph: &PyCsrGraph,
        k: StrictUsize,
        uniform_perplexity: StrictF64,
        backend: &str,
        power: StrictF64,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        row_normalize: bool,
        iterations_stage1: StrictUsize,
        iterations_stage2: StrictUsize,
        iterations_stage3: StrictUsize,
        exaggeration: StrictF64,
        lambda_c_stage1: StrictF64,
        lambda_c_stage2: StrictF64,
        lambda_c_stage3: StrictF64,
        lambda_r_stage3: StrictF64,
        learning_rate: Option<StrictF64>,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_tsnet_config(
            backend,
            0.9,
            5.0,
            1200.0,
            power.0,
            sigma_iters.0,
            sigma_tolerance.0,
            row_normalize,
            iterations_stage1.0,
            iterations_stage2.0,
            iterations_stage3.0,
            exaggeration.0,
            lambda_c_stage1.0,
            lambda_c_stage2.0,
            lambda_c_stage3.0,
            lambda_r_stage3.0,
            learning_rate.map(|v| v.0),
            momentum.0,
            epsilon_r.0,
            theta.0,
            intervals.0,
            interpolation_order.0,
            min_grid_dim.0,
            max_grid_dim.0,
            seed.0,
        )?;
        let g = graph.inner();
        let k_val = k.0;
        let u_perp = uniform_perplexity.0;
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    TsNetSolver::<f64>::from_graph_partial_bfs(g, k_val, u_perp, cfg)
                        .map(TsNetInner::F64)
                }
                ScalarDtype::F32 => {
                    TsNetSolver::<f32>::from_graph_partial_bfs(g, k_val, u_perp, cfg)
                        .map(TsNetInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `TsNetSolver` from a `CsrGraph` using direct row-normalized `A1 + w2 * A2` affinity.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            w2 = StrictF64(0.5),
            max_2hop_per_node = None,
            *,
            backend = "fft",
            iterations_stage1 = StrictUsize(250),
            iterations_stage2 = StrictUsize(250),
            iterations_stage3 = StrictUsize(500),
            exaggeration = StrictF64(4.0),
            lambda_c_stage1 = StrictF64(0.1),
            lambda_c_stage2 = StrictF64(0.01),
            lambda_c_stage3 = StrictF64(0.0),
            lambda_r_stage3 = StrictF64(0.0),
            learning_rate = None,
            momentum = StrictF64(0.8),
            epsilon_r = StrictF64(0.05),
            theta = StrictF64(0.5),
            intervals = StrictUsize(25),
            interpolation_order = StrictUsize(1),
            min_grid_dim = StrictUsize(64),
            max_grid_dim = StrictUsize(512),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, w2=0.5, max_2hop_per_node=None, *, backend='fft', iterations_stage1=250, iterations_stage2=250, iterations_stage3=500, exaggeration=4.0, lambda_c_stage1=0.1, lambda_c_stage2=0.01, lambda_c_stage3=0.0, lambda_r_stage3=0.0, learning_rate=None, momentum=0.8, epsilon_r=0.05, theta=0.5, intervals=25, interpolation_order=1, min_grid_dim=64, max_grid_dim=512, seed=42, dtype='float64')"
    )]
    fn from_graph_row_normalized(
        py: Python<'_>,
        graph: &PyCsrGraph,
        w2: StrictF64,
        max_2hop_per_node: Option<StrictUsize>,
        backend: &str,
        iterations_stage1: StrictUsize,
        iterations_stage2: StrictUsize,
        iterations_stage3: StrictUsize,
        exaggeration: StrictF64,
        lambda_c_stage1: StrictF64,
        lambda_c_stage2: StrictF64,
        lambda_c_stage3: StrictF64,
        lambda_r_stage3: StrictF64,
        learning_rate: Option<StrictF64>,
        momentum: StrictF64,
        epsilon_r: StrictF64,
        theta: StrictF64,
        intervals: StrictUsize,
        interpolation_order: StrictUsize,
        min_grid_dim: StrictUsize,
        max_grid_dim: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_tsnet_config(
            backend,
            0.9,
            5.0,
            1200.0,
            2.0,
            50,
            1e-5,
            false,
            iterations_stage1.0,
            iterations_stage2.0,
            iterations_stage3.0,
            exaggeration.0,
            lambda_c_stage1.0,
            lambda_c_stage2.0,
            lambda_c_stage3.0,
            lambda_r_stage3.0,
            learning_rate.map(|v| v.0),
            momentum.0,
            epsilon_r.0,
            theta.0,
            intervals.0,
            interpolation_order.0,
            min_grid_dim.0,
            max_grid_dim.0,
            seed.0,
        )?;
        let g = graph.inner();
        let w2_val = w2.0;
        let max_2hop_val = max_2hop_per_node.map(|v| v.0);
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    TsNetSolver::<f64>::from_graph_row_normalized(g, w2_val, max_2hop_val, cfg)
                        .map(TsNetInner::F64)
                }
                ScalarDtype::F32 => {
                    TsNetSolver::<f32>::from_graph_row_normalized(g, w2_val, max_2hop_val, cfg)
                        .map(TsNetInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute a single optimization step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` optimization iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run optimization to completion (or early convergence) in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Total scheduled iterations across all stages.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Cumulative number of 2D FFT transforms executed across all steps.
    #[getter]
    fn total_fft_2d_calls(&self) -> u64 {
        match &self.inner {
            TsNetInner::F64(s) => s.total_fft_2d_calls(),
            TsNetInner::F32(s) => s.total_fft_2d_calls(),
        }
    }

    /// Return telemetry counters from the most recent FFT field evaluation, or `None`.
    fn last_fft_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let stats = match &self.inner {
            TsNetInner::F64(s) => s.last_fft_stats(),
            TsNetInner::F32(s) => s.last_fft_stats(),
        };
        match stats {
            Some(st) => {
                let d = PyDict::new(py);
                d.set_item("particle_forward_ffts", st.particle_forward_ffts)?;
                d.set_item("kernel_forward_ffts", st.kernel_forward_ffts)?;
                d.set_item("inverse_ffts", st.inverse_ffts)?;
                d.set_item("total_fft_2d_calls", st.total_fft_2d_calls)?;
                Ok(Some(d))
            }
            None => Ok(None),
        }
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            TsNetInner::F64(s) => s.num_nodes(),
            TsNetInner::F32(s) => s.num_nodes(),
        }
    }

    /// Active or final optimization stage index (`0`, `1`, or `2`).
    #[getter]
    fn stage_index(&self) -> usize {
        match &self.inner {
            TsNetInner::F64(s) => s.stage_index(),
            TsNetInner::F32(s) => s.stage_index(),
        }
    }

    /// Stability-capped learning rate of the current stage.
    #[getter]
    fn current_stage_lr(&self) -> f64 {
        match &self.inner {
            TsNetInner::F64(s) => s.current_stage_lr(),
            TsNetInner::F32(s) => s.current_stage_lr(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `TsNetSolver`.
    fn __repr__(&self) -> String {
        format!(
            "TsNetSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

// ---------------------------------------------------------------------------
// 2. SgdSolver
// ---------------------------------------------------------------------------

fn build_schedule_config(
    t_max: usize,
    eps: f64,
    schedule: &str,
    step_scale: f64,
) -> PyResult<SgdScheduleConfig> {
    let kind = parse_sgd_schedule_kind(schedule).map_err(PyValueError::new_err)?;
    Ok(SgdScheduleConfig {
        t_max,
        eps,
        kind,
        step_scale,
    })
}

fn default_inv_sq_weights(dists: &[f64]) -> Vec<f64> {
    dists
        .iter()
        .map(|&d| {
            if d.is_finite() && d > 0.0 {
                1.0 / (d * d)
            } else {
                0.0
            }
        })
        .collect()
}

fn sparse_barrier_stats_to_py_dict(
    py: Python<'_>,
    stats: &SparseBarrierStats,
) -> PyResult<Py<PyDict>> {
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
    dict.set_item("gamma_eff", stats.gamma_eff)?;
    Ok(dict.unbind())
}

fn dog_stats_to_py_dict(py: Python<'_>, stats: &DogVaradhanStats) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("t1", stats.t1)?;
    dict.set_item("t2", stats.t2)?;
    dict.set_item("alpha", stats.alpha)?;
    dict.set_item("degree", stats.degree)?;
    dict.set_item("delta_psd", stats.delta_psd)?;
    dict.set_item("h_min", stats.h_min)?;
    dict.set_item("k_min", stats.k_min)?;
    dict.set_item("k_med", stats.k_med)?;
    dict.set_item("beta_auto", stats.beta_auto)?;
    dict.set_item("beta_eff", stats.beta_eff)?;
    dict.set_item("n_landmarks", stats.n_landmarks)?;
    dict.set_item("t_coarse", stats.t_coarse)?;
    dict.set_item("used_coarse_scaffold", stats.used_coarse_scaffold)?;
    Ok(dict.unbind())
}

#[derive(Clone)]
enum ExtractedManifold {
    F64(ManifoldLayout2d<f64>, ScalarDtype),
    F32(ManifoldLayout2d<f32>, ScalarDtype),
}

fn extract_manifold_for_solver(
    layout: &Bound<'_, PyManifoldLayout2d>,
    target_dtype: ScalarDtype,
) -> PyResult<ExtractedManifold> {
    let borrowed = layout.try_borrow()?;
    let orig_dtype = borrowed.dtype_enum();
    match (&borrowed.inner, target_dtype) {
        (ManifoldLayout2dInner::F64(m), ScalarDtype::F64) => {
            Ok(ExtractedManifold::F64(m.clone(), orig_dtype))
        }
        (ManifoldLayout2dInner::F32(m), ScalarDtype::F32) => {
            Ok(ExtractedManifold::F32(m.clone(), orig_dtype))
        }
        (ManifoldLayout2dInner::F32(m), ScalarDtype::F64) => {
            let soa64 = layout_f32_to_f64_checked(m.as_layout()).map_err(core_error_to_py)?;
            let m64 = ManifoldLayout2d::from_layout(m.kind(), soa64).map_err(core_error_to_py)?;
            Ok(ExtractedManifold::F64(m64, orig_dtype))
        }
        (ManifoldLayout2dInner::F64(m), ScalarDtype::F32) => {
            let soa32 = layout_f64_to_f32_checked(m.as_layout()).map_err(core_error_to_py)?;
            let m32 = ManifoldLayout2d::from_layout(m.kind(), soa32).map_err(core_error_to_py)?;
            Ok(ExtractedManifold::F32(m32, orig_dtype))
        }
    }
}

fn write_back_manifold(
    layout: &Bound<'_, PyManifoldLayout2d>,
    extracted: ExtractedManifold,
) -> PyResult<()> {
    let mut borrowed = layout.try_borrow_mut()?;
    borrowed.inner = match extracted {
        ExtractedManifold::F64(m64, ScalarDtype::F64) => ManifoldLayout2dInner::F64(m64),
        ExtractedManifold::F32(m32, ScalarDtype::F32) => ManifoldLayout2dInner::F32(m32),
        ExtractedManifold::F64(m64, ScalarDtype::F32) => {
            let soa32 = layout_f64_to_f32_checked(m64.as_layout()).map_err(core_error_to_py)?;
            let m32 = ManifoldLayout2d::from_layout(m64.kind(), soa32).map_err(core_error_to_py)?;
            ManifoldLayout2dInner::F32(m32)
        }
        ExtractedManifold::F32(m32, ScalarDtype::F64) => {
            let soa64 = layout_f32_to_f64_checked(m32.as_layout()).map_err(core_error_to_py)?;
            let m64 = ManifoldLayout2d::from_layout(m32.kind(), soa64).map_err(core_error_to_py)?;
            ManifoldLayout2dInner::F64(m64)
        }
    };
    Ok(())
}

/// Cooperative SoA Stochastic Gradient Descent (SGD) layout solver supporting 3-tier Sparse Barrier
/// SGD, streaming non-edge negative sampling, Euclidean `Layout2d`, and non-Euclidean `ManifoldLayout2d`.
#[pyclass(name = "SgdSolver", module = "egraph._core")]
pub struct PySgdSolver {
    inner: SgdInner,
    barrier_stats: Option<SparseBarrierStats>,
    dog_stats: Option<DogVaradhanStats>,
}

impl PySgdSolver {
    pub(super) fn from_inner(inner: SgdInner) -> Self {
        Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        }
    }

    fn execute_manifold_action(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyManifoldLayout2d>,
        action: SolverAction,
    ) -> PyResult<&'static str> {
        let target_dtype = self.inner.dtype();
        let mut extracted = extract_manifold_for_solver(layout, target_dtype)?;
        let status = py
            .detach(|| match (&mut self.inner, &mut extracted) {
                (SgdInner::F64(s), ExtractedManifold::F64(m, _)) => match action {
                    SolverAction::Step => s.step_manifold(m),
                    SolverAction::RunSteps(steps) => s.run_steps_manifold(m, steps),
                    SolverAction::Run => s.run_manifold(m),
                },
                (SgdInner::F32(s), ExtractedManifold::F32(m, _)) => match action {
                    SolverAction::Step => s.step_manifold(m),
                    SolverAction::RunSteps(steps) => s.run_steps_manifold(m, steps),
                    SolverAction::Run => s.run_manifold(m),
                },
                _ => Err(CoreError::invalid_input(
                    "internal solver manifold dtype mismatch",
                )),
            })
            .map_err(core_error_to_py)?;
        write_back_manifold(layout, extracted)?;
        Ok(solver_status_str(status))
    }
}

#[pymethods]
impl PySgdSolver {
    /// Construct an `SgdSolver` from flat SoA two-sided spring arrays.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            num_nodes,
            src,
            tgt,
            dij,
            dji = None,
            wij = None,
            wji = None,
            *,
            t_max = StrictUsize(60),
            eps = StrictF64(0.01),
            schedule = "exponential",
            step_scale = StrictF64(1.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(num_nodes, src, tgt, dij, dji=None, wij=None, wji=None, *, t_max=60, eps=0.01, schedule='exponential', step_scale=1.0, seed=42, dtype='float64')"
    )]
    fn from_pairs(
        py: Python<'_>,
        num_nodes: StrictUsize,
        src: &Bound<'_, PyAny>,
        tgt: &Bound<'_, PyAny>,
        dij: &Bound<'_, PyAny>,
        dji: Option<&Bound<'_, PyAny>>,
        wij: Option<&Bound<'_, PyAny>>,
        wji: Option<&Bound<'_, PyAny>>,
        t_max: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let sched = build_schedule_config(t_max.0, eps.0, schedule, step_scale.0)?;
        let src_u32 = extract_u32_1d(src, "src")?;
        let tgt_u32 = extract_u32_1d(tgt, "tgt")?;
        let dij_f64 = extract_f64_1d(dij, "dij")?;
        let dji_f64 = match dji {
            Some(obj) if !obj.is_none() => extract_f64_1d(obj, "dji")?,
            _ => dij_f64.clone(),
        };
        let wij_f64 = match wij {
            Some(obj) if !obj.is_none() => extract_f64_1d(obj, "wij")?,
            _ => default_inv_sq_weights(&dij_f64),
        };
        let wji_f64 = match wji {
            Some(obj) if !obj.is_none() => extract_f64_1d(obj, "wji")?,
            _ if wij.is_none_or(|o| o.is_none()) => default_inv_sq_weights(&dji_f64),
            _ => wij_f64.clone(),
        };

        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    SgdSolver::<f64>::from_pairs(
                        num_nodes.0,
                        src_u32,
                        tgt_u32,
                        dij_f64,
                        dji_f64,
                        wij_f64,
                        wji_f64,
                        sched,
                        seed.0,
                    )
                    .map(SgdInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let dij_f32 = vec_f64_to_f32_checked(&dij_f64, "dij")?;
                let dji_f32 = vec_f64_to_f32_checked(&dji_f64, "dji")?;
                let wij_f32 = vec_f64_to_f32_checked(&wij_f64, "wij")?;
                let wji_f32 = vec_f64_to_f32_checked(&wji_f64, "wji")?;
                py.detach(|| {
                    SgdSolver::<f32>::from_pairs(
                        num_nodes.0,
                        src_u32,
                        tgt_u32,
                        dij_f32,
                        dji_f32,
                        wij_f32,
                        wji_f32,
                        sched,
                        seed.0,
                    )
                    .map(SgdInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Construct an `SgdSolver` combining flat SoA two-sided spring pairs and one-sided hinge repulsion pairs.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            num_nodes,
            spring_src,
            spring_tgt,
            spring_dij,
            spring_dji,
            spring_wij,
            spring_wji,
            hinge_src,
            hinge_tgt,
            hinge_dij,
            hinge_dji = None,
            hinge_wij = None,
            hinge_wji = None,
            *,
            t_max = StrictUsize(60),
            eps = StrictF64(0.01),
            schedule = "exponential",
            step_scale = StrictF64(1.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(num_nodes, spring_src, spring_tgt, spring_dij, spring_dji, spring_wij, spring_wji, hinge_src, hinge_tgt, hinge_dij, hinge_dji=None, hinge_wij=None, hinge_wji=None, *, t_max=60, eps=0.01, schedule='exponential', step_scale=1.0, seed=42, dtype='float64')"
    )]
    fn from_pairs_with_hinges(
        py: Python<'_>,
        num_nodes: StrictUsize,
        spring_src: &Bound<'_, PyAny>,
        spring_tgt: &Bound<'_, PyAny>,
        spring_dij: &Bound<'_, PyAny>,
        spring_dji: Option<&Bound<'_, PyAny>>,
        spring_wij: Option<&Bound<'_, PyAny>>,
        spring_wji: Option<&Bound<'_, PyAny>>,
        hinge_src: &Bound<'_, PyAny>,
        hinge_tgt: &Bound<'_, PyAny>,
        hinge_dij: &Bound<'_, PyAny>,
        hinge_dji: Option<&Bound<'_, PyAny>>,
        hinge_wij: Option<&Bound<'_, PyAny>>,
        hinge_wji: Option<&Bound<'_, PyAny>>,
        t_max: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let sched = build_schedule_config(t_max.0, eps.0, schedule, step_scale.0)?;
        let s_src = extract_u32_1d(spring_src, "spring_src")?;
        let s_tgt = extract_u32_1d(spring_tgt, "spring_tgt")?;
        let s_dij = extract_f64_1d(spring_dij, "spring_dij")?;
        let s_dji = match spring_dji {
            Some(obj) if !obj.is_none() => extract_f64_1d(obj, "spring_dji")?,
            _ => s_dij.clone(),
        };
        let s_wij = match spring_wij {
            Some(obj) if !obj.is_none() => extract_f64_1d(obj, "spring_wij")?,
            _ => default_inv_sq_weights(&s_dij),
        };
        let s_wji = match spring_wji {
            Some(obj) if !obj.is_none() => extract_f64_1d(obj, "spring_wji")?,
            _ if spring_wij.is_none_or(|o| o.is_none()) => default_inv_sq_weights(&s_dji),
            _ => s_wij.clone(),
        };

        let h_src = extract_u32_1d(hinge_src, "hinge_src")?;
        let h_tgt = extract_u32_1d(hinge_tgt, "hinge_tgt")?;
        let h_dij = extract_f64_1d(hinge_dij, "hinge_dij")?;
        let h_dji = match hinge_dji {
            Some(obj) if !obj.is_none() => extract_f64_1d(obj, "hinge_dji")?,
            _ => h_dij.clone(),
        };
        let h_wij = match hinge_wij {
            Some(obj) if !obj.is_none() => extract_f64_1d(obj, "hinge_wij")?,
            _ => default_inv_sq_weights(&h_dij),
        };
        let h_wji = match hinge_wji {
            Some(obj) if !obj.is_none() => extract_f64_1d(obj, "hinge_wji")?,
            _ if hinge_wij.is_none_or(|o| o.is_none()) => default_inv_sq_weights(&h_dji),
            _ => h_wij.clone(),
        };

        let inner = match dtype {
            ScalarDtype::F64 => py
                .detach(|| {
                    SgdSolver::<f64>::from_pairs_with_hinges(
                        num_nodes.0,
                        s_src,
                        s_tgt,
                        s_dij,
                        s_dji,
                        s_wij,
                        s_wji,
                        h_src,
                        h_tgt,
                        h_dij,
                        h_dji,
                        h_wij,
                        h_wji,
                        sched,
                        seed.0,
                    )
                    .map(SgdInner::F64)
                })
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => {
                let s_dij_32 = vec_f64_to_f32_checked(&s_dij, "spring_dij")?;
                let s_dji_32 = vec_f64_to_f32_checked(&s_dji, "spring_dji")?;
                let s_wij_32 = vec_f64_to_f32_checked(&s_wij, "spring_wij")?;
                let s_wji_32 = vec_f64_to_f32_checked(&s_wji, "spring_wji")?;
                let h_dij_32 = vec_f64_to_f32_checked(&h_dij, "hinge_dij")?;
                let h_dji_32 = vec_f64_to_f32_checked(&h_dji, "hinge_dji")?;
                let h_wij_32 = vec_f64_to_f32_checked(&h_wij, "hinge_wij")?;
                let h_wji_32 = vec_f64_to_f32_checked(&h_wji, "hinge_wji")?;
                py.detach(|| {
                    SgdSolver::<f32>::from_pairs_with_hinges(
                        num_nodes.0,
                        s_src,
                        s_tgt,
                        s_dij_32,
                        s_dji_32,
                        s_wij_32,
                        s_wji_32,
                        h_src,
                        h_tgt,
                        h_dij_32,
                        h_dji_32,
                        h_wij_32,
                        h_wji_32,
                        sched,
                        seed.0,
                    )
                    .map(SgdInner::F32)
                })
                .map_err(core_error_to_py)?
            }
        };
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Construct a full all-pairs shortest-path `SgdSolver` from `CsrGraph`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, t_max = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, t_max=60, eps=0.01, schedule='exponential', step_scale=1.0, seed=42, dtype='float64')"
    )]
    fn full(
        py: Python<'_>,
        graph: &PyCsrGraph,
        t_max: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let sched = build_schedule_config(t_max.0, eps.0, schedule, step_scale.0)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => SgdSolver::<f64>::full(g, sched, seed.0).map(SgdInner::F64),
                ScalarDtype::F32 => SgdSolver::<f32>::full(g, sched, seed.0).map(SgdInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Construct a Track B 3-tier Sparse Barrier `SgdSolver` directly from precomputed `DogVaradhanLandmarks`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            landmarks,
            pivots = None,
            *,
            max_2hop_per_node = StrictUsize(32),
            hinge_repulsion = true,
            delta_rel = StrictF64(0.25),
            barrier_sym = "sym_max",
            omega_1 = StrictF64(1.0),
            omega_2 = StrictF64(1.0),
            gamma_non2 = StrictF64(1.0),
            ht_repulsion = true,
            min_ratio = StrictF64(0.5),
            eps_dist = StrictF64(1e-6),
            neg_per_node = None,
            num_pivots = StrictUsize(50),
            core_pivots = true,
            t_max = StrictUsize(60),
            eps = StrictF64(0.01),
            schedule = "exponential",
            step_scale = StrictF64(1.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, landmarks, pivots=None, *, max_2hop_per_node=32, hinge_repulsion=True, delta_rel=0.25, barrier_sym='sym_max', omega_1=1.0, omega_2=1.0, gamma_non2=1.0, ht_repulsion=True, min_ratio=0.5, eps_dist=1e-6, neg_per_node=None, num_pivots=50, core_pivots=True, t_max=60, eps=0.01, schedule='exponential', step_scale=1.0, seed=42, dtype='float64')"
    )]
    fn sparse_barrier_from_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &PyDogVaradhanLandmarks,
        pivots: Option<&Bound<'_, PyAny>>,
        max_2hop_per_node: StrictUsize,
        hinge_repulsion: bool,
        delta_rel: StrictF64,
        barrier_sym: &str,
        omega_1: StrictF64,
        omega_2: StrictF64,
        gamma_non2: StrictF64,
        ht_repulsion: bool,
        min_ratio: StrictF64,
        eps_dist: StrictF64,
        neg_per_node: Option<StrictUsize>,
        num_pivots: StrictUsize,
        core_pivots: bool,
        t_max: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let sched = build_schedule_config(t_max.0, eps.0, schedule, step_scale.0)?;
        let sym = parse_barrier_symmetry(barrier_sym).map_err(PyValueError::new_err)?;
        let pivots_vec = match pivots {
            Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "pivots")?),
            _ => None,
        };
        let cfg = SparseBarrierConfig {
            max_2hop_per_node: max_2hop_per_node.0,
            hinge_repulsion,
            delta_rel: delta_rel.0,
            barrier_sym: sym,
            omega_1: omega_1.0,
            omega_2: omega_2.0,
            gamma_non2: gamma_non2.0,
            ht_repulsion,
            min_ratio: min_ratio.0,
            eps_dist: eps_dist.0,
            neg_per_node: neg_per_node.map(|v| v.0),
            num_pivots: num_pivots.0,
            core_pivots,
            schedule: sched,
            seed: seed.0,
        };
        let g = graph.inner();
        let lm = landmarks.inner();
        let (inner, b_stats) = py
            .detach(|| match dtype {
                ScalarDtype::F64 => SgdSolver::<f64>::sparse_barrier_from_landmarks(
                    g,
                    lm,
                    pivots_vec.as_deref(),
                    &cfg,
                )
                .map(|(s, st)| (SgdInner::F64(s), st)),
                ScalarDtype::F32 => SgdSolver::<f32>::sparse_barrier_from_landmarks(
                    g,
                    lm,
                    pivots_vec.as_deref(),
                    &cfg,
                )
                .map(|(s, st)| (SgdInner::F32(s), st)),
            })
            .map_err(core_error_to_py)?;
        Ok(Self {
            inner,
            barrier_stats: Some(b_stats),
            dog_stats: None,
        })
    }

    /// Compute regularized Chebyshev DoG Varadhan landmarks and build a Track B 3-tier Sparse Barrier `SgdSolver`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            degree = None,
            num_landmarks = StrictUsize(64),
            degree_normalize = true,
            p_exp = StrictF64(0.5),
            eps_psd = StrictF64(1e-4),
            t_coarse = StrictF64(48.0),
            t1 = None,
            t2 = None,
            alpha = None,
            beta_floor = None,
            min_degree = StrictUsize(2),
            jl_dim = StrictUsize(64),
            max_2hop_per_node = StrictUsize(32),
            hinge_repulsion = true,
            delta_rel = StrictF64(0.25),
            barrier_sym = "sym_max",
            omega_1 = StrictF64(1.0),
            omega_2 = StrictF64(1.0),
            gamma_non2 = StrictF64(1.0),
            ht_repulsion = true,
            min_ratio = StrictF64(0.5),
            eps_dist = StrictF64(1e-6),
            neg_per_node = None,
            num_pivots = StrictUsize(50),
            core_pivots = true,
            t_max = StrictUsize(60),
            eps = StrictF64(0.01),
            schedule = "exponential",
            step_scale = StrictF64(1.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, degree=None, num_landmarks=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, min_degree=2, jl_dim=64, max_2hop_per_node=32, hinge_repulsion=True, delta_rel=0.25, barrier_sym='sym_max', omega_1=1.0, omega_2=1.0, gamma_non2=1.0, ht_repulsion=True, min_ratio=0.5, eps_dist=1e-6, neg_per_node=None, num_pivots=50, core_pivots=True, t_max=60, eps=0.01, schedule='exponential', step_scale=1.0, seed=42, dtype='float64')"
    )]
    fn sparse_barrier_dog(
        py: Python<'_>,
        graph: &PyCsrGraph,
        degree: Option<StrictUsize>,
        num_landmarks: StrictUsize,
        degree_normalize: bool,
        p_exp: StrictF64,
        eps_psd: StrictF64,
        t_coarse: StrictF64,
        t1: Option<StrictF64>,
        t2: Option<StrictF64>,
        alpha: Option<StrictF64>,
        beta_floor: Option<StrictF64>,
        min_degree: StrictUsize,
        jl_dim: StrictUsize,
        max_2hop_per_node: StrictUsize,
        hinge_repulsion: bool,
        delta_rel: StrictF64,
        barrier_sym: &str,
        omega_1: StrictF64,
        omega_2: StrictF64,
        gamma_non2: StrictF64,
        ht_repulsion: bool,
        min_ratio: StrictF64,
        eps_dist: StrictF64,
        neg_per_node: Option<StrictUsize>,
        num_pivots: StrictUsize,
        core_pivots: bool,
        t_max: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let sched = build_schedule_config(t_max.0, eps.0, schedule, step_scale.0)?;
        let sym = parse_barrier_symmetry(barrier_sym).map_err(PyValueError::new_err)?;
        let dog_cfg = DogVaradhanConfig {
            t1: t1.map(|v| v.0),
            t2: t2.map(|v| v.0),
            alpha: alpha.map(|v| v.0),
            beta_floor: beta_floor.map(|v| v.0),
            degree: degree.map(|v| v.0),
            n_landmarks: num_landmarks.0,
            min_degree: min_degree.0,
            jl_dim: jl_dim.0,
            eps_psd: eps_psd.0,
            degree_normalize,
            p_exp: p_exp.0,
            t_coarse: t_coarse.0,
            seed: seed.0,
            ..DogVaradhanConfig::default()
        };
        let barrier_cfg = SparseBarrierConfig {
            max_2hop_per_node: max_2hop_per_node.0,
            hinge_repulsion,
            delta_rel: delta_rel.0,
            barrier_sym: sym,
            omega_1: omega_1.0,
            omega_2: omega_2.0,
            gamma_non2: gamma_non2.0,
            ht_repulsion,
            min_ratio: min_ratio.0,
            eps_dist: eps_dist.0,
            neg_per_node: neg_per_node.map(|v| v.0),
            num_pivots: num_pivots.0,
            core_pivots,
            schedule: sched,
            seed: seed.0,
        };
        let g = graph.inner();
        let (inner, b_stats, d_stats) = py
            .detach(|| match dtype {
                ScalarDtype::F64 => SgdSolver::<f64>::sparse_barrier_dog(g, &dog_cfg, &barrier_cfg)
                    .map(|(s, b, d)| (SgdInner::F64(s), b, d)),
                ScalarDtype::F32 => SgdSolver::<f32>::sparse_barrier_dog(g, &dog_cfg, &barrier_cfg)
                    .map(|(s, b, d)| (SgdInner::F32(s), b, d)),
            })
            .map_err(core_error_to_py)?;
        Ok(Self {
            inner,
            barrier_stats: Some(b_stats),
            dog_stats: Some(d_stats),
        })
    }

    /// Construct a random-pair sparse `SgdSolver` using exact shortest-path distances.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, pairs_per_node = StrictUsize(30), *, t_max = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, pairs_per_node=30, *, t_max=60, eps=0.01, schedule='exponential', step_scale=1.0, seed=42, dtype='float64')"
    )]
    fn random_pair(
        py: Python<'_>,
        graph: &PyCsrGraph,
        pairs_per_node: StrictUsize,
        t_max: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let sched = build_schedule_config(t_max.0, eps.0, schedule, step_scale.0)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    SgdSolver::<f64>::random_pair(g, pairs_per_node.0, sched, seed.0)
                        .map(SgdInner::F64)
                }
                ScalarDtype::F32 => {
                    SgdSolver::<f32>::random_pair(g, pairs_per_node.0, sched, seed.0)
                        .map(SgdInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Construct an Omega `SgdSolver` using Euclidean distances in a precomputed `(N, D)` embedding.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, embedding, k = StrictUsize(30), min_dist = StrictF64(1e-3), *, t_max = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, embedding, k=30, min_dist=1e-3, *, t_max=60, eps=0.01, schedule='exponential', step_scale=1.0, seed=42, dtype='float64')"
    )]
    fn omega(
        py: Python<'_>,
        graph: &PyCsrGraph,
        embedding: &Bound<'_, PyAny>,
        k: StrictUsize,
        min_dist: StrictF64,
        t_max: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let sched = build_schedule_config(t_max.0, eps.0, schedule, step_scale.0)?;
        let (emb_flat, nrows, dims) = extract_f64_2d(embedding, "embedding")?;
        let g = graph.inner();
        if nrows != g.num_nodes() {
            return Err(PyValueError::new_err(format!(
                "embedding row count {nrows} does not match graph.num_nodes ({})",
                g.num_nodes()
            )));
        }
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    SgdSolver::<f64>::omega(g, &emb_flat, dims, k.0, min_dist.0, sched, seed.0)
                        .map(SgdInner::F64)
                }
                ScalarDtype::F32 => {
                    SgdSolver::<f32>::omega(g, &emb_flat, dims, k.0, min_dist.0, sched, seed.0)
                        .map(SgdInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Construct a minimal-RAM `SgdSolver` storing only 1-hop edges and sampling negatives on-the-fly.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            negatives_per_node = StrictUsize(5),
            target_distance = StrictF64(2.0),
            weight = StrictF64(0.25),
            hinge_only = true,
            t_max = StrictUsize(60),
            eps = StrictF64(0.01),
            schedule = "exponential",
            step_scale = StrictF64(1.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, negatives_per_node=5, target_distance=2.0, weight=0.25, hinge_only=True, t_max=60, eps=0.01, schedule='exponential', step_scale=1.0, seed=42, dtype='float64')"
    )]
    fn streaming_negatives(
        py: Python<'_>,
        graph: &PyCsrGraph,
        negatives_per_node: StrictUsize,
        target_distance: StrictF64,
        weight: StrictF64,
        hinge_only: bool,
        t_max: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let sched = build_schedule_config(t_max.0, eps.0, schedule, step_scale.0)?;
        let neg_cfg = StreamingNegativeConfig {
            negatives_per_node: negatives_per_node.0,
            target_distance: target_distance.0,
            weight: weight.0,
            hinge_only,
        };
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    SgdSolver::<f64>::streaming_negatives(g, neg_cfg, sched, seed.0)
                        .map(SgdInner::F64)
                }
                ScalarDtype::F32 => {
                    SgdSolver::<f32>::streaming_negatives(g, neg_cfg, sched, seed.0)
                        .map(SgdInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        })
    }

    /// Construct a compact-memory edge-incident negative sampling `SNAP-tFDP` `SgdSolver`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            iterations = StrictUsize(200),
            negatives_per_edge = StrictUsize(3),
            learning_rate = StrictF64(1.0),
            alpha = StrictF64(1.0),
            repulsion_scale = StrictF64(1.0),
            rho_start = StrictF64(1.0),
            rho_end = StrictF64(0.1),
            rho_power = StrictF64(1.0),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, iterations=200, negatives_per_edge=3, learning_rate=1.0, alpha=1.0, repulsion_scale=1.0, rho_start=1.0, rho_end=0.1, rho_power=1.0, seed=42, dtype='float64')"
    )]
    fn snap_t_fdp(
        py: Python<'_>,
        graph: &PyCsrGraph,
        iterations: StrictUsize,
        negatives_per_edge: StrictUsize,
        learning_rate: StrictF64,
        alpha: StrictF64,
        repulsion_scale: StrictF64,
        rho_start: StrictF64,
        rho_end: StrictF64,
        rho_power: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        super::v6_m4::build_snap_t_fdp_solver(
            py,
            graph,
            iterations,
            negatives_per_edge,
            learning_rate,
            alpha,
            repulsion_scale,
            rho_start,
            rho_end,
            rho_power,
            seed,
            dtype,
        )
    }

    /// Construct a 3-tier `PaCMAP-Graph` (`NB` / `MN` / `FP`) `SgdSolver`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            n_neighbors = StrictUsize(10),
            mn_ratio = StrictF64(0.5),
            fp_ratio = StrictF64(2.0),
            phase1_iters = StrictUsize(100),
            phase2_iters = StrictUsize(100),
            phase3_iters = StrictUsize(250),
            learning_rate = StrictF64(1.0),
            keep_all_1hop = false,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, phase1_iters=100, phase2_iters=100, phase3_iters=250, learning_rate=1.0, keep_all_1hop=False, seed=42, dtype='float64')"
    )]
    fn pacmap_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        n_neighbors: StrictUsize,
        mn_ratio: StrictF64,
        fp_ratio: StrictF64,
        phase1_iters: StrictUsize,
        phase2_iters: StrictUsize,
        phase3_iters: StrictUsize,
        learning_rate: StrictF64,
        keep_all_1hop: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        super::v6_m4::build_pacmap_graph_solver(
            py,
            graph,
            n_neighbors,
            mn_ratio,
            fp_ratio,
            phase1_iters,
            phase2_iters,
            phase3_iters,
            learning_rate,
            keep_all_1hop,
            seed,
            dtype,
        )
    }

    /// Construct a doubly-stochastic mini-batch diagonal-preconditioned `(SGD)^2` `SgdSolver`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            t_max = StrictUsize(60),
            eps = StrictF64(0.01),
            step_scale = StrictF64(1.0),
            batch_size = StrictUsize(4096),
            damping = StrictF64(1e-4),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, t_max=60, eps=0.01, step_scale=1.0, batch_size=4096, damping=1e-4, seed=42, dtype='float64')"
    )]
    fn sgd2(
        py: Python<'_>,
        graph: &PyCsrGraph,
        t_max: StrictUsize,
        eps: StrictF64,
        step_scale: StrictF64,
        batch_size: StrictUsize,
        damping: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        super::v6_m4::build_sgd2_solver(
            py, graph, t_max, eps, step_scale, batch_size, damping, seed, dtype,
        )
    }

    /// Execute one Euclidean SGD iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` Euclidean SGD iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run Euclidean SGD to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Execute one non-Euclidean manifold SGD iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step_manifold(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyManifoldLayout2d>,
    ) -> PyResult<&'static str> {
        self.execute_manifold_action(py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` non-Euclidean manifold SGD iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps_manifold(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyManifoldLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        self.execute_manifold_action(py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run non-Euclidean manifold SGD to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run_manifold(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyManifoldLayout2d>,
    ) -> PyResult<&'static str> {
        self.execute_manifold_action(py, layout, SolverAction::Run)
    }

    /// Return Sparse Barrier construction diagnostics as a dict, or `None` for non-barrier constructors.
    fn stats(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.barrier_stats
            .as_ref()
            .map(|st| sparse_barrier_stats_to_py_dict(py, st))
            .transpose()
    }

    /// Return regularized DoG Varadhan landmark diagnostics as a dict when built via `sparse_barrier_dog`, or `None`.
    fn dog_stats(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.dog_stats
            .as_ref()
            .map(|st| dog_stats_to_py_dict(py, st))
            .transpose()
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Scheduled total number of iterations (`t_max`).
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in the layout.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            SgdInner::F64(s) => s.num_nodes(),
            SgdInner::F32(s) => s.num_nodes(),
        }
    }

    /// Number of two-sided spring pairs stored in SoA buffers.
    #[getter]
    fn num_springs(&self) -> usize {
        match &self.inner {
            SgdInner::F64(s) => s.num_springs(),
            SgdInner::F32(s) => s.num_springs(),
        }
    }

    /// Number of one-sided hinge repulsion pairs stored in SoA buffers.
    #[getter]
    fn num_hinges(&self) -> usize {
        match &self.inner {
            SgdInner::F64(s) => s.num_hinges(),
            SgdInner::F32(s) => s.num_hinges(),
        }
    }

    /// Learning rate `eta` for the upcoming iteration (`0.0` once finished).
    #[getter]
    fn current_eta(&self) -> f64 {
        match &self.inner {
            SgdInner::F64(s) => s.current_eta(),
            SgdInner::F32(s) => s.current_eta(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `SgdSolver`.
    fn __repr__(&self) -> String {
        format!(
            "SgdSolver(num_nodes={}, springs={}, hinges={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.num_springs(),
            self.num_hinges(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

// ---------------------------------------------------------------------------
// 3. GuMapSolver
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_gumap_config(
    n_epochs: usize,
    negative_sample_rate: usize,
    learning_rate: f64,
    min_dist: f64,
    spread: f64,
    a: Option<f64>,
    b: Option<f64>,
    sigma_iters: usize,
    sigma_tolerance: f64,
    seed: u64,
) -> PyResult<GuMapConfig> {
    let cfg = GuMapConfig {
        n_epochs,
        negative_sample_rate,
        learning_rate,
        min_dist,
        spread,
        a,
        b,
        sigma_iters,
        sigma_tolerance,
        seed,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

/// Cooperative Graph UMAP (`GuMap`) layout solver with `dist_sq.powf(b)` gradient evaluation.
///
/// Supports both `"float64"` and `"float32"` `Layout2d` and releases the Python GIL during stepping.
#[pyclass(name = "GuMapSolver", module = "egraph._core")]
pub struct PyGuMapSolver {
    inner: GuMapInner,
}

#[pymethods]
impl PyGuMapSolver {
    /// Construct a `GuMapSolver` from a precomputed fuzzy simplicial set `SparseProbabilities`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            probs,
            *,
            n_epochs = StrictUsize(200),
            negative_sample_rate = StrictUsize(5),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            sigma_iters = StrictUsize(64),
            sigma_tolerance = StrictF64(1e-5),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(probs, *, n_epochs=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, seed=42, dtype='float64')"
    )]
    fn from_fuzzy_set(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        n_epochs: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_gumap_config(
            n_epochs.0,
            negative_sample_rate.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            sigma_iters.0,
            sigma_tolerance.0,
            seed.0,
        )?;
        let probs_owned = probs.inner().clone();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    GuMapSolver::<f64>::from_fuzzy_set(probs_owned, cfg).map(GuMapInner::F64)
                }
                ScalarDtype::F32 => {
                    GuMapSolver::<f32>::from_fuzzy_set(probs_owned, cfg).map(GuMapInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `GuMapSolver` from a validated `KnnGraph` via `compute_fuzzy_simplicial_set`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            knn,
            *,
            n_epochs = StrictUsize(200),
            negative_sample_rate = StrictUsize(5),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            sigma_iters = StrictUsize(64),
            sigma_tolerance = StrictF64(1e-5),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(knn, *, n_epochs=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn: &PyKnnGraph,
        n_epochs: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_gumap_config(
            n_epochs.0,
            negative_sample_rate.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            sigma_iters.0,
            sigma_tolerance.0,
            seed.0,
        )?;
        let knn_ref = knn.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => GuMapSolver::<f64>::from_knn(knn_ref, cfg).map(GuMapInner::F64),
                ScalarDtype::F32 => GuMapSolver::<f32>::from_knn(knn_ref, cfg).map(GuMapInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `GuMapSolver` from a `CsrGraph` using 1+2-hop priority CSR and regularized DoG Varadhan distances.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            max_2hop_per_node = StrictUsize(32),
            *,
            degree = None,
            num_landmarks = StrictUsize(64),
            degree_normalize = true,
            p_exp = StrictF64(0.5),
            eps_psd = StrictF64(1e-4),
            t_coarse = StrictF64(48.0),
            t1 = None,
            t2 = None,
            alpha = None,
            beta_floor = None,
            min_degree = StrictUsize(2),
            jl_dim = StrictUsize(64),
            n_epochs = StrictUsize(200),
            negative_sample_rate = StrictUsize(5),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            sigma_iters = StrictUsize(64),
            sigma_tolerance = StrictF64(1e-5),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, max_2hop_per_node=32, *, degree=None, num_landmarks=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, min_degree=2, jl_dim=64, n_epochs=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, seed=42, dtype='float64')"
    )]
    fn from_graph_dog(
        py: Python<'_>,
        graph: &PyCsrGraph,
        max_2hop_per_node: StrictUsize,
        degree: Option<StrictUsize>,
        num_landmarks: StrictUsize,
        degree_normalize: bool,
        p_exp: StrictF64,
        eps_psd: StrictF64,
        t_coarse: StrictF64,
        t1: Option<StrictF64>,
        t2: Option<StrictF64>,
        alpha: Option<StrictF64>,
        beta_floor: Option<StrictF64>,
        min_degree: StrictUsize,
        jl_dim: StrictUsize,
        n_epochs: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_gumap_config(
            n_epochs.0,
            negative_sample_rate.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            sigma_iters.0,
            sigma_tolerance.0,
            seed.0,
        )?;
        let dog_cfg = DogVaradhanConfig {
            t1: t1.map(|v| v.0),
            t2: t2.map(|v| v.0),
            alpha: alpha.map(|v| v.0),
            beta_floor: beta_floor.map(|v| v.0),
            degree: degree.map(|v| v.0),
            n_landmarks: num_landmarks.0,
            min_degree: min_degree.0,
            jl_dim: jl_dim.0,
            eps_psd: eps_psd.0,
            degree_normalize,
            p_exp: p_exp.0,
            t_coarse: t_coarse.0,
            seed: seed.0,
            ..DogVaradhanConfig::default()
        };
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    GuMapSolver::<f64>::from_graph_dog(g, max_2hop_per_node.0, &dog_cfg, cfg)
                        .map(GuMapInner::F64)
                }
                ScalarDtype::F32 => {
                    GuMapSolver::<f32>::from_graph_dog(g, max_2hop_per_node.0, &dog_cfg, cfg)
                        .map(GuMapInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `GuMapSolver` from precomputed `DogVaradhanLandmarks` and 1+2-hop priority CSR.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            landmarks,
            max_2hop_per_node = StrictUsize(32),
            *,
            n_epochs = StrictUsize(200),
            negative_sample_rate = StrictUsize(5),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            sigma_iters = StrictUsize(64),
            sigma_tolerance = StrictF64(1e-5),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, landmarks, max_2hop_per_node=32, *, n_epochs=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, seed=42, dtype='float64')"
    )]
    fn from_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &PyDogVaradhanLandmarks,
        max_2hop_per_node: StrictUsize,
        n_epochs: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_gumap_config(
            n_epochs.0,
            negative_sample_rate.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            sigma_iters.0,
            sigma_tolerance.0,
            seed.0,
        )?;
        let g = graph.inner();
        let lm = landmarks.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    GuMapSolver::<f64>::from_landmarks(g, lm, max_2hop_per_node.0, cfg)
                        .map(GuMapInner::F64)
                }
                ScalarDtype::F32 => {
                    GuMapSolver::<f32>::from_landmarks(g, lm, max_2hop_per_node.0, cfg)
                        .map(GuMapInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute a single optimization epoch in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` optimization epochs in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of epochs executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Scheduled total number of epochs (`n_epochs`).
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Number of stored directed positive probability edges.
    #[getter]
    fn num_edges(&self) -> usize {
        match &self.inner {
            GuMapInner::F64(s) => s.num_edges(),
            GuMapInner::F32(s) => s.num_edges(),
        }
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            GuMapInner::F64(s) => s.num_nodes(),
            GuMapInner::F32(s) => s.num_nodes(),
        }
    }

    /// Resolved low-dimensional curve parameters `(a, b)`.
    #[getter]
    fn ab(&self) -> (f64, f64) {
        match &self.inner {
            GuMapInner::F64(s) => s.ab(),
            GuMapInner::F32(s) => s.ab(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `GuMapSolver`.
    fn __repr__(&self) -> String {
        format!(
            "GuMapSolver(num_nodes={}, num_edges={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.num_edges(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

// ---------------------------------------------------------------------------
// 4. SpatialIntruderRefiner
// ---------------------------------------------------------------------------

fn refiner_stats_to_py_dict(py: Python<'_>, stats: &SpatialRefinerStats) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("rounds_executed", stats.rounds_executed)?;
    dict.set_item("n_anchors_total", stats.n_anchors_total)?;
    dict.set_item("n_fn_total", stats.n_fn_total)?;
    dict.set_item("n_fp_total", stats.n_fp_total)?;
    dict.set_item("n_pairs_total", stats.n_pairs_total)?;
    dict.set_item("initial_scale", stats.initial_scale)?;
    dict.set_item("final_scale", stats.final_scale)?;
    dict.set_item("d_floor", stats.d_floor)?;
    dict.set_item("scale_ratio", stats.scale_ratio)?;
    dict.set_item("used_hub_fallback", stats.used_hub_fallback)?;
    dict.set_item("hinge_fp", stats.hinge_fp)?;

    let rounds_list = PyList::empty(py);
    for r in &stats.round_stats {
        let rd = PyDict::new(py);
        rd.set_item("round", r.round)?;
        rd.set_item("n_anchors", r.n_anchors)?;
        rd.set_item("n_fn", r.n_fn)?;
        rd.set_item("n_fp", r.n_fp)?;
        rd.set_item("n_pairs_total", r.n_pairs_total)?;
        rd.set_item("skipped", r.skipped)?;
        rounds_list.append(rd)?;
    }
    dict.set_item("round_stats", rounds_list)?;
    Ok(dict.unbind())
}

/// Cooperative $O(N \log N)$ `StaticKdTree` active-set spatial intruder refiner.
///
/// Detects strictly `> 2`-hop spatial false positives inside each vertex's local spatial neighborhood
/// and resolves them while protecting 1-hop anchors. Releases the Python GIL during stepping.
#[pyclass(name = "SpatialIntruderRefiner", module = "egraph._core")]
pub struct PySpatialIntruderRefiner {
    inner: RefinerInner,
}

#[pymethods]
impl PySpatialIntruderRefiner {
    /// Create a `SpatialIntruderRefiner` precomputing 1+2-hop topology from `graph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            rounds = StrictUsize(4),
            k_cap = StrictUsize(32),
            margin_scale = StrictF64(1.2),
            w_anchor = StrictF64(1.0),
            w_fn = StrictF64(0.0),
            w_fp = StrictF64(0.2),
            fn_shrink = StrictF64(0.88),
            round_decay = StrictF64(0.75),
            t_max = StrictUsize(20),
            eps = StrictF64(0.05),
            hinge_fp = true,
            seed = StrictU64(42),
            *,
            w_anchor2 = StrictF64(0.0),
            k_cap_decay = StrictF64(1.0),
            k_cap_min = StrictUsize(1),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, rounds=4, k_cap=32, margin_scale=1.2, w_anchor=1.0, w_fn=0.0, w_fp=0.2, fn_shrink=0.88, round_decay=0.75, t_max=20, eps=0.05, hinge_fp=True, seed=42, *, w_anchor2=0.0, k_cap_decay=1.0, k_cap_min=1, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        rounds: StrictUsize,
        k_cap: StrictUsize,
        margin_scale: StrictF64,
        w_anchor: StrictF64,
        w_fn: StrictF64,
        w_fp: StrictF64,
        fn_shrink: StrictF64,
        round_decay: StrictF64,
        t_max: StrictUsize,
        eps: StrictF64,
        hinge_fp: bool,
        seed: StrictU64,
        w_anchor2: StrictF64,
        k_cap_decay: StrictF64,
        k_cap_min: StrictUsize,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = SpatialRefinerConfig {
            rounds: rounds.0,
            k_cap: k_cap.0,
            margin_scale: margin_scale.0,
            w_anchor: w_anchor.0,
            w_anchor2: w_anchor2.0,
            w_fn: w_fn.0,
            w_fp: w_fp.0,
            fn_shrink: fn_shrink.0,
            round_decay: round_decay.0,
            k_cap_decay: k_cap_decay.0,
            k_cap_min: k_cap_min.0,
            t_max: t_max.0,
            eps: eps.0,
            hinge_fp,
            seed: seed.0,
        };
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    SpatialIntruderRefiner::<f64>::new(g, cfg).map(RefinerInner::F64)
                }
                ScalarDtype::F32 => {
                    SpatialIntruderRefiner::<f32>::new(g, cfg).map(RefinerInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one spatial intruder refinement round in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` refinement rounds in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run spatial refinement to completion (or early convergence) in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Return cumulative and per-round refinement diagnostics as a dict.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let stats = match &self.inner {
            RefinerInner::F64(r) => r.stats(),
            RefinerInner::F32(r) => r.stats(),
        };
        refiner_stats_to_py_dict(py, stats)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of refinement rounds executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of refinement rounds.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the refiner has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            RefinerInner::F64(r) => r.num_nodes(),
            RefinerInner::F32(r) => r.num_nodes(),
        }
    }

    /// Current dynamic active-set neighborhood cap `k_cap`.
    #[getter]
    fn current_k_cap(&self) -> usize {
        match &self.inner {
            RefinerInner::F64(r) => r.current_k_cap(),
            RefinerInner::F32(r) => r.current_k_cap(),
        }
    }

    /// Update the active-set neighborhood cap `k_cap` for subsequent refinement rounds.
    #[pyo3(signature = (k_cap), text_signature = "($self, k_cap)")]
    fn set_k_cap(&mut self, py: Python<'_>, k_cap: StrictUsize) -> PyResult<()> {
        let cap = k_cap.0;
        py.detach(|| match &mut self.inner {
            RefinerInner::F64(r) => r.set_k_cap(cap),
            RefinerInner::F32(r) => r.set_k_cap(cap),
        })
        .map_err(core_error_to_py)
    }

    /// Refiner scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `SpatialIntruderRefiner`.
    fn __repr__(&self) -> String {
        format!(
            "SpatialIntruderRefiner(num_nodes={}, status='{}', round={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

// ---------------------------------------------------------------------------
// 5. StressMajorizationSolver
// ---------------------------------------------------------------------------

/// Cooperative Gansner-Koren-North (2004) Stress Majorization layout solver backed by Jacobi-PCG.
///
/// Supports both `"float64"` and `"float32"` `Layout2d` and releases the Python GIL during stepping.
#[pyclass(name = "StressMajorizationSolver", module = "egraph._core")]
pub struct PyStressMajorizationSolver {
    inner: StressMajorizationInner,
}

#[pymethods]
impl PyStressMajorizationSolver {
    /// Construct a `StressMajorizationSolver` from a `CsrGraph` using all-pairs shortest paths.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, epsilon = StrictF64(1e-4), max_iterations = StrictUsize(100), pcg_tolerance = StrictF64(1e-6), pcg_max_iterations = None, min_norm = StrictF64(1e-4), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, epsilon=1e-4, max_iterations=100, pcg_tolerance=1e-6, pcg_max_iterations=None, min_norm=1e-4, dtype='float64')"
    )]
    fn from_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        epsilon: StrictF64,
        max_iterations: StrictUsize,
        pcg_tolerance: StrictF64,
        pcg_max_iterations: Option<StrictUsize>,
        min_norm: StrictF64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = StressMajorizationConfig {
            epsilon: epsilon.0,
            max_iterations: max_iterations.0,
            pcg_tolerance: pcg_tolerance.0,
            pcg_max_iterations: pcg_max_iterations.map(|v| v.0),
            min_norm: min_norm.0,
        };
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => StressMajorizationSolver::<f64>::from_graph(g, cfg)
                    .map(StressMajorizationInner::F64),
                ScalarDtype::F32 => StressMajorizationSolver::<f32>::from_graph(g, cfg)
                    .map(StressMajorizationInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `StressMajorizationSolver` from an `(N, N)` distance matrix (where `inf` marks unreachable pairs).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (distances, *, epsilon = StrictF64(1e-4), max_iterations = StrictUsize(100), pcg_tolerance = StrictF64(1e-6), pcg_max_iterations = None, min_norm = StrictF64(1e-4), dtype = ScalarDtype::F64),
        text_signature = "(distances, *, epsilon=1e-4, max_iterations=100, pcg_tolerance=1e-6, pcg_max_iterations=None, min_norm=1e-4, dtype='float64')"
    )]
    fn from_distance_matrix(
        py: Python<'_>,
        distances: &Bound<'_, PyAny>,
        epsilon: StrictF64,
        max_iterations: StrictUsize,
        pcg_tolerance: StrictF64,
        pcg_max_iterations: Option<StrictUsize>,
        min_norm: StrictF64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let (dists_flat, n) = extract_distance_matrix_2d(distances)?;
        let cfg = StressMajorizationConfig {
            epsilon: epsilon.0,
            max_iterations: max_iterations.0,
            pcg_tolerance: pcg_tolerance.0,
            pcg_max_iterations: pcg_max_iterations.map(|v| v.0),
            min_norm: min_norm.0,
        };
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    StressMajorizationSolver::<f64>::from_distance_matrix(n, &dists_flat, cfg)
                        .map(StressMajorizationInner::F64)
                }
                ScalarDtype::F32 => {
                    StressMajorizationSolver::<f32>::from_distance_matrix(n, &dists_flat, cfg)
                        .map(StressMajorizationInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one stress majorization step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` stress majorization iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run stress majorization to completion or convergence in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Evaluate the total weighted stress for `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn compute_stress(&self, py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
        py.detach(|| match (&self.inner, layout.inner()) {
            (StressMajorizationInner::F64(s), Layout2dInner::F64(buf)) => {
                s.compute_stress(&buf.to_layout2d())
            }
            (StressMajorizationInner::F32(s), Layout2dInner::F32(buf)) => {
                s.compute_stress(&buf.to_layout2d())
            }
            (StressMajorizationInner::F64(s), Layout2dInner::F32(buf)) => {
                let soa64 = layout_f32_to_f64_checked(&buf.to_layout2d())?;
                s.compute_stress(&soa64)
            }
            (StressMajorizationInner::F32(s), Layout2dInner::F64(buf)) => {
                let soa32 = layout_f64_to_f32_checked(&buf.to_layout2d())?;
                s.compute_stress(&soa32)
            }
        })
        .map_err(core_error_to_py)
    }

    /// Stress recorded after the most recent executed step, or `None` before stepping.
    #[getter]
    fn last_stress(&self) -> Option<f64> {
        match &self.inner {
            StressMajorizationInner::F64(s) => s.last_stress(),
            StressMajorizationInner::F32(s) => s.last_stress(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of iterations (`max_iterations`).
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            StressMajorizationInner::F64(s) => s.num_nodes(),
            StressMajorizationInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `StressMajorizationSolver`.
    fn __repr__(&self) -> String {
        format!(
            "StressMajorizationSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

// ---------------------------------------------------------------------------
// 6. KamadaKawaiSolver
// ---------------------------------------------------------------------------

/// Cooperative Kamada-Kawai (1989) spring layout solver with Hessian singularity fallback.
///
/// Supports both `"float64"` and `"float32"` `Layout2d` and releases the Python GIL during stepping.
#[pyclass(name = "KamadaKawaiSolver", module = "egraph._core")]
pub struct PyKamadaKawaiSolver {
    inner: KamadaKawaiInner,
}

#[pymethods]
impl PyKamadaKawaiSolver {
    /// Construct a `KamadaKawaiSolver` from a `CsrGraph` using all-pairs shortest paths.
    #[staticmethod]
    #[pyo3(
        signature = (graph, *, eps = StrictF64(1e-2), max_inner_iters = StrictUsize(50), max_outer_iters = None, dtype = ScalarDtype::F64),
        text_signature = "(graph, *, eps=1e-2, max_inner_iters=50, max_outer_iters=None, dtype='float64')"
    )]
    fn from_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        eps: StrictF64,
        max_inner_iters: StrictUsize,
        max_outer_iters: Option<StrictUsize>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = KamadaKawaiConfig {
            eps: eps.0,
            max_inner_iters: max_inner_iters.0,
            max_outer_iters: max_outer_iters.map(|v| v.0),
        };
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    KamadaKawaiSolver::<f64>::from_graph(g, cfg).map(KamadaKawaiInner::F64)
                }
                ScalarDtype::F32 => {
                    KamadaKawaiSolver::<f32>::from_graph(g, cfg).map(KamadaKawaiInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `KamadaKawaiSolver` from an `(N, N)` distance matrix (where `inf` marks unreachable pairs).
    #[staticmethod]
    #[pyo3(
        signature = (distances, *, eps = StrictF64(1e-2), max_inner_iters = StrictUsize(50), max_outer_iters = None, dtype = ScalarDtype::F64),
        text_signature = "(distances, *, eps=1e-2, max_inner_iters=50, max_outer_iters=None, dtype='float64')"
    )]
    fn from_distance_matrix(
        py: Python<'_>,
        distances: &Bound<'_, PyAny>,
        eps: StrictF64,
        max_inner_iters: StrictUsize,
        max_outer_iters: Option<StrictUsize>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let (dists_flat, n) = extract_distance_matrix_2d(distances)?;
        let cfg = KamadaKawaiConfig {
            eps: eps.0,
            max_inner_iters: max_inner_iters.0,
            max_outer_iters: max_outer_iters.map(|v| v.0),
        };
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    KamadaKawaiSolver::<f64>::from_distance_matrix(n, &dists_flat, cfg)
                        .map(KamadaKawaiInner::F64)
                }
                ScalarDtype::F32 => {
                    KamadaKawaiSolver::<f32>::from_distance_matrix(n, &dists_flat, cfg)
                        .map(KamadaKawaiInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one node-relaxation step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` node-relaxation steps in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run Kamada-Kawai optimization to completion or convergence in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of outer steps executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured outer iteration cap (`max_outer_iters` or `(10 * n).max(100)`).
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            KamadaKawaiInner::F64(s) => s.num_nodes(),
            KamadaKawaiInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `KamadaKawaiSolver`.
    fn __repr__(&self) -> String {
        format!(
            "KamadaKawaiSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Register M3 `egraph-clustering`, `ManifoldLayout2d`, and `egraph-layout` solver bindings on `egraph._core`.
pub(super) fn register_m3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyManifoldLayout2d>()?;
    m.add_class::<PyTsNetSolver>()?;
    m.add_class::<PySgdSolver>()?;
    m.add_class::<PyGuMapSolver>()?;
    m.add_class::<PySpatialIntruderRefiner>()?;
    m.add_class::<PyStressMajorizationSolver>()?;
    m.add_class::<PyKamadaKawaiSolver>()?;
    m.add_function(wrap_pyfunction!(poincare_distance, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_distance, m)?)?;
    m.add_function(wrap_pyfunction!(torus_distance, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_torus_point, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans_plus_plus, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_clustering, m)?)?;
    m.add_function(wrap_pyfunction!(louvain_communities, m)?)?;
    m.add_function(wrap_pyfunction!(label_propagation, m)?)?;
    m.add_function(wrap_pyfunction!(coarsen_graph, m)?)?;
    m.add_function(wrap_pyfunction!(project_clustered_layout, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BarrierSymmetry, ManifoldKind, SgdScheduleKind, SolverStatus, TsNetBackend,
        default_inv_sq_weights, manifold_kind_str, parse_barrier_symmetry, parse_manifold_kind,
        parse_sgd_schedule_kind, parse_tsnet_backend, solver_status_str,
    };

    #[test]
    fn test_pure_rust_m3_enum_parsers_and_helpers() {
        assert_eq!(
            parse_manifold_kind("Hyperbolic_Poincare").unwrap(),
            ManifoldKind::HyperbolicPoincare
        );
        assert_eq!(
            parse_manifold_kind("poincare").unwrap(),
            ManifoldKind::HyperbolicPoincare
        );
        assert_eq!(
            parse_manifold_kind("SPHERE").unwrap(),
            ManifoldKind::Spherical
        );
        assert_eq!(
            parse_manifold_kind("torus-2d").unwrap(),
            ManifoldKind::Torus2d
        );
        assert_eq!(
            manifold_kind_str(ManifoldKind::HyperbolicPoincare),
            "hyperbolic"
        );
        assert_eq!(manifold_kind_str(ManifoldKind::Spherical), "spherical");
        assert_eq!(manifold_kind_str(ManifoldKind::Torus2d), "torus");
        assert!(parse_manifold_kind("euclidean").is_err());

        assert_eq!(parse_tsnet_backend("EXACT").unwrap(), TsNetBackend::Exact);
        assert_eq!(
            parse_tsnet_backend("barnes-hut").unwrap(),
            TsNetBackend::BarnesHut
        );
        assert_eq!(parse_tsnet_backend("bh").unwrap(), TsNetBackend::BarnesHut);
        assert_eq!(
            parse_tsnet_backend("fft_interpolation").unwrap(),
            TsNetBackend::FftInterpolation
        );
        assert_eq!(
            parse_tsnet_backend("FIt").unwrap(),
            TsNetBackend::FftInterpolation
        );
        assert_eq!(
            parse_tsnet_backend("linear_fft").unwrap(),
            TsNetBackend::LinearFft
        );
        assert_eq!(parse_tsnet_backend("L").unwrap(), TsNetBackend::LinearFft);
        assert!(parse_tsnet_backend("unknown").is_err());

        assert_eq!(
            parse_sgd_schedule_kind("EXPONENTIAL").unwrap(),
            SgdScheduleKind::Exponential
        );
        assert_eq!(
            parse_sgd_schedule_kind("linear").unwrap(),
            SgdScheduleKind::Linear
        );
        assert_eq!(
            parse_sgd_schedule_kind("Quadratic").unwrap(),
            SgdScheduleKind::Quadratic
        );
        assert_eq!(
            parse_sgd_schedule_kind("reciprocal").unwrap(),
            SgdScheduleKind::Reciprocal
        );
        assert_eq!(
            parse_sgd_schedule_kind("constant").unwrap(),
            SgdScheduleKind::Constant
        );
        assert_eq!(
            parse_sgd_schedule_kind("squared").unwrap(),
            SgdScheduleKind::Squared
        );
        assert!(parse_sgd_schedule_kind("cosine").is_err());

        assert_eq!(
            parse_barrier_symmetry("sym_max").unwrap(),
            BarrierSymmetry::SymMax
        );
        assert_eq!(
            parse_barrier_symmetry("SymAvg").unwrap(),
            BarrierSymmetry::SymAvg
        );
        assert!(parse_barrier_symmetry("sym_min").is_err());

        assert_eq!(solver_status_str(SolverStatus::Initialized), "initialized");
        assert_eq!(solver_status_str(SolverStatus::Running), "running");
        assert_eq!(solver_status_str(SolverStatus::Converged), "converged");
        assert_eq!(solver_status_str(SolverStatus::Completed), "completed");

        let w = default_inv_sq_weights(&[2.0, 0.0, f64::NAN]);
        assert!((w[0] - 0.25).abs() < 1e-12);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);
    }
}
