//! Regularized DoG Varadhan landmark table and ALT lower-bound distance bindings.

use egraph::neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, DogVaradhanStats, RhghLandmarkStats,
    compute_regularized_dog_varadhan_landmarks as nb_compute_regularized_dog_varadhan_landmarks,
    compute_regularized_dog_varadhan_landmarks_from_matrix as nb_compute_regularized_dog_varadhan_landmarks_from_matrix,
};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::invalid_input_err;
use crate::ffi_convert::{StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d};
use crate::linalg::PyCsrMatrix;

#[derive(Clone)]
struct DogLandmarkMeta {
    t1: f64,
    t2: f64,
    alpha: f64,
    degree: usize,
    delta_psd: f64,
    beta_auto: f64,
    beta_eff: f64,
    used_coarse_scaffold: bool,
}

impl From<&DogVaradhanStats> for DogLandmarkMeta {
    fn from(st: &DogVaradhanStats) -> Self {
        Self {
            t1: st.t1,
            t2: st.t2,
            alpha: st.alpha,
            degree: st.degree,
            delta_psd: st.delta_psd,
            beta_auto: st.beta_auto,
            beta_eff: st.beta_eff,
            used_coarse_scaffold: st.used_coarse_scaffold,
        }
    }
}

/// Regularized DoG Varadhan landmark distance table (plus optional coarse diffusive scaffold)
/// supporting fast ALT lower-bound distance queries.
#[pyclass(name = "DogVaradhanLandmarks", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyDogVaradhanLandmarks {
    inner: DogVaradhanLandmarks,
    stats: DogLandmarkMeta,
}

impl PyDogVaradhanLandmarks {
    pub(crate) fn inner(&self) -> &DogVaradhanLandmarks {
        &self.inner
    }

    pub(crate) fn from_rhgh_parts(inner: DogVaradhanLandmarks, rhgh: &RhghLandmarkStats) -> Self {
        let stats = DogLandmarkMeta {
            t1: rhgh.gamma,
            t2: rhgh.gamma,
            alpha: rhgh.p_exp,
            degree: 0,
            delta_psd: 0.0,
            beta_auto: 0.0,
            beta_eff: 0.0,
            used_coarse_scaffold: false,
        };
        Self { inner, stats }
    }
}

#[pymethods]
impl PyDogVaradhanLandmarks {
    /// Number of vertices covered by the landmark table.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of landmark columns in the table.
    #[getter]
    fn num_landmarks(&self) -> usize {
        self.inner.num_landmarks()
    }

    /// Return a `uint32` NumPy copy of selected landmark vertex IDs.
    fn landmarks<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, self.inner.landmarks())
    }

    /// Return the `(num_nodes, num_landmarks)` `float64` DoG Varadhan distance table.
    fn table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = Array2::from_shape_vec(
            (self.inner.num_nodes(), self.inner.num_landmarks()),
            self.inner.d_dog().to_vec(),
        )
        .map_err(|e| invalid_input_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Return the `(num_nodes, num_landmarks)` `float64` coarse heat scaffold table, or `None`.
    fn coarse_table<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let Some(dc) = self.inner.d_coarse() else {
            return Ok(None);
        };
        let arr = Array2::from_shape_vec(
            (self.inner.num_nodes(), self.inner.num_landmarks()),
            dc.to_vec(),
        )
        .map_err(|e| invalid_input_err(e.to_string()))?;
        Ok(Some(PyArray2::from_owned_array(py, arr)))
    }

    /// Effective short-range DoG diffusion time `t1`.
    #[getter]
    fn t1(&self) -> f64 {
        self.stats.t1
    }

    /// Effective long-range DoG diffusion time `t2`.
    #[getter]
    fn t2(&self) -> f64 {
        self.stats.t2
    }

    /// Effective DoG attenuation parameter `alpha`.
    #[getter]
    fn alpha(&self) -> f64 {
        self.stats.alpha
    }

    /// PSD shift `delta_psd` applied during spectral projection.
    #[getter]
    fn delta_psd(&self) -> f64 {
        self.stats.delta_psd
    }

    /// Effective rank-1 positivity floor `beta_eff` actually applied.
    #[getter]
    fn beta_eff(&self) -> f64 {
        self.stats.beta_eff
    }

    /// Effective rank-1 positivity floor `beta_auto`.
    #[getter]
    fn beta_auto(&self) -> f64 {
        self.stats.beta_auto
    }

    /// Chebyshev polynomial degree used for the DoG filter.
    #[getter]
    fn degree(&self) -> usize {
        self.stats.degree
    }

    /// Whether the coarse diffusive heat scaffold was computed and attached.
    #[getter]
    fn used_coarse_scaffold(&self) -> bool {
        self.stats.used_coarse_scaffold
    }

    /// Evaluate landmark ALT lower-bound distances for pairs `(u[i], v[i])`.
    ///
    /// Releases the Python GIL during computation.
    #[pyo3(signature = (u, v), text_signature = "($self, u, v)")]
    fn alt_distances<'py>(
        &self,
        py: Python<'py>,
        u: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let u_vec = extract_u32_1d(u, "u")?;
        let v_vec = extract_u32_1d(v, "v")?;
        let out = py
            .detach(|| self.inner.alt_pair_distances(&u_vec, &v_vec))
            .map_err(core_error_to_py)?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Return the number of vertices in the landmark table.
    fn __len__(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Return a concise string representation of the `DogVaradhanLandmarks`.
    fn __repr__(&self) -> String {
        format!(
            "DogVaradhanLandmarks(num_nodes={}, num_landmarks={}, used_coarse_scaffold={})",
            self.inner.num_nodes(),
            self.inner.num_landmarks(),
            if self.stats.used_coarse_scaffold {
                "True"
            } else {
                "False"
            }
        )
    }
}

/// Compute regularized DoG Varadhan landmark tables and optional coarse diffusive scaffold.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, *, degree = None, num_landmarks = StrictUsize(64), seed = StrictU64(42), degree_normalize = true, p_exp = StrictF64(0.5), eps_psd = StrictF64(1e-4), t_coarse = StrictF64(48.0), t1 = None, t2 = None, alpha = None, beta_floor = None),
    text_signature = "(graph, *, degree=None, num_landmarks=64, seed=42, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None)"
)]
fn compute_regularized_dog_varadhan_landmarks(
    py: Python<'_>,
    graph: &PyCsrGraph,
    degree: Option<StrictUsize>,
    num_landmarks: StrictUsize,
    seed: StrictU64,
    degree_normalize: bool,
    p_exp: StrictF64,
    eps_psd: StrictF64,
    t_coarse: StrictF64,
    t1: Option<StrictF64>,
    t2: Option<StrictF64>,
    alpha: Option<StrictF64>,
    beta_floor: Option<StrictF64>,
) -> PyResult<PyDogVaradhanLandmarks> {
    let cfg = DogVaradhanConfig::new()
        .with_t1(t1.map(|v| v.0))
        .with_t2(t2.map(|v| v.0))
        .with_alpha(alpha.map(|v| v.0))
        .with_beta_floor(beta_floor.map(|v| v.0))
        .with_degree(degree.map(|d| d.0))
        .with_n_landmarks(num_landmarks.0)
        .with_eps_psd(eps_psd.0)
        .with_degree_normalize(degree_normalize)
        .with_p_exp(p_exp.0)
        .with_t_coarse(t_coarse.0)
        .with_seed(seed.0);
    let g = graph.inner();
    let (inner, stats) = py
        .detach(|| nb_compute_regularized_dog_varadhan_landmarks(&*g, &cfg))
        .map_err(core_error_to_py)?;
    Ok(PyDogVaradhanLandmarks {
        inner,
        stats: DogLandmarkMeta::from(&stats),
    })
}

/// Compute regularized DoG Varadhan landmark tables from a custom symmetric PSD `CsrMatrix`
/// with dynamic `[0, lambda_max]` spectrum scaling.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, matrix, *, lambda_max = None, degree = None, num_landmarks = StrictUsize(64), min_degree = StrictUsize(2), jl_dim = StrictUsize(64), seed = StrictU64(42), degree_normalize = true, p_exp = StrictF64(0.5), eps_psd = StrictF64(1e-4), t_coarse = StrictF64(48.0), t1 = None, t2 = None, alpha = None, beta_floor = None),
    text_signature = "(graph, matrix, *, lambda_max=None, degree=None, num_landmarks=64, min_degree=2, jl_dim=64, seed=42, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None)"
)]
fn compute_regularized_dog_varadhan_landmarks_from_matrix(
    py: Python<'_>,
    graph: &PyCsrGraph,
    matrix: &PyCsrMatrix,
    lambda_max: Option<StrictF64>,
    degree: Option<StrictUsize>,
    num_landmarks: StrictUsize,
    min_degree: StrictUsize,
    jl_dim: StrictUsize,
    seed: StrictU64,
    degree_normalize: bool,
    p_exp: StrictF64,
    eps_psd: StrictF64,
    t_coarse: StrictF64,
    t1: Option<StrictF64>,
    t2: Option<StrictF64>,
    alpha: Option<StrictF64>,
    beta_floor: Option<StrictF64>,
) -> PyResult<PyDogVaradhanLandmarks> {
    let cfg = DogVaradhanConfig::new()
        .with_t1(t1.map(|v| v.0))
        .with_t2(t2.map(|v| v.0))
        .with_alpha(alpha.map(|v| v.0))
        .with_beta_floor(beta_floor.map(|v| v.0))
        .with_degree(degree.map(|d| d.0))
        .with_n_landmarks(num_landmarks.0)
        .with_min_degree(min_degree.0)
        .with_jl_dim(jl_dim.0)
        .with_eps_psd(eps_psd.0)
        .with_degree_normalize(degree_normalize)
        .with_p_exp(p_exp.0)
        .with_t_coarse(t_coarse.0)
        .with_seed(seed.0);
    let g = graph.inner();
    let mat = matrix.inner();
    let lmax_opt = lambda_max.map(|v| v.0);
    let (inner, stats) = py
        .detach(|| {
            nb_compute_regularized_dog_varadhan_landmarks_from_matrix(&*g, mat, lmax_opt, &cfg)
        })
        .map_err(core_error_to_py)?;
    Ok(PyDogVaradhanLandmarks {
        inner,
        stats: DogLandmarkMeta::from(&stats),
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDogVaradhanLandmarks>()?;
    m.add_function(wrap_pyfunction!(
        compute_regularized_dog_varadhan_landmarks,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compute_regularized_dog_varadhan_landmarks_from_matrix,
        m
    )?)?;
    Ok(())
}
