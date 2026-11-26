//! `ManifoldLayout2d` Python class binding.

use egraph::core as egraph_core;
use egraph_core::{ManifoldKind, ManifoldLayout2d};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py};
use crate::layout_buffers::{Layout2dInner, PyLayout2d, extract_layout_inner};

use super::geodesic::{
    checked_f64_to_f32, manifold_coords_array, manifold_kind_str, parse_manifold_kind,
};

#[derive(Clone)]
pub(super) enum ManifoldLayout2dInner {
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
    pub(super) fn inner(&self) -> &ManifoldLayout2dInner {
        &self.inner
    }

    pub(super) fn set_inner(&mut self, inner: ManifoldLayout2dInner) {
        self.inner = inner;
    }

    pub(super) fn dtype_enum(&self) -> ScalarDtype {
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

    /// Create a deterministically seeded safe random initialization on `kind`.
    #[staticmethod]
    #[pyo3(
        signature = (kind, n, *, scale = StrictF64(0.25), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(kind, n, *, scale=0.25, seed=42, dtype='float64')"
    )]
    fn random_init(
        py: Python<'_>,
        kind: &str,
        n: StrictUsize,
        scale: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let mk = parse_manifold_kind(kind).map_err(PyValueError::new_err)?;
        let n_val = n.0;
        let scale_val = scale.0;
        let seed_val = seed.0;
        py.detach(|| match dtype {
            ScalarDtype::F64 => {
                ManifoldLayout2d::<f64>::random_init(mk, n_val, scale_val, seed_val)
                    .map(ManifoldLayout2dInner::F64)
            }
            ScalarDtype::F32 => {
                ManifoldLayout2d::<f32>::random_init(mk, n_val, scale_val, seed_val)
                    .map(ManifoldLayout2dInner::F32)
            }
        })
        .map(|inner| Self { inner })
        .map_err(core_error_to_py)
    }

    /// Construct a `ManifoldLayout2d` from a Euclidean `Layout2d` scaled safely inside `max_radius`.
    #[staticmethod]
    #[pyo3(
        signature = (kind, layout, *, max_radius = StrictF64(0.7)),
        text_signature = "(kind, layout, *, max_radius=0.7)"
    )]
    fn from_layout_scaled(
        py: Python<'_>,
        kind: &str,
        layout: &PyLayout2d,
        max_radius: StrictF64,
    ) -> PyResult<Self> {
        let mk = parse_manifold_kind(kind).map_err(PyValueError::new_err)?;
        let mr = max_radius.0;
        py.detach(|| match layout.inner() {
            Layout2dInner::F64(buf) => {
                ManifoldLayout2d::from_layout_scaled(mk, &buf.to_layout2d(), mr)
                    .map(ManifoldLayout2dInner::F64)
            }
            Layout2dInner::F32(buf) => {
                ManifoldLayout2d::from_layout_scaled(mk, &buf.to_layout2d(), mr)
                    .map(ManifoldLayout2dInner::F32)
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
        manifold_coords_array(py, &self.inner)
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
                let x32 = checked_f64_to_f32(x.0, "ManifoldLayout2d.set_point x")?;
                let y32 = checked_f64_to_f32(y.0, "ManifoldLayout2d.set_point y")?;
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
                let vx32 = checked_f64_to_f32(vx.0, "ManifoldLayout2d.apply_tangent_step vx")?;
                let vy32 = checked_f64_to_f32(vy.0, "ManifoldLayout2d.apply_tangent_step vy")?;
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

    /// Compute the closed-form Möbius-equivariant Einstein midpoint centroid on the Poincaré disk.
    fn poincare_centroid(&self) -> PyResult<(f64, f64)> {
        match &self.inner {
            ManifoldLayout2dInner::F64(m) => m.poincare_centroid().map_err(core_error_to_py),
            ManifoldLayout2dInner::F32(m) => m
                .poincare_centroid()
                .map(|(x, y)| (f64::from(x), f64::from(y)))
                .map_err(core_error_to_py),
        }
    }

    /// Isometrically recenter all Poincaré disk points at `(cx, cy)` via Möbius automorphism.
    #[pyo3(signature = (cx, cy), text_signature = "($self, cx, cy)")]
    fn recenter_poincare(&mut self, cx: StrictF64, cy: StrictF64) -> PyResult<()> {
        match &mut self.inner {
            ManifoldLayout2dInner::F64(m) => {
                m.recenter_poincare(cx.0, cy.0).map_err(core_error_to_py)
            }
            ManifoldLayout2dInner::F32(m) => {
                let cx32 = checked_f64_to_f32(cx.0, "ManifoldLayout2d.recenter_poincare cx")?;
                let cy32 = checked_f64_to_f32(cy.0, "ManifoldLayout2d.recenter_poincare cy")?;
                m.recenter_poincare(cx32, cy32).map_err(core_error_to_py)
            }
        }
    }

    /// Isometrically recenter the Poincaré disk layout so vertex `index` moves to `(0, 0)`.
    #[pyo3(signature = (index), text_signature = "($self, index)")]
    fn recenter_poincare_at(&mut self, index: StrictUsize) -> PyResult<()> {
        match &mut self.inner {
            ManifoldLayout2dInner::F64(m) => {
                m.recenter_poincare_at(index.0).map_err(core_error_to_py)
            }
            ManifoldLayout2dInner::F32(m) => {
                m.recenter_poincare_at(index.0).map_err(core_error_to_py)
            }
        }
    }

    /// Isometrically recenter the Poincaré disk layout so its Einstein midpoint lies at `(0, 0)`.
    fn center_poincare_centroid(&mut self) -> PyResult<()> {
        match &mut self.inner {
            ManifoldLayout2dInner::F64(m) => m.center_poincare_centroid().map_err(core_error_to_py),
            ManifoldLayout2dInner::F32(m) => m.center_poincare_centroid().map_err(core_error_to_py),
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
