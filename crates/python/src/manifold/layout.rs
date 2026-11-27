//! `ManifoldLayout2d` Python class binding.

use egraph::core as egraph_core;
use egraph_core::{ManifoldKind, ManifoldLayout2d, ManifoldMetric};
use egraph_ffi::InterleavedLayout2d;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, ffi_error_to_py,
};
use crate::layout_buffers::PyLayout2d;

use super::geodesic::{
    build_core_f32, build_core_f64, inner_from_core_f32, inner_from_core_f64,
    manifold_coords_array, manifold_kind_str, parse_manifold_kind,
};
use super::layout_ops::{
    apply_tangent_step_impl, center_poincare_centroid_impl, from_coords_impl, from_layout_impl,
    from_layout_scaled_impl, project_all_impl, random_init_impl, recenter_poincare_at_impl,
    recenter_poincare_impl, rescale_geodesic_impl, set_point_impl, set_torus_extents_impl,
    to_poincare_layout_impl, torus_auto_pan_impl, torus_spectral_phase_init_impl,
};

#[derive(Clone)]
pub(crate) enum ManifoldLayout2dInner {
    F64 {
        kind: ManifoldKind,
        metric: ManifoldMetric,
        buf: InterleavedLayout2d<f64>,
    },
    F32 {
        kind: ManifoldKind,
        metric: ManifoldMetric,
        buf: InterleavedLayout2d<f32>,
    },
}

/// 2D node coordinate layout constrained to a non-Euclidean manifold (`"hyperbolic"`,
/// `"hyperbolic_lorentz"`, `"spherical"`, or `"torus"`).
#[pyclass(name = "ManifoldLayout2d", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyManifoldLayout2d {
    inner: ManifoldLayout2dInner,
}

impl PyManifoldLayout2d {
    pub(crate) fn inner(&self) -> &ManifoldLayout2dInner {
        &self.inner
    }

    pub(crate) fn inner_mut(&mut self) -> &mut ManifoldLayout2dInner {
        &mut self.inner
    }

    pub(crate) fn dtype_enum(&self) -> ScalarDtype {
        match &self.inner {
            ManifoldLayout2dInner::F64 { .. } => ScalarDtype::F64,
            ManifoldLayout2dInner::F32 { .. } => ScalarDtype::F32,
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
            ScalarDtype::F64 => inner_from_core_f64(ManifoldLayout2d::zeros(mk, n.0)),
            ScalarDtype::F32 => inner_from_core_f32(ManifoldLayout2d::zeros(mk, n.0)),
        };
        Ok(Self { inner })
    }

    /// Create a zero-initialized Poincaré disk `ManifoldLayout2d` for `n` vertices.
    #[staticmethod]
    #[pyo3(signature = (n, dtype = ScalarDtype::F64), text_signature = "(n, dtype='float64')")]
    fn hyperbolic_poincare(n: StrictUsize, dtype: ScalarDtype) -> Self {
        let mk = ManifoldKind::HyperbolicPoincare;
        let inner = match dtype {
            ScalarDtype::F64 => inner_from_core_f64(ManifoldLayout2d::zeros(mk, n.0)),
            ScalarDtype::F32 => inner_from_core_f32(ManifoldLayout2d::zeros(mk, n.0)),
        };
        Self { inner }
    }

    /// Create a zero-initialized Lorentz hyperboloid `ManifoldLayout2d` for `n` vertices.
    #[staticmethod]
    #[pyo3(signature = (n, dtype = ScalarDtype::F64), text_signature = "(n, dtype='float64')")]
    fn hyperbolic_lorentz(n: StrictUsize, dtype: ScalarDtype) -> Self {
        let mk = ManifoldKind::HyperbolicLorentz;
        let inner = match dtype {
            ScalarDtype::F64 => inner_from_core_f64(ManifoldLayout2d::zeros(mk, n.0)),
            ScalarDtype::F32 => inner_from_core_f32(ManifoldLayout2d::zeros(mk, n.0)),
        };
        Self { inner }
    }

    /// Create a zero-initialized spherical `ManifoldLayout2d` for `n` vertices.
    #[staticmethod]
    #[pyo3(signature = (n, dtype = ScalarDtype::F64), text_signature = "(n, dtype='float64')")]
    fn spherical(n: StrictUsize, dtype: ScalarDtype) -> Self {
        let mk = ManifoldKind::Spherical;
        let inner = match dtype {
            ScalarDtype::F64 => inner_from_core_f64(ManifoldLayout2d::zeros(mk, n.0)),
            ScalarDtype::F32 => inner_from_core_f32(ManifoldLayout2d::zeros(mk, n.0)),
        };
        Self { inner }
    }

    /// Create a zero-initialized flat unit torus `ManifoldLayout2d` for `n` vertices.
    #[staticmethod]
    #[pyo3(signature = (n, dtype = ScalarDtype::F64), text_signature = "(n, dtype='float64')")]
    fn torus_2d(n: StrictUsize, dtype: ScalarDtype) -> Self {
        let mk = ManifoldKind::Torus2d;
        let inner = match dtype {
            ScalarDtype::F64 => inner_from_core_f64(ManifoldLayout2d::zeros(mk, n.0)),
            ScalarDtype::F32 => inner_from_core_f32(ManifoldLayout2d::zeros(mk, n.0)),
        };
        Self { inner }
    }

    /// Wrap an existing Euclidean `Layout2d` and project all points onto manifold `kind`.
    #[staticmethod]
    #[pyo3(signature = (kind, layout), text_signature = "(kind, layout)")]
    fn from_layout(py: Python<'_>, kind: &str, layout: &PyLayout2d) -> PyResult<Self> {
        Ok(Self {
            inner: from_layout_impl(py, kind, layout)?,
        })
    }

    /// Construct a `ManifoldLayout2d` from an `(N, 2)` array or sequence and project onto `kind`.
    #[staticmethod]
    #[pyo3(signature = (kind, coords), text_signature = "(kind, coords)")]
    fn from_coords(py: Python<'_>, kind: &str, coords: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: from_coords_impl(py, kind, coords)?,
        })
    }

    /// Create a deterministically seeded safe random initialization on `kind`.
    #[staticmethod]
    #[pyo3(
        signature = (kind, n, *, scale = StrictF64(0.25), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(kind, n, *, scale=0.25, seed=42, dtype='float64')"
    )]
    fn init(
        py: Python<'_>,
        kind: &str,
        n: StrictUsize,
        scale: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        Self::random_init(py, kind, n, scale, seed, dtype)
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
        Ok(Self {
            inner: random_init_impl(py, kind, n, scale, seed, dtype)?,
        })
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
        Ok(Self {
            inner: from_layout_scaled_impl(py, kind, layout, max_radius)?,
        })
    }

    /// Initialize a 2D torus layout from 4-mode spectral phase coordinates of `graph`.
    #[staticmethod]
    #[pyo3(
        signature = (graph, *, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, seed=42, dtype='float64')"
    )]
    fn torus_spectral_phase_init(
        py: Python<'_>,
        graph: &PyCsrGraph,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: torus_spectral_phase_init_impl(py, graph, seed, dtype)?,
        })
    }

    /// Return a deep copy of this `ManifoldLayout2d` preserving `kind`, `metric`, and `dtype`.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Canonical manifold kind name.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            ManifoldLayout2dInner::F64 { kind, .. } => manifold_kind_str(*kind),
            ManifoldLayout2dInner::F32 { kind, .. } => manifold_kind_str(*kind),
        }
    }

    /// Scalar coordinate precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype_enum().as_str()
    }

    /// Physical geodesic scale factor (`> 0`).
    #[getter]
    fn scale(&self) -> f64 {
        match &self.inner {
            ManifoldLayout2dInner::F64 { metric, .. } => metric.scale,
            ManifoldLayout2dInner::F32 { metric, .. } => metric.scale,
        }
    }

    /// Physical aspect ratio (`Lx / Ly` on `Torus2d`, `> 0`).
    #[getter]
    fn aspect(&self) -> f64 {
        match &self.inner {
            ManifoldLayout2dInner::F64 { metric, .. } => metric.aspect,
            ManifoldLayout2dInner::F32 { metric, .. } => metric.aspect,
        }
    }

    /// Return the physical metric parameters `(scale, aspect)`.
    fn metric(&self) -> (f64, f64) {
        (self.scale(), self.aspect())
    }

    /// Set the physical geodesic scale factor (`> 0`) without altering stored coordinates.
    #[pyo3(signature = (scale), text_signature = "($self, scale)")]
    fn set_scale(&mut self, scale: StrictF64) -> PyResult<()> {
        if !scale.0.is_finite() || scale.0 <= 0.0 {
            return Err(PyValueError::new_err(
                "scale must be finite and strictly positive",
            ));
        }
        match &mut self.inner {
            ManifoldLayout2dInner::F64 { metric, .. } => metric.scale = scale.0,
            ManifoldLayout2dInner::F32 { metric, .. } => metric.scale = scale.0,
        }
        Ok(())
    }

    /// Set the physical aspect ratio (`> 0`) without altering stored coordinates.
    #[pyo3(signature = (aspect), text_signature = "($self, aspect)")]
    fn set_aspect(&mut self, aspect: StrictF64) -> PyResult<()> {
        if !aspect.0.is_finite() || aspect.0 <= 0.0 {
            return Err(PyValueError::new_err(
                "aspect must be finite and strictly positive",
            ));
        }
        match &mut self.inner {
            ManifoldLayout2dInner::F64 { metric, .. } => metric.aspect = aspect.0,
            ManifoldLayout2dInner::F32 { metric, .. } => metric.aspect = aspect.0,
        }
        Ok(())
    }

    /// Set physical torus extents `(lx, ly)` (`scale = ly`, `aspect = lx / ly`).
    #[pyo3(signature = (lx, ly), text_signature = "($self, lx, ly)")]
    fn set_torus_extents(&mut self, lx: StrictF64, ly: StrictF64) -> PyResult<()> {
        set_torus_extents_impl(&mut self.inner, lx.0, ly.0)
    }

    /// Rescale hyperbolic coordinates radially around the Einstein midpoint to `new_scale`.
    #[pyo3(signature = (new_scale), text_signature = "($self, new_scale)")]
    fn rescale_geodesic(&mut self, new_scale: StrictF64) -> PyResult<()> {
        rescale_geodesic_impl(&mut self.inner, new_scale.0)
    }

    /// Convert a hyperbolic (Lorentz or Poincaré) layout into a Poincaré disk `ManifoldLayout2d`.
    fn to_poincare_layout(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            inner: to_poincare_layout_impl(py, &self.inner)?,
        })
    }

    /// Return a rigidly panned `Torus2d` layout minimizing boundary-wrapped edges for `graph`.
    #[pyo3(signature = (graph), text_signature = "($self, graph)")]
    fn torus_auto_pan(&self, py: Python<'_>, graph: &PyCsrGraph) -> PyResult<Self> {
        Ok(Self {
            inner: torus_auto_pan_impl(py, &self.inner, graph)?,
        })
    }

    /// Number of vertices in the manifold layout.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.__len__()
    }

    /// Return `True` if the manifold layout contains zero vertices.
    fn is_empty(&self) -> bool {
        self.__len__() == 0
    }

    /// Return the number of vertices in the manifold layout.
    fn __len__(&self) -> usize {
        match &self.inner {
            ManifoldLayout2dInner::F64 { buf, .. } => buf.len(),
            ManifoldLayout2dInner::F32 { buf, .. } => buf.len(),
        }
    }

    /// Return an `(N, 2)` NumPy coordinate array (zero-copy writable view when `copy=False`).
    #[pyo3(signature = (copy = false), text_signature = "($self, copy=False)")]
    fn coords<'py>(slf: Bound<'py, Self>, copy: bool) -> PyResult<Bound<'py, PyAny>> {
        manifold_coords_array(slf, copy)
    }

    /// Convert to a Euclidean `Layout2d` copy with matching precision.
    fn to_layout(&self) -> PyLayout2d {
        match &self.inner {
            ManifoldLayout2dInner::F64 { buf, .. } => {
                PyLayout2d::from_f64_layout(&buf.to_layout2d())
            }
            ManifoldLayout2dInner::F32 { buf, .. } => {
                PyLayout2d::from_f32_layout(&buf.to_layout2d())
            }
        }
    }

    /// Return the first coordinate (`x` or longitude or toroidal `x`) of vertex `i`.
    #[pyo3(signature = (i), text_signature = "($self, i)")]
    fn x(&self, i: StrictUsize) -> PyResult<f64> {
        let n = self.__len__();
        if i.0 >= n {
            return Err(PyIndexError::new_err(format!(
                "ManifoldLayout2d::x index {} out of bounds for length {n}",
                i.0
            )));
        }
        match &self.inner {
            ManifoldLayout2dInner::F64 { buf, .. } => buf.x(i.0).map_err(ffi_error_to_py),
            ManifoldLayout2dInner::F32 { buf, .. } => {
                buf.x(i.0).map(f64::from).map_err(ffi_error_to_py)
            }
        }
    }

    /// Return the second coordinate (`y` or colatitude or toroidal `y`) of vertex `i`.
    #[pyo3(signature = (i), text_signature = "($self, i)")]
    fn y(&self, i: StrictUsize) -> PyResult<f64> {
        let n = self.__len__();
        if i.0 >= n {
            return Err(PyIndexError::new_err(format!(
                "ManifoldLayout2d::y index {} out of bounds for length {n}",
                i.0
            )));
        }
        match &self.inner {
            ManifoldLayout2dInner::F64 { buf, .. } => buf.y(i.0).map_err(ffi_error_to_py),
            ManifoldLayout2dInner::F32 { buf, .. } => {
                buf.y(i.0).map(f64::from).map_err(ffi_error_to_py)
            }
        }
    }

    /// Return `True` if every coordinate in the manifold layout is finite.
    fn is_all_finite(&self) -> bool {
        match &self.inner {
            ManifoldLayout2dInner::F64 { buf, .. } => buf.is_all_finite(),
            ManifoldLayout2dInner::F32 { buf, .. } => buf.is_all_finite(),
        }
    }

    /// Set vertex `index`'s coordinates, projecting onto the manifold domain.
    #[pyo3(signature = (index, x, y), text_signature = "($self, index, x, y)")]
    fn set_point(&mut self, index: StrictUsize, x: StrictF64, y: StrictF64) -> PyResult<()> {
        set_point_impl(&mut self.inner, index.0, x.0, y.0)
    }

    /// Compute the exact geodesic distance between vertices `i` and `j` on the manifold.
    #[pyo3(signature = (i, j), text_signature = "($self, i, j)")]
    fn geodesic_distance(&self, i: StrictUsize, j: StrictUsize) -> PyResult<f64> {
        match &self.inner {
            ManifoldLayout2dInner::F64 { kind, metric, buf } => {
                let m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
                m.geodesic_distance(i.0, j.0).map_err(core_error_to_py)
            }
            ManifoldLayout2dInner::F32 { kind, metric, buf } => {
                let m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
                m.geodesic_distance(i.0, j.0)
                    .map(f64::from)
                    .map_err(core_error_to_py)
            }
        }
    }

    /// Compute the Riemannian logarithmic map (tangent vector at `i` pointing toward `j`).
    #[pyo3(signature = (i, j), text_signature = "($self, i, j)")]
    fn tangent_delta(&self, i: StrictUsize, j: StrictUsize) -> PyResult<(f64, f64)> {
        match &self.inner {
            ManifoldLayout2dInner::F64 { kind, metric, buf } => {
                let m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
                m.tangent_delta(i.0, j.0).map_err(core_error_to_py)
            }
            ManifoldLayout2dInner::F32 { kind, metric, buf } => {
                let m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
                m.tangent_delta(i.0, j.0)
                    .map(|(vx, vy)| (f64::from(vx), f64::from(vy)))
                    .map_err(core_error_to_py)
            }
        }
    }

    /// Advance vertex `i` along the geodesic defined by tangent vector `(vx, vy)`.
    #[pyo3(signature = (i, vx, vy), text_signature = "($self, i, vx, vy)")]
    fn apply_tangent_step(&mut self, i: StrictUsize, vx: StrictF64, vy: StrictF64) -> PyResult<()> {
        apply_tangent_step_impl(&mut self.inner, i.0, vx.0, vy.0)
    }

    /// Project all coordinates in place onto the manifold's canonical domain.
    fn project_all(&mut self) -> PyResult<()> {
        project_all_impl(&mut self.inner)
    }

    /// Compute the closed-form Möbius-equivariant Einstein midpoint centroid on the Poincaré disk.
    fn poincare_centroid(&self) -> PyResult<(f64, f64)> {
        match &self.inner {
            ManifoldLayout2dInner::F64 { kind, metric, buf } => {
                let m = build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?;
                m.poincare_centroid().map_err(core_error_to_py)
            }
            ManifoldLayout2dInner::F32 { kind, metric, buf } => {
                let m = build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?;
                m.poincare_centroid()
                    .map(|(x, y)| (f64::from(x), f64::from(y)))
                    .map_err(core_error_to_py)
            }
        }
    }

    /// Isometrically recenter all Poincaré disk points at `(cx, cy)` via Möbius automorphism.
    #[pyo3(signature = (cx, cy), text_signature = "($self, cx, cy)")]
    fn recenter_poincare(&mut self, cx: StrictF64, cy: StrictF64) -> PyResult<()> {
        recenter_poincare_impl(&mut self.inner, cx.0, cy.0)
    }

    /// Isometrically recenter the Poincaré disk layout so vertex `index` moves to `(0, 0)`.
    #[pyo3(signature = (index), text_signature = "($self, index)")]
    fn recenter_poincare_at(&mut self, index: StrictUsize) -> PyResult<()> {
        recenter_poincare_at_impl(&mut self.inner, index.0)
    }

    /// Isometrically recenter the Poincaré disk layout so its Einstein midpoint lies at `(0, 0)`.
    fn center_poincare_centroid(&mut self) -> PyResult<()> {
        center_poincare_centroid_impl(&mut self.inner)
    }

    /// Return a concise string representation of the `ManifoldLayout2d`.
    fn __repr__(&self) -> String {
        format!(
            "ManifoldLayout2d(kind='{}', len={}, dtype='{}')",
            self.kind(),
            self.__len__(),
            self.dtype()
        )
    }
}
