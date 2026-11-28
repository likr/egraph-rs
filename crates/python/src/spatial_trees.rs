//! 3D Barnes-Hut octree (`ArenaOctree3d`) binding.

use egraph::spatial::ArenaOctree3d;
use numpy::PyArray2;
use numpy::ndarray::Array2;
use pyo3::prelude::*;

use crate::array_extract::{extract_f64_1d, extract_f64_2d_raw};
use crate::errors::{dimension_mismatch_err, invalid_input_err};
use crate::ffi_convert::{ScalarDtype, StrictF64, core_error_to_py, extract_optional_weights_1d};
use crate::layout_nd::{LayoutNdInner, PyLayoutNd};

enum OctreeInner {
    F64(ArenaOctree3d<f64>),
    F32(ArenaOctree3d<f32>),
}

/// Flat arena-allocated 3D Barnes-Hut octree for Coulomb, MaxEnt, KL, and entropy repulsion.
#[pyclass(name = "ArenaOctree3d", module = "egraph._core")]
pub struct PyArenaOctree3d {
    inner: OctreeInner,
}

#[derive(FromPyObject)]
enum OctreeBuildInput<'py> {
    Layout(Bound<'py, PyLayoutNd>),
    Coords(Bound<'py, PyAny>),
}

fn octree_to_py_array<'py>(
    py: Python<'py>,
    n: usize,
    flat: Vec<f64>,
    dtype: ScalarDtype,
) -> PyResult<Bound<'py, PyAny>> {
    match dtype {
        ScalarDtype::F64 => {
            let arr = Array2::from_shape_vec((n, 3), flat)
                .map_err(|e| invalid_input_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr).into_any())
        }
        ScalarDtype::F32 => {
            let flat32: Vec<f32> = flat.into_iter().map(|v| v as f32).collect();
            let arr = Array2::from_shape_vec((n, 3), flat32)
                .map_err(|e| invalid_input_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr).into_any())
        }
    }
}

#[pymethods]
impl PyArenaOctree3d {
    /// Construct an empty reusable 3D Barnes-Hut octree (`dtype="float64"` or `"float32"`).
    #[new]
    #[pyo3(signature = (dtype = ScalarDtype::F64), text_signature = "(dtype='float64')")]
    fn new(dtype: ScalarDtype) -> Self {
        let inner = match dtype {
            ScalarDtype::F64 => OctreeInner::F64(ArenaOctree3d::<f64>::new()),
            ScalarDtype::F32 => OctreeInner::F32(ArenaOctree3d::<f32>::new()),
        };
        Self { inner }
    }

    /// Number of indexed points in the octree.
    #[getter]
    fn point_count(&self) -> usize {
        match &self.inner {
            OctreeInner::F64(t) => t.point_count(),
            OctreeInner::F32(t) => t.point_count(),
        }
    }

    /// Number of allocated internal and leaf arena nodes in the octree.
    #[getter]
    fn node_count(&self) -> usize {
        match &self.inner {
            OctreeInner::F64(t) => t.node_count(),
            OctreeInner::F32(t) => t.node_count(),
        }
    }

    /// Scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            OctreeInner::F64(_) => ScalarDtype::F64.as_str(),
            OctreeInner::F32(_) => ScalarDtype::F32.as_str(),
        }
    }

    /// Return the number of indexed points in the octree.
    fn __len__(&self) -> usize {
        self.point_count()
    }

    /// Clear all points and tree nodes while retaining allocated arena buffers.
    fn clear(&mut self) {
        match &mut self.inner {
            OctreeInner::F64(t) => t.clear(),
            OctreeInner::F32(t) => t.clear(),
        }
    }

    /// Rebuild the 3D octree from a 3D `LayoutNd` or `(N, 3)` array/sequence with optional `masses`.
    #[pyo3(
        signature = (coords_or_layout, *, masses = None),
        text_signature = "($self, coords_or_layout, *, masses=None)"
    )]
    fn build(
        &mut self,
        py: Python<'_>,
        coords_or_layout: OctreeBuildInput<'_>,
        masses: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let masses_opt = extract_optional_weights_1d(masses, "masses")?;
        match coords_or_layout {
            OctreeBuildInput::Layout(lnd) => {
                let borrowed = lnd.try_borrow()?;
                let nd_ref: &PyLayoutNd = &borrowed;
                py.detach(|| match (&mut self.inner, nd_ref.inner()) {
                    (OctreeInner::F64(t), LayoutNdInner::F64(nd)) => match masses_opt.as_deref() {
                        Some(m) => t.build_weighted(nd, m),
                        None => t.build(nd),
                    },
                    (OctreeInner::F32(t), LayoutNdInner::F32(nd)) => match masses_opt.as_deref() {
                        Some(m) => t.build_weighted(nd, m),
                        None => t.build(nd),
                    },
                    (OctreeInner::F64(t), LayoutNdInner::F32(nd)) => {
                        if nd.dims() != 3 {
                            return Err(egraph::core::CoreError::dimension_mismatch(
                                3,
                                nd.dims(),
                                "ArenaOctree3d::build",
                            ));
                        }
                        let flat: Vec<f64> = nd.coords().iter().map(|&v| f64::from(v)).collect();
                        match masses_opt.as_deref() {
                            Some(m) => t.build_weighted_flat(&flat, nd.num_nodes(), m),
                            None => t.build_flat(&flat, nd.num_nodes()),
                        }
                    }
                    (OctreeInner::F32(t), LayoutNdInner::F64(nd)) => {
                        if nd.dims() != 3 {
                            return Err(egraph::core::CoreError::dimension_mismatch(
                                3,
                                nd.dims(),
                                "ArenaOctree3d::build",
                            ));
                        }
                        let flat: Vec<f32> = nd.coords().iter().map(|&v| v as f32).collect();
                        match masses_opt.as_deref() {
                            Some(m) => t.build_weighted_flat(&flat, nd.num_nodes(), m),
                            None => t.build_flat(&flat, nd.num_nodes()),
                        }
                    }
                })
                .map_err(core_error_to_py)
            }
            OctreeBuildInput::Coords(obj) => {
                let (flat, nrows, ncols) = extract_f64_2d_raw(&obj, "coords")?;
                if ncols != 3 {
                    return Err(dimension_mismatch_err(format!(
                        "ArenaOctree3d.build requires shape (N, 3), got ({nrows}, {ncols})"
                    )));
                }
                py.detach(|| match &mut self.inner {
                    OctreeInner::F64(t) => match masses_opt.as_deref() {
                        Some(m) => t.build_weighted_flat(&flat, nrows, m),
                        None => t.build_flat(&flat, nrows),
                    },
                    OctreeInner::F32(t) => {
                        let flat32: Vec<f32> = flat.iter().map(|&v| v as f32).collect();
                        match masses_opt.as_deref() {
                            Some(m) => t.build_weighted_flat(&flat32, nrows, m),
                            None => t.build_flat(&flat32, nrows),
                        }
                    }
                })
                .map_err(core_error_to_py)
            }
        }
    }

    /// Rebuild the 3D octree from separate 1D `x`, `y`, `z` coordinate arrays and optional `masses`.
    #[pyo3(
        signature = (x, y, z, *, masses = None),
        text_signature = "($self, x, y, z, *, masses=None)"
    )]
    fn build_xyz(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        z: &Bound<'_, PyAny>,
        masses: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let xv = extract_f64_1d(x, "x")?;
        let yv = extract_f64_1d(y, "y")?;
        let zv = extract_f64_1d(z, "z")?;
        let masses_opt = extract_optional_weights_1d(masses, "masses")?;
        py.detach(|| match &mut self.inner {
            OctreeInner::F64(t) => match masses_opt.as_deref() {
                Some(m) => t.build_weighted_xyz(&xv, &yv, &zv, m),
                None => t.build_xyz(&xv, &yv, &zv),
            },
            OctreeInner::F32(t) => {
                let x32: Vec<f32> = xv.iter().map(|&v| v as f32).collect();
                let y32: Vec<f32> = yv.iter().map(|&v| v as f32).collect();
                let z32: Vec<f32> = zv.iter().map(|&v| v as f32).collect();
                match masses_opt.as_deref() {
                    Some(m) => t.build_weighted_xyz(&x32, &y32, &z32, m),
                    None => t.build_xyz(&x32, &y32, &z32),
                }
            }
        })
        .map_err(core_error_to_py)
    }

    /// Compute weighted Barnes-Hut Coulomb repulsive forces of shape `(N, 3)`.
    #[pyo3(
        signature = (*, theta = StrictF64(0.5), eps_sq = StrictF64(1e-6)),
        text_signature = "($self, *, theta=0.5, eps_sq=1e-6)"
    )]
    fn compute_coulomb_repulsion<'py>(
        &self,
        py: Python<'py>,
        theta: StrictF64,
        eps_sq: StrictF64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = self.point_count();
        let (flat, dtype) = py
            .detach(|| match &self.inner {
                OctreeInner::F64(t) => {
                    let mut out = vec![0.0_f64; n * 3];
                    t.compute_coulomb_repulsion_flat(theta.0, eps_sq.0, &mut out)?;
                    Ok((out, ScalarDtype::F64))
                }
                OctreeInner::F32(t) => {
                    let mut out = vec![0.0_f32; n * 3];
                    t.compute_coulomb_repulsion_flat(theta.0 as f32, eps_sq.0 as f32, &mut out)?;
                    Ok((out.into_iter().map(f64::from).collect(), ScalarDtype::F32))
                }
            })
            .map_err(core_error_to_py)?;
        octree_to_py_array(py, n, flat, dtype)
    }

    /// Compute weighted Barnes-Hut generalized MaxEnt `q`-power repulsion forces of shape `(N, 3)`.
    #[pyo3(
        signature = (q, *, theta = StrictF64(0.5), eps_sq = StrictF64(1e-6)),
        text_signature = "($self, q, *, theta=0.5, eps_sq=1e-6)"
    )]
    fn compute_maxent_repulsion<'py>(
        &self,
        py: Python<'py>,
        q: StrictF64,
        theta: StrictF64,
        eps_sq: StrictF64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = self.point_count();
        let (flat, dtype) = py
            .detach(|| match &self.inner {
                OctreeInner::F64(t) => {
                    let mut out = vec![0.0_f64; n * 3];
                    t.compute_maxent_repulsion_flat(theta.0, q.0, eps_sq.0, &mut out)?;
                    Ok((out, ScalarDtype::F64))
                }
                OctreeInner::F32(t) => {
                    let mut out = vec![0.0_f32; n * 3];
                    t.compute_maxent_repulsion_flat(
                        theta.0 as f32,
                        q.0,
                        eps_sq.0 as f32,
                        &mut out,
                    )?;
                    Ok((out.into_iter().map(f64::from).collect(), ScalarDtype::F32))
                }
            })
            .map_err(core_error_to_py)?;
        octree_to_py_array(py, n, flat, dtype)
    }

    /// Compute normalized Barnes-Hut KL repulsive forces `(forces (N, 3), total_z)`.
    #[pyo3(signature = (*, theta = StrictF64(0.5)), text_signature = "($self, *, theta=0.5)")]
    fn compute_kl_repulsion<'py>(
        &self,
        py: Python<'py>,
        theta: StrictF64,
    ) -> PyResult<(Bound<'py, PyAny>, f64)> {
        let n = self.point_count();
        let (flat, total_z, dtype) = py
            .detach(|| match &self.inner {
                OctreeInner::F64(t) => {
                    let mut out = vec![0.0_f64; n * 3];
                    let z = t.compute_kl_repulsion_flat(theta.0, &mut out)?;
                    Ok((out, z, ScalarDtype::F64))
                }
                OctreeInner::F32(t) => {
                    let mut out = vec![0.0_f32; n * 3];
                    let z = t.compute_kl_repulsion_flat(theta.0 as f32, &mut out)?;
                    Ok((
                        out.into_iter().map(f64::from).collect(),
                        f64::from(z),
                        ScalarDtype::F32,
                    ))
                }
            })
            .map_err(core_error_to_py)?;
        let arr = octree_to_py_array(py, n, flat, dtype)?;
        Ok((arr, total_z))
    }

    /// Compute Barnes-Hut entropy repulsive forces of shape `(N, 3)`.
    #[pyo3(
        signature = (*, theta = StrictF64(0.5), epsilon_r = StrictF64(0.05)),
        text_signature = "($self, *, theta=0.5, epsilon_r=0.05)"
    )]
    fn compute_entropy_repulsion<'py>(
        &self,
        py: Python<'py>,
        theta: StrictF64,
        epsilon_r: StrictF64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = self.point_count();
        let (flat, dtype) = py
            .detach(|| match &self.inner {
                OctreeInner::F64(t) => {
                    let mut out = vec![0.0_f64; n * 3];
                    t.compute_entropy_repulsion_flat(theta.0, epsilon_r.0, &mut out)?;
                    Ok((out, ScalarDtype::F64))
                }
                OctreeInner::F32(t) => {
                    let mut out = vec![0.0_f32; n * 3];
                    t.compute_entropy_repulsion_flat(theta.0 as f32, epsilon_r.0 as f32, &mut out)?;
                    Ok((out.into_iter().map(f64::from).collect(), ScalarDtype::F32))
                }
            })
            .map_err(core_error_to_py)?;
        octree_to_py_array(py, n, flat, dtype)
    }

    /// Return a concise string representation of the `ArenaOctree3d`.
    fn __repr__(&self) -> String {
        format!(
            "ArenaOctree3d(point_count={}, node_count={}, dtype='{}')",
            self.point_count(),
            self.node_count(),
            self.dtype()
        )
    }
}

/// Register `ArenaOctree3d` on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArenaOctree3d>()?;
    Ok(())
}
