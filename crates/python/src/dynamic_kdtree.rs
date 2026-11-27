//! Runtime-dimension dynamic KD-tree (`DynamicKdTree`) Python binding.

use egraph::spatial::DynamicKdTree;
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::array_extract::{extract_f64_1d, extract_f64_2d};
use crate::ffi_convert::{StrictF64, StrictUsize, core_error_to_py};
use crate::layout_nd::{LayoutNdInner, PyLayoutNd};

#[derive(FromPyObject)]
enum KdTreeInput<'py> {
    Layout(Bound<'py, PyLayoutNd>),
    Coords(Bound<'py, PyAny>),
}

/// Runtime-dimension flat median-split KD-tree supporting `k`-NN and radius queries.
#[pyclass(name = "DynamicKdTree", module = "egraph._core")]
pub struct PyDynamicKdTree {
    inner: DynamicKdTree<f64>,
}

type IndexDistPair<'py> = (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>>);

fn pairs_to_index_dist<'py>(py: Python<'py>, pairs: Vec<(usize, f64)>) -> IndexDistPair<'py> {
    let mut idxs = Vec::with_capacity(pairs.len());
    let mut dists = Vec::with_capacity(pairs.len());
    for (i, d2) in pairs {
        idxs.push(i);
        dists.push(d2);
    }
    (PyArray1::from_vec(py, idxs), PyArray1::from_vec(py, dists))
}

#[pymethods]
impl PyDynamicKdTree {
    /// Build a `DynamicKdTree` from a `LayoutNd` or `(N, D)` coordinate array/sequence.
    #[new]
    #[pyo3(signature = (coords_or_layout), text_signature = "(coords_or_layout)")]
    fn new(py: Python<'_>, coords_or_layout: KdTreeInput<'_>) -> PyResult<Self> {
        let inner = match coords_or_layout {
            KdTreeInput::Layout(lnd) => {
                let borrowed = lnd.try_borrow()?;
                let nd_ref: &PyLayoutNd = &borrowed;
                py.detach(|| match nd_ref.inner() {
                    LayoutNdInner::F64(nd) => DynamicKdTree::from_layout_nd(nd),
                    LayoutNdInner::F32(nd) => {
                        let coords64: Vec<f64> =
                            nd.coords().iter().map(|&v| f64::from(v)).collect();
                        DynamicKdTree::from_flat(&coords64, nd.num_nodes(), nd.dims())
                    }
                })
                .map_err(core_error_to_py)?
            }
            KdTreeInput::Coords(obj) => {
                let (flat, nrows, ncols) = extract_f64_2d(&obj, "coords")?;
                py.detach(|| DynamicKdTree::from_flat(&flat, nrows, ncols))
                    .map_err(core_error_to_py)?
            }
        };
        Ok(Self { inner })
    }

    /// Coordinate dimensionality `D`.
    #[getter]
    fn dims(&self) -> usize {
        self.inner.dims()
    }

    /// Number of points stored in the tree.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.len()
    }

    /// Return the number of points stored in the tree.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Return `True` if the tree contains zero points.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return the `D`-dimensional `float64` coordinate vector for point `index`.
    #[pyo3(signature = (index), text_signature = "($self, index)")]
    fn point<'py>(
        &self,
        py: Python<'py>,
        index: StrictUsize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pt = self.inner.point(index.0).map_err(core_error_to_py)?;
        Ok(PyArray1::from_slice(py, pt))
    }

    /// Find the `k` nearest points `(indices, dist_sq)` to `query`.
    #[pyo3(signature = (query, k), text_signature = "($self, query, k)")]
    fn k_nearest<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'_, PyAny>,
        k: StrictUsize,
    ) -> PyResult<IndexDistPair<'py>> {
        let q = extract_f64_1d(query, "query")?;
        let pairs = py
            .detach(|| self.inner.k_nearest(&q, k.0))
            .map_err(core_error_to_py)?;
        Ok(pairs_to_index_dist(py, pairs))
    }

    /// Find the `k` nearest points `(indices, dist_sq)` to point `index`, excluding `index`.
    #[pyo3(signature = (index, k), text_signature = "($self, index, k)")]
    fn k_nearest_excluding_self<'py>(
        &self,
        py: Python<'py>,
        index: StrictUsize,
        k: StrictUsize,
    ) -> PyResult<IndexDistPair<'py>> {
        let pairs = py
            .detach(|| self.inner.k_nearest_excluding_self(index.0, k.0))
            .map_err(core_error_to_py)?;
        Ok(pairs_to_index_dist(py, pairs))
    }

    /// Find the single nearest point `(index, dist_sq)` to point `index` excluding itself, or `None`.
    #[pyo3(signature = (index), text_signature = "($self, index)")]
    fn nearest_excluding_self(
        &self,
        py: Python<'_>,
        index: StrictUsize,
    ) -> PyResult<Option<(usize, f64)>> {
        py.detach(|| self.inner.nearest_excluding_self(index.0))
            .map_err(core_error_to_py)
    }

    /// Find all points `(indices, dist_sq)` within Euclidean `radius` of `query`.
    #[pyo3(signature = (query, radius), text_signature = "($self, query, radius)")]
    fn within_radius<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'_, PyAny>,
        radius: StrictF64,
    ) -> PyResult<IndexDistPair<'py>> {
        let q = extract_f64_1d(query, "query")?;
        let pairs = py
            .detach(|| self.inner.within_radius(&q, radius.0))
            .map_err(core_error_to_py)?;
        Ok(pairs_to_index_dist(py, pairs))
    }

    /// Find all points `(indices, dist_sq)` with squared distance `<= radius_sq` from `query`.
    #[pyo3(signature = (query, radius_sq), text_signature = "($self, query, radius_sq)")]
    fn within_radius_sq<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'_, PyAny>,
        radius_sq: StrictF64,
    ) -> PyResult<IndexDistPair<'py>> {
        let q = extract_f64_1d(query, "query")?;
        let pairs = py
            .detach(|| self.inner.within_radius_sq(&q, radius_sq.0))
            .map_err(core_error_to_py)?;
        Ok(pairs_to_index_dist(py, pairs))
    }

    /// Return a concise string representation of the `DynamicKdTree`.
    fn __repr__(&self) -> String {
        format!(
            "DynamicKdTree(num_nodes={}, dims={})",
            self.inner.len(),
            self.inner.dims()
        )
    }
}

/// Register `DynamicKdTree` on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDynamicKdTree>()?;
    Ok(())
}
