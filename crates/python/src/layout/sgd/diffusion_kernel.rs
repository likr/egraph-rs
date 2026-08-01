use crate::{
    distance_matrix::PyLaplacian,
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::PySgd,
    FloatType,
};
use petgraph::visit::{EdgeRef, IntoNodeIdentifiers};
use petgraph_layout_sgd::PivotDiffusionSgd;
use petgraph_linalg_diffusion_kernel::{
    DiffusionKernel, LowRankDiffusionKernel, MultiscaleDiffusionKernel,
    PivotDiffusionDistanceMatrix,
};
use pyo3::prelude::*;

/// Python class for querying diffusion kernel matrix elements
#[pyclass]
#[pyo3(name = "DiffusionKernel")]
pub struct PyDiffusionKernel {
    pub(crate) kernel: DiffusionKernel<FloatType>,
}

#[pymethods]
impl PyDiffusionKernel {
    /// Creates a new DiffusionKernel with automatic lambda_max estimation
    #[new]
    fn new(
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        num_vectors: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = DiffusionKernel::new(&laplacian.matrix, t, degree, num_vectors, rng.get_mut());

        Ok(PyDiffusionKernel { kernel })
    }

    /// Creates a new DiffusionKernel with externally provided lambda_max
    #[staticmethod]
    fn new_with_lambda_max(
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        lambda_max: FloatType,
        num_vectors: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = DiffusionKernel::new_with_lambda_max(
            &laplacian.matrix,
            t,
            degree,
            lambda_max,
            num_vectors,
            rng.get_mut(),
        );

        Ok(PyDiffusionKernel { kernel })
    }

    /// Queries the (i, j) element of the diffusion kernel matrix
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.get(i, j)
    }

    /// Computes the exact single-source heat diffusion vector K e_pivot for a given pivot node
    #[staticmethod]
    fn single_source_heat_vector(
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        pivot: usize,
    ) -> Vec<FloatType> {
        DiffusionKernel::single_source_heat_vector(&laplacian.matrix, t, degree, pivot).to_vec()
    }

    /// Computes the distance vector from a pivot to all nodes using single-source exact heat diffusion
    fn pivot_distance_vector(
        &self,
        laplacian: &PyLaplacian,
        t: FloatType,
        degree: usize,
        pivot: usize,
    ) -> Vec<FloatType> {
        PivotDiffusionDistanceMatrix::pivot_distance_vector(
            &laplacian.matrix,
            &self.kernel,
            t,
            degree,
            pivot,
        )
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.n()
    }
}

/// Python class for computing multiscale diffusion distance between nodes
#[pyclass]
#[pyo3(name = "MultiscaleDiffusionKernel")]
pub struct PyMultiscaleDiffusionKernel {
    pub(crate) kernel: MultiscaleDiffusionKernel,
}

#[pymethods]
impl PyMultiscaleDiffusionKernel {
    /// Creates a new MultiscaleDiffusionKernel and builds its Hutchinson index
    #[new]
    fn new(
        laplacian: &PyLaplacian,
        alpha: FloatType,
        num_samples: usize,
        tol: FloatType,
        max_iter: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let mut kernel = MultiscaleDiffusionKernel::new(
            laplacian.matrix.clone(),
            alpha,
            num_samples,
            tol,
            max_iter,
        );

        kernel.build_index_with_rng(rng.get_mut());

        Ok(PyMultiscaleDiffusionKernel { kernel })
    }

    /// Queries the multiscale diffusion distance between nodes i and j
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.sample_distance(i, j).unwrap_or(0.0)
    }

    /// Computes multiscale diffusion distance between nodes i and j
    fn sample_distance(&self, i: usize, j: usize) -> PyResult<FloatType> {
        self.kernel
            .sample_distance(i, j)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Re-builds the Hutchinson index using a custom random number generator
    fn build_index(&mut self, rng: &mut crate::rng::PyRng) {
        self.kernel.build_index_with_rng(rng.get_mut());
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.n()
    }
}

/// Python class for creating Pivot-based Diffusion SGD layout instances
#[pyclass]
#[pyo3(name = "PivotDiffusionSgd")]
pub struct PyPivotDiffusionSgd {
    builder: PivotDiffusionSgd<FloatType>,
}

#[pymethods]
impl PyPivotDiffusionSgd {
    /// Creates a new PivotDiffusionSgd builder
    #[new]
    fn new() -> Self {
        Self {
            builder: PivotDiffusionSgd::new(),
        }
    }

    /// Sets the diffusion time parameter t
    fn t(mut slf: PyRefMut<Self>, t: FloatType) -> Py<Self> {
        slf.builder.t(t);
        slf.into()
    }

    /// Sets the degree of Chebyshev polynomial approximation
    fn degree(mut slf: PyRefMut<Self>, degree: usize) -> Py<Self> {
        slf.builder.degree(degree);
        slf.into()
    }

    /// Sets the number of vectors used for Hutchinson diagonal trace estimation
    fn num_vectors(mut slf: PyRefMut<Self>, num_vectors: usize) -> Py<Self> {
        slf.builder.num_vectors(num_vectors);
        slf.into()
    }

    /// Sets the number of pivot nodes
    fn h(mut slf: PyRefMut<Self>, h: usize) -> Py<Self> {
        slf.builder.h(h);
        slf.into()
    }

    /// Builds an SGD instance with heat diffusion distances and max-min random pivot sampling
    fn build(
        &self,
        graph: &PyGraphAdapter,
        laplacian: &PyLaplacian,
        f: &Bound<PyAny>,
        rng: &mut crate::rng::PyRng,
    ) -> PySgd {
        PySgd::new_with_sgd(match graph.graph() {
            GraphType::Graph(native_graph) => self.builder.build(
                native_graph,
                &laplacian.matrix,
                |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                rng.get_mut(),
            ),
            _ => panic!("unsupported graph type"),
        })
    }

    /// Builds an SGD instance with pre-selected pivot nodes
    fn build_with_pivot(
        &self,
        graph: &PyGraphAdapter,
        laplacian: &PyLaplacian,
        f: &Bound<PyAny>,
        pivot: Vec<usize>,
        rng: &mut crate::rng::PyRng,
    ) -> PySgd {
        PySgd::new_with_sgd(match graph.graph() {
            GraphType::Graph(native_graph) => {
                let nodes = native_graph.node_identifiers().collect::<Vec<_>>();
                let pivot_nodes = pivot.iter().map(|&i| nodes[i]).collect::<Vec<_>>();
                self.builder.build_with_pivot(
                    native_graph,
                    &laplacian.matrix,
                    |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                    &pivot_nodes,
                    rng.get_mut(),
                )
            }
            _ => panic!("unsupported graph type"),
        })
    }
}

/// Python class for querying low-rank spectral heat kernel matrix elements and heat diffusion distances
#[pyclass]
#[pyo3(name = "LowRankDiffusionKernel")]
pub struct PyLowRankDiffusionKernel {
    pub(crate) kernel: LowRankDiffusionKernel<FloatType>,
}

#[pymethods]
impl PyLowRankDiffusionKernel {
    /// Creates a new LowRankDiffusionKernel by computing smallest eigenvalues and eigenvectors
    #[new]
    fn new(
        laplacian: &PyLaplacian,
        t: FloatType,
        rank: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = LowRankDiffusionKernel::new(&laplacian.matrix, t, rank, rng.get_mut());
        Ok(PyLowRankDiffusionKernel { kernel })
    }

    /// Creates a LowRankDiffusionKernel with custom eigensolver iteration and tolerance parameters
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    fn new_with_params(
        laplacian: &PyLaplacian,
        t: FloatType,
        rank: usize,
        shift: FloatType,
        eigenvalue_max_iterations: usize,
        cg_max_iterations: usize,
        eigenvalue_tolerance: FloatType,
        cg_tolerance: FloatType,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = LowRankDiffusionKernel::new_with_params(
            &laplacian.matrix,
            t,
            rank,
            shift,
            eigenvalue_max_iterations,
            cg_max_iterations,
            eigenvalue_tolerance,
            cg_tolerance,
            rng.get_mut(),
        );
        Ok(PyLowRankDiffusionKernel { kernel })
    }

    /// Queries the (i, j) element of the low-rank heat kernel matrix
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.get(i, j)
    }

    /// Computes the heat diffusion distance between node i and node j
    fn distance(&self, i: usize, j: usize) -> FloatType {
        self.kernel.distance(i, j)
    }

    /// Computes the single-source heat diffusion distance vector from a pivot node
    fn pivot_distance_vector(&self, pivot: usize) -> Vec<FloatType> {
        self.kernel.pivot_distance_vector(pivot)
    }

    /// Returns the number of nodes in the graph
    fn n(&self) -> usize {
        self.kernel.n()
    }

    /// Returns the diffusion time parameter t
    fn t(&self) -> FloatType {
        self.kernel.t()
    }

    /// Returns the approximation rank r
    fn rank(&self) -> usize {
        self.kernel.rank()
    }

    /// Returns a list of computed eigenvalues
    fn eigenvalues(&self) -> Vec<FloatType> {
        self.kernel.eigenvalues().to_vec()
    }
}
