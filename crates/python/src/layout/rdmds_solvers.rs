use crate::FloatType;
use petgraph_linalg_rdmds::solvers::{CgSolver, Ic0CgSolver, JacobiCgSolver, LinearSolver};
use petgraph_distance::SparseSymmetricMatrix;
use ndarray::Array1;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "CgSolver")]
#[derive(Clone)]
pub struct PyCgSolver {
    pub solver: CgSolver<FloatType>,
}

#[pymethods]
impl PyCgSolver {
    #[new]
    #[pyo3(signature = (max_iterations=100, tolerance=1e-4))]
    fn new(max_iterations: usize, tolerance: FloatType) -> Self {
        Self {
            solver: CgSolver { max_iterations, tolerance },
        }
    }
}

#[pyclass]
#[pyo3(name = "JacobiCgSolver")]
#[derive(Clone)]
pub struct PyJacobiCgSolver {
    pub solver: JacobiCgSolver<FloatType>,
}

#[pymethods]
impl PyJacobiCgSolver {
    #[new]
    #[pyo3(signature = (max_iterations=100, tolerance=1e-4))]
    fn new(max_iterations: usize, tolerance: FloatType) -> Self {
        Self {
            solver: JacobiCgSolver { max_iterations, tolerance },
        }
    }
}

#[pyclass]
#[pyo3(name = "Ic0CgSolver")]
#[derive(Clone)]
pub struct PyIc0CgSolver {
    pub solver: Ic0CgSolver<FloatType>,
}

#[pymethods]
impl PyIc0CgSolver {
    #[new]
    #[pyo3(signature = (max_iterations=100, tolerance=1e-4))]
    fn new(max_iterations: usize, tolerance: FloatType) -> Self {
        Self {
            solver: Ic0CgSolver { max_iterations, tolerance },
        }
    }
}

#[derive(FromPyObject)]
pub enum PySolverEnum {
    Cg(PyCgSolver),
    JacobiCg(PyJacobiCgSolver),
    Ic0Cg(PyIc0CgSolver),
}

impl LinearSolver<FloatType> for PySolverEnum {
    fn solve(&self, matrix: &SparseSymmetricMatrix<FloatType>, b: &Array1<FloatType>, x: &mut Array1<FloatType>) -> usize {
        match self {
            PySolverEnum::Cg(s) => s.solver.solve(matrix, b, x),
            PySolverEnum::JacobiCg(s) => s.solver.solve(matrix, b, x),
            PySolverEnum::Ic0Cg(s) => s.solver.solve(matrix, b, x),
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCgSolver>()?;
    m.add_class::<PyJacobiCgSolver>()?;
    m.add_class::<PyIc0CgSolver>()?;
    Ok(())
}
