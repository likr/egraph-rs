use crate::FloatType;
use petgraph_linalg_rdmds::solvers::{
    AmgCgSolver, AmgCgSolverInstance, CgSolver, CgSolverInstance, Ic0CgSolver, Ic0CgSolverInstance,
    JacobiCgSolver, JacobiCgSolverInstance, LinearSolver, LinearSolverBuilder,
};
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

#[pyclass]
#[pyo3(name = "AmgCgSolver")]
#[derive(Clone)]
pub struct PyAmgCgSolver {
    pub solver: AmgCgSolver<FloatType>,
}

#[pymethods]
impl PyAmgCgSolver {
    #[new]
    #[pyo3(signature = (max_iterations=100, tolerance=1e-4))]
    fn new(max_iterations: usize, tolerance: FloatType) -> Self {
        Self {
            solver: AmgCgSolver { max_iterations, tolerance },
        }
    }
}

#[derive(FromPyObject)]
pub enum PySolverEnum {
    Cg(PyCgSolver),
    JacobiCg(PyJacobiCgSolver),
    Ic0Cg(PyIc0CgSolver),
    AmgCg(PyAmgCgSolver),
}

pub enum PySolverInstanceEnum {
    Cg(CgSolverInstance<FloatType>),
    JacobiCg(JacobiCgSolverInstance<FloatType>),
    Ic0Cg(Ic0CgSolverInstance<FloatType>),
    AmgCg(AmgCgSolverInstance<FloatType>),
}

impl LinearSolverBuilder<FloatType> for PySolverEnum {
    type Solver = PySolverInstanceEnum;

    fn build(&self, matrix: SparseSymmetricMatrix<FloatType>) -> Self::Solver {
        match self {
            PySolverEnum::Cg(s) => PySolverInstanceEnum::Cg(s.solver.build(matrix)),
            PySolverEnum::JacobiCg(s) => PySolverInstanceEnum::JacobiCg(s.solver.build(matrix)),
            PySolverEnum::Ic0Cg(s) => PySolverInstanceEnum::Ic0Cg(s.solver.build(matrix)),
            PySolverEnum::AmgCg(s) => PySolverInstanceEnum::AmgCg(s.solver.build(matrix)),
        }
    }
}

impl LinearSolver<FloatType> for PySolverInstanceEnum {
    fn matrix(&self) -> &SparseSymmetricMatrix<FloatType> {
        match self {
            PySolverInstanceEnum::Cg(s) => s.matrix(),
            PySolverInstanceEnum::JacobiCg(s) => s.matrix(),
            PySolverInstanceEnum::Ic0Cg(s) => s.matrix(),
            PySolverInstanceEnum::AmgCg(s) => s.matrix(),
        }
    }

    fn solve(&self, b: &Array1<FloatType>, x: &mut Array1<FloatType>) -> usize {
        match self {
            PySolverInstanceEnum::Cg(s) => s.solve(b, x),
            PySolverInstanceEnum::JacobiCg(s) => s.solve(b, x),
            PySolverInstanceEnum::Ic0Cg(s) => s.solve(b, x),
            PySolverInstanceEnum::AmgCg(s) => s.solve(b, x),
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCgSolver>()?;
    m.add_class::<PyJacobiCgSolver>()?;
    m.add_class::<PyIc0CgSolver>()?;
    m.add_class::<PyAmgCgSolver>()?;
    Ok(())
}
