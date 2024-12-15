use pyo3::{prelude::*, types::PyType};
use rand::prelude::*;

#[pyclass]
#[pyo3(name = "Rng")]
pub struct PyRng {
    rng: StdRng,
}

impl PyRng {
    pub fn get_mut(&mut self) -> &mut StdRng {
        &mut self.rng
    }
}

#[pymethods]
impl PyRng {
    #[new]
    fn new() -> PyRng {
        PyRng {
            rng: StdRng::from_entropy(),
        }
    }

    #[classmethod]
    fn seed_from(_cls: &Bound<PyType>, seed: u64) -> PyRng {
        PyRng {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyRng>()?;
    Ok(())
}
