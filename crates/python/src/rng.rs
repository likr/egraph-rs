use pyo3::{prelude::*, types::PyType};
use rand::prelude::*;

/// Python class for random number generation
///
/// This class provides a wrapper around Rust's cryptographically secure random number generator.
/// It can be used to provide deterministic randomness for layout algorithms when seeded.
#[pyclass]
#[pyo3(name = "Rng")]
pub struct PyRng {
    rng: StdRng,
}

impl PyRng {
    /// Returns a mutable reference to the underlying random number generator
    pub fn get_mut(&mut self) -> &mut StdRng {
        &mut self.rng
    }
}

#[pymethods]
impl PyRng {
    /// Creates a new random number generator from system entropy
    ///
    /// This constructor creates a random number generator that is seeded with random
    /// data from the operating system's entropy source, making it suitable for
    /// most applications where unpredictability is desired.
    #[new]
    fn new() -> PyRng {
        PyRng {
            rng: StdRng::from_entropy(),
        }
    }

    /// Creates a new random number generator with a specific seed
    ///
    /// This class method creates a random number generator that is seeded with
    /// a specific value, making it produce a deterministic sequence of numbers.
    /// This is useful for reproducibility in algorithms.
    ///
    /// # Parameters
    /// * `seed` - The seed value to use
    #[classmethod]
    fn seed_from(_cls: &Bound<PyType>, seed: u64) -> PyRng {
        PyRng {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

/// Registers random number generator classes with the Python module
pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyRng>()?;
    Ok(())
}
