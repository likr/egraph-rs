/// Random number generation utilities for the Python bindings
///
/// This module provides a wrapper around Rust's high-quality random number generator,
/// making it available to Python code. It's particularly useful for graph layout algorithms
/// that require randomization (like initial node placement or stochastic optimization).
///
/// The module offers both entropy-based (truly random) initialization and seed-based
/// (deterministic) initialization, allowing for reproducible results when desired.
///
/// # Example use cases
///
/// - Initial node placement in layout algorithms
/// - Randomized optimization methods like SGD
/// - Sampling pivot nodes in sparse approximation algorithms
/// - Reproducible layouts for benchmarking and comparison
use pyo3::{prelude::*, types::PyType};
use rand::prelude::*;

/// Python class for random number generation
///
/// This class provides a wrapper around Rust's cryptographically secure random number generator.
/// It can be used to provide deterministic randomness for layout algorithms when seeded,
/// which is essential for reproducible results in visualization and optimization tasks.
///
/// The class offers two constructors:
/// - `Rng()` - Creates a generator from system entropy (non-deterministic)
/// - `Rng.seed_from(seed)` - Creates a generator with a specific seed (deterministic)
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
    /// :param seed: The seed value to use
    /// :type seed: int
    /// :return: A new random number generator with the specified seed
    /// :rtype: Rng
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
