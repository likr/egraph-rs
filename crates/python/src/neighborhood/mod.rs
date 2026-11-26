//! Neighborhood graph construction, adaptive perplexity, probability, and DoG Varadhan bindings.

mod dog_varadhan;
mod knn;
mod probabilities;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use dog_varadhan::PyDogVaradhanLandmarks;
pub(crate) use types::{PyKnnGraph, PySparseProbabilities};

use pyo3::prelude::*;

/// Register neighborhood graph, probability, and DoG Varadhan bindings on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    types::register(m)?;
    knn::register(m)?;
    probabilities::register(m)?;
    dog_varadhan::register(m)?;
    Ok(())
}
