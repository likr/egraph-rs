mod drawing;
mod drawing_euclidean;
mod drawing_euclidean_2d;
mod drawing_torus_2d;

pub use drawing::*;
pub use drawing_euclidean::*;
pub use drawing_euclidean_2d::*;
pub use drawing_torus_2d::*;

use pyo3::prelude::*;

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyDrawing>()?;
    m.add_class::<PyDrawingEuclidean2d>()?;
    m.add_class::<PyDrawingEuclidean>()?;
    m.add_class::<PyDrawingTorus2d>()?;
    Ok(())
}
