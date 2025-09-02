//! Python wrapper classes for ndarray types
//!
//! This module provides PyArray1 and PyArray2 classes that wrap ndarray::Array1 and ndarray::Array2
//! to provide a Python-native interface without depending on numpy.

use crate::FloatType;
use ndarray::{Array1, Array2};
use pyo3::prelude::*;

/// Python wrapper for ndarray::Array1<FloatType>
///
/// Provides a 1-dimensional array interface with indexing, iteration, and basic operations.
#[pyclass]
#[pyo3(name = "Array1")]
pub struct PyArray1 {
    pub(crate) array: Array1<FloatType>,
}

impl PyArray1 {
    /// Create a new PyArray1 from an ndarray::Array1
    pub fn new(array: Array1<FloatType>) -> Self {
        Self { array }
    }

    /// Get a reference to the internal ndarray
    pub fn as_array(&self) -> &Array1<FloatType> {
        &self.array
    }

    /// Convert to owned ndarray
    pub fn into_array(self) -> Array1<FloatType> {
        self.array
    }
}

#[pymethods]
impl PyArray1 {
    /// Get the length of the array
    ///
    /// :return: Number of elements
    /// :rtype: int
    fn __len__(&self) -> usize {
        self.array.len()
    }

    /// Get an element by index
    ///
    /// :param index: Index of the element
    /// :type index: int
    /// :return: The element at the given index
    /// :rtype: float
    /// :raises IndexError: If index is out of bounds
    fn __getitem__(&self, index: isize) -> PyResult<FloatType> {
        let len = self.array.len() as isize;
        let idx = if index < 0 { len + index } else { index };

        if idx < 0 || idx >= len {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of bounds for array of length {}",
                index, len
            )))
        } else {
            Ok(self.array[idx as usize])
        }
    }

    /// Set an element by index
    ///
    /// :param index: Index of the element
    /// :type index: int
    /// :param value: Value to set
    /// :type value: float
    /// :raises IndexError: If index is out of bounds
    fn __setitem__(&mut self, index: isize, value: FloatType) -> PyResult<()> {
        let len = self.array.len() as isize;
        let idx = if index < 0 { len + index } else { index };

        if idx < 0 || idx >= len {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of bounds for array of length {}",
                index, len
            )))
        } else {
            self.array[idx as usize] = value;
            Ok(())
        }
    }

    /// Return a string representation of the array
    ///
    /// :return: String representation
    /// :rtype: str
    fn __repr__(&self) -> String {
        format!("Array1({:?})", self.array.as_slice().unwrap_or(&[]))
    }

    /// Iterate over the array elements
    ///
    /// :return: Iterator over elements
    fn __iter__(slf: PyRef<Self>) -> PyArray1Iterator {
        PyArray1Iterator {
            array: slf.array.clone(),
            index: 0,
        }
    }

    /// Convert to a Python list
    ///
    /// :return: List of elements
    /// :rtype: list[float]
    fn to_list(&self) -> Vec<FloatType> {
        self.array.to_vec()
    }
}

/// Iterator for PyArray1
#[pyclass]
pub struct PyArray1Iterator {
    array: Array1<FloatType>,
    index: usize,
}

#[pymethods]
impl PyArray1Iterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<FloatType> {
        if slf.index < slf.array.len() {
            let value = slf.array[slf.index];
            slf.index += 1;
            Some(value)
        } else {
            None
        }
    }
}

/// Python wrapper for ndarray::Array2<FloatType>
///
/// Provides a 2-dimensional array interface with indexing, shape access, and basic operations.
#[pyclass]
#[pyo3(name = "Array2")]
pub struct PyArray2 {
    pub(crate) array: Array2<FloatType>,
}

impl PyArray2 {
    /// Create a new PyArray2 from an ndarray::Array2
    pub fn new(array: Array2<FloatType>) -> Self {
        Self { array }
    }

    /// Get a reference to the internal ndarray
    pub fn as_array(&self) -> &Array2<FloatType> {
        &self.array
    }

    /// Convert to owned ndarray
    pub fn into_array(self) -> Array2<FloatType> {
        self.array
    }
}

#[pymethods]
impl PyArray2 {
    /// Get the shape of the array
    ///
    /// :return: Tuple of (rows, columns)
    /// :rtype: tuple[int, int]
    #[getter]
    fn shape(&self) -> (usize, usize) {
        let shape = self.array.dim();
        (shape.0, shape.1)
    }

    /// Get the number of rows
    ///
    /// :return: Number of rows
    /// :rtype: int
    #[getter]
    fn nrows(&self) -> usize {
        self.array.nrows()
    }

    /// Get the number of columns
    ///
    /// :return: Number of columns
    /// :rtype: int
    #[getter]
    fn ncols(&self) -> usize {
        self.array.ncols()
    }

    /// Get an element by (row, column) indices
    ///
    /// :param index: Tuple of (row, column) indices
    /// :type index: tuple[int, int]
    /// :return: The element at the given indices
    /// :rtype: float
    /// :raises IndexError: If indices are out of bounds
    fn __getitem__(&self, index: (isize, isize)) -> PyResult<FloatType> {
        let (rows, cols) = (self.array.nrows() as isize, self.array.ncols() as isize);
        let (row_idx, col_idx) = index;

        let row = if row_idx < 0 { rows + row_idx } else { row_idx };
        let col = if col_idx < 0 { cols + col_idx } else { col_idx };

        if row < 0 || row >= rows || col < 0 || col >= cols {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index ({}, {}) out of bounds for array of shape ({}, {})",
                row_idx, col_idx, rows, cols
            )))
        } else {
            Ok(self.array[[row as usize, col as usize]])
        }
    }

    /// Set an element by (row, column) indices
    ///
    /// :param index: Tuple of (row, column) indices
    /// :type index: tuple[int, int]
    /// :param value: Value to set
    /// :type value: float
    /// :raises IndexError: If indices are out of bounds
    fn __setitem__(&mut self, index: (isize, isize), value: FloatType) -> PyResult<()> {
        let (rows, cols) = (self.array.nrows() as isize, self.array.ncols() as isize);
        let (row_idx, col_idx) = index;

        let row = if row_idx < 0 { rows + row_idx } else { row_idx };
        let col = if col_idx < 0 { cols + col_idx } else { col_idx };

        if row < 0 || row >= rows || col < 0 || col >= cols {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index ({}, {}) out of bounds for array of shape ({}, {})",
                row_idx, col_idx, rows, cols
            )))
        } else {
            self.array[[row as usize, col as usize]] = value;
            Ok(())
        }
    }

    /// Return a string representation of the array
    ///
    /// :return: String representation
    /// :rtype: str
    fn __repr__(&self) -> String {
        format!(
            "Array2(shape=({}, {}))",
            self.array.nrows(),
            self.array.ncols()
        )
    }

    /// Get a row as a new Array1
    ///
    /// :param index: Row index
    /// :type index: int
    /// :return: Row as Array1
    /// :rtype: Array1
    /// :raises IndexError: If index is out of bounds
    fn row(&self, index: isize) -> PyResult<PyArray1> {
        let rows = self.array.nrows() as isize;
        let row_idx = if index < 0 { rows + index } else { index };

        if row_idx < 0 || row_idx >= rows {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Row index {} out of bounds for array with {} rows",
                index, rows
            )))
        } else {
            let row = self.array.row(row_idx as usize).to_owned();
            Ok(PyArray1::new(row))
        }
    }

    /// Get a column as a new Array1
    ///
    /// :param index: Column index
    /// :type index: int
    /// :return: Column as Array1
    /// :rtype: Array1
    /// :raises IndexError: If index is out of bounds
    fn column(&self, index: isize) -> PyResult<PyArray1> {
        let cols = self.array.ncols() as isize;
        let col_idx = if index < 0 { cols + index } else { index };

        if col_idx < 0 || col_idx >= cols {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Column index {} out of bounds for array with {} columns",
                index, cols
            )))
        } else {
            let col = self.array.column(col_idx as usize).to_owned();
            Ok(PyArray1::new(col))
        }
    }

    /// Convert to a nested Python list
    ///
    /// :return: Nested list representation
    /// :rtype: list[list[float]]
    fn to_list(&self) -> Vec<Vec<FloatType>> {
        self.array.outer_iter().map(|row| row.to_vec()).collect()
    }
}

/// Register the array module with Python
pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyArray1>()?;
    m.add_class::<PyArray2>()?;
    m.add_class::<PyArray1Iterator>()?;
    Ok(())
}
