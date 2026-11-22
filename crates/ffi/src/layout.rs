//! Interleaved 2D coordinate buffer exposing raw linear-memory pointers for zero-copy FFI.

use egraph_core::{Layout2d, Scalar};

use crate::error::{ErrorKind, FfiError, FfiResult};

/// Contiguous interleaved `[x0, y0, x1, y1, ...]` coordinate buffer for Python/WASM zero-copy views.
///
/// Host bindings expose `coords_ptr()` and `coords_len()` so JavaScript (`Float32Array` over
/// WebAssembly linear memory) and Python (`NumPy` `(N, 2)` view) can inspect and draw coordinates
/// without per-vertex FFI calls or extra copies.
#[derive(Debug, Clone, PartialEq)]
pub struct InterleavedLayout2d<S: Scalar = f64> {
    coords: Vec<S>,
}

impl<S: Scalar> InterleavedLayout2d<S> {
    /// Creates a zero-initialized interleaved buffer for `n` vertices (`2 * n` scalars).
    ///
    /// # Panics
    ///
    /// Panics if `n * 2` overflows `usize`. Use [`InterleavedLayout2d::try_zeros`] for a fallible
    /// alternative.
    pub fn zeros(n: usize) -> Self {
        let len = n.checked_mul(2).expect("vertex count * 2 overflows usize");
        Self {
            coords: vec![S::ZERO; len],
        }
    }

    /// Fallible constructor creating a zero-initialized interleaved buffer for `n` vertices.
    pub fn try_zeros(n: usize) -> FfiResult<Self> {
        let len = n
            .checked_mul(2)
            .ok_or_else(|| FfiError::new(ErrorKind::InvalidInput, "vertex count overflow"))?;
        Ok(Self {
            coords: vec![S::ZERO; len],
        })
    }

    /// Creates an interleaved buffer by copying an even-length `[x0, y0, ...]` slice.
    pub fn from_interleaved(coords: &[S]) -> FfiResult<Self> {
        if !coords.len().is_multiple_of(2) {
            return Err(FfiError::new(
                ErrorKind::DimensionMismatch,
                format!(
                    "expected even-length interleaved coordinates, got {}",
                    coords.len()
                ),
            ));
        }
        Ok(Self {
            coords: coords.to_vec(),
        })
    }

    /// Creates an interleaved buffer initialized from a core SoA [`Layout2d<S>`].
    pub fn from_layout2d(layout: &Layout2d<S>) -> Self {
        Self {
            coords: layout.to_interleaved(),
        }
    }

    /// Returns the number of vertices (`coords_len() / 2`).
    #[inline]
    pub fn len(&self) -> usize {
        self.coords.len() / 2
    }

    /// Returns `true` if the buffer stores zero vertices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Returns a const pointer to the contiguous `[x0, y0, ...]` buffer.
    #[inline]
    pub fn coords_ptr(&self) -> *const S {
        self.coords.as_ptr()
    }

    /// Returns a mutable pointer to the contiguous `[x0, y0, ...]` buffer.
    #[inline]
    pub fn coords_mut_ptr(&mut self) -> *mut S {
        self.coords.as_mut_ptr()
    }

    /// Returns the total scalar length of the interleaved buffer (`2 * len()`).
    #[inline]
    pub fn coords_len(&self) -> usize {
        self.coords.len()
    }

    /// Returns an immutable slice over the interleaved `[x0, y0, ...]` buffer.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        &self.coords
    }

    /// Returns a mutable slice over the interleaved `[x0, y0, ...]` buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        &mut self.coords
    }

    /// Converts this interleaved buffer into an SoA [`Layout2d<S>`] for solver execution.
    pub fn to_layout2d(&self) -> Layout2d<S> {
        Layout2d::from_interleaved(&self.coords)
            .expect("InterleavedLayout2d invariant guarantees even length")
    }

    /// Copies coordinates from an SoA [`Layout2d<S>`] into this buffer in place.
    ///
    /// Writing in place preserves the buffer's existing allocation (when unchanged in length)
    /// for stable zero-copy host views.
    pub fn copy_from_layout2d(&mut self, layout: &Layout2d<S>) -> FfiResult<()> {
        layout
            .write_interleaved(&mut self.coords)
            .map_err(Into::into)
    }

    /// Returns the `x` coordinate of vertex `index`.
    #[inline]
    pub fn x(&self, index: usize) -> FfiResult<S> {
        self.check_index(index)?;
        Ok(self.coords[index * 2])
    }

    /// Returns the `y` coordinate of vertex `index`.
    #[inline]
    pub fn y(&self, index: usize) -> FfiResult<S> {
        self.check_index(index)?;
        Ok(self.coords[index * 2 + 1])
    }

    /// Sets the `x` coordinate of vertex `index`.
    #[inline]
    pub fn set_x(&mut self, index: usize, value: S) -> FfiResult<()> {
        self.check_index(index)?;
        self.coords[index * 2] = value;
        Ok(())
    }

    /// Sets the `y` coordinate of vertex `index`.
    #[inline]
    pub fn set_y(&mut self, index: usize, value: S) -> FfiResult<()> {
        self.check_index(index)?;
        self.coords[index * 2 + 1] = value;
        Ok(())
    }

    /// Returns `true` if every coordinate scalar in the buffer is finite.
    #[inline]
    pub fn is_all_finite(&self) -> bool {
        self.coords.iter().all(|v| v.is_finite())
    }

    #[inline]
    fn check_index(&self, index: usize) -> FfiResult<()> {
        let n = self.len();
        if index >= n {
            return Err(FfiError::new(
                ErrorKind::IndexOutOfBounds,
                format!("vertex index {index} out of bounds for len {n}"),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleaved_layout_roundtrip_and_ptrs() {
        let mut buf = InterleavedLayout2d::<f32>::zeros(3);
        assert_eq!(buf.len(), 3);
        assert!(!buf.is_empty());
        assert_eq!(buf.coords_len(), 6);
        assert_eq!(buf.coords_ptr(), buf.as_slice().as_ptr());
        assert_eq!(buf.coords_mut_ptr(), buf.as_mut_slice().as_mut_ptr());

        buf.set_x(1, 2.5).unwrap();
        buf.set_y(1, -3.5).unwrap();
        assert_eq!(buf.x(1).unwrap(), 2.5);
        assert_eq!(buf.y(1).unwrap(), -3.5);
        assert!(buf.is_all_finite());

        let mut soa = buf.to_layout2d();
        assert_eq!(soa.x(), &[0.0, 2.5, 0.0]);
        assert_eq!(soa.y(), &[0.0, -3.5, 0.0]);

        soa.x_mut()[2] = 9.0;
        buf.copy_from_layout2d(&soa).unwrap();
        assert_eq!(buf.as_slice(), &[0.0, 0.0, 2.5, -3.5, 9.0, 0.0]);
    }

    #[test]
    fn test_interleaved_layout_validation_and_non_finite() {
        let odd = InterleavedLayout2d::<f64>::from_interleaved(&[1.0, 2.0, 3.0]).unwrap_err();
        assert_eq!(odd.kind, ErrorKind::DimensionMismatch);

        let mut buf = InterleavedLayout2d::from_interleaved(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(buf.x(2).unwrap_err().kind, ErrorKind::IndexOutOfBounds);
        assert_eq!(
            buf.set_y(2, 0.0).unwrap_err().kind,
            ErrorKind::IndexOutOfBounds
        );

        let mismatch = Layout2d::<f64>::zeros(3);
        assert_eq!(
            buf.copy_from_layout2d(&mismatch).unwrap_err().kind,
            ErrorKind::DimensionMismatch
        );

        buf.as_mut_slice()[3] = f64::INFINITY;
        assert!(!buf.is_all_finite());
    }
}
