//! Interleaved 3D coordinate buffer exposing raw linear-memory pointers for zero-copy FFI.

use egraph_core::{Layout3d, Scalar};

use crate::error::{ErrorKind, FfiError, FfiResult};

/// Contiguous interleaved `[x0, y0, z0, x1, y1, z1, ...]` coordinate buffer for Python/WASM
/// zero-copy views.
///
/// Host bindings expose `coords_ptr()` and `coords_len()` so JavaScript (`Float32Array` over
/// WebAssembly linear memory) and Python (`NumPy` `(N, 3)` view) can inspect and update 3D
/// coordinates without per-vertex FFI calls or extra copies.
#[derive(Debug, Clone, PartialEq)]
pub struct InterleavedLayout3d<S: Scalar = f64> {
    coords: Vec<S>,
}

impl<S: Scalar> InterleavedLayout3d<S> {
    /// Creates a zero-initialized interleaved buffer for `n` vertices (`3 * n` scalars).
    ///
    /// # Panics
    ///
    /// Panics if `n * 3` overflows `usize`. Use [`InterleavedLayout3d::try_zeros`] for a fallible
    /// alternative.
    pub fn zeros(n: usize) -> Self {
        let len = n.checked_mul(3).expect("vertex count * 3 overflows usize");
        Self {
            coords: vec![S::ZERO; len],
        }
    }

    /// Fallible constructor creating a zero-initialized interleaved buffer for `n` vertices.
    pub fn try_zeros(n: usize) -> FfiResult<Self> {
        let len = n
            .checked_mul(3)
            .ok_or_else(|| FfiError::new(ErrorKind::InvalidInput, "vertex count overflow"))?;
        Ok(Self {
            coords: vec![S::ZERO; len],
        })
    }

    /// Creates an interleaved buffer by copying a `[x0, y0, z0, ...]` slice whose length is a
    /// multiple of 3 and whose coordinates are all finite.
    pub fn from_interleaved(coords: &[S]) -> FfiResult<Self> {
        if !coords.len().is_multiple_of(3) {
            return Err(FfiError::new(
                ErrorKind::DimensionMismatch,
                format!(
                    "expected interleaved 3D coordinates of length divisible by 3, got {}",
                    coords.len()
                ),
            ));
        }
        if !coords.iter().all(|v| v.is_finite()) {
            return Err(FfiError::new(
                ErrorKind::NonFiniteValue,
                "InterleavedLayout3d::from_interleaved contains non-finite coordinate",
            ));
        }
        Ok(Self {
            coords: coords.to_vec(),
        })
    }

    /// Creates an interleaved buffer initialized from a core SoA [`Layout3d<S>`].
    pub fn from_layout3d(layout: &Layout3d<S>) -> Self {
        Self {
            coords: layout.to_interleaved(),
        }
    }

    /// Returns the number of vertices (`coords_len() / 3`).
    #[inline]
    pub fn len(&self) -> usize {
        self.coords.len() / 3
    }

    /// Returns `true` if the buffer stores zero vertices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Returns a const pointer to the contiguous `[x0, y0, z0, ...]` buffer.
    #[inline]
    pub fn coords_ptr(&self) -> *const S {
        self.coords.as_ptr()
    }

    /// Returns a mutable pointer to the contiguous `[x0, y0, z0, ...]` buffer.
    #[inline]
    pub fn coords_mut_ptr(&mut self) -> *mut S {
        self.coords.as_mut_ptr()
    }

    /// Returns the total scalar length of the interleaved buffer (`3 * len()`).
    #[inline]
    pub fn coords_len(&self) -> usize {
        self.coords.len()
    }

    /// Returns an immutable slice over the interleaved `[x0, y0, z0, ...]` buffer.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        &self.coords
    }

    /// Returns a mutable slice over the interleaved `[x0, y0, z0, ...]` buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        &mut self.coords
    }

    /// Converts this interleaved buffer into an SoA [`Layout3d<S>`] for solver execution.
    pub fn to_layout3d(&self) -> Layout3d<S> {
        Layout3d::from_interleaved(&self.coords)
            .expect("InterleavedLayout3d invariant guarantees length is a multiple of 3")
    }

    /// Copies coordinates from an SoA [`Layout3d<S>`] into this buffer in place.
    ///
    /// Writing in place preserves the buffer's existing allocation (when unchanged in length)
    /// for stable zero-copy host views.
    pub fn copy_from_layout3d(&mut self, layout: &Layout3d<S>) -> FfiResult<()> {
        layout
            .write_interleaved(&mut self.coords)
            .map_err(Into::into)
    }

    /// Returns the `x` coordinate of vertex `index`.
    #[inline]
    pub fn x(&self, index: usize) -> FfiResult<S> {
        self.check_index(index)?;
        Ok(self.coords[index * 3])
    }

    /// Returns the `y` coordinate of vertex `index`.
    #[inline]
    pub fn y(&self, index: usize) -> FfiResult<S> {
        self.check_index(index)?;
        Ok(self.coords[index * 3 + 1])
    }

    /// Returns the `z` coordinate of vertex `index`.
    #[inline]
    pub fn z(&self, index: usize) -> FfiResult<S> {
        self.check_index(index)?;
        Ok(self.coords[index * 3 + 2])
    }

    /// Sets the `x` coordinate of vertex `index`.
    #[inline]
    pub fn set_x(&mut self, index: usize, value: S) -> FfiResult<()> {
        self.check_index(index)?;
        self.coords[index * 3] = value;
        Ok(())
    }

    /// Sets the `y` coordinate of vertex `index`.
    #[inline]
    pub fn set_y(&mut self, index: usize, value: S) -> FfiResult<()> {
        self.check_index(index)?;
        self.coords[index * 3 + 1] = value;
        Ok(())
    }

    /// Sets the `z` coordinate of vertex `index`.
    #[inline]
    pub fn set_z(&mut self, index: usize, value: S) -> FfiResult<()> {
        self.check_index(index)?;
        self.coords[index * 3 + 2] = value;
        Ok(())
    }

    /// Returns `true` if every coordinate scalar in the buffer is finite.
    #[inline]
    pub fn is_all_finite(&self) -> bool {
        self.coords.iter().all(|v| v.is_finite())
    }

    /// Validates that every coordinate scalar in the buffer is finite.
    #[inline]
    pub fn validate_finite(&self) -> FfiResult<()> {
        if self.is_all_finite() {
            Ok(())
        } else {
            Err(FfiError::new(
                ErrorKind::NonFiniteValue,
                "InterleavedLayout3d contains non-finite coordinate",
            ))
        }
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
    fn test_interleaved_layout3d_roundtrip_and_ptrs() {
        let mut buf = InterleavedLayout3d::<f32>::zeros(3);
        assert_eq!(buf.len(), 3);
        assert!(!buf.is_empty());
        assert_eq!(buf.coords_len(), 9);
        assert_eq!(buf.coords_ptr(), buf.as_slice().as_ptr());
        assert_eq!(buf.coords_mut_ptr(), buf.as_mut_slice().as_mut_ptr());

        buf.set_x(1, 2.5).unwrap();
        buf.set_y(1, -3.5).unwrap();
        buf.set_z(1, 7.25).unwrap();
        assert_eq!(buf.x(1).unwrap(), 2.5);
        assert_eq!(buf.y(1).unwrap(), -3.5);
        assert_eq!(buf.z(1).unwrap(), 7.25);
        assert!(buf.is_all_finite());
        buf.validate_finite().unwrap();

        let mut soa = buf.to_layout3d();
        assert_eq!(soa.x(), &[0.0, 2.5, 0.0]);
        assert_eq!(soa.y(), &[0.0, -3.5, 0.0]);
        assert_eq!(soa.z(), &[0.0, 7.25, 0.0]);

        soa.z_mut()[2] = 9.0;
        buf.copy_from_layout3d(&soa).unwrap();
        assert_eq!(
            buf.as_slice(),
            &[0.0, 0.0, 0.0, 2.5, -3.5, 7.25, 0.0, 0.0, 9.0]
        );

        let from_soa = InterleavedLayout3d::from_layout3d(&soa);
        assert_eq!(from_soa, buf);

        let empty = InterleavedLayout3d::<f64>::try_zeros(0).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.coords_len(), 0);
    }

    #[test]
    fn test_interleaved_layout3d_validation_and_non_finite() {
        let bad_len_1 = InterleavedLayout3d::<f64>::from_interleaved(&[1.0, 2.0]).unwrap_err();
        assert_eq!(bad_len_1.kind, ErrorKind::DimensionMismatch);
        let bad_len_2 =
            InterleavedLayout3d::<f64>::from_interleaved(&[1.0, 2.0, 3.0, 4.0]).unwrap_err();
        assert_eq!(bad_len_2.kind, ErrorKind::DimensionMismatch);

        let non_finite =
            InterleavedLayout3d::<f64>::from_interleaved(&[1.0, f64::NAN, 3.0]).unwrap_err();
        assert_eq!(non_finite.kind, ErrorKind::NonFiniteValue);

        let overflow = InterleavedLayout3d::<f64>::try_zeros(usize::MAX).unwrap_err();
        assert_eq!(overflow.kind, ErrorKind::InvalidInput);

        let mut buf =
            InterleavedLayout3d::from_interleaved(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(buf.x(2).unwrap_err().kind, ErrorKind::IndexOutOfBounds);
        assert_eq!(buf.y(2).unwrap_err().kind, ErrorKind::IndexOutOfBounds);
        assert_eq!(buf.z(2).unwrap_err().kind, ErrorKind::IndexOutOfBounds);
        assert_eq!(
            buf.set_x(2, 0.0).unwrap_err().kind,
            ErrorKind::IndexOutOfBounds
        );
        assert_eq!(
            buf.set_y(2, 0.0).unwrap_err().kind,
            ErrorKind::IndexOutOfBounds
        );
        assert_eq!(
            buf.set_z(2, 0.0).unwrap_err().kind,
            ErrorKind::IndexOutOfBounds
        );

        let mismatch = Layout3d::<f64>::zeros(3);
        assert_eq!(
            buf.copy_from_layout3d(&mismatch).unwrap_err().kind,
            ErrorKind::DimensionMismatch
        );

        buf.as_mut_slice()[5] = f64::INFINITY;
        assert!(!buf.is_all_finite());
        assert_eq!(
            buf.validate_finite().unwrap_err().kind,
            ErrorKind::NonFiniteValue
        );
    }
}
