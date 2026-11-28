//! Flat polyline buffer validation and interleaved 2D/3D views for [`EdgePolylines`].

use egraph_core::{EdgePolylines, Layout2d, Layout3d, LayoutStorage, Scalar};

use crate::error::{ErrorKind, FfiError, FfiResult};
use crate::layout::InterleavedLayout2d;
use crate::layout3d::InterleavedLayout3d;

/// Validates `edge_offsets` against a control-point count `num_points`.
///
/// Checks that `edge_offsets` is non-empty, starts at `0`, is monotonically non-decreasing,
/// and ends at `num_points`.
pub fn validate_edge_offsets(edge_offsets: &[u32], num_points: usize) -> FfiResult<()> {
    if edge_offsets.is_empty() {
        return Err(FfiError::new(
            ErrorKind::InvalidInput,
            "EdgePolylines edge_offsets must contain at least 1 element",
        ));
    }
    if edge_offsets[0] != 0 {
        return Err(FfiError::new(
            ErrorKind::InvalidInput,
            format!(
                "EdgePolylines edge_offsets[0] must be 0, got {}",
                edge_offsets[0]
            ),
        ));
    }
    for (edge, win) in edge_offsets.windows(2).enumerate() {
        if win[0] > win[1] {
            return Err(FfiError::new(
                ErrorKind::InvalidInput,
                format!(
                    "EdgePolylines edge_offsets must be monotonic at edge {edge}: {} > {}",
                    win[0], win[1]
                ),
            ));
        }
    }
    let last = *edge_offsets.last().unwrap() as usize;
    if last != num_points {
        return Err(FfiError::new(
            ErrorKind::DimensionMismatch,
            format!(
                "dimension mismatch in EdgePolylines::edge_offsets: expected {last}, got {num_points}"
            ),
        ));
    }
    Ok(())
}

/// Validates `edge_offsets` and `points` invariants for any [`LayoutStorage`] implementation,
/// including coordinate finiteness.
pub fn validate_edge_polylines_parts<L: LayoutStorage>(
    edge_offsets: &[u32],
    points: &L,
) -> FfiResult<()> {
    validate_edge_offsets(edge_offsets, points.len())?;
    points.validate_finite().map_err(Into::into)
}

/// Contiguous interleaved 2D polyline buffer (`edge_offsets` + `[x0, y0, ...]`) for zero-copy
/// Python/WASM views.
#[derive(Debug, Clone, PartialEq)]
pub struct InterleavedEdgePolylines2d<S: Scalar = f64> {
    edge_offsets: Vec<u32>,
    points: InterleavedLayout2d<S>,
}

impl<S: Scalar> Default for InterleavedEdgePolylines2d<S> {
    fn default() -> Self {
        Self {
            edge_offsets: vec![0],
            points: InterleavedLayout2d::zeros(0),
        }
    }
}

impl<S: Scalar> InterleavedEdgePolylines2d<S> {
    /// Creates an empty 2D polyline buffer with zero edges and zero points.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Constructs a 2D polyline buffer from `edge_offsets` and interleaved `[x0, y0, ...]`
    /// control-point coordinates, validating offset invariants, even length, and finiteness.
    pub fn from_parts(edge_offsets: &[u32], coords: &[S]) -> FfiResult<Self> {
        let points = InterleavedLayout2d::from_interleaved(coords)?;
        points.validate_finite()?;
        validate_edge_offsets(edge_offsets, points.len())?;
        Ok(Self {
            edge_offsets: edge_offsets.to_vec(),
            points,
        })
    }

    /// Creates a 2D interleaved polyline buffer from a core [`EdgePolylines<Layout2d<S>>`].
    pub fn from_core(polylines: &EdgePolylines<Layout2d<S>>) -> Self {
        Self {
            edge_offsets: polylines.edge_offsets().to_vec(),
            points: InterleavedLayout2d::from_layout2d(polylines.points()),
        }
    }

    /// Converts this interleaved buffer into a core [`EdgePolylines<Layout2d<S>>`].
    pub fn to_core(&self) -> FfiResult<EdgePolylines<Layout2d<S>>> {
        EdgePolylines::from_parts(&self.edge_offsets, self.points.to_layout2d()).map_err(Into::into)
    }

    /// Replaces this buffer's contents from a core [`EdgePolylines<Layout2d<S>>`].
    ///
    /// When `edge_offsets` and `points` lengths match the existing allocation (such as intra-cycle
    /// updates in iterative bundling), coordinates and offsets are copied in place without
    /// reallocating.
    pub fn replace_from_core(&mut self, polylines: &EdgePolylines<Layout2d<S>>) {
        let new_offsets = polylines.edge_offsets();
        if self.edge_offsets.len() == new_offsets.len() {
            self.edge_offsets.copy_from_slice(new_offsets);
        } else {
            self.edge_offsets.clear();
            self.edge_offsets.extend_from_slice(new_offsets);
        }

        if self.points.len() == polylines.total_points() {
            self.points
                .copy_from_layout2d(polylines.points())
                .expect("point length checked immediately above");
        } else {
            self.points = InterleavedLayout2d::from_layout2d(polylines.points());
        }
    }

    /// Returns the number of polylines (edges).
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edge_offsets.len().saturating_sub(1)
    }

    /// Returns the total number of control points across all edges.
    #[inline]
    pub fn total_points(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the buffer stores zero edges.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_edges() == 0
    }

    /// Returns the `edge_offsets` slice of length `num_edges() + 1`.
    #[inline]
    pub fn edge_offsets(&self) -> &[u32] {
        &self.edge_offsets
    }

    /// Returns a const pointer to the contiguous `edge_offsets` buffer.
    #[inline]
    pub fn offsets_ptr(&self) -> *const u32 {
        self.edge_offsets.as_ptr()
    }

    /// Returns the length of `edge_offsets` (`num_edges() + 1`).
    #[inline]
    pub fn offsets_len(&self) -> usize {
        self.edge_offsets.len()
    }

    /// Returns a shared reference to the underlying [`InterleavedLayout2d<S>`] points buffer.
    #[inline]
    pub fn points(&self) -> &InterleavedLayout2d<S> {
        &self.points
    }

    /// Returns a mutable reference to the underlying [`InterleavedLayout2d<S>`] points buffer.
    #[inline]
    pub fn points_mut(&mut self) -> &mut InterleavedLayout2d<S> {
        &mut self.points
    }

    /// Returns a const pointer to the contiguous `[x0, y0, ...]` control-point buffer.
    #[inline]
    pub fn points_ptr(&self) -> *const S {
        self.points.coords_ptr()
    }

    /// Returns a mutable pointer to the contiguous `[x0, y0, ...]` control-point buffer.
    #[inline]
    pub fn points_mut_ptr(&mut self) -> *mut S {
        self.points.coords_mut_ptr()
    }

    /// Returns the total scalar length of the interleaved control-point buffer (`2 * total_points()`).
    #[inline]
    pub fn points_len(&self) -> usize {
        self.points.coords_len()
    }

    /// Returns an immutable slice over the interleaved `[x0, y0, ...]` control-point buffer.
    #[inline]
    pub fn points_as_slice(&self) -> &[S] {
        self.points.as_slice()
    }

    /// Returns a mutable slice over the interleaved `[x0, y0, ...]` control-point buffer.
    #[inline]
    pub fn points_as_mut_slice(&mut self) -> &mut [S] {
        self.points.as_mut_slice()
    }

    /// Returns `true` if every control-point coordinate is finite.
    #[inline]
    pub fn is_all_finite(&self) -> bool {
        self.points.is_all_finite()
    }

    /// Validates that every control-point coordinate in the buffer is finite.
    #[inline]
    pub fn validate_finite(&self) -> FfiResult<()> {
        self.points.validate_finite()
    }
}

/// Contiguous interleaved 3D polyline buffer (`edge_offsets` + `[x0, y0, z0, ...]`) for zero-copy
/// Python/WASM views.
#[derive(Debug, Clone, PartialEq)]
pub struct InterleavedEdgePolylines3d<S: Scalar = f64> {
    edge_offsets: Vec<u32>,
    points: InterleavedLayout3d<S>,
}

impl<S: Scalar> Default for InterleavedEdgePolylines3d<S> {
    fn default() -> Self {
        Self {
            edge_offsets: vec![0],
            points: InterleavedLayout3d::zeros(0),
        }
    }
}

impl<S: Scalar> InterleavedEdgePolylines3d<S> {
    /// Creates an empty 3D polyline buffer with zero edges and zero points.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Constructs a 3D polyline buffer from `edge_offsets` and interleaved `[x0, y0, z0, ...]`
    /// control-point coordinates, validating offset invariants, multiple-of-3 length, and
    /// finiteness.
    pub fn from_parts(edge_offsets: &[u32], coords: &[S]) -> FfiResult<Self> {
        let points = InterleavedLayout3d::from_interleaved(coords)?;
        validate_edge_offsets(edge_offsets, points.len())?;
        Ok(Self {
            edge_offsets: edge_offsets.to_vec(),
            points,
        })
    }

    /// Creates a 3D interleaved polyline buffer from a core [`EdgePolylines<Layout3d<S>>`].
    pub fn from_core(polylines: &EdgePolylines<Layout3d<S>>) -> Self {
        Self {
            edge_offsets: polylines.edge_offsets().to_vec(),
            points: InterleavedLayout3d::from_layout3d(polylines.points()),
        }
    }

    /// Converts this interleaved buffer into a core [`EdgePolylines<Layout3d<S>>`].
    pub fn to_core(&self) -> FfiResult<EdgePolylines<Layout3d<S>>> {
        EdgePolylines::from_parts(&self.edge_offsets, self.points.to_layout3d()).map_err(Into::into)
    }

    /// Replaces this buffer's contents from a core [`EdgePolylines<Layout3d<S>>`].
    ///
    /// When `edge_offsets` and `points` lengths match the existing allocation (such as intra-cycle
    /// updates in iterative bundling), coordinates and offsets are copied in place without
    /// reallocating.
    pub fn replace_from_core(&mut self, polylines: &EdgePolylines<Layout3d<S>>) {
        let new_offsets = polylines.edge_offsets();
        if self.edge_offsets.len() == new_offsets.len() {
            self.edge_offsets.copy_from_slice(new_offsets);
        } else {
            self.edge_offsets.clear();
            self.edge_offsets.extend_from_slice(new_offsets);
        }

        if self.points.len() == polylines.total_points() {
            self.points
                .copy_from_layout3d(polylines.points())
                .expect("point length checked immediately above");
        } else {
            self.points = InterleavedLayout3d::from_layout3d(polylines.points());
        }
    }

    /// Returns the number of polylines (edges).
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edge_offsets.len().saturating_sub(1)
    }

    /// Returns the total number of control points across all edges.
    #[inline]
    pub fn total_points(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the buffer stores zero edges.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_edges() == 0
    }

    /// Returns the `edge_offsets` slice of length `num_edges() + 1`.
    #[inline]
    pub fn edge_offsets(&self) -> &[u32] {
        &self.edge_offsets
    }

    /// Returns a const pointer to the contiguous `edge_offsets` buffer.
    #[inline]
    pub fn offsets_ptr(&self) -> *const u32 {
        self.edge_offsets.as_ptr()
    }

    /// Returns the length of `edge_offsets` (`num_edges() + 1`).
    #[inline]
    pub fn offsets_len(&self) -> usize {
        self.edge_offsets.len()
    }

    /// Returns a shared reference to the underlying [`InterleavedLayout3d<S>`] points buffer.
    #[inline]
    pub fn points(&self) -> &InterleavedLayout3d<S> {
        &self.points
    }

    /// Returns a mutable reference to the underlying [`InterleavedLayout3d<S>`] points buffer.
    #[inline]
    pub fn points_mut(&mut self) -> &mut InterleavedLayout3d<S> {
        &mut self.points
    }

    /// Returns a const pointer to the contiguous `[x0, y0, z0, ...]` control-point buffer.
    #[inline]
    pub fn points_ptr(&self) -> *const S {
        self.points.coords_ptr()
    }

    /// Returns a mutable pointer to the contiguous `[x0, y0, z0, ...]` control-point buffer.
    #[inline]
    pub fn points_mut_ptr(&mut self) -> *mut S {
        self.points.coords_mut_ptr()
    }

    /// Returns the total scalar length of the interleaved control-point buffer (`3 * total_points()`).
    #[inline]
    pub fn points_len(&self) -> usize {
        self.points.coords_len()
    }

    /// Returns an immutable slice over the interleaved `[x0, y0, z0, ...]` control-point buffer.
    #[inline]
    pub fn points_as_slice(&self) -> &[S] {
        self.points.as_slice()
    }

    /// Returns a mutable slice over the interleaved `[x0, y0, z0, ...]` control-point buffer.
    #[inline]
    pub fn points_as_mut_slice(&mut self) -> &mut [S] {
        self.points.as_mut_slice()
    }

    /// Returns `true` if every control-point coordinate is finite.
    #[inline]
    pub fn is_all_finite(&self) -> bool {
        self.points.is_all_finite()
    }

    /// Validates that every control-point coordinate in the buffer is finite.
    #[inline]
    pub fn validate_finite(&self) -> FfiResult<()> {
        self.points.validate_finite()
    }
}

#[cfg(test)]
mod tests;
