//! Contiguous Structure-of-Arrays (`Layout2d`) and arbitrary-dimensional (`LayoutNd`) buffers.

use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;

/// Structure-of-Arrays (SoA) 2D node layout storing contiguous `x` and `y` coordinate slices.
///
/// Storing `x` and `y` in separate contiguous vectors improves cache locality and SIMD
/// auto-vectorization inside inner layout loops compared to AoS pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct Layout2d<S: Scalar = f64> {
    x: Vec<S>,
    y: Vec<S>,
}

impl<S: Scalar> Layout2d<S> {
    /// Creates a zero-initialized 2D layout for `n` vertices.
    pub fn zeros(n: usize) -> Self {
        Self {
            x: vec![S::ZERO; n],
            y: vec![S::ZERO; n],
        }
    }

    /// Creates a 2D layout from owned `x` and `y` vectors, checking equal lengths.
    pub fn from_xy(x: Vec<S>, y: Vec<S>) -> CoreResult<Self> {
        if x.len() != y.len() {
            return Err(CoreError::dimension_mismatch(
                x.len(),
                y.len(),
                "Layout2d::from_xy",
            ));
        }
        Ok(Self { x, y })
    }

    /// Creates a 2D layout by copying separate `x` and `y` slices.
    pub fn from_xy_slices(x: &[S], y: &[S]) -> CoreResult<Self> {
        if x.len() != y.len() {
            return Err(CoreError::dimension_mismatch(
                x.len(),
                y.len(),
                "Layout2d::from_xy_slices",
            ));
        }
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
        })
    }

    /// Creates a 2D layout from interleaved `[x0, y0, x1, y1, ...]` coordinates.
    pub fn from_interleaved(coords: &[S]) -> CoreResult<Self> {
        if !coords.len().is_multiple_of(2) {
            return Err(CoreError::dimension_mismatch(
                coords.len() + 1,
                coords.len(),
                "Layout2d::from_interleaved",
            ));
        }
        let n = coords.len() / 2;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for &[px, py] in coords.as_chunks::<2>().0 {
            x.push(px);
            y.push(py);
        }
        Ok(Self { x, y })
    }

    /// Writes SoA coordinates into an interleaved `[x0, y0, x1, y1, ...]` output slice.
    pub fn write_interleaved(&self, out: &mut [S]) -> CoreResult<()> {
        let expected = self.len() * 2;
        if out.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                out.len(),
                "Layout2d::write_interleaved",
            ));
        }
        for (i, [ox, oy]) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            *ox = self.x[i];
            *oy = self.y[i];
        }
        Ok(())
    }

    /// Returns a newly allocated interleaved `[x0, y0, x1, y1, ...]` vector.
    pub fn to_interleaved(&self) -> Vec<S> {
        let mut out = Vec::with_capacity(self.len() * 2);
        for i in 0..self.len() {
            out.push(self.x[i]);
            out.push(self.y[i]);
        }
        out
    }

    /// Returns the number of vertices in the layout.
    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Returns `true` if the layout has zero vertices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Returns an immutable slice of `x` coordinates.
    #[inline]
    pub fn x(&self) -> &[S] {
        &self.x
    }

    /// Returns a mutable slice of `x` coordinates.
    #[inline]
    pub fn x_mut(&mut self) -> &mut [S] {
        &mut self.x
    }

    /// Returns an immutable slice of `y` coordinates.
    #[inline]
    pub fn y(&self) -> &[S] {
        &self.y
    }

    /// Returns a mutable slice of `y` coordinates.
    #[inline]
    pub fn y_mut(&mut self) -> &mut [S] {
        &mut self.y
    }

    /// Returns disjoint mutable slices `(&mut [S], &mut [S])` for simultaneous `x` and `y` updates.
    #[inline]
    pub fn xy_mut(&mut self) -> (&mut [S], &mut [S]) {
        (&mut self.x, &mut self.y)
    }

    /// Returns `true` if every coordinate in `x` and `y` is finite (not `NaN` or `±Inf`).
    #[inline]
    pub fn is_all_finite(&self) -> bool {
        self.x.iter().all(|v| v.is_finite()) && self.y.iter().all(|v| v.is_finite())
    }

    /// Validates that every coordinate in the layout is finite.
    #[inline]
    pub fn validate_finite(&self) -> CoreResult<()> {
        if self.is_all_finite() {
            Ok(())
        } else {
            Err(CoreError::non_finite("Layout2d"))
        }
    }

    /// Centers coordinates at zero mean and scales isotropically so the global coordinate
    /// standard deviation $\sqrt{\frac{1}{2N}\sum_i ((x_i - \bar{x})^2 + (y_i - \bar{y})^2)}$
    /// equals `1.0`.
    ///
    /// Accumulates sums in `f64` to preserve precision when `S = f32`. For `n <= 1` or
    /// collapsed layouts (`std <= 1e-12`), coordinates are centered at `(0, 0)` without
    /// amplifying numerical noise.
    pub fn center_and_normalize_std(&mut self) -> CoreResult<()> {
        self.center_and_normalize_to_std(S::ONE)
    }

    /// Centers coordinates at zero mean and scales isotropically to `target_std` (`> 0`).
    pub fn center_and_normalize_to_std(&mut self, target_std: S) -> CoreResult<()> {
        let target = target_std.to_f64();
        if !target.is_finite() || target <= 0.0 {
            return Err(CoreError::invalid_input(
                "target_std must be finite and strictly positive",
            ));
        }
        if !self.is_all_finite() {
            return Err(CoreError::non_finite("Layout2d::center_and_normalize_std"));
        }
        let n = self.len();
        if n == 0 {
            return Ok(());
        }
        if n == 1 {
            self.x[0] = S::ZERO;
            self.y[0] = S::ZERO;
            return Ok(());
        }

        let n_f64 = n as f64;
        let mean_x = self.x.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
        let mean_y = self.y.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;

        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            let dx = self.x[i].to_f64() - mean_x;
            let dy = self.y[i].to_f64() - mean_y;
            self.x[i] = S::from_f64(dx);
            self.y[i] = S::from_f64(dy);
            sum_sq += dx * dx + dy * dy;
        }

        let std_dev = (sum_sq / (2.0 * n_f64)).sqrt();
        if std_dev <= 1e-12 {
            self.x.fill(S::ZERO);
            self.y.fill(S::ZERO);
            return Ok(());
        }

        let scale = target / std_dev;
        for i in 0..n {
            self.x[i] = S::from_f64(self.x[i].to_f64() * scale);
            self.y[i] = S::from_f64(self.y[i].to_f64() * scale);
        }
        Ok(())
    }
}

/// Flat row-major `d`-dimensional coordinate layout (`num_nodes * dims`).
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutNd<S: Scalar = f64> {
    coords: Vec<S>,
    num_nodes: usize,
    dims: usize,
}

impl<S: Scalar> LayoutNd<S> {
    /// Creates a zero-initialized `dims`-dimensional layout for `num_nodes` vertices.
    pub fn zeros(num_nodes: usize, dims: usize) -> CoreResult<Self> {
        if dims == 0 {
            return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
        }
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("LayoutNd size overflow"))?;
        Ok(Self {
            coords: vec![S::ZERO; total],
            num_nodes,
            dims,
        })
    }

    /// Creates a `LayoutNd` from a flat row-major coordinate vector of length `num_nodes * dims`.
    pub fn from_flat(num_nodes: usize, dims: usize, coords: Vec<S>) -> CoreResult<Self> {
        if dims == 0 {
            return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
        }
        let expected = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("LayoutNd size overflow"))?;
        if coords.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                coords.len(),
                "LayoutNd::from_flat",
            ));
        }
        Ok(Self {
            coords,
            num_nodes,
            dims,
        })
    }

    /// Returns the dimensionality of the embedding space.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Returns the number of vertices in the layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns `true` if the layout contains zero vertices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_nodes == 0
    }

    /// Returns an immutable slice over the flat row-major coordinate buffer.
    #[inline]
    pub fn coords(&self) -> &[S] {
        &self.coords
    }

    /// Returns a mutable slice over the flat row-major coordinate buffer.
    #[inline]
    pub fn coords_mut(&mut self) -> &mut [S] {
        &mut self.coords
    }

    /// Returns the `dims`-element coordinate slice for `node`.
    pub fn node(&self, node: usize) -> CoreResult<&[S]> {
        if node >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.num_nodes,
                "LayoutNd::node",
            ));
        }
        let start = node * self.dims;
        Ok(&self.coords[start..start + self.dims])
    }

    /// Returns a mutable `dims`-element coordinate slice for `node`.
    pub fn node_mut(&mut self, node: usize) -> CoreResult<&mut [S]> {
        if node >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.num_nodes,
                "LayoutNd::node_mut",
            ));
        }
        let start = node * self.dims;
        Ok(&mut self.coords[start..start + self.dims])
    }

    /// Returns `true` if all coordinates are finite.
    #[inline]
    pub fn is_all_finite(&self) -> bool {
        self.coords.iter().all(|v| v.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout2d_interleaved_roundtrip_and_mutators() {
        let mut layout = Layout2d::from_interleaved(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(layout.len(), 2);
        assert!(!layout.is_empty());
        assert_eq!(layout.x(), &[1.0, 3.0]);
        assert_eq!(layout.y(), &[2.0, 4.0]);

        layout.x_mut()[0] = 5.0;
        layout.y_mut()[1] = 6.0;
        let mut out = vec![0.0_f64; 4];
        layout.write_interleaved(&mut out).unwrap();
        assert_eq!(out, vec![5.0, 2.0, 3.0, 6.0]);

        assert!(matches!(
            Layout2d::<f64>::from_interleaved(&[1.0, 2.0, 3.0]),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            layout.write_interleaved(&mut [0.0; 2]),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            Layout2d::from_xy(vec![1.0], vec![1.0, 2.0]),
            Err(CoreError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_layout2d_center_and_normalize_std() {
        let mut layout = Layout2d::from_xy(
            vec![10.0_f64, 12.0, 14.0, 10.0],
            vec![-5.0_f64, -3.0, -5.0, -7.0],
        )
        .unwrap();
        layout.center_and_normalize_std().unwrap();
        let n = layout.len() as f64;
        let mean_x = layout.x().iter().sum::<f64>() / n;
        let mean_y = layout.y().iter().sum::<f64>() / n;
        assert!(mean_x.abs() < 1e-12);
        assert!(mean_y.abs() < 1e-12);
        let var: f64 = layout
            .x()
            .iter()
            .zip(layout.y().iter())
            .map(|(&x, &y)| x * x + y * y)
            .sum::<f64>()
            / (2.0 * n);
        assert!((var.sqrt() - 1.0).abs() < 1e-12);

        // Collapsed / 1-node / empty / non-finite edge cases.
        let mut collapsed = Layout2d::from_xy(vec![3.0, 3.0], vec![7.0, 7.0]).unwrap();
        collapsed.center_and_normalize_std().unwrap();
        assert_eq!(collapsed.x(), &[0.0, 0.0]);
        assert_eq!(collapsed.y(), &[0.0, 0.0]);

        let mut single = Layout2d::from_xy(vec![9.0], vec![-4.0]).unwrap();
        single.center_and_normalize_std().unwrap();
        assert_eq!(single.x(), &[0.0]);
        assert_eq!(single.y(), &[0.0]);

        let mut empty = Layout2d::<f32>::zeros(0);
        assert!(empty.is_empty());
        empty.center_and_normalize_std().unwrap();

        let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap();
        assert!(!bad.is_all_finite());
        assert!(matches!(
            bad.center_and_normalize_std(),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_layout_nd() {
        let mut nd = LayoutNd::<f32>::zeros(2, 3).unwrap();
        assert_eq!(nd.dims(), 3);
        assert_eq!(nd.num_nodes(), 2);
        assert!(nd.is_all_finite());
        nd.node_mut(1).unwrap().copy_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(nd.node(1).unwrap(), &[1.0, 2.0, 3.0]);
        assert!(matches!(
            LayoutNd::<f64>::zeros(2, 0),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            nd.node(2),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
    }
}
