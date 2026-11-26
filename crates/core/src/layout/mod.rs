//! Contiguous Structure-of-Arrays (`Layout2d`) and arbitrary-dimensional (`LayoutNd`) buffers.

mod nd;

#[cfg(test)]
mod tests;

pub use nd::LayoutNd;

use crate::error::{CoreError, CoreResult};
use crate::rng::DeterministicRng;
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

    /// Creates a deterministically seeded uniform random 2D layout in `[-0.5 * scale, 0.5 * scale)^2`.
    pub fn init(num_nodes: usize, seed: u64, scale: f64) -> CoreResult<Self> {
        if !scale.is_finite() || !S::from_f64(scale).is_finite() {
            return Err(CoreError::non_finite("Layout2d::init::scale"));
        }
        if scale <= 0.0 {
            return Err(CoreError::invalid_input("scale must be strictly positive"));
        }
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut x = Vec::with_capacity(num_nodes);
        let mut y = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            x.push(S::from_f64((rng.next_f64() - 0.5) * scale));
            y.push(S::from_f64((rng.next_f64() - 0.5) * scale));
        }
        Ok(Self { x, y })
    }

    /// Returns a cloned copy of the layout.
    #[inline]
    pub fn copy(&self) -> Self {
        self.clone()
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
