use crate::error::{CoreError, CoreResult};
use crate::rng::DeterministicRng;
use crate::scalar::Scalar;

/// Flat row-major `d`-dimensional coordinate layout (`num_nodes * dims`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Creates a deterministically seeded uniform random `dims`-dimensional layout in
    /// `[-0.5 * scale, 0.5 * scale)^dims`.
    pub fn init(num_nodes: usize, dims: usize, seed: u64, scale: f64) -> CoreResult<Self> {
        if dims == 0 {
            return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
        }
        if !scale.is_finite() || !S::from_f64(scale).is_finite() {
            return Err(CoreError::non_finite("LayoutNd::init::scale"));
        }
        if scale <= 0.0 {
            return Err(CoreError::invalid_input("scale must be strictly positive"));
        }
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("LayoutNd size overflow"))?;
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut coords = Vec::with_capacity(total);
        for _ in 0..total {
            coords.push(S::from_f64((rng.next_f64() - 0.5) * scale));
        }
        Ok(Self {
            coords,
            num_nodes,
            dims,
        })
    }

    /// Creates a `LayoutNd` by copying a flat row-major slice of length `num_nodes * dims`.
    pub fn from_flat(num_nodes: usize, dims: usize, coords: &[S]) -> CoreResult<Self> {
        Self::from_flat_vec(num_nodes, dims, coords.to_vec())
    }

    /// Creates a `LayoutNd` from an owned flat row-major vector of length `num_nodes * dims`.
    pub fn from_flat_vec(num_nodes: usize, dims: usize, coords: Vec<S>) -> CoreResult<Self> {
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

    /// Returns the number of vertices in the layout (alias matching `Layout2d::len`).
    #[inline]
    pub fn len(&self) -> usize {
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

    /// Returns the coordinate value for `(node, dim)`.
    pub fn get(&self, node: usize, dim: usize) -> CoreResult<S> {
        if node >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.num_nodes,
                "LayoutNd::get",
            ));
        }
        if dim >= self.dims {
            return Err(CoreError::index_out_of_bounds(
                dim,
                self.dims,
                "LayoutNd::get",
            ));
        }
        Ok(self.coords[node * self.dims + dim])
    }

    /// Sets the coordinate value for `(node, dim)` after validating bounds and finiteness.
    pub fn set(&mut self, node: usize, dim: usize, val: S) -> CoreResult<()> {
        if node >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.num_nodes,
                "LayoutNd::set",
            ));
        }
        if dim >= self.dims {
            return Err(CoreError::index_out_of_bounds(
                dim,
                self.dims,
                "LayoutNd::set",
            ));
        }
        if !val.is_finite() {
            return Err(CoreError::non_finite("LayoutNd::set"));
        }
        self.coords[node * self.dims + dim] = val;
        Ok(())
    }

    /// Returns `true` if all coordinates are finite.
    #[inline]
    pub fn is_all_finite(&self) -> bool {
        self.coords.iter().all(|v| v.is_finite())
    }

    /// Validates that every coordinate in the layout is finite.
    #[inline]
    pub fn validate_finite(&self) -> CoreResult<()> {
        if self.is_all_finite() {
            Ok(())
        } else {
            Err(CoreError::non_finite("LayoutNd"))
        }
    }

    /// Centers coordinates at zero mean per axis and scales isotropically so the global
    /// coordinate standard deviation $\sqrt{\frac{1}{ND}\sum_{i,a} (x_{i,a} - \bar{x}_a)^2}$
    /// equals `1.0`.
    pub fn center_and_normalize_std(&mut self) -> CoreResult<()> {
        self.center_and_normalize_to_std(1.0)
    }

    /// Centers coordinates at zero mean per axis and scales isotropically to `target_std` (`> 0`).
    pub fn center_and_normalize_to_std(&mut self, target_std: f64) -> CoreResult<()> {
        if !target_std.is_finite() || !S::from_f64(target_std).is_finite() {
            return Err(CoreError::non_finite(
                "LayoutNd::center_and_normalize_to_std::target_std",
            ));
        }
        if target_std <= 0.0 {
            return Err(CoreError::invalid_input(
                "target_std must be finite and strictly positive",
            ));
        }
        if !self.is_all_finite() {
            return Err(CoreError::non_finite("LayoutNd::center_and_normalize_std"));
        }
        if self.num_nodes == 0 {
            return Ok(());
        }
        if self.num_nodes == 1 {
            self.coords.fill(S::ZERO);
            return Ok(());
        }

        let n = self.num_nodes;
        let d = self.dims;
        let n_f64 = n as f64;
        let mut means = vec![0.0_f64; d];
        for i in 0..n {
            let base = i * d;
            for axis in 0..d {
                means[axis] += self.coords[base + axis].to_f64();
            }
        }
        for mean in &mut means {
            *mean /= n_f64;
        }

        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            let base = i * d;
            let mut node_sq = 0.0_f64;
            for axis in 0..d {
                let centered = self.coords[base + axis].to_f64() - means[axis];
                node_sq += centered * centered;
            }
            sum_sq += node_sq;
        }

        let std_dev = (sum_sq / ((n * d) as f64)).sqrt();
        if std_dev <= 1e-12 {
            self.coords.fill(S::ZERO);
            return Ok(());
        }

        let scale = target_std / std_dev;
        if !std_dev.is_finite() || !scale.is_finite() {
            return Err(CoreError::non_finite(
                "LayoutNd::center_and_normalize_to_std",
            ));
        }
        for i in 0..n {
            let base = i * d;
            for axis in 0..d {
                let scaled = S::from_f64((self.coords[base + axis].to_f64() - means[axis]) * scale);
                if !scaled.is_finite() {
                    return Err(CoreError::non_finite(
                        "LayoutNd::center_and_normalize_to_std",
                    ));
                }
            }
        }
        for i in 0..n {
            let base = i * d;
            for axis in 0..d {
                self.coords[base + axis] =
                    S::from_f64((self.coords[base + axis].to_f64() - means[axis]) * scale);
            }
        }
        Ok(())
    }
}
