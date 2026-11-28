use super::validate_bijection;
use crate::error::{CoreError, CoreResult};
use crate::rng::DeterministicRng;
use crate::scalar::Scalar;

/// Structure-of-Arrays (SoA) 3D node layout storing contiguous `x`, `y`, and `z` coordinate slices.
///
/// Storing `x`, `y`, and `z` in separate contiguous vectors aligns directly with 3D spatial
/// octrees and WebGL/GPU-friendly vertex buffers while enabling simultaneous multi-axis mutable
/// borrows.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Layout3d<S: Scalar = f64> {
    x: Vec<S>,
    y: Vec<S>,
    z: Vec<S>,
}

impl<S: Scalar> Layout3d<S> {
    /// Creates a zero-initialized 3D layout for `n` vertices.
    pub fn zeros(n: usize) -> Self {
        Self {
            x: vec![S::ZERO; n],
            y: vec![S::ZERO; n],
            z: vec![S::ZERO; n],
        }
    }

    /// Creates a deterministically seeded uniform random 3D layout in
    /// `[-0.5 * scale, 0.5 * scale)^3`.
    pub fn init(num_nodes: usize, seed: u64, scale: f64) -> CoreResult<Self> {
        if !scale.is_finite() || !S::from_f64(scale).is_finite() {
            return Err(CoreError::non_finite("Layout3d::init::scale"));
        }
        if scale <= 0.0 {
            return Err(CoreError::invalid_input("scale must be strictly positive"));
        }
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut x = Vec::with_capacity(num_nodes);
        let mut y = Vec::with_capacity(num_nodes);
        let mut z = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            x.push(S::from_f64((rng.next_f64() - 0.5) * scale));
            y.push(S::from_f64((rng.next_f64() - 0.5) * scale));
            z.push(S::from_f64((rng.next_f64() - 0.5) * scale));
        }
        Ok(Self { x, y, z })
    }

    /// Creates a 3D layout from owned `x`, `y`, and `z` vectors, checking equal lengths.
    pub fn from_xyz(x: Vec<S>, y: Vec<S>, z: Vec<S>) -> CoreResult<Self> {
        if x.len() != y.len() {
            return Err(CoreError::dimension_mismatch(
                x.len(),
                y.len(),
                "Layout3d::from_xyz::y",
            ));
        }
        if x.len() != z.len() {
            return Err(CoreError::dimension_mismatch(
                x.len(),
                z.len(),
                "Layout3d::from_xyz::z",
            ));
        }
        Ok(Self { x, y, z })
    }

    /// Creates a 3D layout by copying separate `x`, `y`, and `z` slices.
    pub fn from_xyz_slices(x: &[S], y: &[S], z: &[S]) -> CoreResult<Self> {
        if x.len() != y.len() {
            return Err(CoreError::dimension_mismatch(
                x.len(),
                y.len(),
                "Layout3d::from_xyz_slices::y",
            ));
        }
        if x.len() != z.len() {
            return Err(CoreError::dimension_mismatch(
                x.len(),
                z.len(),
                "Layout3d::from_xyz_slices::z",
            ));
        }
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            z: z.to_vec(),
        })
    }

    /// Creates a 3D layout from interleaved `[x0, y0, z0, x1, y1, z1, ...]` coordinates.
    pub fn from_interleaved(coords: &[S]) -> CoreResult<Self> {
        if !coords.len().is_multiple_of(3) {
            return Err(CoreError::dimension_mismatch(
                coords.len() + (3 - coords.len() % 3),
                coords.len(),
                "Layout3d::from_interleaved",
            ));
        }
        let n = coords.len() / 3;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for &[px, py, pz] in coords.as_chunks::<3>().0 {
            x.push(px);
            y.push(py);
            z.push(pz);
        }
        Ok(Self { x, y, z })
    }

    /// Writes SoA coordinates into an interleaved `[x0, y0, z0, x1, y1, z1, ...]` output slice.
    pub fn write_interleaved(&self, out: &mut [S]) -> CoreResult<()> {
        let expected = self.len() * 3;
        if out.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                out.len(),
                "Layout3d::write_interleaved",
            ));
        }
        for (i, [ox, oy, oz]) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            *ox = self.x[i];
            *oy = self.y[i];
            *oz = self.z[i];
        }
        Ok(())
    }

    /// Returns a newly allocated interleaved `[x0, y0, z0, x1, y1, z1, ...]` vector.
    pub fn to_interleaved(&self) -> Vec<S> {
        let mut out = Vec::with_capacity(self.len() * 3);
        for i in 0..self.len() {
            out.push(self.x[i]);
            out.push(self.y[i]);
            out.push(self.z[i]);
        }
        out
    }

    /// Returns the number of vertices in the layout.
    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Returns the number of vertices in the layout (alias matching `LayoutNd::num_nodes`).
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.x.len()
    }

    /// Returns `true` if the layout has zero vertices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Returns the `[x, y, z]` coordinates for `node`.
    #[inline]
    pub fn get(&self, node: usize) -> CoreResult<[S; 3]> {
        if node >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.len(),
                "Layout3d::get",
            ));
        }
        Ok([self.x[node], self.y[node], self.z[node]])
    }

    /// Sets the `[x, y, z]` coordinates for `node` after validating bounds and finiteness.
    #[inline]
    pub fn set(&mut self, node: usize, x: S, y: S, z: S) -> CoreResult<()> {
        if node >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.len(),
                "Layout3d::set",
            ));
        }
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(CoreError::non_finite("Layout3d::set"));
        }
        self.x[node] = x;
        self.y[node] = y;
        self.z[node] = z;
        Ok(())
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

    /// Returns an immutable slice of `z` coordinates.
    #[inline]
    pub fn z(&self) -> &[S] {
        &self.z
    }

    /// Returns a mutable slice of `z` coordinates.
    #[inline]
    pub fn z_mut(&mut self) -> &mut [S] {
        &mut self.z
    }

    /// Returns disjoint mutable slices `(&mut [S], &mut [S], &mut [S])` for simultaneous
    /// `x`, `y`, and `z` updates.
    #[inline]
    pub fn xyz_mut(&mut self) -> (&mut [S], &mut [S], &mut [S]) {
        (&mut self.x, &mut self.y, &mut self.z)
    }

    /// Returns `true` if every coordinate in `x`, `y`, and `z` is finite.
    #[inline]
    pub fn is_all_finite(&self) -> bool {
        self.x.iter().all(|v| v.is_finite())
            && self.y.iter().all(|v| v.is_finite())
            && self.z.iter().all(|v| v.is_finite())
    }

    /// Validates that every coordinate in the layout is finite.
    #[inline]
    pub fn validate_finite(&self) -> CoreResult<()> {
        if self.is_all_finite() {
            Ok(())
        } else {
            Err(CoreError::non_finite("Layout3d"))
        }
    }

    /// Centers coordinates at zero mean per axis and scales isotropically so the global
    /// coordinate standard deviation $\sqrt{\frac{1}{3N}\sum_i ((x_i - \bar{x})^2 + (y_i - \bar{y})^2 + (z_i - \bar{z})^2)}$
    /// equals `1.0`.
    pub fn center_and_normalize_std(&mut self) -> CoreResult<()> {
        self.center_and_normalize_to_std(1.0)
    }

    /// Centers coordinates at zero mean per axis and scales isotropically to `target_std` (`> 0`).
    pub fn center_and_normalize_to_std(&mut self, target_std: f64) -> CoreResult<()> {
        if !target_std.is_finite() || !S::from_f64(target_std).is_finite() {
            return Err(CoreError::non_finite(
                "Layout3d::center_and_normalize_to_std::target_std",
            ));
        }
        if target_std <= 0.0 {
            return Err(CoreError::invalid_input(
                "target_std must be finite and strictly positive",
            ));
        }
        if !self.is_all_finite() {
            return Err(CoreError::non_finite("Layout3d::center_and_normalize_std"));
        }
        let n = self.len();
        if n == 0 {
            return Ok(());
        }
        if n == 1 {
            self.x[0] = S::ZERO;
            self.y[0] = S::ZERO;
            self.z[0] = S::ZERO;
            return Ok(());
        }

        let n_f64 = n as f64;
        let mean_x = self.x.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
        let mean_y = self.y.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;
        let mean_z = self.z.iter().map(|v| v.to_f64()).sum::<f64>() / n_f64;

        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            let dx = self.x[i].to_f64() - mean_x;
            let dy = self.y[i].to_f64() - mean_y;
            let dz = self.z[i].to_f64() - mean_z;
            sum_sq += dx * dx + dy * dy + dz * dz;
        }

        let std_dev = (sum_sq / (3.0 * n_f64)).sqrt();
        if std_dev <= 1e-12 {
            self.x.fill(S::ZERO);
            self.y.fill(S::ZERO);
            self.z.fill(S::ZERO);
            return Ok(());
        }

        let scale = target_std / std_dev;
        if !std_dev.is_finite() || !scale.is_finite() {
            return Err(CoreError::non_finite(
                "Layout3d::center_and_normalize_to_std",
            ));
        }
        for i in 0..n {
            let sx = S::from_f64((self.x[i].to_f64() - mean_x) * scale);
            let sy = S::from_f64((self.y[i].to_f64() - mean_y) * scale);
            let sz = S::from_f64((self.z[i].to_f64() - mean_z) * scale);
            if !sx.is_finite() || !sy.is_finite() || !sz.is_finite() {
                return Err(CoreError::non_finite(
                    "Layout3d::center_and_normalize_to_std",
                ));
            }
        }
        for i in 0..n {
            self.x[i] = S::from_f64((self.x[i].to_f64() - mean_x) * scale);
            self.y[i] = S::from_f64((self.y[i].to_f64() - mean_y) * scale);
            self.z[i] = S::from_f64((self.z[i].to_f64() - mean_z) * scale);
        }
        Ok(())
    }

    /// Returns a vertex-permuted layout where `out[old_to_new[old_i]] = self[old_i]`.
    pub fn permute(&self, old_to_new: &[u32]) -> CoreResult<Self> {
        let n = self.len();
        validate_bijection(n, old_to_new, "Layout3d::permute")?;
        let mut x = vec![S::ZERO; n];
        let mut y = vec![S::ZERO; n];
        let mut z = vec![S::ZERO; n];
        for (old_i, &new_i_u32) in old_to_new.iter().enumerate() {
            let new_i = new_i_u32 as usize;
            x[new_i] = self.x[old_i];
            y[new_i] = self.y[old_i];
            z[new_i] = self.z[old_i];
        }
        Ok(Self { x, y, z })
    }

    /// Restores original vertex ordering from a permuted layout: `out[old_i] = self[old_to_new[old_i]]`.
    pub fn unpermute(&self, old_to_new: &[u32]) -> CoreResult<Self> {
        let n = self.len();
        validate_bijection(n, old_to_new, "Layout3d::unpermute")?;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for &new_i_u32 in old_to_new {
            let new_i = new_i_u32 as usize;
            x.push(self.x[new_i]);
            y.push(self.y[new_i]);
            z.push(self.z[new_i]);
        }
        Ok(Self { x, y, z })
    }
}
