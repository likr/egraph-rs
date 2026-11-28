use egraph_core::{CoreError, CoreResult};

pub(super) const MAX_SFC_ORDER: u32 = 15;

/// Space-filling curve mapping family for [`SfcLayout`](super::SfcLayout).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SfcCurveKind {
    /// Locality-preserving 2D Hilbert curve.
    #[default]
    Hilbert,
    /// Z-order (Morton) bit-interleaved 2D curve.
    Morton,
}

/// Hyperparameters for [`SfcLayout`](super::SfcLayout).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SfcConfig {
    /// Space-filling curve family (`default: SfcCurveKind::Hilbert`).
    pub curve: SfcCurveKind,
    /// Target bounding square side length in `[-scale / 2, scale / 2]^2` (`default: 100.0`).
    pub scale: f64,
    /// Optional explicit curve resolution order $k \in [1, 15]$ yielding a $2^k \times 2^k$ grid
    /// (`default: None`, automatically choosing the smallest $k \ge 1$ with $2^{2k} \ge N$).
    pub resolution_order: Option<u32>,
    /// Deterministic PRNG seed passed to Louvain community coarsening (`default: 42`).
    pub seed: u64,
}

impl Default for SfcConfig {
    fn default() -> Self {
        Self {
            curve: SfcCurveKind::Hilbert,
            scale: 100.0,
            resolution_order: None,
            seed: 42,
        }
    }
}

impl SfcConfig {
    /// Creates a new [`SfcConfig`] initialized to [`Self::default`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the space-filling curve family `curve`.
    #[inline]
    pub fn with_curve(mut self, curve: SfcCurveKind) -> Self {
        self.curve = curve;
        self
    }

    /// Sets the target bounding square side length `scale`.
    #[inline]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Sets the optional explicit curve resolution order `resolution_order`.
    #[inline]
    pub fn with_resolution_order(mut self, resolution_order: Option<u32>) -> Self {
        self.resolution_order = resolution_order;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validates all space-filling curve layout parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.scale.is_finite() {
            return Err(CoreError::non_finite("SfcConfig::scale"));
        }
        if self.scale <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "SfcConfig::scale must be > 0.0, got {}",
                self.scale
            )));
        }
        if let Some(ord) = self.resolution_order
            && !(1..=MAX_SFC_ORDER).contains(&ord)
        {
            return Err(CoreError::invalid_input(format!(
                "SfcConfig::resolution_order must be in [1, {MAX_SFC_ORDER}], got {ord}"
            )));
        }
        Ok(())
    }
}
