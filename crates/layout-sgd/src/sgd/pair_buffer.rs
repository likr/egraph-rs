use egraph_core::{CoreError, CoreResult, Scalar};

use super::pairs::{INDEX_MASK, PackedSymPair};

/// Potential interaction kind for a buffered or solver-stored SGD pair.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SgdPairKind {
    /// Two-sided (possibly directionally weighted) spring interaction.
    Spring,
    /// One-sided hinge repulsion active only when current distance is below target distance.
    Hinge,
}

/// Decoded read-only SGD pair record decoupled from internal bit-packed storage.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SgdPairRecord<S: Scalar = f64> {
    /// First endpoint vertex index.
    pub u: u32,
    /// Second endpoint vertex index.
    pub v: u32,
    /// Interaction kind (`Spring` or `Hinge`).
    pub kind: SgdPairKind,
    /// Target distance used when updating endpoint `u` relative to `v`.
    pub d_uv: S,
    /// Target distance used when updating endpoint `v` relative to `u`.
    pub d_vu: S,
    /// Weight applied when updating endpoint `u`.
    pub w_uv: S,
    /// Weight applied when updating endpoint `v`.
    pub w_vu: S,
}

impl<S: Scalar> SgdPairRecord<S> {
    #[inline]
    pub(super) const fn new(
        u: u32,
        v: u32,
        kind: SgdPairKind,
        d_uv: S,
        d_vu: S,
        w_uv: S,
        w_vu: S,
    ) -> Self {
        Self {
            u,
            v,
            kind,
            d_uv,
            d_vu,
            w_uv,
            w_vu,
        }
    }
}

/// Reusable zero-copy buffer of symmetric two-sided spring and one-sided hinge pairs for
/// constructing [`super::SgdSolver`] instances without exposing packed internal representations.
#[derive(Debug, Clone, Default)]
pub struct SgdPairBuffer<S: Scalar = f64> {
    pairs: Vec<PackedSymPair<S>>,
    num_springs: usize,
    num_hinges: usize,
}

impl<S: Scalar> SgdPairBuffer<S> {
    /// Creates an empty [`SgdPairBuffer`].
    #[inline]
    pub fn new() -> Self {
        Self {
            pairs: Vec::new(),
            num_springs: 0,
            num_hinges: 0,
        }
    }

    /// Creates an empty [`SgdPairBuffer`] with preallocated capacity for `cap` pairs.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            pairs: Vec::with_capacity(cap),
            num_springs: 0,
            num_hinges: 0,
        }
    }

    fn push_pair_inner(
        &mut self,
        u: u32,
        v: u32,
        is_hinge: bool,
        d: S,
        w: S,
        ctx: &'static str,
    ) -> CoreResult<()> {
        if u > INDEX_MASK || v > INDEX_MASK {
            return Err(CoreError::invalid_input(format!(
                "{ctx}: vertex index exceeds packed u32 limit ({INDEX_MASK})"
            )));
        }
        if u == v {
            return Err(CoreError::invalid_input(format!(
                "{ctx}: self-loop pair ({u}, {u}) is not permitted"
            )));
        }
        if !d.is_finite() || !w.is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
        if d <= S::ZERO {
            return Err(CoreError::invalid_input(format!(
                "{ctx}: target distance must be > 0"
            )));
        }
        if w < S::ZERO {
            return Err(CoreError::invalid_input(format!(
                "{ctx}: weight must be >= 0"
            )));
        }
        self.pairs.push(PackedSymPair::new(u, v, is_hinge, d, w));
        if is_hinge {
            self.num_hinges += 1;
        } else {
            self.num_springs += 1;
        }
        Ok(())
    }

    /// Appends a symmetric two-sided spring pair `(u, v)` with target distance `d > 0` and weight `w >= 0`.
    #[inline]
    pub fn push_spring(&mut self, u: u32, v: u32, d: S, w: S) -> CoreResult<()> {
        self.push_pair_inner(u, v, false, d, w, "SgdPairBuffer::push_spring")
    }

    /// Appends a symmetric one-sided hinge repulsion pair `(u, v)` active when distance `< d` (`d > 0`, `w >= 0`).
    #[inline]
    pub fn push_hinge(&mut self, u: u32, v: u32, d: S, w: S) -> CoreResult<()> {
        self.push_pair_inner(u, v, true, d, w, "SgdPairBuffer::push_hinge")
    }

    /// Clamps stored pair weights symmetrically around their median so `w_max / w_min <= max_ratio`.
    pub fn condition_weights(&mut self, max_ratio: f64) -> CoreResult<()> {
        if !max_ratio.is_finite() {
            return Err(CoreError::non_finite(
                "SgdPairBuffer::condition_weights::max_ratio",
            ));
        }
        if max_ratio < 1.0 {
            return Err(CoreError::invalid_input(format!(
                "SgdPairBuffer::condition_weights: max_ratio must be >= 1.0, got {max_ratio}"
            )));
        }
        if self.pairs.len() <= 1 {
            return Ok(());
        }
        let mut w_min = f64::INFINITY;
        let mut w_max = 0.0_f64;
        for p in &self.pairs {
            let w = p.w.to_f64();
            if w < w_min {
                w_min = w;
            }
            if w > w_max {
                w_max = w;
            }
        }
        if w_max <= max_ratio * w_min || w_min <= 0.0 {
            return Ok(());
        }
        let half_span = max_ratio.sqrt();
        let mid = self.pairs.len() / 2;
        let (_, mid_elem, _) = self
            .pairs
            .select_nth_unstable_by(mid, |a, b| a.w.to_f64().total_cmp(&b.w.to_f64()));
        let mid_w = mid_elem.w.to_f64();
        let w_med = if self.pairs.len() % 2 == 1 {
            mid_w
        } else {
            let mut left_max = f64::NEG_INFINITY;
            for p in &self.pairs[..mid] {
                let w = p.w.to_f64();
                if w > left_max {
                    left_max = w;
                }
            }
            0.5 * (left_max + mid_w)
        };
        let lo = w_med / half_span;
        let hi = w_med * half_span;
        for p in &mut self.pairs {
            let clamped = p.w.to_f64().clamp(lo, hi);
            let clamped_s = S::from_f64(clamped);
            if clamped_s.is_finite() && clamped_s > S::ZERO {
                p.w = clamped_s;
            }
        }
        Ok(())
    }

    /// Returns the total number of buffered spring and hinge pairs.
    #[inline]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Returns `true` if the buffer contains zero pairs.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Returns the number of buffered two-sided spring pairs.
    #[inline]
    pub fn num_springs(&self) -> usize {
        self.num_springs
    }

    /// Returns the number of buffered one-sided hinge pairs.
    #[inline]
    pub fn num_hinges(&self) -> usize {
        self.num_hinges
    }

    /// Returns a read-only iterator over buffered pair records in insertion order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = SgdPairRecord<S>> + '_ {
        self.pairs.iter().map(|&p| p.to_record())
    }

    /// Clears all buffered pairs while retaining allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.pairs.clear();
        self.num_springs = 0;
        self.num_hinges = 0;
    }

    #[inline]
    pub(super) fn into_parts(self) -> (Vec<PackedSymPair<S>>, usize, usize) {
        (self.pairs, self.num_springs, self.num_hinges)
    }
}
