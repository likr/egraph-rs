//! Fixed-node (pinning) constraint mask shared across solvers and layout transforms.

use crate::error::{CoreError, CoreResult};

/// Boolean mask identifying pinned (fixed-position) vertices across `0..len()`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PinMask {
    pinned: Vec<bool>,
}

impl PinMask {
    /// Creates an all-unpinned mask for `num_nodes` vertices.
    pub fn unpinned(num_nodes: usize) -> Self {
        Self {
            pinned: vec![false; num_nodes],
        }
    }

    /// Creates a pin mask by copying a boolean slice.
    pub fn from_bools(pinned: &[bool]) -> Self {
        Self {
            pinned: pinned.to_vec(),
        }
    }

    /// Creates a pin mask for `num_nodes` vertices with `pinned_indices` marked pinned.
    pub fn from_indices(num_nodes: usize, pinned_indices: &[u32]) -> CoreResult<Self> {
        let mut pinned = vec![false; num_nodes];
        for &idx in pinned_indices {
            let i = idx as usize;
            if i >= num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    i,
                    num_nodes,
                    "PinMask::from_indices",
                ));
            }
            pinned[i] = true;
        }
        Ok(Self { pinned })
    }

    /// Returns the total number of vertices tracked by this mask.
    #[inline]
    pub fn len(&self) -> usize {
        self.pinned.len()
    }

    /// Returns `true` if `self.len() == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pinned.is_empty()
    }

    /// Returns the count of pinned vertices.
    #[inline]
    pub fn num_pinned(&self) -> usize {
        self.pinned.iter().filter(|&&p| p).count()
    }

    /// Returns `true` if `node` is pinned.
    ///
    /// # Panics
    ///
    /// Panics if `node >= self.len()`.
    #[inline]
    pub fn is_pinned(&self, node: usize) -> bool {
        self.pinned[node]
    }

    /// Fallible variant of [`PinMask::is_pinned`] returning [`CoreError::IndexOutOfBounds`].
    #[inline]
    pub fn try_is_pinned(&self, node: usize) -> CoreResult<bool> {
        if node >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.len(),
                "PinMask::try_is_pinned",
            ));
        }
        Ok(self.pinned[node])
    }

    /// Sets the pinned status of `node`.
    #[inline]
    pub fn set_pinned(&mut self, node: usize, pinned: bool) -> CoreResult<()> {
        if node >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.len(),
                "PinMask::set_pinned",
            ));
        }
        self.pinned[node] = pinned;
        Ok(())
    }

    /// Iterates over pinned vertex indices in ascending order.
    #[inline]
    pub fn iter_pinned(&self) -> impl Iterator<Item = usize> + '_ {
        self.pinned
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| p.then_some(i))
    }

    /// Returns the underlying boolean mask slice.
    #[inline]
    pub fn as_slice(&self) -> &[bool] {
        &self.pinned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pin_mask_constructors_and_queries() {
        let mut mask = PinMask::unpinned(4);
        assert_eq!(mask.len(), 4);
        assert!(!mask.is_empty());
        assert_eq!(mask.num_pinned(), 0);
        assert_eq!(mask.as_slice(), &[false, false, false, false]);

        mask.set_pinned(1, true).unwrap();
        mask.set_pinned(3, true).unwrap();
        assert!(mask.is_pinned(1));
        assert!(!mask.is_pinned(2));
        assert!(mask.try_is_pinned(3).unwrap());
        assert_eq!(mask.num_pinned(), 2);
        assert_eq!(mask.iter_pinned().collect::<Vec<_>>(), vec![1, 3]);

        let from_idx = PinMask::from_indices(4, &[3, 1, 3]).unwrap();
        assert_eq!(from_idx, mask);

        let from_bools = PinMask::from_bools(&[false, true, false, true]);
        assert_eq!(from_bools, mask);

        let empty = PinMask::unpinned(0);
        assert!(empty.is_empty());
        assert_eq!(empty.num_pinned(), 0);

        assert!(matches!(
            PinMask::from_indices(3, &[3]),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            mask.try_is_pinned(4),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            mask.set_pinned(4, true),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
    }
}
