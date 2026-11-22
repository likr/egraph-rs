//! Epoch-based `VisitStamp` for $O(1)$ amortized clear across repeated graph traversals.

/// Epoch-stamped visitation tracker supporting $O(1)$ clear between traversals.
///
/// Instead of zero-filling a `Vec<bool>` of size $N$ for every BFS/DFS source node,
/// `VisitStamp` increments a `u32` epoch counter. Only once every `u32::MAX` clears does
/// it reset the backing array to `0`.
#[derive(Debug, Clone)]
pub struct VisitStamp {
    stamps: Vec<u32>,
    epoch: u32,
}

impl VisitStamp {
    /// Creates a new `VisitStamp` tracking `len` vertices, all initially unvisited.
    pub fn new(len: usize) -> Self {
        Self {
            stamps: vec![0; len],
            epoch: 1,
        }
    }

    /// Returns the number of tracked slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.stamps.len()
    }

    /// Returns `true` if no slots are tracked.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.stamps.is_empty()
    }

    /// Returns the current active epoch identifier.
    #[inline]
    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Clears all visited marks in $O(1)$ time (amortized across $2^{32}-1$ clears).
    #[inline]
    pub fn clear(&mut self) {
        if self.epoch == u32::MAX {
            self.stamps.fill(0);
            self.epoch = 1;
        } else {
            self.epoch += 1;
        }
    }

    /// Resizes the tracker to `new_len` and clears all visited marks.
    pub fn resize_and_clear(&mut self, new_len: usize) {
        self.stamps.resize(new_len, 0);
        self.clear();
    }

    /// Returns `true` if `index` has been marked visited in the current epoch.
    #[inline]
    pub fn is_visited(&self, index: usize) -> bool {
        self.stamps[index] == self.epoch
    }

    /// Marks `index` as visited in the current epoch and returns `true` if it was newly visited.
    #[inline]
    pub fn visit(&mut self, index: usize) -> bool {
        let slot = &mut self.stamps[index];
        if *slot == self.epoch {
            false
        } else {
            *slot = self.epoch;
            true
        }
    }

    /// Marks `index` as visited in the current epoch.
    #[inline]
    pub fn mark(&mut self, index: usize) {
        self.stamps[index] = self.epoch;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visit_stamp_o1_clear_and_wrap() {
        let mut stamp = VisitStamp::new(5);
        assert_eq!(stamp.len(), 5);
        assert!(!stamp.is_empty());
        assert!(!stamp.is_visited(2));
        assert!(stamp.visit(2));
        assert!(!stamp.visit(2));
        assert!(stamp.is_visited(2));

        stamp.clear();
        assert!(!stamp.is_visited(2));
        assert!(stamp.visit(2));

        // Simulate epoch wrap-around at u32::MAX without needing 4B iterations.
        stamp.epoch = u32::MAX;
        stamp.mark(4);
        assert!(stamp.is_visited(4));
        stamp.clear();
        assert_eq!(stamp.epoch(), 1);
        assert!(!stamp.is_visited(4));
        assert!(!stamp.is_visited(2));
    }

    #[test]
    fn test_visit_stamp_empty_and_resize() {
        let mut empty = VisitStamp::new(0);
        assert!(empty.is_empty());
        empty.clear();
        empty.resize_and_clear(3);
        assert_eq!(empty.len(), 3);
        assert!(empty.visit(0));
        assert!(empty.visit(2));
    }
}
