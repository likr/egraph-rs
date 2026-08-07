//! Kernel traits.

/// A kernel matrix that computes elements from specific pivot nodes.
pub trait PivotedKernel<S> {
    /// Returns the indices of the pivot nodes.
    fn pivots(&self) -> &[usize];

    /// Queries the (pivot_idx, j) element of the kernel matrix, where `pivot_idx` is the index into the `pivots()` array.
    fn get_from_pivot(&self, pivot_idx: usize, j: usize) -> S;
}
