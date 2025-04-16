use std::collections::HashSet;

/// Represents a set of variables that move rigidly together, connected by active constraints.
/// The block structure evolves as constraints are satisfied during the projection process.
/// See Section 3.2 and Figure 9 in the IPSEP-COLA paper [1].
pub struct Block {
    /// Indices (`usize`) of variables belonging to this block.
    pub variables: HashSet<usize>,
    /// The reference position of the block (`posn` in the paper).
    pub position: f32,
    /// Indices (`usize`) of constraints active within this block, forming a spanning tree.
    /// An active constraint `c = (u, v, gap)` implies `position(u) + gap = position(v)`.
    pub active: HashSet<usize>,
}
