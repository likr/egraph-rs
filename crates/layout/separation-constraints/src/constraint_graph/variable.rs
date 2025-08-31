/// Represents a variable (typically a node's coordinate in one dimension) within the QPSC problem.
/// Its position is determined by the block it belongs to and its offset within that block.
/// See Section 3.2 and Figure 9 in the IPSEP-COLA paper [1].
pub struct Variable<S> {
    /// Index of the [`Block`] this variable belongs to.
    pub block: usize,
    /// Displacement from the [`Block::position`] of its containing block.
    pub offset: S,
}
