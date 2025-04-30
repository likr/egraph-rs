/// Represents a separation constraint `variables[left] + gap <= variables[right]`.
/// Enforces minimum separation between node pairs in one dimension.
///
/// Separation constraints are useful for enforcing layout properties such as:
/// - Minimum distance between nodes
/// - Hierarchical relationships (e.g., parent above child)
/// - Alignment requirements (e.g., nodes at same level)
/// - Non-overlap between elements with extent
///
/// In graph layouts, variables typically correspond to node coordinates in a specific dimension.
/// For example, in a 2D layout, a constraint might enforce that node A is at least 50 pixels
/// to the left of node B.
///
/// See Section 2 in the IPSEP-COLA paper [2].
#[derive(Clone, Debug)]
pub struct Constraint {
    /// Index (`usize`) of the variable on the left side.
    /// This corresponds to node indices in the original graph.
    pub left: usize,

    /// Index (`usize`) of the variable on the right side.
    /// This corresponds to node indices in the original graph.
    pub right: usize,

    /// Minimum required separation (`a` in `u + a <= v`).
    /// Units are the same as the drawing coordinates (typically pixels or other distance units).
    pub gap: f32,
}

impl Constraint {
    /// Creates a new separation constraint.
    ///
    /// # Arguments
    ///
    /// * `left` - The index of the variable that should be on the left side of the constraint
    /// * `right` - The index of the variable that should be on the right side of the constraint
    /// * `gap` - The minimum required separation between the two variables
    ///
    /// # Returns
    ///
    /// A new `Constraint` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use petgraph_layout_separation_constraints::Constraint;
    ///
    /// // Create a constraint that node 0 must be at least 5.0 units to the left of node 1
    /// let constraint = Constraint::new(0, 1, 5.0);
    /// ```
    pub fn new(left: usize, right: usize, gap: f32) -> Self {
        Constraint { left, right, gap }
    }
}
