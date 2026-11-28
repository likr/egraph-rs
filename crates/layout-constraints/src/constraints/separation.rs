use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

/// One-dimensional separation constraint enforcing `coord[left] + gap <= coord[right]`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SeparationConstraint<S: Scalar = f64> {
    pub left: u32,
    pub right: u32,
    pub gap: S,
}

impl<S: Scalar> SeparationConstraint<S> {
    /// Creates a validated separation constraint `coord[left] + gap <= coord[right]`.
    #[inline]
    pub fn new(left: u32, right: u32, gap: S) -> CoreResult<Self> {
        let constraint = Self { left, right, gap };
        constraint.validate()?;
        Ok(constraint)
    }

    /// Validates that `gap` is finite.
    #[inline]
    pub fn validate(&self) -> CoreResult<()> {
        if !self.gap.is_finite() {
            return Err(CoreError::non_finite("SeparationConstraint::gap"));
        }
        Ok(())
    }

    /// Replaces `left` on this constraint.
    #[inline]
    pub fn with_left(mut self, left: u32) -> Self {
        self.left = left;
        self
    }

    /// Replaces `right` on this constraint.
    #[inline]
    pub fn with_right(mut self, right: u32) -> Self {
        self.right = right;
        self
    }

    /// Validates and replaces `gap` on this constraint.
    #[inline]
    pub fn with_gap(mut self, gap: S) -> CoreResult<Self> {
        self.gap = gap;
        self.validate()?;
        Ok(self)
    }

    /// Returns the left-hand variable index.
    #[inline]
    pub fn left(&self) -> u32 {
        self.left
    }

    /// Returns the right-hand variable index.
    #[inline]
    pub fn right(&self) -> u32 {
        self.right
    }

    /// Returns the required separation gap.
    #[inline]
    pub fn gap(&self) -> S {
        self.gap
    }
}

/// Generates hierarchical vertical separation constraints `y[u] + gap <= y[v]` for edges `(u, v)`
/// spanning layers `layers[u] < layers[v]`, where `gap = min_layer_distance * (layers[v] - layers[u])`.
pub fn generate_layered_constraints<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layers: &[u32],
    min_layer_distance: S,
) -> CoreResult<Vec<SeparationConstraint<S>>> {
    if !graph.is_directed() {
        return Err(CoreError::unsupported(
            "generate_layered_constraints requires a directed CsrGraph",
        ));
    }
    let n = graph.num_nodes();
    if layers.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            layers.len(),
            "generate_layered_constraints::layers",
        ));
    }
    if !min_layer_distance.is_finite() {
        return Err(CoreError::non_finite(
            "generate_layered_constraints::min_layer_distance",
        ));
    }
    if min_layer_distance < S::ZERO {
        return Err(CoreError::invalid_input("min_layer_distance must be >= 0"));
    }

    let mut constraints = Vec::new();
    for u in 0..n {
        let lu = layers[u];
        for &v in graph.neighbors(u) {
            let lv = layers[v as usize];
            if lu < lv {
                let span = (lv - lu) as usize;
                let gap = min_layer_distance * S::from_usize(span);
                constraints.push(SeparationConstraint::new(u as u32, v, gap)?);
            }
        }
    }
    Ok(constraints)
}
