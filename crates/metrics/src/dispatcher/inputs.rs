use std::fmt;

use egraph_core::{Layout2d, LayoutStorage, NodeExtents2d};

/// Optional auxiliary inputs passed to [`super::evaluate_metrics_with_inputs`].
#[non_exhaustive]
pub struct MetricInputs<'a, L: LayoutStorage + ?Sized = Layout2d<f64>> {
    /// Optional cluster/community label slice of length `N`.
    pub labels: Option<&'a [u32]>,
    /// Optional positive finite vertex weights of length `N` for `ClusterAreaFaithfulness`
    /// (mutually exclusive with `cluster_weights`).
    pub node_weights: Option<&'a [f64]>,
    /// Optional positive finite cluster weights of length `K` for `ClusterAreaFaithfulness`
    /// (mutually exclusive with `node_weights`).
    pub cluster_weights: Option<&'a [f64]>,
    /// Optional reference layout of length `N` required by `PrismDistortion` and `NorMetrics`.
    pub original_layout: Option<&'a L>,
    /// Optional per-node 2D half-extents of length `N` for `SprawlterNn` and `NorMetrics`.
    pub sprawlter_extents: Option<NodeExtents2d<'a, L::Scalar>>,
}

impl<'a, L: LayoutStorage + ?Sized> Default for MetricInputs<'a, L> {
    fn default() -> Self {
        Self {
            labels: None,
            node_weights: None,
            cluster_weights: None,
            original_layout: None,
            sprawlter_extents: None,
        }
    }
}

impl<'a, L: LayoutStorage + ?Sized> Clone for MetricInputs<'a, L> {
    fn clone(&self) -> Self {
        Self {
            labels: self.labels,
            node_weights: self.node_weights,
            cluster_weights: self.cluster_weights,
            original_layout: self.original_layout,
            sprawlter_extents: self.sprawlter_extents,
        }
    }
}

impl<'a, L: LayoutStorage + ?Sized> fmt::Debug for MetricInputs<'a, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetricInputs")
            .field("labels", &self.labels)
            .field("node_weights", &self.node_weights)
            .field("cluster_weights", &self.cluster_weights)
            .field(
                "original_layout_len",
                &self.original_layout.map(|l| l.len()),
            )
            .field("sprawlter_extents", &self.sprawlter_extents)
            .finish()
    }
}

impl<'a, L: LayoutStorage + ?Sized> MetricInputs<'a, L> {
    /// Creates empty [`MetricInputs`] with all optional fields set to `None`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the optional cluster/community label slice of length `N`.
    #[inline]
    pub fn with_labels(mut self, labels: Option<&'a [u32]>) -> Self {
        self.labels = labels;
        self
    }

    /// Sets the optional positive finite vertex weights for `ClusterAreaFaithfulness`.
    #[inline]
    pub fn with_node_weights(mut self, node_weights: Option<&'a [f64]>) -> Self {
        self.node_weights = node_weights;
        self
    }

    /// Sets the optional positive finite cluster weights for `ClusterAreaFaithfulness`.
    #[inline]
    pub fn with_cluster_weights(mut self, cluster_weights: Option<&'a [f64]>) -> Self {
        self.cluster_weights = cluster_weights;
        self
    }

    /// Sets the optional reference layout required by `PrismDistortion` and `NorMetrics`.
    #[inline]
    pub fn with_original_layout(mut self, original_layout: Option<&'a L>) -> Self {
        self.original_layout = original_layout;
        self
    }

    /// Sets the optional per-node 2D half-extents for `SprawlterNn` and `NorMetrics`.
    #[inline]
    pub fn with_sprawlter_extents(
        mut self,
        sprawlter_extents: Option<NodeExtents2d<'a, L::Scalar>>,
    ) -> Self {
        self.sprawlter_extents = sprawlter_extents;
        self
    }
}
