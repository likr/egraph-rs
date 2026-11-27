//! Unified clustering result and trait abstractions for graph community detection and
//! feature-space clustering.

use egraph_core::{CoreResult, CsrGraph, Scalar};

/// Unified result returned by [`CommunityDetector`] and [`Clusterer`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterResult {
    /// Canonical cluster or community assignment in `0..num_clusters` for each item.
    pub labels: Vec<u32>,
    /// Number of distinct non-empty clusters or communities.
    pub num_clusters: usize,
    /// Optional algorithm-specific objective or quality score (`Some(modularity)` for Louvain,
    /// `Some(quality)` for Leiden, `Some(inertia)` for K-Means++, `None` otherwise).
    pub quality: Option<f64>,
    /// Number of iterations or multi-level passes executed.
    pub iterations: usize,
    /// Whether the algorithm reached its convergence or fixed-point stopping criterion.
    pub converged: bool,
}

impl ClusterResult {
    /// Creates a new [`ClusterResult`].
    pub fn new(
        labels: Vec<u32>,
        num_clusters: usize,
        quality: Option<f64>,
        iterations: usize,
        converged: bool,
    ) -> Self {
        Self {
            labels,
            num_clusters,
            quality,
            iterations,
            converged,
        }
    }

    /// Returns the canonical `0..num_clusters` cluster labels slice.
    #[inline]
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// Consumes this result and returns the owned cluster labels vector.
    #[inline]
    pub fn into_labels(self) -> Vec<u32> {
        self.labels
    }

    /// Returns the number of distinct clusters or communities.
    #[inline]
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Returns the optional algorithm-specific quality or objective score.
    #[inline]
    pub fn quality(&self) -> Option<f64> {
        self.quality
    }

    /// Returns the number of iterations or passes executed.
    #[inline]
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns `true` if the algorithm reached its convergence criterion.
    #[inline]
    pub fn converged(&self) -> bool {
        self.converged
    }
}

/// Graph community detection over an undirected [`CsrGraph`].
///
/// Implemented by [`Louvain`](crate::Louvain), [`Leiden`](crate::Leiden),
/// [`SpectralClustering`](crate::SpectralClustering), and
/// [`LabelPropagation`](crate::LabelPropagation).
pub trait CommunityDetector<W: Scalar = f64> {
    /// Detects communities on an undirected [`CsrGraph`] and returns a unified [`ClusterResult`].
    fn detect(&self, graph: &CsrGraph<W>) -> CoreResult<ClusterResult>;
}

impl<W: Scalar, T: CommunityDetector<W> + ?Sized> CommunityDetector<W> for &T {
    #[inline]
    fn detect(&self, graph: &CsrGraph<W>) -> CoreResult<ClusterResult> {
        (**self).detect(graph)
    }
}

impl<W: Scalar, T: CommunityDetector<W> + ?Sized> CommunityDetector<W> for &mut T {
    #[inline]
    fn detect(&self, graph: &CsrGraph<W>) -> CoreResult<ClusterResult> {
        (**self).detect(graph)
    }
}

impl<W: Scalar, T: CommunityDetector<W> + ?Sized> CommunityDetector<W> for Box<T> {
    #[inline]
    fn detect(&self, graph: &CsrGraph<W>) -> CoreResult<ClusterResult> {
        (**self).detect(graph)
    }
}

/// Feature-space clustering over flat row-major point coordinates.
///
/// Implemented by [`KMeansPlusPlus`](crate::KMeansPlusPlus).
pub trait Clusterer {
    /// Clusters `num_points` row-major points of dimension `dims` stored in `points_flat`.
    fn cluster(
        &self,
        points_flat: &[f64],
        num_points: usize,
        dims: usize,
    ) -> CoreResult<ClusterResult>;
}

impl<T: Clusterer + ?Sized> Clusterer for &T {
    #[inline]
    fn cluster(
        &self,
        points_flat: &[f64],
        num_points: usize,
        dims: usize,
    ) -> CoreResult<ClusterResult> {
        (**self).cluster(points_flat, num_points, dims)
    }
}

impl<T: Clusterer + ?Sized> Clusterer for &mut T {
    #[inline]
    fn cluster(
        &self,
        points_flat: &[f64],
        num_points: usize,
        dims: usize,
    ) -> CoreResult<ClusterResult> {
        (**self).cluster(points_flat, num_points, dims)
    }
}

impl<T: Clusterer + ?Sized> Clusterer for Box<T> {
    #[inline]
    fn cluster(
        &self,
        points_flat: &[f64],
        num_points: usize,
        dims: usize,
    ) -> CoreResult<ClusterResult> {
        (**self).cluster(points_flat, num_points, dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        KMeansConfig, KMeansPlusPlus, LabelPropagation, LabelPropagationConfig, Leiden,
        LeidenConfig, Louvain, LouvainConfig, SpectralClustering, SpectralClusteringConfig,
    };

    fn make_barbell_f64() -> CsrGraph<f64> {
        let mut edges = Vec::new();
        for i in 0..5u32 {
            for j in (i + 1)..5 {
                edges.push((i, j));
                edges.push((5 + i, 5 + j));
            }
        }
        edges.push((4, 5));
        CsrGraph::from_edges(10, &edges, None, false, false).unwrap()
    }

    #[test]
    fn test_community_detector_polymorphic_and_blanket_impls() {
        let graph = make_barbell_f64();
        let mut louvain_det = Louvain::new(LouvainConfig::default());
        let detectors: Vec<Box<dyn CommunityDetector<f64>>> = vec![
            Box::new(Louvain::default()),
            Box::new(Leiden::new(LeidenConfig::default())),
            Box::new(SpectralClustering::new(SpectralClusteringConfig::default())),
            Box::new(LabelPropagation::new(LabelPropagationConfig::default())),
        ];

        for det in &detectors {
            let res = det.detect(&graph).unwrap();
            assert_eq!(res.num_clusters(), 2);
            assert_eq!(res.labels().len(), 10);
            assert_eq!(&res.labels()[..5], &[0, 0, 0, 0, 0]);
            assert_eq!(&res.labels()[5..], &[1, 1, 1, 1, 1]);
            assert!(res.converged());
        }

        // Verify &T, &mut T, and Box<T> blanket implementations plus into_labels().
        fn run_detector<D: CommunityDetector<f64>>(det: D, g: &CsrGraph<f64>) -> ClusterResult {
            det.detect(g).unwrap()
        }

        let r_ref = run_detector(&louvain_det, &graph);
        let r_mut = run_detector(&mut louvain_det, &graph);
        let r_box = run_detector(Box::new(louvain_det), &graph);
        assert_eq!(r_ref, r_mut);
        assert_eq!(r_mut, r_box);
        assert!(r_ref.quality().is_some());
        assert!(r_ref.iterations() >= 1);
        assert_eq!(r_ref.clone().into_labels(), r_ref.labels());
    }

    #[test]
    fn test_clusterer_polymorphic_and_blanket_impls() {
        fn run_clusterer<C: Clusterer>(c: C, pts: &[f64]) -> ClusterResult {
            c.cluster(pts, 6, 2).unwrap()
        }

        let pts = [
            0.0, 0.0, 0.1, -0.1, -0.1, 0.1, // cluster 0
            10.0, 10.0, 10.1, 9.9, 9.9, 10.1, // cluster 1
        ];
        let mut km = KMeansPlusPlus::new(KMeansConfig::new().with_k(2).with_seed(7));
        let boxed: Box<dyn Clusterer> = Box::new(km.clone());
        let r_dyn = boxed.cluster(&pts, 6, 2).unwrap();
        let r_ref = run_clusterer(&km, &pts);
        let r_mut = run_clusterer(&mut km, &pts);
        let r_box = run_clusterer(Box::new(km), &pts);

        assert_eq!(r_dyn, r_ref);
        assert_eq!(r_ref, r_mut);
        assert_eq!(r_mut, r_box);
        assert_eq!(r_dyn.num_clusters(), 2);
        assert_eq!(r_dyn.labels(), &[0, 0, 0, 1, 1, 1]);
        assert!(r_dyn.quality().unwrap() < 0.5);
        assert!(r_dyn.iterations() >= 1);
        assert!(r_dyn.converged());
    }
}
