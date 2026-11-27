use super::TORUS_DEG_CV_THRESHOLD;
use super::config::SuggestedGeometry;

/// Classifies a target 2D geometry from link resistance curvature and vertex degree regularity.
#[allow(clippy::too_many_arguments)]
pub fn classify_geometry_from_curvature(
    num_edges: usize,
    mean_link_curvature: f64,
    std_link_curvature: f64,
    leaf_fraction: f64,
    deg_cv: f64,
    hyperbolic_threshold: f64,
    spherical_threshold: f64,
    torus_std_threshold: f64,
) -> SuggestedGeometry {
    classify_geometry_from_curvature_with_sampling(
        num_edges,
        mean_link_curvature,
        std_link_curvature,
        leaf_fraction,
        deg_cv,
        hyperbolic_threshold,
        spherical_threshold,
        torus_std_threshold,
        true,
        1,
        false,
    )
}

/// Classifies a target 2D geometry from link resistance curvature, degree regularity, JL
/// sampling regime (`used_exact`, `jl_dim`), and closed-surface topological predicate.
///
/// Both exact and JL modes use the sharp `hyperbolic_threshold` and `spherical_threshold` bounds
/// directly (without shifting mean thresholds), because the Jacobi $D^{-1}$ control variate and
/// per-component Foster normalization eliminate global Foster drift and local correlation bias.
/// When `used_exact` is `false`, only `torus_std_threshold` is scaled by $O(1/\sqrt{k_{\mathrm{JL}}})$
/// to absorb the residual per-edge JL curvature spread on flat regular tori.
#[allow(clippy::too_many_arguments)]
pub fn classify_geometry_from_curvature_with_sampling(
    num_edges: usize,
    mean_link_curvature: f64,
    std_link_curvature: f64,
    leaf_fraction: f64,
    deg_cv: f64,
    hyperbolic_threshold: f64,
    spherical_threshold: f64,
    torus_std_threshold: f64,
    used_exact: bool,
    jl_dim: usize,
    is_closed_spherical_surface: bool,
) -> SuggestedGeometry {
    if num_edges == 0 {
        return SuggestedGeometry::Euclidean;
    }
    if is_closed_spherical_surface {
        return SuggestedGeometry::Spherical;
    }
    let jl_scale = if used_exact {
        0.0
    } else {
        (64.0 / (jl_dim.max(1) as f64)).sqrt()
    };
    let is_regular_closed = leaf_fraction == 0.0 && deg_cv < TORUS_DEG_CV_THRESHOLD;
    let effective_torus_std_threshold = if used_exact {
        torus_std_threshold
    } else {
        // Residual JL resistance variance after the Jacobi control variate scales as
        // 1 / sqrt(jl_dim) across curvature-flow steps on regular flat tori.
        torus_std_threshold * (1.0 + 8.0 * jl_scale)
    };
    if mean_link_curvature < hyperbolic_threshold {
        return SuggestedGeometry::HyperbolicPoincare;
    }
    if mean_link_curvature > spherical_threshold {
        return SuggestedGeometry::Spherical;
    }
    if is_regular_closed && std_link_curvature <= effective_torus_std_threshold {
        SuggestedGeometry::Torus2d
    } else {
        SuggestedGeometry::Euclidean
    }
}
