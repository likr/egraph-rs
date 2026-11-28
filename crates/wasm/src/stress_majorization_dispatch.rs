//! Internal `(target, backend)` option parsing and enum construction helpers
//! for `StressMajorizationSolver`.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldLayout2d, SolverStatus};
use egraph::layout::{
    EuclideanPcg, GeodesicGradient, StressMajorizationConfig, StressMajorizationSolver,
};
use wasm_bindgen::JsValue;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, invalid_input_err, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, normalize_enum_token, opt_f64_prop, opt_maybe_usize_prop,
    opt_string_prop, opt_target_prop, opt_usize_prop,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum StressBackendKind {
    EuclideanPcg,
    GeodesicGradient,
}

pub(crate) fn parse_stress_options(
    options: &JsValue,
    ctx: &str,
) -> Result<(StressMajorizationConfig, JsTargetKind, StressBackendKind), JsValue> {
    let mut cfg = StressMajorizationConfig::default();
    let has_opts = check_options_arg(options, ctx)?;
    let target = if has_opts {
        opt_target_prop(options)?
    } else {
        JsTargetKind::Layout2d
    };
    let backend_opt = if has_opts {
        opt_string_prop(options, "backend")?
    } else {
        None
    };
    let backend = match (target, backend_opt) {
        (JsTargetKind::Manifold2d, None) => StressBackendKind::GeodesicGradient,
        (JsTargetKind::Manifold2d, Some(ref b)) => match normalize_enum_token(b)?.as_str() {
            "geodesicgradient" | "geodesic" => StressBackendKind::GeodesicGradient,
            "euclideanpcg" | "pcg" => {
                return Err(unsupported_err(
                    "StressMajorizationSolver target 'manifold2d' does not support backend 'euclideanPcg'",
                ));
            }
            _ => {
                return Err(invalid_input_err(format!(
                    "unknown StressMajorizationSolver backend {b:?}; expected 'geodesicGradient'"
                )));
            }
        },
        (_, None) => StressBackendKind::EuclideanPcg,
        (_, Some(ref b)) => match normalize_enum_token(b)?.as_str() {
            "euclideanpcg" | "pcg" => StressBackendKind::EuclideanPcg,
            "geodesicgradient" | "geodesic" => StressBackendKind::GeodesicGradient,
            _ => {
                return Err(invalid_input_err(format!(
                    "unknown StressMajorizationSolver backend {b:?}; expected 'euclideanPcg' or 'geodesicGradient'"
                )));
            }
        },
    };
    if has_opts {
        cfg.epsilon = opt_f64_prop(options, "epsilon", cfg.epsilon)?;
        cfg.max_iterations = opt_usize_prop(options, "maxIterations", cfg.max_iterations)?;
        cfg.pcg_tolerance = opt_f64_prop(options, "pcgTolerance", cfg.pcg_tolerance)?;
        cfg.pcg_max_iterations = opt_maybe_usize_prop(options, "pcgMaxIterations")?;
        cfg.min_norm = opt_f64_prop(options, "minNorm", cfg.min_norm)?;
    }
    Ok((cfg, target, backend))
}

#[derive(Debug, Clone)]
pub(crate) enum StressMajInner {
    Dim2Pcg(StressMajorizationSolver<Layout2d<f32>, EuclideanPcg>),
    Dim2Geo(StressMajorizationSolver<Layout2d<f32>, GeodesicGradient>),
    Dim3Pcg(StressMajorizationSolver<Layout3d<f32>, EuclideanPcg>),
    Dim3Geo(StressMajorizationSolver<Layout3d<f32>, GeodesicGradient>),
    DimNPcg(StressMajorizationSolver<LayoutNd<f32>, EuclideanPcg>),
    DimNGeo(StressMajorizationSolver<LayoutNd<f32>, GeodesicGradient>),
    Manifold(StressMajorizationSolver<ManifoldLayout2d<f32>, GeodesicGradient>),
}

pub(crate) fn build_stress_from_graph(
    graph: &JsCsrGraph,
    cfg: StressMajorizationConfig,
    target: JsTargetKind,
    backend: StressBackendKind,
) -> Result<StressMajInner, JsValue> {
    let g = graph.inner();
    let inner = match (target, backend) {
        (JsTargetKind::Layout2d, StressBackendKind::EuclideanPcg) => StressMajInner::Dim2Pcg(
            StressMajorizationSolver::<Layout2d<f32>, EuclideanPcg>::from_graph(g, cfg)
                .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::Layout2d, StressBackendKind::GeodesicGradient) => StressMajInner::Dim2Geo(
            StressMajorizationSolver::<Layout2d<f32>, GeodesicGradient>::from_graph(g, cfg)
                .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::Layout3d, StressBackendKind::EuclideanPcg) => StressMajInner::Dim3Pcg(
            StressMajorizationSolver::<Layout3d<f32>, EuclideanPcg>::from_graph(g, cfg)
                .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::Layout3d, StressBackendKind::GeodesicGradient) => StressMajInner::Dim3Geo(
            StressMajorizationSolver::<Layout3d<f32>, GeodesicGradient>::from_graph(g, cfg)
                .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::LayoutNd, StressBackendKind::EuclideanPcg) => StressMajInner::DimNPcg(
            StressMajorizationSolver::<LayoutNd<f32>, EuclideanPcg>::from_graph(g, cfg)
                .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::LayoutNd, StressBackendKind::GeodesicGradient) => StressMajInner::DimNGeo(
            StressMajorizationSolver::<LayoutNd<f32>, GeodesicGradient>::from_graph(g, cfg)
                .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::Manifold2d, _) => StressMajInner::Manifold(
            StressMajorizationSolver::<ManifoldLayout2d<f32>, GeodesicGradient>::from_graph(g, cfg)
                .map_err(core_err_to_js)?,
        ),
    };
    Ok(inner)
}

pub(crate) fn build_stress_from_distances(
    n: usize,
    dists_flat: &[f64],
    cfg: StressMajorizationConfig,
    target: JsTargetKind,
    backend: StressBackendKind,
) -> Result<StressMajInner, JsValue> {
    let inner = match (target, backend) {
        (JsTargetKind::Layout2d, StressBackendKind::EuclideanPcg) => StressMajInner::Dim2Pcg(
            StressMajorizationSolver::<Layout2d<f32>, EuclideanPcg>::from_distance_matrix(
                n, dists_flat, cfg,
            )
            .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::Layout2d, StressBackendKind::GeodesicGradient) => StressMajInner::Dim2Geo(
            StressMajorizationSolver::<Layout2d<f32>, GeodesicGradient>::from_distance_matrix(
                n, dists_flat, cfg,
            )
            .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::Layout3d, StressBackendKind::EuclideanPcg) => StressMajInner::Dim3Pcg(
            StressMajorizationSolver::<Layout3d<f32>, EuclideanPcg>::from_distance_matrix(
                n, dists_flat, cfg,
            )
            .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::Layout3d, StressBackendKind::GeodesicGradient) => StressMajInner::Dim3Geo(
            StressMajorizationSolver::<Layout3d<f32>, GeodesicGradient>::from_distance_matrix(
                n, dists_flat, cfg,
            )
            .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::LayoutNd, StressBackendKind::EuclideanPcg) => StressMajInner::DimNPcg(
            StressMajorizationSolver::<LayoutNd<f32>, EuclideanPcg>::from_distance_matrix(
                n, dists_flat, cfg,
            )
            .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::LayoutNd, StressBackendKind::GeodesicGradient) => StressMajInner::DimNGeo(
            StressMajorizationSolver::<LayoutNd<f32>, GeodesicGradient>::from_distance_matrix(
                n, dists_flat, cfg,
            )
            .map_err(core_err_to_js)?,
        ),
        (JsTargetKind::Manifold2d, _) => StressMajInner::Manifold(
            StressMajorizationSolver::<ManifoldLayout2d<f32>, GeodesicGradient>::from_distance_matrix(
                n, dists_flat, cfg,
            )
            .map_err(core_err_to_js)?,
        ),
    };
    Ok(inner)
}

impl StressMajInner {
    pub(crate) fn target_str(&self) -> &'static str {
        match self {
            Self::Dim2Pcg(_) | Self::Dim2Geo(_) => "layout2d",
            Self::Dim3Pcg(_) | Self::Dim3Geo(_) => "layout3d",
            Self::DimNPcg(_) | Self::DimNGeo(_) => "layoutNd",
            Self::Manifold(_) => "manifold2d",
        }
    }

    pub(crate) fn backend_str(&self) -> &'static str {
        match self {
            Self::Dim2Pcg(_) | Self::Dim3Pcg(_) | Self::DimNPcg(_) => "euclideanPcg",
            Self::Dim2Geo(_) | Self::Dim3Geo(_) | Self::DimNGeo(_) | Self::Manifold(_) => {
                "geodesicGradient"
            }
        }
    }

    pub(crate) fn last_stress(&self) -> Option<f64> {
        match self {
            Self::Dim2Pcg(s) => s.last_stress(),
            Self::Dim2Geo(s) => s.last_stress(),
            Self::Dim3Pcg(s) => s.last_stress(),
            Self::Dim3Geo(s) => s.last_stress(),
            Self::DimNPcg(s) => s.last_stress(),
            Self::DimNGeo(s) => s.last_stress(),
            Self::Manifold(s) => s.last_stress(),
        }
    }

    pub(crate) fn status(&self) -> SolverStatus {
        match self {
            Self::Dim2Pcg(s) => s.status(),
            Self::Dim2Geo(s) => s.status(),
            Self::Dim3Pcg(s) => s.status(),
            Self::Dim3Geo(s) => s.status(),
            Self::DimNPcg(s) => s.status(),
            Self::DimNGeo(s) => s.status(),
            Self::Manifold(s) => s.status(),
        }
    }

    pub(crate) fn iteration(&self) -> usize {
        match self {
            Self::Dim2Pcg(s) => s.iteration(),
            Self::Dim2Geo(s) => s.iteration(),
            Self::Dim3Pcg(s) => s.iteration(),
            Self::Dim3Geo(s) => s.iteration(),
            Self::DimNPcg(s) => s.iteration(),
            Self::DimNGeo(s) => s.iteration(),
            Self::Manifold(s) => s.iteration(),
        }
    }

    pub(crate) fn total_iterations(&self) -> Option<usize> {
        match self {
            Self::Dim2Pcg(s) => s.total_iterations(),
            Self::Dim2Geo(s) => s.total_iterations(),
            Self::Dim3Pcg(s) => s.total_iterations(),
            Self::Dim3Geo(s) => s.total_iterations(),
            Self::DimNPcg(s) => s.total_iterations(),
            Self::DimNGeo(s) => s.total_iterations(),
            Self::Manifold(s) => s.total_iterations(),
        }
    }

    pub(crate) fn num_nodes(&self) -> usize {
        match self {
            Self::Dim2Pcg(s) => s.num_nodes(),
            Self::Dim2Geo(s) => s.num_nodes(),
            Self::Dim3Pcg(s) => s.num_nodes(),
            Self::Dim3Geo(s) => s.num_nodes(),
            Self::DimNPcg(s) => s.num_nodes(),
            Self::DimNGeo(s) => s.num_nodes(),
            Self::Manifold(s) => s.num_nodes(),
        }
    }
}
