//! WebAssembly `LayoutGenerator` class bindings for `ClassicalMds`, `PivotMds`, `Hde`, `RdMds`,
//! and `KorenSpectral` across `layout2d`, `layout3d`, and `layoutNd`.

use egraph::core::{Layout2d, Layout3d, LayoutGenerator, LayoutNd};
use egraph::linalg::{
    ClassicalMds, ClassicalMdsConfig, Hde, HdeConfig, KorenSpectral, KorenSpectralConfig, PivotMds,
    PivotMdsConfig, RdMds, RdMdsConfig,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::hde::parse_hde_config;
use crate::js_convert::{core_err_to_js, parse_u32_sequence, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, get_prop, opt_bool_prop, opt_f64_prop, opt_maybe_usize_prop,
    opt_seed_prop, opt_target_prop, opt_usize_prop,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::linalg_mds::parse_optional_target_std;

fn reject_manifold_generator(cls: &str, target: JsTargetKind) -> Result<(), JsValue> {
    if target == JsTargetKind::Manifold2d {
        return Err(unsupported_err(format!(
            "{cls} only supports Euclidean targets ('layout2d', 'layout3d', 'layoutNd'), not 'manifold2d'"
        )));
    }
    Ok(())
}

/// Classical MDS `LayoutGenerator` supporting `"layout2d"`, `"layout3d"`, and `"layoutNd"`.
#[wasm_bindgen(js_name = ClassicalMds)]
#[derive(Debug, Clone)]
pub struct JsClassicalMds {
    target: JsTargetKind,
    config: ClassicalMdsConfig,
}

#[wasm_bindgen(js_class = ClassicalMds)]
impl JsClassicalMds {
    #[wasm_bindgen(constructor)]
    pub fn new(options: &JsValue) -> Result<JsClassicalMds, JsValue> {
        let mut target = JsTargetKind::Layout2d;
        let mut config = ClassicalMdsConfig::default();
        if check_options_arg(options, "ClassicalMds")? {
            target = opt_target_prop(options)?;
            reject_manifold_generator("ClassicalMds", target)?;
            config.dims = opt_maybe_usize_prop(options, "dims")?;
            config.eig_eps = opt_f64_prop(options, "eigEps", config.eig_eps)?;
            config.tol = opt_f64_prop(options, "tol", config.tol)?;
            config.max_sweeps = opt_usize_prop(options, "maxSweeps", config.max_sweeps)?;
        }
        Ok(Self { target, config })
    }

    pub fn generate(&self, graph: &JsCsrGraph) -> Result<JsValue, JsValue> {
        match self.target {
            JsTargetKind::Layout2d => {
                let l = ClassicalMds::<Layout2d<f32>, f32>::from_config(
                    graph.inner(),
                    self.config.clone(),
                )
                .generate()
                .map_err(core_err_to_js)?;
                Ok(JsLayout2d::from_layout2d(&l).into())
            }
            JsTargetKind::Layout3d => {
                let l = ClassicalMds::<Layout3d<f32>, f32>::from_config(
                    graph.inner(),
                    self.config.clone(),
                )
                .generate()
                .map_err(core_err_to_js)?;
                Ok(JsLayout3d::from_layout3d(&l).into())
            }
            JsTargetKind::LayoutNd => {
                let l = ClassicalMds::<LayoutNd<f32>, f32>::from_config(
                    graph.inner(),
                    self.config.clone(),
                )
                .generate()
                .map_err(core_err_to_js)?;
                Ok(JsLayoutNd::from_inner(l).into())
            }
            JsTargetKind::Manifold2d => unreachable!(),
        }
    }
}

/// Core-First MaxMin PivotMDS `LayoutGenerator` supporting `"layout2d"`, `"layout3d"`, and `"layoutNd"`.
#[wasm_bindgen(js_name = PivotMds)]
#[derive(Debug, Clone)]
pub struct JsPivotMds {
    target: JsTargetKind,
    config: PivotMdsConfig,
    pivots: Option<Vec<u32>>,
}

#[wasm_bindgen(js_class = PivotMds)]
impl JsPivotMds {
    #[wasm_bindgen(constructor)]
    pub fn new(options: &JsValue) -> Result<JsPivotMds, JsValue> {
        let mut target = JsTargetKind::Layout2d;
        let mut config = PivotMdsConfig::default();
        let mut pivots = None;
        if check_options_arg(options, "PivotMds")? {
            target = opt_target_prop(options)?;
            reject_manifold_generator("PivotMds", target)?;
            config.dims = opt_maybe_usize_prop(options, "dims")?;
            config.num_pivots = opt_usize_prop(options, "numPivots", config.num_pivots)?;
            config.min_degree = opt_usize_prop(options, "minDegree", config.min_degree)?;
            config.seed = opt_seed_prop(options)?;
            let ts_val = get_prop(options, "targetStd")?;
            if !ts_val.is_undefined() {
                config.target_std = parse_optional_target_std(&ts_val)?;
            }
            let piv_val = get_prop(options, "pivots")?;
            if !piv_val.is_undefined() && !piv_val.is_null() {
                pivots = Some(parse_u32_sequence(&piv_val, "pivots")?);
            }
        }
        Ok(Self {
            target,
            config,
            pivots,
        })
    }

    pub fn generate(&self, graph: &JsCsrGraph) -> Result<JsValue, JsValue> {
        match self.target {
            JsTargetKind::Layout2d => {
                let mut generator =
                    PivotMds::<Layout2d<f32>, f32>::from_config(graph.inner(), self.config.clone());
                if let Some(ref p) = self.pivots {
                    generator = generator.with_pivots(p);
                }
                let l = generator.generate().map_err(core_err_to_js)?;
                Ok(JsLayout2d::from_layout2d(&l).into())
            }
            JsTargetKind::Layout3d => {
                let mut generator =
                    PivotMds::<Layout3d<f32>, f32>::from_config(graph.inner(), self.config.clone());
                if let Some(ref p) = self.pivots {
                    generator = generator.with_pivots(p);
                }
                let l = generator.generate().map_err(core_err_to_js)?;
                Ok(JsLayout3d::from_layout3d(&l).into())
            }
            JsTargetKind::LayoutNd => {
                let mut generator =
                    PivotMds::<LayoutNd<f32>, f32>::from_config(graph.inner(), self.config.clone());
                if let Some(ref p) = self.pivots {
                    generator = generator.with_pivots(p);
                }
                let l = generator.generate().map_err(core_err_to_js)?;
                Ok(JsLayoutNd::from_inner(l).into())
            }
            JsTargetKind::Manifold2d => unreachable!(),
        }
    }
}

/// Harel-Koren High-Dimensional Embedding (`HDE`) `LayoutGenerator` supporting
/// `"layout2d"`, `"layout3d"`, and `"layoutNd"`.
#[wasm_bindgen(js_name = Hde)]
#[derive(Debug, Clone)]
pub struct JsHde {
    target: JsTargetKind,
    config: HdeConfig,
    pivots: Option<Vec<u32>>,
}

#[wasm_bindgen(js_class = Hde)]
impl JsHde {
    #[wasm_bindgen(constructor)]
    pub fn new(options: &JsValue) -> Result<JsHde, JsValue> {
        let mut target = JsTargetKind::Layout2d;
        if check_options_arg(options, "Hde")? {
            target = opt_target_prop(options)?;
            reject_manifold_generator("Hde", target)?;
        }
        let (config, pivots) = parse_hde_config(options, true, "Hde")?;
        Ok(Self {
            target,
            config,
            pivots,
        })
    }

    pub fn generate(&self, graph: &JsCsrGraph) -> Result<JsValue, JsValue> {
        match self.target {
            JsTargetKind::Layout2d => {
                let mut generator =
                    Hde::<Layout2d<f32>, f32>::from_config(graph.inner(), self.config.clone());
                if let Some(ref p) = self.pivots {
                    generator = generator.with_pivots(p);
                }
                let l = generator.generate().map_err(core_err_to_js)?;
                Ok(JsLayout2d::from_layout2d(&l).into())
            }
            JsTargetKind::Layout3d => {
                let mut generator =
                    Hde::<Layout3d<f32>, f32>::from_config(graph.inner(), self.config.clone());
                if let Some(ref p) = self.pivots {
                    generator = generator.with_pivots(p);
                }
                let l = generator.generate().map_err(core_err_to_js)?;
                Ok(JsLayout3d::from_layout3d(&l).into())
            }
            JsTargetKind::LayoutNd => {
                let mut generator =
                    Hde::<LayoutNd<f32>, f32>::from_config(graph.inner(), self.config.clone());
                if let Some(ref p) = self.pivots {
                    generator = generator.with_pivots(p);
                }
                let l = generator.generate().map_err(core_err_to_js)?;
                Ok(JsLayoutNd::from_inner(l).into())
            }
            JsTargetKind::Manifold2d => unreachable!(),
        }
    }
}

/// Resistance-distance MDS (`RdMds`) `LayoutGenerator` supporting
/// `"layout2d"`, `"layout3d"`, and `"layoutNd"`.
#[wasm_bindgen(js_name = RdMds)]
#[derive(Debug, Clone)]
pub struct JsRdMds {
    target: JsTargetKind,
    config: RdMdsConfig,
}

#[wasm_bindgen(js_class = RdMds)]
impl JsRdMds {
    #[wasm_bindgen(constructor)]
    pub fn new(options: &JsValue) -> Result<JsRdMds, JsValue> {
        let mut target = JsTargetKind::Layout2d;
        let mut config = RdMdsConfig::default();
        if check_options_arg(options, "RdMds")? {
            target = opt_target_prop(options)?;
            reject_manifold_generator("RdMds", target)?;
            config.dims = opt_maybe_usize_prop(options, "dims")?;
            config.eig_eps = opt_f64_prop(options, "eigEps", config.eig_eps)?;
            config.tol = opt_f64_prop(options, "tol", config.tol)?;
            config.max_iters = opt_usize_prop(options, "maxIterations", config.max_iters)?;
            config.seed = opt_seed_prop(options)?;
            config.degree_normalize =
                opt_bool_prop(options, "degreeNormalize", config.degree_normalize)?;
            let ts_val = get_prop(options, "targetStd")?;
            if !ts_val.is_undefined() {
                config.target_std = parse_optional_target_std(&ts_val)?;
            }
        }
        Ok(Self { target, config })
    }

    pub fn generate(&self, graph: &JsCsrGraph) -> Result<JsValue, JsValue> {
        match self.target {
            JsTargetKind::Layout2d => {
                let l =
                    RdMds::<Layout2d<f32>, f32>::from_config(graph.inner(), self.config.clone())
                        .generate()
                        .map_err(core_err_to_js)?;
                Ok(JsLayout2d::from_layout2d(&l).into())
            }
            JsTargetKind::Layout3d => {
                let l =
                    RdMds::<Layout3d<f32>, f32>::from_config(graph.inner(), self.config.clone())
                        .generate()
                        .map_err(core_err_to_js)?;
                Ok(JsLayout3d::from_layout3d(&l).into())
            }
            JsTargetKind::LayoutNd => {
                let l =
                    RdMds::<LayoutNd<f32>, f32>::from_config(graph.inner(), self.config.clone())
                        .generate()
                        .map_err(core_err_to_js)?;
                Ok(JsLayoutNd::from_inner(l).into())
            }
            JsTargetKind::Manifold2d => unreachable!(),
        }
    }
}

/// Koren degree-normalized spectral `LayoutGenerator` supporting
/// `"layout2d"`, `"layout3d"`, and `"layoutNd"`.
#[wasm_bindgen(js_name = KorenSpectral)]
#[derive(Debug, Clone)]
pub struct JsKorenSpectral {
    target: JsTargetKind,
    config: KorenSpectralConfig,
}

#[wasm_bindgen(js_class = KorenSpectral)]
impl JsKorenSpectral {
    #[wasm_bindgen(constructor)]
    pub fn new(options: &JsValue) -> Result<JsKorenSpectral, JsValue> {
        let mut target = JsTargetKind::Layout2d;
        let mut config = KorenSpectralConfig::default();
        if check_options_arg(options, "KorenSpectral")? {
            target = opt_target_prop(options)?;
            reject_manifold_generator("KorenSpectral", target)?;
            config.dims = opt_maybe_usize_prop(options, "dims")?;
            let ts_val = get_prop(options, "targetStd")?;
            if !ts_val.is_undefined() {
                config.target_std = parse_optional_target_std(&ts_val)?;
            }
            config.tol = opt_f64_prop(options, "tol", config.tol)?;
            config.max_iters = opt_usize_prop(options, "maxIterations", config.max_iters)?;
            config.seed = opt_seed_prop(options)?;
        }
        Ok(Self { target, config })
    }

    pub fn generate(&self, graph: &JsCsrGraph) -> Result<JsValue, JsValue> {
        match self.target {
            JsTargetKind::Layout2d => {
                let l = KorenSpectral::<Layout2d<f32>, f32>::from_config(
                    graph.inner(),
                    self.config.clone(),
                )
                .generate()
                .map_err(core_err_to_js)?;
                Ok(JsLayout2d::from_layout2d(&l).into())
            }
            JsTargetKind::Layout3d => {
                let l = KorenSpectral::<Layout3d<f32>, f32>::from_config(
                    graph.inner(),
                    self.config.clone(),
                )
                .generate()
                .map_err(core_err_to_js)?;
                Ok(JsLayout3d::from_layout3d(&l).into())
            }
            JsTargetKind::LayoutNd => {
                let l = KorenSpectral::<LayoutNd<f32>, f32>::from_config(
                    graph.inner(),
                    self.config.clone(),
                )
                .generate()
                .map_err(core_err_to_js)?;
                Ok(JsLayoutNd::from_inner(l).into())
            }
            JsTargetKind::Manifold2d => unreachable!(),
        }
    }
}
