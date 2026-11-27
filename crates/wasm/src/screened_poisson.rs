//! Screened Poisson unilateral upper-bound 2D density field bindings (`f32`).

use egraph::spatial::{
    DensityFieldKernel, ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput,
    ScreenedPoissonStats,
};
use wasm_bindgen::prelude::*;

use crate::js_convert::core_err_to_js;
use crate::js_options::{
    check_options_arg, normalize_enum_token, opt_bool_prop, opt_f64_prop, opt_maybe_f64_prop,
    opt_string_prop, opt_usize_prop,
};
use crate::layout_buffers::JsLayout2d;

const DEFAULT_GRID_SIZE: usize = 512;

fn parse_density_field_kernel(raw: &str) -> Result<DensityFieldKernel, JsError> {
    match normalize_enum_token(raw).as_str() {
        "screenedpoisson" | "sp" => Ok(DensityFieldKernel::ScreenedPoisson),
        "singlegaussian" | "gaussian" | "sg" => Ok(DensityFieldKernel::SingleGaussian),
        _ => Err(JsError::new(&format!(
            "unknown density field kernel {raw:?}; expected 'screenedPoisson' or 'singleGaussian'"
        ))),
    }
}

fn density_field_kernel_js_str(kernel: DensityFieldKernel) -> &'static str {
    match kernel {
        DensityFieldKernel::ScreenedPoisson => "screenedPoisson",
        DensityFieldKernel::SingleGaussian => "singleGaussian",
    }
}

pub(crate) fn parse_screened_poisson_config(
    options: &JsValue,
    ctx: &str,
) -> Result<ScreenedPoissonConfig, JsError> {
    let mut cfg = ScreenedPoissonConfig {
        grid_size: DEFAULT_GRID_SIZE,
        ..ScreenedPoissonConfig::default()
    };
    if check_options_arg(options, ctx)? {
        if let Some(k_str) = opt_string_prop(options, "kernel")? {
            cfg.kernel = parse_density_field_kernel(&k_str)?;
        }
        cfg.grid_size = opt_usize_prop(options, "gridSize", cfg.grid_size)?;
        cfg.padding_factor = opt_usize_prop(options, "paddingFactor", cfg.padding_factor)?;
        cfg.margin_ratio = opt_f64_prop(options, "marginRatio", cfg.margin_ratio)?;
        cfg.sigma_ratio = opt_f64_prop(options, "sigmaRatio", cfg.sigma_ratio)?;
        cfg.sigma = opt_maybe_f64_prop(options, "sigma")?;
        cfg.screening_length_ratio =
            opt_f64_prop(options, "screeningLengthRatio", cfg.screening_length_ratio)?;
        cfg.kappa = opt_maybe_f64_prop(options, "kappa")?;
        cfg.rho_max_factor = opt_f64_prop(options, "rhoMaxFactor", cfg.rho_max_factor)?;
        cfg.rho_max = opt_maybe_f64_prop(options, "rhoMax")?;
        cfg.exact_adjoint = opt_bool_prop(options, "exactAdjoint", cfg.exact_adjoint)?;
    }
    cfg.validate().map_err(core_err_to_js)?;
    Ok(cfg)
}

/// Output forces and telemetry produced by `ScreenedPoissonField2d` and `ScreenedPoissonRefiner`.
#[wasm_bindgen(js_name = ScreenedPoissonResult)]
#[derive(Debug, Clone)]
pub struct JsScreenedPoissonResult {
    force_x: Vec<f32>,
    force_y: Vec<f32>,
    stats: ScreenedPoissonStats,
}

impl JsScreenedPoissonResult {
    pub(crate) fn from_output(out: ScreenedPoissonOutput<f32>) -> Self {
        Self {
            force_x: out.force_x,
            force_y: out.force_y,
            stats: out.stats,
        }
    }

    pub(crate) fn from_stats(stats: ScreenedPoissonStats) -> Self {
        Self {
            force_x: Vec::new(),
            force_y: Vec::new(),
            stats,
        }
    }
}

#[wasm_bindgen(js_class = ScreenedPoissonResult)]
impl JsScreenedPoissonResult {
    /// Returns the per-node $x$-component descent forces (`Float32Array`).
    #[wasm_bindgen(js_name = forceX)]
    pub fn force_x(&self) -> Vec<f32> {
        self.force_x.clone()
    }

    /// Returns the per-node $y$-component descent forces (`Float32Array`).
    #[wasm_bindgen(js_name = forceY)]
    pub fn force_y(&self) -> Vec<f32> {
        self.force_y.clone()
    }

    /// Discrete Parseval variational energy $\mathcal{L}_{\mathrm{dens}} \ge 0$.
    #[wasm_bindgen(getter)]
    pub fn energy(&self) -> f64 {
        self.stats.energy
    }

    /// Peak smoothed density $\max_p \rho_V(p)$ on the grid.
    #[wasm_bindgen(getter, js_name = maxDensity)]
    pub fn max_density(&self) -> f64 {
        self.stats.max_density
    }

    /// Fraction of active interior grid cells where $\rho_V > \rho_{\max}$.
    #[wasm_bindgen(getter, js_name = excessFraction)]
    pub fn excess_fraction(&self) -> f64 {
        self.stats.excess_fraction
    }

    /// Integrated excess density mass $\int \delta\rho(p)\,dp \in [0, 1]$.
    #[wasm_bindgen(getter, js_name = excessMass)]
    pub fn excess_mass(&self) -> f64 {
        self.stats.excess_mass
    }

    /// Effective Gaussian smoothing width $\sigma_0$ used.
    #[wasm_bindgen(getter, js_name = sigmaUsed)]
    pub fn sigma_used(&self) -> f64 {
        self.stats.sigma_used
    }

    /// Effective screening wavenumber $\kappa_0$ used.
    #[wasm_bindgen(getter, js_name = kappaUsed)]
    pub fn kappa_used(&self) -> f64 {
        self.stats.kappa_used
    }

    /// Effective upper density threshold $\rho_{\max}$ used.
    #[wasm_bindgen(getter, js_name = rhoMaxUsed)]
    pub fn rho_max_used(&self) -> f64 {
        self.stats.rho_max_used
    }

    /// Active interior grid dimension $G$.
    #[wasm_bindgen(getter, js_name = gridDim)]
    pub fn grid_dim(&self) -> usize {
        self.stats.grid_dim
    }

    /// Padded FFT dimension $G_{\mathrm{pad}}$.
    #[wasm_bindgen(getter, js_name = padDim)]
    pub fn pad_dim(&self) -> usize {
        self.stats.pad_dim
    }

    /// Number of 2D R2C/C2R FFT calls executed (`0`, `2`, `4`, or `6`).
    #[wasm_bindgen(getter, js_name = fft2dCalls)]
    pub fn fft_2d_calls(&self) -> f64 {
        self.stats.fft_2d_calls as f64
    }

    /// `true` when zero grid cells exceeded $\rho_{\max}$ and the solve exited early.
    #[wasm_bindgen(getter, js_name = earlyExitNoExcess)]
    pub fn early_exit_no_excess(&self) -> bool {
        self.stats.early_exit_no_excess
    }
}

/// Reusable `f32` R2C-FFT Screened Poisson unilateral upper-bound 2D density field evaluator.
#[wasm_bindgen(js_name = ScreenedPoissonField2d)]
#[derive(Debug, Clone)]
pub struct JsScreenedPoissonField2d {
    inner: ScreenedPoissonField2d<f32>,
}

#[wasm_bindgen(js_class = ScreenedPoissonField2d)]
impl JsScreenedPoissonField2d {
    /// Allocates a `ScreenedPoissonField2d` evaluator workspace (`f32`-backed, default `gridSize = 512`).
    #[wasm_bindgen(constructor)]
    pub fn new(options: &JsValue) -> Result<JsScreenedPoissonField2d, JsError> {
        let cfg = parse_screened_poisson_config(options, "ScreenedPoissonField2d")?;
        let inner = ScreenedPoissonField2d::<f32>::new(cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Evaluates unilateral density energy and descent forces over `layout`'s adaptive bounding box.
    pub fn evaluate(&mut self, layout: &JsLayout2d) -> Result<JsScreenedPoissonResult, JsError> {
        let soa = layout.inner().to_layout2d();
        let mut out = ScreenedPoissonOutput::new();
        self.inner
            .evaluate(&soa, &mut out)
            .map_err(core_err_to_js)?;
        Ok(JsScreenedPoissonResult::from_output(out))
    }

    /// Evaluates unilateral density energy and descent forces inside explicit bounds
    /// `(minX, minY, maxX, maxY)`.
    #[wasm_bindgen(js_name = evaluateInBounds)]
    pub fn evaluate_in_bounds(
        &mut self,
        layout: &JsLayout2d,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Result<JsScreenedPoissonResult, JsError> {
        let soa = layout.inner().to_layout2d();
        let mut out = ScreenedPoissonOutput::new();
        self.inner
            .evaluate_in_bounds(&soa, [min_x, min_y, max_x, max_y], &mut out)
            .map_err(core_err_to_js)?;
        Ok(JsScreenedPoissonResult::from_output(out))
    }

    /// Returns a flat row-major `[densityGridDim * densityGridDim]` copy of the last evaluated
    /// smoothed density grid (`Float32Array`), or an empty `Float32Array` before evaluation.
    #[wasm_bindgen(js_name = densityGrid)]
    pub fn density_grid(&self) -> Result<Vec<f32>, JsError> {
        Ok(self
            .inner
            .density_grid()
            .map(|slice| slice.to_vec())
            .unwrap_or_default())
    }

    /// Returns the active density grid side length (`0` before evaluation or for empty layouts).
    #[wasm_bindgen(getter, js_name = densityGridDim)]
    pub fn density_grid_dim(&self) -> usize {
        self.inner.density_grid_dim()
    }

    /// Returns total heap bytes allocated by internal `f32` grids, bitmasks, and FFT tables.
    #[wasm_bindgen(js_name = workspaceMemoryBytes)]
    pub fn workspace_memory_bytes(&self) -> usize {
        self.inner.workspace_memory_bytes()
    }

    /// Returns the active interior grid dimension $G$.
    #[wasm_bindgen(getter, js_name = gridSize)]
    pub fn grid_size(&self) -> usize {
        self.inner.config().grid_size
    }

    /// Returns the zero-padding multiplier (`1` or `2`).
    #[wasm_bindgen(getter, js_name = paddingFactor)]
    pub fn padding_factor(&self) -> usize {
        self.inner.config().padding_factor
    }

    /// Returns the active kernel name (`"screenedPoisson"` or `"singleGaussian"`).
    #[wasm_bindgen(getter)]
    pub fn kernel(&self) -> String {
        density_field_kernel_js_str(self.inner.config().kernel).to_owned()
    }

    /// Returns whether exact variational adjoint masking is enabled.
    #[wasm_bindgen(getter, js_name = exactAdjoint)]
    pub fn exact_adjoint(&self) -> bool {
        self.inner.config().exact_adjoint
    }
}
