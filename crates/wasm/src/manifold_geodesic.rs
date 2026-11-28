//! WebAssembly geodesic distance, midpoint, exponential/logarithmic map, and torus
//! spectral-phase bindings across hyperbolic, spherical, and flat-torus manifolds.

use egraph::core::{
    ManifoldLayout2d, lorentz_distance, lorentz_einstein_midpoint, lorentz_exp_map,
    lorentz_log_map, lorentz_to_poincare_point, poincare_distance, poincare_einstein_midpoint,
    poincare_mobius_add, spherical_distance, torus_distance, wrap_torus_point,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_seed_u64};
use crate::manifold::JsManifoldLayout2d;

/// Computes the Poincaré disk hyperbolic geodesic distance between `(x1, y1)` and `(x2, y2)`.
#[wasm_bindgen(js_name = poincareDistance)]
pub fn js_poincare_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    poincare_distance(x1, y1, x2, y2)
}

/// Computes Poincaré disk Möbius addition `(x1, y1) (+) (x2, y2)` inside the open unit disk, returning `[x, y]`.
#[wasm_bindgen(js_name = poincareMobiusAdd)]
pub fn js_poincare_mobius_add(x1: f64, y1: f64, x2: f64, y2: f64) -> Vec<f64> {
    let (rx, ry) = poincare_mobius_add(x1, y1, x2, y2);
    vec![rx, ry]
}

/// Computes the closed-form Möbius-equivariant Einstein midpoint `[cx, cy]` of Poincaré disk points `(xs, ys)`.
#[wasm_bindgen(js_name = poincareEinsteinMidpoint)]
pub fn js_poincare_einstein_midpoint(xs: &[f64], ys: &[f64]) -> Result<Vec<f64>, JsValue> {
    let (cx, cy) = poincare_einstein_midpoint(xs, ys).map_err(core_err_to_js)?;
    Ok(vec![cx, cy])
}

/// Computes the Lorentz hyperboloid geodesic distance between `(x1, y1)` and `(x2, y2)`.
#[wasm_bindgen(js_name = lorentzDistance)]
pub fn js_lorentz_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    lorentz_distance(x1, y1, x2, y2)
}

/// Stereographically maps Lorentz hyperboloid spatial coordinates `(x, y)` to Poincaré disk `[px, py]`.
#[wasm_bindgen(js_name = lorentzToPoincarePoint)]
pub fn js_lorentz_to_poincare_point(x: f64, y: f64) -> Vec<f64> {
    let (px, py) = lorentz_to_poincare_point(x, y);
    vec![px, py]
}

/// Evaluates the Lorentz hyperboloid exponential map at `(x, y)` with tangent `(vx, vy)`, returning `[nx, ny]`.
#[wasm_bindgen(js_name = lorentzExpMap)]
pub fn js_lorentz_exp_map(x: f64, y: f64, vx: f64, vy: f64) -> Vec<f64> {
    let (nx, ny) = lorentz_exp_map(x, y, vx, vy);
    vec![nx, ny]
}

/// Evaluates the Lorentz hyperboloid logarithmic map at `(x1, y1)` pointing toward `(x2, y2)`, returning `[vx, vy]`.
#[wasm_bindgen(js_name = lorentzLogMap)]
pub fn js_lorentz_log_map(x1: f64, y1: f64, x2: f64, y2: f64) -> Vec<f64> {
    let (vx, vy) = lorentz_log_map(x1, y1, x2, y2);
    vec![vx, vy]
}

/// Computes the Einstein midpoint `[cx, cy]` of Lorentz hyperboloid spatial coordinates `(xs, ys)`.
#[wasm_bindgen(js_name = lorentzEinsteinMidpoint)]
pub fn js_lorentz_einstein_midpoint(xs: &[f64], ys: &[f64]) -> Result<Vec<f64>, JsValue> {
    let (cx, cy) = lorentz_einstein_midpoint(xs, ys).map_err(core_err_to_js)?;
    Ok(vec![cx, cy])
}

/// Computes the great-circle geodesic distance on the unit sphere in `[0, pi]`.
#[wasm_bindgen(js_name = sphericalDistance)]
pub fn js_spherical_distance(lon1: f64, colat1: f64, lon2: f64, colat2: f64) -> f64 {
    spherical_distance(lon1, colat1, lon2, colat2)
}

/// Computes the geodesic distance on the flat unit torus `[0, 1)^2`.
#[wasm_bindgen(js_name = torusDistance)]
pub fn js_torus_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    torus_distance(x1, y1, x2, y2)
}

/// Wraps a 2D coordinate pair onto the flat unit torus `[0, 1)^2`, returning `[x, y]`.
#[wasm_bindgen(js_name = wrapTorusPoint)]
pub fn js_wrap_torus_point(x: f64, y: f64) -> Vec<f64> {
    let (wx, wy) = wrap_torus_point(x, y);
    vec![wx, wy]
}

/// Initializes a flat 2D torus `ManifoldLayout2d` via 4-mode spectral phase coordinates of $L_{\mathrm{sym}}$.
#[wasm_bindgen(js_name = torusSpectralPhaseInit)]
pub fn js_torus_spectral_phase_init(
    graph: &JsCsrGraph,
    seed: &JsValue,
) -> Result<JsManifoldLayout2d, JsValue> {
    let seed_u64 = parse_seed_u64(seed)?;
    let inner = ManifoldLayout2d::<f32>::torus_spectral_phase_init(graph.inner(), seed_u64)
        .map_err(core_err_to_js)?;
    Ok(JsManifoldLayout2d::from_inner(inner))
}
