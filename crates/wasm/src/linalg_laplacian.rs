//! WebAssembly bindings for CSR graph Laplacians and Chebyshev shifted Laplacians.

use egraph::linalg::{
    LaplacianKind, combinatorial_laplacian, laplacian, random_walk_laplacian,
    shifted_normalized_laplacian, symmetric_normalized_laplacian,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, invalid_input_err, set_js_prop};
use crate::js_options::normalize_enum_token;
use crate::linalg_matrix::JsCsrMatrix;

fn parse_laplacian_kind(raw: &str) -> Result<LaplacianKind, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "combinatorial" | "unnormalized" | "standard" => Ok(LaplacianKind::Combinatorial),
        "symmetricnormalized" | "symnorm" | "normalized" | "sym" => {
            Ok(LaplacianKind::SymmetricNormalized)
        }
        "randomwalk" | "rw" => Ok(LaplacianKind::RandomWalk),
        _ => Err(invalid_input_err(format!(
            "unknown Laplacian kind {raw:?}; expected 'combinatorial', 'symmetricNormalized', or 'randomWalk'"
        ))),
    }
}

/// Builds a `CsrMatrix` graph Laplacian of the requested `kind` (default `"symmetricNormalized"`).
#[wasm_bindgen(js_name = laplacian)]
pub fn js_laplacian(graph: &JsCsrGraph, kind: Option<String>) -> Result<JsCsrMatrix, JsValue> {
    let k = match kind {
        Some(raw) => parse_laplacian_kind(&raw)?,
        None => LaplacianKind::SymmetricNormalized,
    };
    Ok(JsCsrMatrix::from_inner(laplacian(graph.inner(), k)))
}

/// Builds the combinatorial graph Laplacian $L = D - A$ as a `CsrMatrix`.
#[wasm_bindgen(js_name = combinatorialLaplacian)]
pub fn js_combinatorial_laplacian(graph: &JsCsrGraph) -> JsCsrMatrix {
    JsCsrMatrix::from_inner(combinatorial_laplacian(graph.inner()))
}

/// Builds the symmetric normalized Laplacian $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$
/// and inverse square-root degrees `{ matrix, invSqrtDeg }`.
#[wasm_bindgen(js_name = symmetricNormalizedLaplacian)]
pub fn js_symmetric_normalized_laplacian(graph: &JsCsrGraph) -> Result<JsValue, JsValue> {
    let (mat, inv_sqrt_deg) = symmetric_normalized_laplacian(graph.inner());
    let obj = js_sys::Object::new();
    set_js_prop(&obj, "matrix", &JsCsrMatrix::from_inner(mat).into())?;
    let inv_arr = js_sys::Float64Array::from(inv_sqrt_deg.as_slice());
    set_js_prop(&obj, "invSqrtDeg", &inv_arr.into())?;
    Ok(obj.into())
}

/// Builds the random-walk normalized graph Laplacian $L_{\mathrm{rw}} = D^{-1}(D - A)$ as a `CsrMatrix`.
#[wasm_bindgen(js_name = randomWalkLaplacian)]
pub fn js_random_walk_laplacian(graph: &JsCsrGraph) -> JsCsrMatrix {
    JsCsrMatrix::from_inner(random_walk_laplacian(graph.inner()))
}

/// Builds the Chebyshev-scaled shifted Laplacian $\frac{2}{\lambda_{\max}} L_{\mathrm{sym}} - I$.
#[wasm_bindgen(js_name = shiftedNormalizedLaplacian)]
pub fn js_shifted_normalized_laplacian(
    l_sym: &JsCsrMatrix,
    lambda_max: Option<f64>,
) -> Result<JsCsrMatrix, JsValue> {
    let mat = shifted_normalized_laplacian(l_sym.inner(), lambda_max.unwrap_or(2.0))
        .map_err(core_err_to_js)?;
    Ok(JsCsrMatrix::from_inner(mat))
}
