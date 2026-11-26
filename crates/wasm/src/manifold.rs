//! WebAssembly bindings for non-Euclidean `ManifoldLayout2d` (`f32`) and geodesic utilities.

use egraph::core::{
    CoreError, Layout2d, ManifoldKind, ManifoldLayout2d, poincare_distance, poincare_mobius_add,
    spherical_distance, torus_distance, wrap_torus_point,
};
use wasm_bindgen::prelude::*;

use crate::js_convert::{core_err_to_js, parse_non_negative_usize, parse_seed_u64};
use crate::js_options::normalize_enum_token;
use crate::layout_buffers::JsLayout2d;

fn parse_manifold_kind(raw: &str) -> Result<ManifoldKind, JsError> {
    match normalize_enum_token(raw).as_str() {
        "hyperbolic" | "hyperbolicpoincare" | "poincare" => Ok(ManifoldKind::HyperbolicPoincare),
        "spherical" | "sphere" => Ok(ManifoldKind::Spherical),
        "torus" | "torus2d" => Ok(ManifoldKind::Torus2d),
        _ => Err(JsError::new(&format!(
            "unknown manifold kind {raw:?}; expected 'hyperbolic', 'spherical', or 'torus'"
        ))),
    }
}

fn manifold_kind_str(kind: ManifoldKind) -> &'static str {
    match kind {
        ManifoldKind::HyperbolicPoincare => "hyperbolic",
        ManifoldKind::Spherical => "spherical",
        ManifoldKind::Torus2d => "torus",
        _ => "unknown",
    }
}

/// Non-Euclidean 2D node layout (`f32`) constrained to `"hyperbolic"`, `"spherical"`, or `"torus"`.
#[wasm_bindgen(js_name = ManifoldLayout2d)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsManifoldLayout2d {
    inner: ManifoldLayout2d<f32>,
}

impl JsManifoldLayout2d {
    pub(crate) fn inner(&self) -> &ManifoldLayout2d<f32> {
        &self.inner
    }

    pub(crate) fn inner_mut(&mut self) -> &mut ManifoldLayout2d<f32> {
        &mut self.inner
    }
}

#[wasm_bindgen(js_class = ManifoldLayout2d)]
impl JsManifoldLayout2d {
    /// Creates a zero-initialized `ManifoldLayout2d` for `n` vertices on geometry `kind`.
    pub fn zeros(kind: &str, n: f64) -> Result<JsManifoldLayout2d, JsError> {
        let m_kind = parse_manifold_kind(kind)?;
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self {
            inner: ManifoldLayout2d::<f32>::zeros(m_kind, count),
        })
    }

    /// Creates a zero-initialized Poincaré disk `ManifoldLayout2d` for `n` vertices.
    #[wasm_bindgen(js_name = hyperbolicPoincare)]
    pub fn hyperbolic_poincare(n: f64) -> Result<JsManifoldLayout2d, JsError> {
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self {
            inner: ManifoldLayout2d::<f32>::zeros(ManifoldKind::HyperbolicPoincare, count),
        })
    }

    /// Creates a zero-initialized spherical `ManifoldLayout2d` for `n` vertices.
    pub fn spherical(n: f64) -> Result<JsManifoldLayout2d, JsError> {
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self {
            inner: ManifoldLayout2d::<f32>::zeros(ManifoldKind::Spherical, count),
        })
    }

    /// Creates a zero-initialized flat unit torus `ManifoldLayout2d` for `n` vertices.
    #[wasm_bindgen(js_name = torus2d)]
    pub fn torus_2d(n: f64) -> Result<JsManifoldLayout2d, JsError> {
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self {
            inner: ManifoldLayout2d::<f32>::zeros(ManifoldKind::Torus2d, count),
        })
    }

    /// Creates a `ManifoldLayout2d` from a Euclidean `Layout2d`, projecting points onto `kind`.
    #[wasm_bindgen(js_name = fromLayout)]
    pub fn from_layout(kind: &str, layout: &JsLayout2d) -> Result<JsManifoldLayout2d, JsError> {
        let m_kind = parse_manifold_kind(kind)?;
        let soa = layout.inner().to_layout2d();
        let inner = ManifoldLayout2d::<f32>::from_layout(m_kind, soa).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Creates a `ManifoldLayout2d` from flat interleaved `[x0, y0, ...]` coordinates, projecting onto `kind`.
    #[wasm_bindgen(js_name = fromCoords)]
    pub fn from_coords(kind: &str, coords: &[f32]) -> Result<JsManifoldLayout2d, JsError> {
        let m_kind = parse_manifold_kind(kind)?;
        let soa = Layout2d::<f32>::from_interleaved(coords).map_err(core_err_to_js)?;
        let inner = ManifoldLayout2d::<f32>::from_layout(m_kind, soa).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Creates a deterministically seeded safe random initialization on `kind`.
    #[wasm_bindgen(js_name = randomInit)]
    pub fn random_init(
        kind: &str,
        n: f64,
        scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsManifoldLayout2d, JsError> {
        let m_kind = parse_manifold_kind(kind)?;
        let count = parse_non_negative_usize(n, "n")?;
        let scale_val = scale.unwrap_or(0.25);
        let seed_u64 = parse_seed_u64(seed)?;
        let inner = ManifoldLayout2d::<f32>::random_init(m_kind, count, scale_val, seed_u64)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Creates a `ManifoldLayout2d` from a Euclidean `Layout2d` scaled safely inside `maxRadius` (default `0.7`).
    #[wasm_bindgen(js_name = fromLayoutScaled)]
    pub fn from_layout_scaled(
        kind: &str,
        layout: &JsLayout2d,
        max_radius: Option<f64>,
    ) -> Result<JsManifoldLayout2d, JsError> {
        let m_kind = parse_manifold_kind(kind)?;
        let mr = max_radius.unwrap_or(0.7);
        let soa = layout.inner().to_layout2d();
        let inner = ManifoldLayout2d::<f32>::from_layout_scaled(m_kind, &soa, mr)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the canonical manifold kind string: `"hyperbolic"`, `"spherical"`, or `"torus"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        manifold_kind_str(self.inner.kind()).to_owned()
    }

    /// Alias for `kind` matching `ManifoldLayout2d.manifold`.
    #[wasm_bindgen(getter)]
    pub fn manifold(&self) -> String {
        manifold_kind_str(self.inner.kind()).to_owned()
    }

    /// Returns the number of vertices in the manifold layout.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of vertices in the manifold layout.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the manifold layout contains zero vertices.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a flat interleaved `[x0, y0, ...]` `Float32Array` copy of all coordinates.
    pub fn coords(&self) -> Vec<f32> {
        self.inner.as_layout().to_interleaved()
    }

    /// Converts this manifold layout to a `Layout2d` copy.
    #[wasm_bindgen(js_name = toLayout)]
    pub fn to_layout(&self) -> JsLayout2d {
        JsLayout2d::from_layout2d(self.inner.as_layout())
    }

    /// Computes the exact geodesic distance between vertices `i` and `j`.
    #[wasm_bindgen(js_name = geodesicDistance)]
    pub fn geodesic_distance(&self, i: f64, j: f64) -> Result<f32, JsError> {
        let ii = parse_non_negative_usize(i, "i")?;
        let jj = parse_non_negative_usize(j, "j")?;
        self.inner.geodesic_distance(ii, jj).map_err(core_err_to_js)
    }

    /// Computes the Riemannian logarithmic map `[vx, vy]` at vertex `i` pointing toward `j`.
    #[wasm_bindgen(js_name = tangentDelta)]
    pub fn tangent_delta(&self, i: f64, j: f64) -> Result<Vec<f32>, JsError> {
        let ii = parse_non_negative_usize(i, "i")?;
        let jj = parse_non_negative_usize(j, "j")?;
        let (vx, vy) = self.inner.tangent_delta(ii, jj).map_err(core_err_to_js)?;
        Ok(vec![vx, vy])
    }

    /// Advances vertex `i` along the geodesic defined by tangent vector `(vx, vy)`.
    #[wasm_bindgen(js_name = applyTangentStep)]
    pub fn apply_tangent_step(&mut self, i: f64, vx: f32, vy: f32) -> Result<(), JsError> {
        let idx = parse_non_negative_usize(i, "i")?;
        self.inner
            .apply_tangent_step(idx, vx, vy)
            .map_err(core_err_to_js)
    }

    /// Re-projects all coordinates onto the manifold's canonical domain after checking finiteness.
    #[wasm_bindgen(js_name = projectAll)]
    pub fn project_all(&mut self) -> Result<(), JsError> {
        self.inner.project_all().map_err(core_err_to_js)
    }

    /// Returns the first coordinate (`x` or `lon`) of vertex `index`.
    pub fn x(&self, index: f64) -> Result<f32, JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        if idx >= self.inner.len() {
            return Err(core_err_to_js(CoreError::index_out_of_bounds(
                idx,
                self.inner.len(),
                "ManifoldLayout2d::x",
            )));
        }
        Ok(self.inner.x()[idx])
    }

    /// Returns the second coordinate (`y` or `colat`) of vertex `index`.
    pub fn y(&self, index: f64) -> Result<f32, JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        if idx >= self.inner.len() {
            return Err(core_err_to_js(CoreError::index_out_of_bounds(
                idx,
                self.inner.len(),
                "ManifoldLayout2d::y",
            )));
        }
        Ok(self.inner.y()[idx])
    }

    /// Sets vertex `index`'s coordinates `(x, y)` and projects onto the manifold domain.
    #[wasm_bindgen(js_name = setPoint)]
    pub fn set_point(&mut self, index: f64, x: f32, y: f32) -> Result<(), JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.set_point(idx, x, y).map_err(core_err_to_js)
    }

    /// Returns `true` if every coordinate in the manifold layout is finite.
    #[wasm_bindgen(js_name = isAllFinite)]
    pub fn is_all_finite(&self) -> bool {
        self.inner.as_layout().is_all_finite()
    }

    /// Computes the closed-form Möbius-equivariant Einstein midpoint `[cx, cy]` on the Poincaré disk.
    #[wasm_bindgen(js_name = poincareCentroid)]
    pub fn poincare_centroid(&self) -> Result<Vec<f64>, JsError> {
        let (cx, cy) = self.inner.poincare_centroid().map_err(core_err_to_js)?;
        Ok(vec![f64::from(cx), f64::from(cy)])
    }

    /// Isometrically recenters all Poincaré disk points at `(cx, cy)` via Möbius automorphism.
    #[wasm_bindgen(js_name = recenterPoincare)]
    pub fn recenter_poincare(&mut self, cx: f64, cy: f64) -> Result<(), JsError> {
        if !cx.is_finite() || !cy.is_finite() {
            return Err(core_err_to_js(CoreError::non_finite(
                "ManifoldLayout2d::recenterPoincare",
            )));
        }
        let cx32 = cx as f32;
        let cy32 = cy as f32;
        if !cx32.is_finite() || !cy32.is_finite() {
            return Err(core_err_to_js(CoreError::invalid_input(
                "recenterPoincare coordinate overflows float32",
            )));
        }
        self.inner
            .recenter_poincare(cx32, cy32)
            .map_err(core_err_to_js)
    }

    /// Isometrically recenters the Poincaré disk layout so vertex `index` moves to `(0, 0)`.
    #[wasm_bindgen(js_name = recenterPoincareAt)]
    pub fn recenter_poincare_at(&mut self, index: f64) -> Result<(), JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.recenter_poincare_at(idx).map_err(core_err_to_js)
    }

    /// Isometrically recenters the Poincaré disk layout so its Einstein midpoint lies at `(0, 0)`.
    #[wasm_bindgen(js_name = centerPoincareCentroid)]
    pub fn center_poincare_centroid(&mut self) -> Result<(), JsError> {
        self.inner
            .center_poincare_centroid()
            .map_err(core_err_to_js)
    }
}

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
