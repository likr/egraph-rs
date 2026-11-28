//! WebAssembly bindings for non-Euclidean `ManifoldLayout2d` (`f32`) with shadow-synced zero-copy views.

use egraph::core::{CoreError, Layout2d, ManifoldKind, ManifoldLayout2d};
use egraph_ffi::InterleavedLayout2d;
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    DEFAULT_SEED, core_err_to_js, ffi_err_to_js, invalid_input_err, parse_non_negative_usize,
    parse_seed_u64,
};
use crate::js_options::{check_options_arg, normalize_enum_token, opt_f64_prop, opt_seed_prop};
use crate::layout_buffers::JsLayout2d;

pub(crate) fn parse_manifold_kind(raw: &str) -> Result<ManifoldKind, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "hyperbolic" | "hyperbolicpoincare" | "poincare" => Ok(ManifoldKind::HyperbolicPoincare),
        "hyperboliclorentz" | "lorentz" | "hyperboloid" => Ok(ManifoldKind::HyperbolicLorentz),
        "spherical" | "sphere" => Ok(ManifoldKind::Spherical),
        "torus" | "torus2d" => Ok(ManifoldKind::Torus2d),
        _ => Err(invalid_input_err(format!(
            "unknown manifold kind {raw:?}; expected 'hyperbolicPoincare', 'hyperbolicLorentz', 'spherical', or 'torus'"
        ))),
    }
}

pub(crate) fn manifold_kind_str(kind: ManifoldKind) -> &'static str {
    match kind {
        ManifoldKind::HyperbolicPoincare => "hyperbolicPoincare",
        ManifoldKind::HyperbolicLorentz => "hyperbolicLorentz",
        ManifoldKind::Spherical => "spherical",
        ManifoldKind::Torus2d => "torus",
        _ => "unknown",
    }
}

/// Non-Euclidean 2D node layout (`f32`) constrained to `"hyperbolicPoincare"`,
/// `"hyperbolicLorentz"`, `"spherical"`, or `"torus"`, exposing zero-copy `coordsView()` via a
/// synced interleaved mirror.
#[wasm_bindgen(js_name = ManifoldLayout2d)]
#[derive(Debug, Clone)]
pub struct JsManifoldLayout2d {
    inner: ManifoldLayout2d<f32>,
    shadow: InterleavedLayout2d<f32>,
}

impl JsManifoldLayout2d {
    pub(crate) fn from_inner(inner: ManifoldLayout2d<f32>) -> Self {
        let shadow = InterleavedLayout2d::from_layout2d(inner.as_layout());
        Self { inner, shadow }
    }

    pub(crate) fn inner(&self) -> &ManifoldLayout2d<f32> {
        &self.inner
    }

    pub(crate) fn inner_mut(&mut self) -> &mut ManifoldLayout2d<f32> {
        &mut self.inner
    }

    pub(crate) fn sync_shadow(&mut self) -> Result<(), JsValue> {
        self.shadow
            .copy_from_layout2d(self.inner.as_layout())
            .map_err(ffi_err_to_js)
    }
}

#[wasm_bindgen(js_class = ManifoldLayout2d)]
impl JsManifoldLayout2d {
    /// Creates a zero-initialized `ManifoldLayout2d` for `n` vertices on geometry `kind`.
    pub fn zeros(kind: &str, n: f64) -> Result<JsManifoldLayout2d, JsValue> {
        let m_kind = parse_manifold_kind(kind)?;
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self::from_inner(ManifoldLayout2d::<f32>::zeros(
            m_kind, count,
        )))
    }

    /// Creates a deterministically seeded `ManifoldLayout2d` for `n` vertices on geometry `kind`.
    pub fn init(kind: &str, n: f64, options: &JsValue) -> Result<JsManifoldLayout2d, JsValue> {
        let m_kind = parse_manifold_kind(kind)?;
        let count = parse_non_negative_usize(n, "n")?;
        let (seed_u64, scale_val) = if check_options_arg(options, "ManifoldLayout2d.init")? {
            (
                opt_seed_prop(options)?,
                opt_f64_prop(options, "scale", 0.25)?,
            )
        } else {
            (DEFAULT_SEED, 0.25)
        };
        let inner = ManifoldLayout2d::<f32>::init(m_kind, count, seed_u64, scale_val)
            .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Creates a zero-initialized Poincaré disk `ManifoldLayout2d` for `n` vertices.
    #[wasm_bindgen(js_name = hyperbolicPoincare)]
    pub fn hyperbolic_poincare(n: f64) -> Result<JsManifoldLayout2d, JsValue> {
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self::from_inner(ManifoldLayout2d::<f32>::zeros(
            ManifoldKind::HyperbolicPoincare,
            count,
        )))
    }

    /// Creates a zero-initialized Lorentz hyperboloid `ManifoldLayout2d` for `n` vertices.
    #[wasm_bindgen(js_name = hyperbolicLorentz)]
    pub fn hyperbolic_lorentz(n: f64) -> Result<JsManifoldLayout2d, JsValue> {
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self::from_inner(ManifoldLayout2d::<f32>::zeros(
            ManifoldKind::HyperbolicLorentz,
            count,
        )))
    }

    /// Creates a zero-initialized spherical `ManifoldLayout2d` for `n` vertices.
    pub fn spherical(n: f64) -> Result<JsManifoldLayout2d, JsValue> {
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self::from_inner(ManifoldLayout2d::<f32>::zeros(
            ManifoldKind::Spherical,
            count,
        )))
    }

    /// Creates a zero-initialized flat unit torus `ManifoldLayout2d` for `n` vertices.
    #[wasm_bindgen(js_name = torus2d)]
    pub fn torus_2d(n: f64) -> Result<JsManifoldLayout2d, JsValue> {
        let count = parse_non_negative_usize(n, "n")?;
        Ok(Self::from_inner(ManifoldLayout2d::<f32>::zeros(
            ManifoldKind::Torus2d,
            count,
        )))
    }

    /// Initializes a flat 2D torus layout via 4-mode spectral phase coordinates of $L_{\mathrm{sym}}$.
    #[wasm_bindgen(js_name = torusSpectralPhaseInit)]
    pub fn torus_spectral_phase_init(
        graph: &JsCsrGraph,
        seed: &JsValue,
    ) -> Result<JsManifoldLayout2d, JsValue> {
        let seed_u64 = parse_seed_u64(seed)?;
        let inner = ManifoldLayout2d::<f32>::torus_spectral_phase_init(graph.inner(), seed_u64)
            .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Creates a `ManifoldLayout2d` from a Euclidean `Layout2d`, projecting points onto `kind`.
    #[wasm_bindgen(js_name = fromLayout)]
    pub fn from_layout(kind: &str, layout: &JsLayout2d) -> Result<JsManifoldLayout2d, JsValue> {
        let m_kind = parse_manifold_kind(kind)?;
        let soa = layout.inner().to_layout2d();
        let inner = ManifoldLayout2d::<f32>::from_layout(m_kind, soa).map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Creates a `ManifoldLayout2d` from flat interleaved `[x0, y0, ...]` coordinates, projecting onto `kind`.
    #[wasm_bindgen(js_name = fromCoords)]
    pub fn from_coords(kind: &str, coords: &[f32]) -> Result<JsManifoldLayout2d, JsValue> {
        let m_kind = parse_manifold_kind(kind)?;
        let soa = Layout2d::<f32>::from_interleaved(coords).map_err(core_err_to_js)?;
        let inner = ManifoldLayout2d::<f32>::from_layout(m_kind, soa).map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Creates a deterministically seeded safe random initialization on `kind`.
    #[wasm_bindgen(js_name = randomInit)]
    pub fn random_init(
        kind: &str,
        n: f64,
        options: &JsValue,
    ) -> Result<JsManifoldLayout2d, JsValue> {
        let m_kind = parse_manifold_kind(kind)?;
        let count = parse_non_negative_usize(n, "n")?;
        let (scale_val, seed_u64) = if check_options_arg(options, "ManifoldLayout2d.randomInit")? {
            (
                opt_f64_prop(options, "scale", 0.25)?,
                opt_seed_prop(options)?,
            )
        } else {
            (0.25, DEFAULT_SEED)
        };
        let inner = ManifoldLayout2d::<f32>::random_init(m_kind, count, scale_val, seed_u64)
            .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Creates a `ManifoldLayout2d` from a Euclidean `Layout2d` scaled safely inside `maxRadius` (default `0.7`).
    #[wasm_bindgen(js_name = fromLayoutScaled)]
    pub fn from_layout_scaled(
        kind: &str,
        layout: &JsLayout2d,
        max_radius: Option<f64>,
    ) -> Result<JsManifoldLayout2d, JsValue> {
        let m_kind = parse_manifold_kind(kind)?;
        let mr = max_radius.unwrap_or(0.7);
        let soa = layout.inner().to_layout2d();
        let inner = ManifoldLayout2d::<f32>::from_layout_scaled(m_kind, &soa, mr)
            .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Returns a deep copy of this `ManifoldLayout2d`.
    pub fn copy(&self) -> JsManifoldLayout2d {
        self.clone()
    }

    /// Returns the canonical manifold kind string: `"hyperbolicPoincare"`, `"hyperbolicLorentz"`, `"spherical"`, or `"torus"`.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        manifold_kind_str(self.inner.kind()).to_owned()
    }

    /// Returns the physical geodesic scale factor (`> 0`).
    #[wasm_bindgen(getter)]
    pub fn scale(&self) -> f64 {
        self.inner.scale()
    }

    /// Returns the physical aspect ratio (`Lx / Ly` on `Torus2d`, `> 0`).
    #[wasm_bindgen(getter)]
    pub fn aspect(&self) -> f64 {
        self.inner.aspect()
    }

    /// Returns `{ scale, aspect }` metric parameters.
    pub fn metric(&self) -> Result<JsValue, JsValue> {
        let m = self.inner.metric();
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("scale"),
            &JsValue::from_f64(m.scale),
        )
        .map_err(|e| invalid_input_err(format!("failed to set metric.scale: {e:?}")))?;
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("aspect"),
            &JsValue::from_f64(m.aspect),
        )
        .map_err(|e| invalid_input_err(format!("failed to set metric.aspect: {e:?}")))?;
        Ok(obj.into())
    }

    /// Sets the physical metric scale factor (`> 0`) without mutating stored coordinates.
    #[wasm_bindgen(js_name = setScale)]
    pub fn set_scale(&mut self, scale: f64) -> Result<(), JsValue> {
        self.inner.set_scale(scale).map_err(core_err_to_js)
    }

    /// Sets the physical aspect ratio (`> 0`) without mutating stored coordinates.
    #[wasm_bindgen(js_name = setAspect)]
    pub fn set_aspect(&mut self, aspect: f64) -> Result<(), JsValue> {
        self.inner.set_aspect(aspect).map_err(core_err_to_js)
    }

    /// Sets physical rectangular torus extents `(lx, ly)` without mutating `[0, 1)^2` coordinates.
    #[wasm_bindgen(js_name = setTorusExtents)]
    pub fn set_torus_extents(&mut self, lx: f64, ly: f64) -> Result<(), JsValue> {
        self.inner.set_torus_extents(lx, ly).map_err(core_err_to_js)
    }

    /// Updates the geodesic scale to `newScale`, rescaling hyperbolic points radially around the Einstein midpoint.
    #[wasm_bindgen(js_name = rescaleGeodesic)]
    pub fn rescale_geodesic(&mut self, new_scale: f64) -> Result<(), JsValue> {
        self.inner
            .rescale_geodesic(new_scale)
            .map_err(core_err_to_js)?;
        self.sync_shadow()
    }

    /// Converts a hyperbolic Lorentz or Poincaré layout into a Poincaré disk `ManifoldLayout2d`.
    #[wasm_bindgen(js_name = toPoincareLayout)]
    pub fn to_poincare_layout(&self) -> Result<JsManifoldLayout2d, JsValue> {
        let inner = self.inner.to_poincare_layout().map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Returns a rigidly panned `Torus2d` `ManifoldLayout2d` minimizing boundary-wrapped edges.
    #[wasm_bindgen(js_name = torusAutoPan)]
    pub fn torus_auto_pan(&self, graph: &JsCsrGraph) -> Result<JsManifoldLayout2d, JsValue> {
        let inner = self
            .inner
            .torus_auto_pan(graph.inner())
            .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
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

    /// Returns the byte offset of the synced interleaved `f32` shadow buffer in linear memory.
    #[wasm_bindgen(js_name = coordsPtr)]
    pub fn coords_ptr(&self) -> usize {
        self.shadow.coords_ptr() as usize
    }

    /// Returns the total scalar length of the synced interleaved shadow buffer (`2 * len()`).
    #[wasm_bindgen(js_name = coordsLen)]
    pub fn coords_len(&self) -> usize {
        self.shadow.coords_len()
    }

    /// Returns a flat interleaved `[x0, y0, ...]` `Float32Array` copy of all coordinates.
    pub fn coords(&self) -> Vec<f32> {
        self.shadow.as_slice().to_vec()
    }

    /// Converts this manifold layout to a `Layout2d` copy.
    #[wasm_bindgen(js_name = toLayout)]
    pub fn to_layout(&self) -> JsLayout2d {
        JsLayout2d::from_layout2d(self.inner.as_layout())
    }

    /// Centers and scales coordinates to `targetStd` (default `1.0`) then projects onto the manifold domain.
    #[wasm_bindgen(js_name = normalizeStd)]
    pub fn normalize_std(&mut self, target_std: Option<f64>) -> Result<(), JsValue> {
        let t_std = target_std.unwrap_or(1.0);
        if !t_std.is_finite() || t_std <= 0.0 {
            return Err(core_err_to_js(CoreError::invalid_input(
                "targetStd must be finite and strictly positive",
            )));
        }
        let mut soa = self.inner.as_layout().clone();
        soa.center_and_normalize_to_std(t_std)
            .map_err(core_err_to_js)?;
        let metric = self.inner.metric();
        let kind = self.inner.kind();
        let mut updated =
            ManifoldLayout2d::<f32>::from_layout(kind, soa).map_err(core_err_to_js)?;
        updated.set_scale(metric.scale).map_err(core_err_to_js)?;
        updated.set_aspect(metric.aspect).map_err(core_err_to_js)?;
        self.inner = updated;
        self.sync_shadow()
    }

    /// Computes the exact geodesic distance between vertices `i` and `j`.
    #[wasm_bindgen(js_name = geodesicDistance)]
    pub fn geodesic_distance(&self, i: f64, j: f64) -> Result<f32, JsValue> {
        let ii = parse_non_negative_usize(i, "i")?;
        let jj = parse_non_negative_usize(j, "j")?;
        self.inner.geodesic_distance(ii, jj).map_err(core_err_to_js)
    }

    /// Computes the Riemannian logarithmic map `[vx, vy]` at vertex `i` pointing toward `j`.
    #[wasm_bindgen(js_name = tangentDelta)]
    pub fn tangent_delta(&self, i: f64, j: f64) -> Result<Vec<f32>, JsValue> {
        let ii = parse_non_negative_usize(i, "i")?;
        let jj = parse_non_negative_usize(j, "j")?;
        let (vx, vy) = self.inner.tangent_delta(ii, jj).map_err(core_err_to_js)?;
        Ok(vec![vx, vy])
    }

    /// Advances vertex `i` along the geodesic defined by tangent vector `(vx, vy)`.
    #[wasm_bindgen(js_name = applyTangentStep)]
    pub fn apply_tangent_step(&mut self, i: f64, vx: f32, vy: f32) -> Result<(), JsValue> {
        let idx = parse_non_negative_usize(i, "i")?;
        self.inner
            .apply_tangent_step(idx, vx, vy)
            .map_err(core_err_to_js)?;
        self.sync_shadow()
    }

    /// Re-projects all coordinates onto the manifold's canonical domain after checking finiteness.
    #[wasm_bindgen(js_name = projectAll)]
    pub fn project_all(&mut self) -> Result<(), JsValue> {
        self.inner.project_all().map_err(core_err_to_js)?;
        self.sync_shadow()
    }

    /// Returns the first coordinate (`x` or `lon`) of vertex `index`.
    pub fn x(&self, index: f64) -> Result<f32, JsValue> {
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
    pub fn y(&self, index: f64) -> Result<f32, JsValue> {
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
    pub fn set_point(&mut self, index: f64, x: f32, y: f32) -> Result<(), JsValue> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner.set_point(idx, x, y).map_err(core_err_to_js)?;
        self.sync_shadow()
    }

    /// Returns `true` if every coordinate in the manifold layout is finite.
    #[wasm_bindgen(js_name = isAllFinite)]
    pub fn is_all_finite(&self) -> bool {
        self.inner.as_layout().is_all_finite()
    }

    /// Computes the closed-form Möbius-equivariant Einstein midpoint `[cx, cy]` on the Poincaré disk.
    #[wasm_bindgen(js_name = poincareCentroid)]
    pub fn poincare_centroid(&self) -> Result<Vec<f64>, JsValue> {
        let (cx, cy) = self.inner.poincare_centroid().map_err(core_err_to_js)?;
        Ok(vec![f64::from(cx), f64::from(cy)])
    }

    /// Isometrically recenters all Poincaré disk points at `(cx, cy)` via Möbius automorphism.
    #[wasm_bindgen(js_name = recenterPoincare)]
    pub fn recenter_poincare(&mut self, cx: f64, cy: f64) -> Result<(), JsValue> {
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
            .map_err(core_err_to_js)?;
        self.sync_shadow()
    }

    /// Isometrically recenters the Poincaré disk layout so vertex `index` moves to `(0, 0)`.
    #[wasm_bindgen(js_name = recenterPoincareAt)]
    pub fn recenter_poincare_at(&mut self, index: f64) -> Result<(), JsValue> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner
            .recenter_poincare_at(idx)
            .map_err(core_err_to_js)?;
        self.sync_shadow()
    }

    /// Isometrically recenters the Poincaré disk layout so its Einstein midpoint lies at `(0, 0)`.
    #[wasm_bindgen(js_name = centerPoincareCentroid)]
    pub fn center_poincare_centroid(&mut self) -> Result<(), JsValue> {
        self.inner
            .center_poincare_centroid()
            .map_err(core_err_to_js)?;
        self.sync_shadow()
    }
}
