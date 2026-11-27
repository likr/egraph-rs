//! WebAssembly bindings for zero-allocation 3D Barnes-Hut `ArenaOctree3d` and arbitrary-dimensional `DynamicKdTree`.

use egraph::core::CoreError;
use egraph::spatial::{ArenaOctree3d, DynamicKdTree};
use wasm_bindgen::prelude::*;

use crate::js_convert::{
    core_err_to_js, parse_f32_sequence, parse_f64_sequence, parse_non_negative_usize, set_f64_prop,
    set_js_prop,
};
use crate::layout_buffers::JsLayoutNd;

const DEFAULT_THETA: f32 = 0.5;
const DEFAULT_EPS_SQ: f32 = 1e-6;
const DEFAULT_MAXENT_Q: f64 = 1.0;
const DEFAULT_ENTROPY_EPS_R: f32 = 0.05;

fn kd_hits_to_js_array(hits: &[(usize, f64)]) -> Result<js_sys::Array, JsError> {
    let arr = js_sys::Array::new_with_length(hits.len() as u32);
    for (pos, &(idx, dist_sq)) in hits.iter().enumerate() {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "index", idx as f64)?;
        set_f64_prop(&obj, "distSq", dist_sq)?;
        arr.set(pos as u32, obj.into());
    }
    Ok(arr)
}

/// Reusable single-threaded 3D Barnes-Hut octree (`f32`) for Coulomb, generalized MaxEnt,
/// Student-$t$ KL, and entropy repulsive force approximations.
#[wasm_bindgen(js_name = ArenaOctree3d)]
#[derive(Debug, Clone)]
pub struct JsArenaOctree3d {
    inner: ArenaOctree3d<f32>,
}

impl Default for JsArenaOctree3d {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen(js_class = ArenaOctree3d)]
impl JsArenaOctree3d {
    /// Constructs an empty reusable `ArenaOctree3d` workspace.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: ArenaOctree3d::new(),
        }
    }

    /// Returns the number of points currently indexed in the octree.
    #[wasm_bindgen(getter, js_name = pointCount)]
    pub fn point_count(&self) -> usize {
        self.inner.point_count()
    }

    /// Returns the total number of internal and leaf octree nodes allocated.
    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Clears indexed points while retaining allocated octree node and coordinate capacities.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Builds the octree from flat `[x0, y0, z0, ...]` coordinates (unit mass `1.0` per point).
    #[wasm_bindgen(js_name = _buildFlat)]
    pub fn build(&mut self, coords: &JsValue) -> Result<(), JsError> {
        let flat = parse_f32_sequence(coords, "coords")?;
        if !flat.len().is_multiple_of(3) {
            return Err(core_err_to_js(CoreError::dimension_mismatch(
                flat.len() + (3 - flat.len() % 3),
                flat.len(),
                "ArenaOctree3d::build",
            )));
        }
        let n = flat.len() / 3;
        self.inner.build_flat(&flat, n).map_err(core_err_to_js)
    }

    /// Builds the octree directly from a 3D `LayoutNd`.
    #[wasm_bindgen(js_name = buildNd)]
    pub fn build_nd(&mut self, layout: &JsLayoutNd) -> Result<(), JsError> {
        self.inner.build(layout.inner()).map_err(core_err_to_js)
    }

    /// Builds the weighted octree from flat `[x0, y0, z0, ...]` coordinates and per-point `masses`.
    #[wasm_bindgen(js_name = _buildWeightedFlat)]
    pub fn build_weighted(&mut self, coords: &JsValue, masses: &JsValue) -> Result<(), JsError> {
        let flat = parse_f32_sequence(coords, "coords")?;
        let m = parse_f64_sequence(masses, "masses")?;
        if !flat.len().is_multiple_of(3) {
            return Err(core_err_to_js(CoreError::dimension_mismatch(
                flat.len() + (3 - flat.len() % 3),
                flat.len(),
                "ArenaOctree3d::buildWeighted",
            )));
        }
        let n = flat.len() / 3;
        self.inner
            .build_weighted_flat(&flat, n, &m)
            .map_err(core_err_to_js)
    }

    /// Builds the weighted octree directly from a 3D `LayoutNd` and per-point `masses`.
    #[wasm_bindgen(js_name = buildWeightedNd)]
    pub fn build_weighted_nd(
        &mut self,
        layout: &JsLayoutNd,
        masses: &[f64],
    ) -> Result<(), JsError> {
        self.inner
            .build_weighted(layout.inner(), masses)
            .map_err(core_err_to_js)
    }

    /// Builds the octree from separate `(x, y, z)` arrays with optional `masses`.
    #[wasm_bindgen(js_name = buildXyz)]
    pub fn build_xyz(
        &mut self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        masses: Option<Vec<f64>>,
    ) -> Result<(), JsError> {
        match masses {
            Some(m) => self
                .inner
                .build_weighted_xyz(x, y, z, &m)
                .map_err(core_err_to_js),
            None => self.inner.build_xyz(x, y, z).map_err(core_err_to_js),
        }
    }

    /// Evaluates Barnes-Hut Coulomb repulsion returning flat `[fx0, fy0, fz0, ...]` (`Float32Array`).
    #[wasm_bindgen(js_name = computeCoulombRepulsion)]
    pub fn compute_coulomb_repulsion(
        &self,
        theta: Option<f32>,
        eps_sq: Option<f32>,
    ) -> Result<Vec<f32>, JsError> {
        let n = self.inner.point_count();
        let mut out = vec![0.0_f32; n.saturating_mul(3)];
        self.inner
            .compute_coulomb_repulsion_flat(
                theta.unwrap_or(DEFAULT_THETA),
                eps_sq.unwrap_or(DEFAULT_EPS_SQ),
                &mut out,
            )
            .map_err(core_err_to_js)?;
        Ok(out)
    }

    /// Evaluates Barnes-Hut generalized MaxEnt $q$-power repulsion returning flat `[fx0, fy0, fz0, ...]`.
    #[wasm_bindgen(js_name = computeMaxentRepulsion)]
    pub fn compute_maxent_repulsion(
        &self,
        theta: Option<f32>,
        q: Option<f64>,
        eps_sq: Option<f32>,
    ) -> Result<Vec<f32>, JsError> {
        let n = self.inner.point_count();
        let mut out = vec![0.0_f32; n.saturating_mul(3)];
        self.inner
            .compute_maxent_repulsion_flat(
                theta.unwrap_or(DEFAULT_THETA),
                q.unwrap_or(DEFAULT_MAXENT_Q),
                eps_sq.unwrap_or(DEFAULT_EPS_SQ),
                &mut out,
            )
            .map_err(core_err_to_js)?;
        Ok(out)
    }

    /// Evaluates normalized Barnes-Hut Student-$t$ KL repulsion returning flat `[fx0, fy0, fz0, ...]`.
    #[wasm_bindgen(js_name = computeKlRepulsion)]
    pub fn compute_kl_repulsion(&self, theta: Option<f32>) -> Result<Vec<f32>, JsError> {
        let n = self.inner.point_count();
        let mut out = vec![0.0_f32; n.saturating_mul(3)];
        self.inner
            .compute_kl_repulsion_flat(theta.unwrap_or(DEFAULT_THETA), &mut out)
            .map_err(core_err_to_js)?;
        Ok(out)
    }

    /// Evaluates Barnes-Hut entropy repulsion returning flat `[fx0, fy0, fz0, ...]`.
    #[wasm_bindgen(js_name = computeEntropyRepulsion)]
    pub fn compute_entropy_repulsion(
        &self,
        theta: Option<f32>,
        epsilon_r: Option<f32>,
    ) -> Result<Vec<f32>, JsError> {
        let n = self.inner.point_count();
        let mut out = vec![0.0_f32; n.saturating_mul(3)];
        self.inner
            .compute_entropy_repulsion_flat(
                theta.unwrap_or(DEFAULT_THETA),
                epsilon_r.unwrap_or(DEFAULT_ENTROPY_EPS_R),
                &mut out,
            )
            .map_err(core_err_to_js)?;
        Ok(out)
    }
}

/// Arbitrary-dimensional $k$-d tree (`f32`) supporting $k$-NN and fixed-radius spatial queries.
#[wasm_bindgen(js_name = DynamicKdTree)]
#[derive(Debug, Clone)]
pub struct JsDynamicKdTree {
    inner: DynamicKdTree<f32>,
}

#[wasm_bindgen(js_class = DynamicKdTree)]
impl JsDynamicKdTree {
    /// Constructs a `DynamicKdTree` from flat row-major coordinates of shape `(n, dims)`.
    #[wasm_bindgen(constructor)]
    pub fn new(coords: &[f32], n: f64, dims: f64) -> Result<JsDynamicKdTree, JsError> {
        Self::from_flat(coords, n, dims)
    }

    /// Constructs a `DynamicKdTree` directly from a `LayoutNd`.
    #[wasm_bindgen(js_name = fromLayoutNd)]
    pub fn from_layout_nd(layout: &JsLayoutNd) -> Result<JsDynamicKdTree, JsError> {
        let inner = DynamicKdTree::<f32>::from_layout_nd(layout.inner()).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `DynamicKdTree` from flat row-major coordinates of shape `(n, dims)`.
    #[wasm_bindgen(js_name = fromFlat)]
    pub fn from_flat(coords: &[f32], n: f64, dims: f64) -> Result<JsDynamicKdTree, JsError> {
        let num_nodes = parse_non_negative_usize(n, "n")?;
        let d = parse_non_negative_usize(dims, "dims")?;
        let inner =
            DynamicKdTree::<f32>::from_flat(coords, num_nodes, d).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the spatial dimensionality `D` of indexed points.
    #[wasm_bindgen(getter)]
    pub fn dims(&self) -> usize {
        self.inner.dims()
    }

    /// Returns the number of points indexed in the tree.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of points indexed in the tree.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the tree contains zero points.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a `Float64Array` copy of point `index`'s coordinates.
    pub fn point(&self, index: f64) -> Result<Vec<f64>, JsError> {
        let idx = parse_non_negative_usize(index, "index")?;
        self.inner
            .point(idx)
            .map(|slice| slice.to_vec())
            .map_err(core_err_to_js)
    }

    /// Returns up to `k` nearest points `Array<{ index, distSq }>` to `query`.
    #[wasm_bindgen(js_name = kNearest)]
    pub fn k_nearest(&self, query: &[f32], k: f64) -> Result<js_sys::Array, JsError> {
        let k_u = parse_non_negative_usize(k, "k")?;
        let hits = self.inner.k_nearest(query, k_u).map_err(core_err_to_js)?;
        kd_hits_to_js_array(&hits)
    }

    /// Returns up to `k` nearest points `Array<{ index, distSq }>` to `pointIdx`, excluding self.
    #[wasm_bindgen(js_name = kNearestExcludingSelf)]
    pub fn k_nearest_excluding_self(
        &self,
        point_idx: f64,
        k: f64,
    ) -> Result<js_sys::Array, JsError> {
        let idx = parse_non_negative_usize(point_idx, "pointIdx")?;
        let k_u = parse_non_negative_usize(k, "k")?;
        let hits = self
            .inner
            .k_nearest_excluding_self(idx, k_u)
            .map_err(core_err_to_js)?;
        kd_hits_to_js_array(&hits)
    }

    /// Returns `{ index, distSq }` for the nearest neighbor of `pointIdx`, or `undefined` if `numNodes <= 1`.
    #[wasm_bindgen(js_name = nearestExcludingSelf)]
    pub fn nearest_excluding_self(&self, point_idx: f64) -> Result<JsValue, JsError> {
        let idx = parse_non_negative_usize(point_idx, "pointIdx")?;
        let hit = self
            .inner
            .nearest_excluding_self(idx)
            .map_err(core_err_to_js)?;
        match hit {
            None => Ok(JsValue::UNDEFINED),
            Some((nbr, dist_sq)) => {
                let obj = js_sys::Object::new();
                set_f64_prop(&obj, "index", nbr as f64)?;
                set_f64_prop(&obj, "distSq", dist_sq)?;
                let _ = &set_js_prop;
                Ok(obj.into())
            }
        }
    }

    /// Returns all points `Array<{ index, distSq }>` within Euclidean `radius` of `query`.
    #[wasm_bindgen(js_name = withinRadius)]
    pub fn within_radius(&self, query: &[f32], radius: f64) -> Result<js_sys::Array, JsError> {
        let hits = self
            .inner
            .within_radius(query, radius)
            .map_err(core_err_to_js)?;
        kd_hits_to_js_array(&hits)
    }

    /// Returns all points `Array<{ index, distSq }>` within squared distance `radiusSq` of `query`.
    #[wasm_bindgen(js_name = withinRadiusSq)]
    pub fn within_radius_sq(
        &self,
        query: &[f32],
        radius_sq: f64,
    ) -> Result<js_sys::Array, JsError> {
        let hits = self
            .inner
            .within_radius_sq(query, radius_sq)
            .map_err(core_err_to_js)?;
        kd_hits_to_js_array(&hits)
    }
}
