//! WebAssembly bindings for non-Euclidean crossings and clipped torus edge segments.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, set_f64_prop, set_js_prop};
use crate::layout_buffers::JsLayout2d;
use crate::manifold::JsManifoldLayout2d;
use crate::metrics_crossings::JsCrossingPair;
use egraph::metrics::{
    TorusCrossing, TorusView, crossing_edges_hyperbolic, crossing_edges_spherical,
    crossing_edges_torus, hyperbolic_crossing_number, spherical_crossing_number,
    torus_crossing_number,
};
use egraph::spatial::{torus_edge_segments, torus_layout_edge_segments};
use wasm_bindgen::prelude::*;

/// Counts non-adjacent edge sub-segment crossings on the flat unit torus `[0, 1)^2` for `Layout2d`.
#[wasm_bindgen(js_name = _torusCrossingNumber2d)]
pub fn js_torus_crossing_number_2d(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<usize, JsValue> {
    let soa = layout.inner().to_layout2d();
    let view = TorusView::new(&soa, 1.0, 1.0).map_err(core_err_to_js)?;
    torus_crossing_number(graph.inner(), &view).map_err(core_err_to_js)
}

/// Counts non-adjacent edge sub-segment crossings on the flat torus for `ManifoldLayout2d` (`torus2d`).
#[wasm_bindgen(js_name = _torusCrossingNumberManifold)]
pub fn js_torus_crossing_number_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<usize, JsValue> {
    torus_crossing_number(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

/// Periodic `[0, 1)^2` torus crossing edge sub-segment pair returned by `crossingEdgesTorus`.
#[wasm_bindgen(js_name = TorusCrossing)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsTorusCrossing {
    inner: TorusCrossing,
}

impl JsTorusCrossing {
    pub(crate) fn from_inner(inner: TorusCrossing) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = TorusCrossing)]
impl JsTorusCrossing {
    #[wasm_bindgen(getter, js_name = edgeA)]
    pub fn edge_a(&self) -> usize {
        self.inner.edge_a
    }

    #[wasm_bindgen(getter, js_name = edgeB)]
    pub fn edge_b(&self) -> usize {
        self.inner.edge_b
    }

    #[wasm_bindgen(getter)]
    pub fn u1(&self) -> u32 {
        self.inner.u1
    }

    #[wasm_bindgen(getter)]
    pub fn v1(&self) -> u32 {
        self.inner.v1
    }

    #[wasm_bindgen(getter)]
    pub fn u2(&self) -> u32 {
        self.inner.u2
    }

    #[wasm_bindgen(getter)]
    pub fn v2(&self) -> u32 {
        self.inner.v2
    }

    #[wasm_bindgen(getter, js_name = angleRad)]
    pub fn angle_rad(&self) -> Option<f64> {
        self.inner.angle_rad
    }

    #[wasm_bindgen(getter, js_name = cosSq)]
    pub fn cos_sq(&self) -> f64 {
        self.inner.cos_sq
    }

    /// Returns `[x1, y1, x2, y2]` coordinates of the first crossing sub-segment.
    #[wasm_bindgen(js_name = segA)]
    pub fn seg_a(&self) -> Vec<f64> {
        self.inner.seg_a.to_vec()
    }

    /// Returns `[x1, y1, x2, y2]` coordinates of the second crossing sub-segment.
    #[wasm_bindgen(js_name = segB)]
    pub fn seg_b(&self) -> Vec<f64> {
        self.inner.seg_b.to_vec()
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsValue> {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "edgeA", self.inner.edge_a as f64)?;
        set_f64_prop(&obj, "edgeB", self.inner.edge_b as f64)?;
        set_f64_prop(&obj, "u1", self.inner.u1 as f64)?;
        set_f64_prop(&obj, "v1", self.inner.v1 as f64)?;
        set_f64_prop(&obj, "u2", self.inner.u2 as f64)?;
        set_f64_prop(&obj, "v2", self.inner.v2 as f64)?;
        set_js_prop(
            &obj,
            "segA",
            &js_sys::Float64Array::from(self.inner.seg_a.as_slice()).into(),
        )?;
        set_js_prop(
            &obj,
            "segB",
            &js_sys::Float64Array::from(self.inner.seg_b.as_slice()).into(),
        )?;
        if let Some(ang) = self.inner.angle_rad {
            set_f64_prop(&obj, "angleRad", ang)?;
        }
        set_f64_prop(&obj, "cosSq", self.inner.cos_sq)?;
        Ok(obj)
    }
}

/// Finds all crossing edge sub-segment pairs on the unit torus `[0, 1)^2` for `Layout2d`.
#[wasm_bindgen(js_name = _crossingEdgesTorus2d)]
pub fn js_crossing_edges_torus_2d(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<Vec<JsTorusCrossing>, JsValue> {
    let soa = layout.inner().to_layout2d();
    let view = TorusView::new(&soa, 1.0, 1.0).map_err(core_err_to_js)?;
    let crossings = crossing_edges_torus(graph.inner(), &view).map_err(core_err_to_js)?;
    Ok(crossings
        .into_iter()
        .map(JsTorusCrossing::from_inner)
        .collect())
}

/// Finds all crossing edge sub-segment pairs on the flat torus for `ManifoldLayout2d` (`torus2d`).
#[wasm_bindgen(js_name = _crossingEdgesTorusManifold)]
pub fn js_crossing_edges_torus_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<Vec<JsTorusCrossing>, JsValue> {
    let crossings = crossing_edges_torus(graph.inner(), layout.inner()).map_err(core_err_to_js)?;
    Ok(crossings
        .into_iter()
        .map(JsTorusCrossing::from_inner)
        .collect())
}

/// Splits a 2D torus edge `(x1, y1) -> (x2, y2)` into `[x1, y1, x2, y2, ...]` sub-segments in `[0, 1)^2`.
#[wasm_bindgen(js_name = torusEdgeSegments)]
pub fn js_torus_edge_segments(x1: f64, y1: f64, x2: f64, y2: f64) -> Result<Vec<f64>, JsValue> {
    let segs = torus_edge_segments(x1, y1, x2, y2).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(segs.len() * 4);
    for s in segs {
        flat.extend_from_slice(&s);
    }
    Ok(flat)
}

/// Returns flat `[x1, y1, x2, y2, ...]` clipped torus edge sub-segments for `Layout2d`.
#[wasm_bindgen(js_name = _torusLayoutEdgeSegments2d)]
pub fn js_torus_layout_edge_segments_2d(
    graph: &JsCsrGraph,
    layout: &JsLayout2d,
) -> Result<Vec<f64>, JsValue> {
    let soa = layout.inner().to_layout2d();
    let view = TorusView::new(&soa, 1.0, 1.0).map_err(core_err_to_js)?;
    let segs = torus_layout_edge_segments(graph.inner(), &view).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(segs.len() * 4);
    for s in segs {
        flat.extend_from_slice(&s);
    }
    Ok(flat)
}

/// Returns flat `[x1, y1, x2, y2, ...]` clipped torus edge sub-segments for `ManifoldLayout2d`.
#[wasm_bindgen(js_name = _torusLayoutEdgeSegmentsManifold)]
pub fn js_torus_layout_edge_segments_manifold(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<Vec<f64>, JsValue> {
    let segs = torus_layout_edge_segments(graph.inner(), layout.inner()).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(segs.len() * 4);
    for s in segs {
        flat.extend_from_slice(&s);
    }
    Ok(flat)
}

/// Counts exact great-circle geodesic arc crossings on `ManifoldKind::Spherical`.
#[wasm_bindgen(js_name = sphericalCrossingNumber)]
pub fn js_spherical_crossing_number(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<usize, JsValue> {
    spherical_crossing_number(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

/// Finds all shortest great-circle geodesic arc crossing pairs on `ManifoldKind::Spherical`.
#[wasm_bindgen(js_name = crossingEdgesSpherical)]
pub fn js_crossing_edges_spherical(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<Vec<JsCrossingPair>, JsValue> {
    let pairs = crossing_edges_spherical(graph.inner(), layout.inner()).map_err(core_err_to_js)?;
    Ok(pairs.into_iter().map(JsCrossingPair::from_inner).collect())
}

/// Counts exact hyperbolic geodesic crossings on `HyperbolicPoincare` or `HyperbolicLorentz`.
#[wasm_bindgen(js_name = hyperbolicCrossingNumber)]
pub fn js_hyperbolic_crossing_number(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<usize, JsValue> {
    hyperbolic_crossing_number(graph.inner(), layout.inner()).map_err(core_err_to_js)
}

/// Finds all hyperbolic geodesic crossing pairs on `HyperbolicPoincare` or `HyperbolicLorentz`.
#[wasm_bindgen(js_name = crossingEdgesHyperbolic)]
pub fn js_crossing_edges_hyperbolic(
    graph: &JsCsrGraph,
    layout: &JsManifoldLayout2d,
) -> Result<Vec<JsCrossingPair>, JsValue> {
    let pairs = crossing_edges_hyperbolic(graph.inner(), layout.inner()).map_err(core_err_to_js)?;
    Ok(pairs.into_iter().map(JsCrossingPair::from_inner).collect())
}
