//! WebAssembly linear-memory `EdgePolylines` bindings across 2D, 3D, and N-D layouts.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, error_with_code, ffi_err_to_js};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use egraph::core::{EdgePolylines, Layout2d, Layout3d, LayoutNd};
use egraph_ffi::{InterleavedEdgePolylines2d, InterleavedEdgePolylines3d};
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub(crate) enum EdgePolylinesInner {
    Dim2(InterleavedEdgePolylines2d<f32>),
    Dim3(InterleavedEdgePolylines3d<f32>),
    DimN(EdgePolylines<LayoutNd<f32>>),
}

/// Contiguous edge-polyline buffer (`edgeOffsets` + interleaved control points `f32`)
/// across `"layout2d"`, `"layout3d"`, and `"layoutNd"`, exposing `edgeOffsetsView()` and
/// `pointsView()`.
#[wasm_bindgen(js_name = EdgePolylines)]
#[derive(Debug, Clone)]
pub struct JsEdgePolylines {
    inner: EdgePolylinesInner,
}

impl JsEdgePolylines {
    pub(crate) fn inner(&self) -> &EdgePolylinesInner {
        &self.inner
    }

    pub(crate) fn inner_mut(&mut self) -> &mut EdgePolylinesInner {
        &mut self.inner
    }
}

#[wasm_bindgen(js_class = EdgePolylines)]
impl JsEdgePolylines {
    #[wasm_bindgen(js_name = _straightLineFromGraph2d)]
    pub fn straight_line_from_graph_2d(
        graph: &JsCsrGraph,
        layout: &JsLayout2d,
    ) -> Result<JsEdgePolylines, JsValue> {
        let soa = layout.inner().to_layout2d();
        let poly = EdgePolylines::<Layout2d<f32>>::straight_line_from_graph(graph.inner(), &soa)
            .map_err(core_err_to_js)?;
        Ok(Self {
            inner: EdgePolylinesInner::Dim2(InterleavedEdgePolylines2d::from_core(&poly)),
        })
    }

    #[wasm_bindgen(js_name = _straightLineFromGraph3d)]
    pub fn straight_line_from_graph_3d(
        graph: &JsCsrGraph,
        layout: &JsLayout3d,
    ) -> Result<JsEdgePolylines, JsValue> {
        let soa = layout.inner().to_layout3d();
        let poly = EdgePolylines::<Layout3d<f32>>::straight_line_from_graph(graph.inner(), &soa)
            .map_err(core_err_to_js)?;
        Ok(Self {
            inner: EdgePolylinesInner::Dim3(InterleavedEdgePolylines3d::from_core(&poly)),
        })
    }

    #[wasm_bindgen(js_name = _straightLineFromGraphNd)]
    pub fn straight_line_from_graph_nd(
        graph: &JsCsrGraph,
        layout: &JsLayoutNd,
    ) -> Result<JsEdgePolylines, JsValue> {
        let poly =
            EdgePolylines::<LayoutNd<f32>>::straight_line_from_graph(graph.inner(), layout.inner())
                .map_err(core_err_to_js)?;
        Ok(Self {
            inner: EdgePolylinesInner::DimN(poly),
        })
    }

    #[wasm_bindgen(js_name = _fromParts2d)]
    pub fn from_parts_2d(
        edge_offsets: &[u32],
        points: &JsLayout2d,
    ) -> Result<JsEdgePolylines, JsValue> {
        let poly = InterleavedEdgePolylines2d::from_parts(edge_offsets, points.inner().as_slice())
            .map_err(ffi_err_to_js)?;
        Ok(Self {
            inner: EdgePolylinesInner::Dim2(poly),
        })
    }

    #[wasm_bindgen(js_name = _fromParts3d)]
    pub fn from_parts_3d(
        edge_offsets: &[u32],
        points: &JsLayout3d,
    ) -> Result<JsEdgePolylines, JsValue> {
        let poly = InterleavedEdgePolylines3d::from_parts(edge_offsets, points.inner().as_slice())
            .map_err(ffi_err_to_js)?;
        Ok(Self {
            inner: EdgePolylinesInner::Dim3(poly),
        })
    }

    #[wasm_bindgen(js_name = _fromPartsNd)]
    pub fn from_parts_nd(
        edge_offsets: &[u32],
        points: &JsLayoutNd,
    ) -> Result<JsEdgePolylines, JsValue> {
        let poly = EdgePolylines::from_parts(edge_offsets, points.inner().clone())
            .map_err(core_err_to_js)?;
        Ok(Self {
            inner: EdgePolylinesInner::DimN(poly),
        })
    }

    /// Returns a deep copy of this `EdgePolylines`.
    pub fn copy(&self) -> JsEdgePolylines {
        self.clone()
    }

    /// Returns the number of polylines (edges) stored in this buffer.
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        match &self.inner {
            EdgePolylinesInner::Dim2(p) => p.num_edges(),
            EdgePolylinesInner::Dim3(p) => p.num_edges(),
            EdgePolylinesInner::DimN(p) => p.num_edges(),
        }
    }

    /// Returns the total number of control points across all edges.
    #[wasm_bindgen(getter, js_name = totalPoints)]
    pub fn total_points(&self) -> usize {
        match &self.inner {
            EdgePolylinesInner::Dim2(p) => p.total_points(),
            EdgePolylinesInner::Dim3(p) => p.total_points(),
            EdgePolylinesInner::DimN(p) => p.total_points(),
        }
    }

    /// Returns the coordinate dimensionality (`2`, `3`, or `LayoutNd.dims`).
    #[wasm_bindgen(getter)]
    pub fn dims(&self) -> usize {
        match &self.inner {
            EdgePolylinesInner::Dim2(_) => 2,
            EdgePolylinesInner::Dim3(_) => 3,
            EdgePolylinesInner::DimN(p) => p.points().dims(),
        }
    }

    /// Returns the backing layout target string (`"layout2d"`, `"layout3d"`, or `"layoutNd"`).
    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        match &self.inner {
            EdgePolylinesInner::Dim2(_) => "layout2d".to_owned(),
            EdgePolylinesInner::Dim3(_) => "layout3d".to_owned(),
            EdgePolylinesInner::DimN(_) => "layoutNd".to_owned(),
        }
    }

    /// Returns `true` if there are zero edges.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.num_edges() == 0
    }

    /// Returns the linear-memory byte offset of `edgeOffsets` (`Uint32Array`).
    #[wasm_bindgen(js_name = edgeOffsetsPtr)]
    pub fn edge_offsets_ptr(&self) -> usize {
        match &self.inner {
            EdgePolylinesInner::Dim2(p) => p.offsets_ptr() as usize,
            EdgePolylinesInner::Dim3(p) => p.offsets_ptr() as usize,
            EdgePolylinesInner::DimN(p) => p.edge_offsets().as_ptr() as usize,
        }
    }

    /// Returns the scalar length of `edgeOffsets` (`numEdges + 1`).
    #[wasm_bindgen(js_name = edgeOffsetsLen)]
    pub fn edge_offsets_len(&self) -> usize {
        match &self.inner {
            EdgePolylinesInner::Dim2(p) => p.offsets_len(),
            EdgePolylinesInner::Dim3(p) => p.offsets_len(),
            EdgePolylinesInner::DimN(p) => p.edge_offsets().len(),
        }
    }

    /// Returns a freshly allocated `Uint32Array` copy of `edgeOffsets`.
    #[wasm_bindgen(js_name = copyEdgeOffsets)]
    pub fn copy_edge_offsets(&self) -> Vec<u32> {
        match &self.inner {
            EdgePolylinesInner::Dim2(p) => p.edge_offsets().to_vec(),
            EdgePolylinesInner::Dim3(p) => p.edge_offsets().to_vec(),
            EdgePolylinesInner::DimN(p) => p.edge_offsets().to_vec(),
        }
    }

    /// Returns the linear-memory byte offset of the interleaved control points (`Float32Array`).
    #[wasm_bindgen(js_name = pointsPtr)]
    pub fn points_ptr(&self) -> usize {
        match &self.inner {
            EdgePolylinesInner::Dim2(p) => p.points_ptr() as usize,
            EdgePolylinesInner::Dim3(p) => p.points_ptr() as usize,
            EdgePolylinesInner::DimN(p) => p.points().coords().as_ptr() as usize,
        }
    }

    /// Returns the total scalar length of the interleaved control-point buffer (`totalPoints * dims`).
    #[wasm_bindgen(js_name = pointsLen)]
    pub fn points_len(&self) -> usize {
        match &self.inner {
            EdgePolylinesInner::Dim2(p) => p.points_len(),
            EdgePolylinesInner::Dim3(p) => p.points_len(),
            EdgePolylinesInner::DimN(p) => p.points().coords().len(),
        }
    }

    /// Returns a freshly allocated `Float32Array` copy of the interleaved control-point buffer.
    #[wasm_bindgen(js_name = copyPoints)]
    pub fn copy_points(&self) -> Vec<f32> {
        match &self.inner {
            EdgePolylinesInner::Dim2(p) => p.points_as_slice().to_vec(),
            EdgePolylinesInner::Dim3(p) => p.points_as_slice().to_vec(),
            EdgePolylinesInner::DimN(p) => p.points().coords().to_vec(),
        }
    }

    /// Returns the control-point `[start, end)` range for `edge`.
    #[wasm_bindgen(js_name = edgePointRange)]
    pub fn edge_point_range(&self, edge: usize) -> Result<Vec<u32>, JsValue> {
        let range = match &self.inner {
            EdgePolylinesInner::Dim2(p) => {
                if edge >= p.num_edges() {
                    return Err(error_with_code(
                        "IndexOutOfBounds",
                        &format!(
                            "edge index {edge} out of bounds for {} edges",
                            p.num_edges()
                        ),
                    ));
                }
                let offs = p.edge_offsets();
                offs[edge]..offs[edge + 1]
            }
            EdgePolylinesInner::Dim3(p) => {
                if edge >= p.num_edges() {
                    return Err(error_with_code(
                        "IndexOutOfBounds",
                        &format!(
                            "edge index {edge} out of bounds for {} edges",
                            p.num_edges()
                        ),
                    ));
                }
                let offs = p.edge_offsets();
                offs[edge]..offs[edge + 1]
            }
            EdgePolylinesInner::DimN(p) => {
                let r = p.edge_point_range(edge).map_err(core_err_to_js)?;
                (r.start as u32)..(r.end as u32)
            }
        };
        Ok(vec![range.start, range.end])
    }

    /// Returns `true` if every control-point coordinate is finite.
    #[wasm_bindgen(js_name = isAllFinite)]
    pub fn is_all_finite(&self) -> bool {
        match &self.inner {
            EdgePolylinesInner::Dim2(p) => p.is_all_finite(),
            EdgePolylinesInner::Dim3(p) => p.is_all_finite(),
            EdgePolylinesInner::DimN(p) => p.points().is_all_finite(),
        }
    }
}
