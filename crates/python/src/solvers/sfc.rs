//! Space-filling curve (`SfcLayout`) layout generator bindings (`LayoutGenerator`).

use std::sync::Arc;

use egraph::core::{CsrGraph, LayoutGenerator};
use egraph::layout::{SfcConfig, SfcCurveKind, SfcLayout};
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, invalid_input_err, unsupported_operation_err};
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, normalize_enum_key};
use crate::layout_buffers::PyLayout2d;

pub(super) fn parse_sfc_curve_kind(raw: &str) -> Result<SfcCurveKind, String> {
    match normalize_enum_key(raw).as_str() {
        "hilbert" => Ok(SfcCurveKind::Hilbert),
        "morton" | "zorder" => Ok(SfcCurveKind::Morton),
        _ => Err(format!(
            "unknown SFC curve {raw:?}; expected 'hilbert' or 'morton'"
        )),
    }
}

/// Community-ordered Space-Filling Curve (`SFC`) layout generator implementing `LayoutGenerator`.
///
/// Precomputes a hierarchical Louvain community vertex ordering and maps ranks injectively
/// onto a `2^k x 2^k` Hilbert or Morton (`"hilbert"` / `"morton"`) grid inside
/// `[-scale / 2, scale / 2]^2`, guaranteeing zero vertex co-location in both `"float64"`
/// and `"float32"` while releasing the Python GIL.
///
/// Note (`muelder2008_space_filling` `MISSING_PDF`): Muelder & Ma (2008) is marked
/// `MISSING_PDF` in the project literature archive, so exact clustering, ordering, and
/// post-refinement parameters from that paper are not claimed; this generator provides a
/// canonical Louvain community-ordered Hilbert/Morton space-filling curve baseline.
#[pyclass(name = "SfcLayout", module = "egraph._core")]
pub struct PySfcLayout {
    graph: Arc<CsrGraph<f64>>,
    config: SfcConfig,
    vertex_order: Vec<u32>,
    dtype: ScalarDtype,
}

#[pymethods]
impl PySfcLayout {
    #[new]
    #[pyo3(
        signature = (graph, *, curve = "hilbert", scale = StrictF64(100.0), resolution_order = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, curve='hilbert', scale=100.0, resolution_order=None, seed=42, dtype='float64')"
    )]
    fn new(
        graph: &PyCsrGraph,
        curve: &str,
        scale: StrictF64,
        resolution_order: Option<StrictUsize>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_curve = parse_sfc_curve_kind(curve).map_err(unsupported_operation_err)?;
        let res_order = match resolution_order {
            Some(ord) => Some(u32::try_from(ord.0).map_err(|_| {
                invalid_input_err(format!(
                    "resolution_order must be in [1, 15], got {}",
                    ord.0
                ))
            })?),
            None => None,
        };
        let cfg = SfcConfig::new()
            .with_curve(parsed_curve)
            .with_scale(scale.0)
            .with_resolution_order(res_order)
            .with_seed(seed.0);
        cfg.validate().map_err(core_error_to_py)?;
        let g = Arc::new(graph.inner().into_owned());
        // Validate capacity up front by generating a lightweight dry check on order / graph weights.
        if let Some(ord) = res_order {
            let total_cells = 1u64 << (2 * ord.min(31));
            if (g.num_nodes() as u64) > total_cells {
                return Err(invalid_input_err(format!(
                    "SfcConfig::resolution_order={ord} provides {total_cells} cells, insufficient for {} vertices",
                    g.num_nodes()
                )));
            }
        }
        Ok(Self {
            graph: g,
            config: cfg,
            vertex_order: Vec::new(),
            dtype,
        })
    }

    /// Generate a 2D Space-Filling Curve `Layout2d` while releasing the Python GIL.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn generate(&mut self, py: Python<'_>) -> PyResult<PyLayout2d> {
        let g = Arc::clone(&self.graph);
        let cfg = self.config.clone();
        match self.dtype {
            ScalarDtype::F64 => {
                let (layout, order) = py
                    .detach(|| {
                        let mut generator = SfcLayout::<f64, f64>::from_config(&*g, cfg);
                        let layout = generator.generate()?;
                        let order = generator.vertex_order().to_vec();
                        Ok::<_, egraph::core::CoreError>((layout, order))
                    })
                    .map_err(core_error_to_py)?;
                self.vertex_order = order;
                Ok(PyLayout2d::from_f64_layout(&layout))
            }
            ScalarDtype::F32 => {
                let (layout, order) = py
                    .detach(|| {
                        let mut generator = SfcLayout::<f32, f64>::from_config(&*g, cfg);
                        let layout = generator.generate()?;
                        let order = generator.vertex_order().to_vec();
                        Ok::<_, egraph::core::CoreError>((layout, order))
                    })
                    .map_err(core_error_to_py)?;
                self.vertex_order = order;
                Ok(PyLayout2d::from_f32_layout(&layout))
            }
        }
    }

    /// Return a `uint32` NumPy copy of the community-ordered vertex permutation along the curve.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn vertex_order<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.vertex_order)
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "SfcLayout(num_nodes={}, scale={}, dtype='{}')",
            self.num_nodes(),
            self.config.scale,
            self.dtype()
        )
    }
}
