//! WebAssembly bindings for dataset catalog metadata and deterministic synthetic graph generators.

use egraph::dataset::{
    DatasetInfo, dataset_catalog, dataset_info, generate_hyperbolic_tree, generate_layered_dag,
    generate_sbm_graph, generate_spherical_grid, generate_spherical_icosphere, generate_torus_grid,
    generate_weighted_graph, list_datasets,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    DEFAULT_SEED, core_err_to_js, csr_f64_to_f32, parse_non_negative_usize, set_f64_prop,
    set_js_prop,
};
use crate::js_options::{check_options_arg, opt_f64_prop, opt_seed_prop, opt_usize_prop};

const DEFAULT_WEIGHTED_AVG_DEGREE: usize = 4;
const DEFAULT_MIN_WEIGHT: f64 = 0.1;
const DEFAULT_MAX_WEIGHT: f64 = 1.0;
const DEFAULT_DAG_EDGE_PROB: f64 = 0.35;
const DEFAULT_SBM_P_INTRA: f64 = 0.5;
const DEFAULT_SBM_P_INTER: f64 = 0.05;
const DEFAULT_ICOSPHERE_SUBDIVISIONS: usize = 2;

/// Static metadata describing a built-in benchmark dataset.
#[wasm_bindgen(js_name = DatasetInfo)]
#[derive(Debug, Clone)]
pub struct JsDatasetInfo {
    info: DatasetInfo,
}

#[wasm_bindgen(js_class = DatasetInfo)]
impl JsDatasetInfo {
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.info.name.to_owned()
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.info.num_nodes
    }

    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.info.num_edges
    }

    #[wasm_bindgen(getter)]
    pub fn source(&self) -> String {
        self.info.source.to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn description(&self) -> String {
        self.info.description.to_owned()
    }

    /// Converts metadata into a plain JavaScript object `{ name, numNodes, numEdges, source, description }`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<JsValue, JsValue> {
        let obj = js_sys::Object::new();
        set_js_prop(&obj, "name", &JsValue::from_str(self.info.name))?;
        set_f64_prop(&obj, "numNodes", self.info.num_nodes as f64)?;
        set_f64_prop(&obj, "numEdges", self.info.num_edges as f64)?;
        set_js_prop(&obj, "source", &JsValue::from_str(self.info.source))?;
        set_js_prop(
            &obj,
            "description",
            &JsValue::from_str(self.info.description),
        )?;
        Ok(obj.into())
    }
}

/// Returns the canonical names of all built-in benchmark datasets (`string[]`).
#[wasm_bindgen(js_name = listDatasets)]
pub fn js_list_datasets() -> Vec<String> {
    list_datasets()
        .iter()
        .map(|&name| name.to_owned())
        .collect()
}

/// Returns static `DatasetInfo` records for all built-in benchmark datasets.
#[wasm_bindgen(js_name = datasetCatalog)]
pub fn js_dataset_catalog() -> js_sys::Array {
    let entries = dataset_catalog();
    let arr = js_sys::Array::new_with_length(entries.len() as u32);
    for (idx, info) in entries.iter().enumerate() {
        arr.set(idx as u32, JsDatasetInfo { info: info.clone() }.into());
    }
    arr
}

/// Looks up static `DatasetInfo` for a built-in dataset by canonical name, or `undefined`.
#[wasm_bindgen(js_name = datasetInfo)]
pub fn js_dataset_info(name: &str) -> Option<JsDatasetInfo> {
    dataset_info(name)
        .cloned()
        .map(|info| JsDatasetInfo { info })
}

/// Generates a deterministic undirected weighted `CsrGraph` with edge weights in `[minWeight, maxWeight]`.
#[wasm_bindgen(js_name = generateWeightedGraph)]
pub fn js_generate_weighted_graph(
    num_nodes: f64,
    options: &JsValue,
) -> Result<JsCsrGraph, JsValue> {
    let n = parse_non_negative_usize(num_nodes, "numNodes")?;
    let (deg, min_w, max_w, seed_u64) = if check_options_arg(options, "generateWeightedGraph")? {
        (
            opt_usize_prop(options, "avgDegree", DEFAULT_WEIGHTED_AVG_DEGREE)?,
            opt_f64_prop(options, "minWeight", DEFAULT_MIN_WEIGHT)?,
            opt_f64_prop(options, "maxWeight", DEFAULT_MAX_WEIGHT)?,
            opt_seed_prop(options)?,
        )
    } else {
        (
            DEFAULT_WEIGHTED_AVG_DEGREE,
            DEFAULT_MIN_WEIGHT,
            DEFAULT_MAX_WEIGHT,
            DEFAULT_SEED,
        )
    };
    let g64 = generate_weighted_graph(n, deg, min_w, max_w, seed_u64).map_err(core_err_to_js)?;
    Ok(JsCsrGraph::from_inner(csr_f64_to_f32(&g64)?))
}

/// Generates a deterministic layered Directed Acyclic Graph (`CsrGraph`).
#[wasm_bindgen(js_name = generateLayeredDag)]
pub fn js_generate_layered_dag(
    num_layers: f64,
    nodes_per_layer: f64,
    options: &JsValue,
) -> Result<JsCsrGraph, JsValue> {
    let layers = parse_non_negative_usize(num_layers, "numLayers")?;
    let per_layer = parse_non_negative_usize(nodes_per_layer, "nodesPerLayer")?;
    let (edge_prob, seed_u64) = if check_options_arg(options, "generateLayeredDag")? {
        (
            opt_f64_prop(options, "edgeProb", DEFAULT_DAG_EDGE_PROB)?,
            opt_seed_prop(options)?,
        )
    } else {
        (DEFAULT_DAG_EDGE_PROB, DEFAULT_SEED)
    };
    let g64 =
        generate_layered_dag(layers, per_layer, edge_prob, seed_u64).map_err(core_err_to_js)?;
    Ok(JsCsrGraph::from_inner(csr_f64_to_f32(&g64)?))
}

/// Generates a deterministic Stochastic Block Model graph returning `{ graph, labels }`.
#[wasm_bindgen(js_name = generateSbmGraph)]
pub fn js_generate_sbm_graph(block_sizes: &[u32], options: &JsValue) -> Result<JsValue, JsValue> {
    let sizes: Vec<usize> = block_sizes.iter().map(|&s| s as usize).collect();
    let (p_intra, p_inter, seed_u64) = if check_options_arg(options, "generateSbmGraph")? {
        (
            opt_f64_prop(options, "pIntra", DEFAULT_SBM_P_INTRA)?,
            opt_f64_prop(options, "pInter", DEFAULT_SBM_P_INTER)?,
            opt_seed_prop(options)?,
        )
    } else {
        (DEFAULT_SBM_P_INTRA, DEFAULT_SBM_P_INTER, DEFAULT_SEED)
    };
    let (g64, labels) =
        generate_sbm_graph(&sizes, p_intra, p_inter, seed_u64).map_err(core_err_to_js)?;
    let graph = JsCsrGraph::from_inner(csr_f64_to_f32(&g64)?);
    let obj = js_sys::Object::new();
    set_js_prop(&obj, "graph", &graph.into())?;
    let labels_arr = js_sys::Uint32Array::from(labels.as_slice());
    set_js_prop(&obj, "labels", &labels_arr.into())?;
    Ok(obj.into())
}

/// Generates a periodic 2D torus mesh `CsrGraph` of size `rows x cols` (`>= 3x3`).
#[wasm_bindgen(js_name = generateTorusGrid)]
pub fn js_generate_torus_grid(
    rows: f64,
    cols: f64,
    triangulated: Option<bool>,
) -> Result<JsCsrGraph, JsValue> {
    let r = parse_non_negative_usize(rows, "rows")?;
    let c = parse_non_negative_usize(cols, "cols")?;
    let g64 = generate_torus_grid(r, c, triangulated.unwrap_or(false)).map_err(core_err_to_js)?;
    Ok(JsCsrGraph::from_inner(csr_f64_to_f32(&g64)?))
}

/// Generates a subdivided icosphere spherical geodesic mesh `CsrGraph`.
#[wasm_bindgen(js_name = generateSphericalIcosphere)]
pub fn js_generate_spherical_icosphere(subdivisions: Option<f64>) -> Result<JsCsrGraph, JsValue> {
    let sub = subdivisions
        .map(|s| parse_non_negative_usize(s, "subdivisions"))
        .transpose()?
        .unwrap_or(DEFAULT_ICOSPHERE_SUBDIVISIONS);
    let g64 = generate_spherical_icosphere(sub).map_err(core_err_to_js)?;
    Ok(JsCsrGraph::from_inner(csr_f64_to_f32(&g64)?))
}

/// Generates a closed UV spherical grid mesh `CsrGraph` with poles and `latitudes x longitudes` rings.
#[wasm_bindgen(js_name = generateSphericalGrid)]
pub fn js_generate_spherical_grid(latitudes: f64, longitudes: f64) -> Result<JsCsrGraph, JsValue> {
    let lats = parse_non_negative_usize(latitudes, "latitudes")?;
    let lons = parse_non_negative_usize(longitudes, "longitudes")?;
    let g64 = generate_spherical_grid(lats, lons).map_err(core_err_to_js)?;
    Ok(JsCsrGraph::from_inner(csr_f64_to_f32(&g64)?))
}

/// Generates a regular hyperbolic branching tree `CsrGraph` with `branchingFactor` and `depth`.
#[wasm_bindgen(js_name = generateHyperbolicTree)]
pub fn js_generate_hyperbolic_tree(
    branching_factor: f64,
    depth: f64,
) -> Result<JsCsrGraph, JsValue> {
    let bf = parse_non_negative_usize(branching_factor, "branchingFactor")?;
    let d = parse_non_negative_usize(depth, "depth")?;
    let g64 = generate_hyperbolic_tree(bf, d).map_err(core_err_to_js)?;
    Ok(JsCsrGraph::from_inner(csr_f64_to_f32(&g64)?))
}
