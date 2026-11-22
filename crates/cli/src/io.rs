//! Graph and layout I/O utilities for `egraph-cli`.
//!
//! Supports embedded benchmark datasets, JSON (CSR, edge-list, and node-link
//! coordinates), Matrix Market (`.mtx`), and CSR edge-list CSV (`.csv`).

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use egraph::prelude::{CoreError, CsrGraph, Layout2d, parse_csr_csv};
use serde::{Deserialize, Serialize};

/// Unified error type returned by CLI I/O and execution helpers.
#[derive(Debug)]
pub enum CliError {
    /// Filesystem or standard I/O error.
    Io(io::Error),
    /// JSON serialization or deserialization error.
    Json(serde_json::Error),
    /// Core `egraph` computation or validation error.
    Core(CoreError),
    /// Invalid CLI flag, dataset name, or input file syntax.
    InvalidInput(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Json(err) => write!(f, "JSON error: {err}"),
            Self::Core(err) => write!(f, "egraph error: {err}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::Core(err) => Some(err),
            Self::InvalidInput(_) => None,
        }
    }
}

impl From<io::Error> for CliError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for CliError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl From<CoreError> for CliError {
    fn from(err: CoreError) -> Self {
        Self::Core(err)
    }
}

/// Result alias for CLI operations.
pub type CliResult<T> = Result<T, CliError>;

#[derive(Debug, Deserialize)]
struct CsrJsonInput {
    #[serde(default)]
    num_nodes: Option<usize>,
    indptr: Vec<u32>,
    indices: Vec<u32>,
    #[serde(default)]
    weights: Option<Vec<f64>>,
    #[serde(default)]
    directed: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct EdgeListJsonInput {
    num_nodes: usize,
    edges: Vec<[u32; 2]>,
    #[serde(default)]
    weights: Option<Vec<f64>>,
    #[serde(default)]
    directed: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct NodeLinkNode {
    id: serde_json::Value,
    #[serde(default)]
    x: Option<f64>,
    #[serde(default)]
    y: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct NodeLinkEdge {
    source: serde_json::Value,
    target: serde_json::Value,
    #[serde(default)]
    weight: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct NodeLinkJsonInput {
    nodes: Vec<NodeLinkNode>,
    links: Vec<NodeLinkEdge>,
    #[serde(default)]
    directed: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct LayoutCoordsObject {
    coords: Vec<[f64; 2]>,
}

#[derive(Debug, Deserialize)]
struct LayoutSplitObject {
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct LayoutNodesObject {
    nodes: Vec<LayoutNodePoint>,
}

#[derive(Debug, Deserialize)]
struct LayoutNodePoint {
    x: f64,
    y: f64,
}

/// Loads one of the built-in benchmark datasets as an undirected [`CsrGraph<f64>`].
pub fn load_dataset(name: &str) -> CliResult<CsrGraph<f64>> {
    let key = name.trim().to_ascii_lowercase();
    match key.as_str() {
        "1138_bus" => Ok(egraph::dataset::dataset_1138_bus_csr()),
        "3_elt" | "3elt" => Ok(egraph::dataset::dataset_3_elt_csr()),
        "dwt_1005" => Ok(egraph::dataset::dataset_dwt_1005_csr()),
        "dwt_2680" => Ok(egraph::dataset::dataset_dwt_2680_csr()),
        "poli" => Ok(egraph::dataset::dataset_poli_csr()),
        "qh882" => Ok(egraph::dataset::dataset_qh882_csr()),
        "uspower_grid" | "uspowergrid" => Ok(egraph::dataset::dataset_uspower_grid_csr()),
        "karate" => Ok(egraph::dataset::dataset_karate_csr()),
        "lesmis" => Ok(egraph::dataset::dataset_lesmis_csr()),
        _ => Err(CliError::InvalidInput(format!(
            "unknown dataset '{name}'; expected one of: 1138_bus, 3_elt, dwt_1005, dwt_2680, poli, qh882, uspower_grid, karate, lesmis"
        ))),
    }
}

/// Loads a [`CsrGraph<f64>`] and optional embedded initial [`Layout2d<f64>`] from `.json`,
/// `.mtx`, or `.csv` files.
pub fn load_graph_from_path(
    path: &Path,
    directed_override: Option<bool>,
) -> CliResult<(CsrGraph<f64>, Option<Layout2d<f64>>)> {
    let content = fs::read_to_string(path)?;
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "json" => parse_json_graph(&content, directed_override),
        "mtx" => {
            let directed = directed_override.unwrap_or(false);
            let graph = parse_mtx_graph(&content, directed)?;
            Ok((graph, None))
        }
        "csv" => {
            let directed = directed_override.unwrap_or(false);
            let graph = parse_csr_csv(&content, directed, true)?;
            Ok((graph, None))
        }
        other => Err(CliError::InvalidInput(format!(
            "unsupported graph file extension '{other}' for '{}'; expected .json, .mtx, or .csv",
            path.display()
        ))),
    }
}

/// Resolves a graph either from `--input <FILE>` or `--dataset <NAME>`.
pub fn resolve_graph(
    input: Option<&Path>,
    dataset: Option<&str>,
    directed_override: Option<bool>,
) -> CliResult<(CsrGraph<f64>, Option<Layout2d<f64>>, String)> {
    match (input, dataset) {
        (Some(path), None) => {
            let (graph, init_layout) = load_graph_from_path(path, directed_override)?;
            Ok((graph, init_layout, path.display().to_string()))
        }
        (None, Some(name)) => {
            let graph = load_dataset(name)?;
            Ok((graph, None, format!("dataset:{name}")))
        }
        (Some(_), Some(_)) => Err(CliError::InvalidInput(
            "specify either --input or --dataset, not both".to_string(),
        )),
        (None, None) => Err(CliError::InvalidInput(
            "either --input <FILE> or --dataset <NAME> is required".to_string(),
        )),
    }
}

fn node_id_key(value: &serde_json::Value) -> CliResult<String> {
    match value {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        other => Err(CliError::InvalidInput(format!(
            "node id must be string or number, got {other}"
        ))),
    }
}

fn parse_json_graph(
    content: &str,
    directed_override: Option<bool>,
) -> CliResult<(CsrGraph<f64>, Option<Layout2d<f64>>)> {
    let val: serde_json::Value = serde_json::from_str(content)?;

    if val.get("indptr").is_some() && val.get("indices").is_some() {
        let csr: CsrJsonInput = serde_json::from_value(val)?;
        let directed = directed_override.unwrap_or(csr.directed.unwrap_or(false));
        let graph = CsrGraph::from_csr(
            &csr.indptr,
            &csr.indices,
            csr.weights.as_deref(),
            directed,
            true,
        )?;
        if let Some(expected_n) = csr.num_nodes
            && expected_n != graph.num_nodes()
        {
            return Err(CliError::Core(CoreError::dimension_mismatch(
                expected_n,
                graph.num_nodes(),
                "parse_json_graph::csr.num_nodes",
            )));
        }
        return Ok((graph, None));
    }

    if val.get("edges").is_some() {
        let edge_list: EdgeListJsonInput = serde_json::from_value(val)?;
        let directed = directed_override.unwrap_or(edge_list.directed.unwrap_or(false));
        let pairs: Vec<(u32, u32)> = edge_list.edges.into_iter().map(|[u, v]| (u, v)).collect();
        let graph = CsrGraph::from_edges(
            edge_list.num_nodes,
            &pairs,
            edge_list.weights.as_deref(),
            directed,
            true,
        )?;
        return Ok((graph, None));
    }

    if val.get("nodes").is_some() && val.get("links").is_some() {
        let nl: NodeLinkJsonInput = serde_json::from_value(val)?;
        let directed = directed_override.unwrap_or(nl.directed.unwrap_or(false));
        let num_nodes = nl.nodes.len();
        let mut id_map = HashMap::with_capacity(num_nodes);
        let mut xs = Vec::with_capacity(num_nodes);
        let mut ys = Vec::with_capacity(num_nodes);
        let mut all_have_coords = !nl.nodes.is_empty();

        for (idx, node) in nl.nodes.iter().enumerate() {
            let key = node_id_key(&node.id)?;
            if id_map.insert(key.clone(), idx as u32).is_some() {
                return Err(CliError::InvalidInput(format!(
                    "duplicate node id '{key}' in node-link JSON"
                )));
            }
            match (node.x, node.y) {
                (Some(x), Some(y)) => {
                    xs.push(x);
                    ys.push(y);
                }
                _ => {
                    all_have_coords = false;
                }
            }
        }

        let mut pairs = Vec::with_capacity(nl.links.len());
        let mut weights = Vec::with_capacity(nl.links.len());
        let mut has_any_weight = false;
        for link in &nl.links {
            let src_key = node_id_key(&link.source)?;
            let tgt_key = node_id_key(&link.target)?;
            let u = *id_map.get(&src_key).ok_or_else(|| {
                CliError::InvalidInput(format!("link source '{src_key}' not found in nodes"))
            })?;
            let v = *id_map.get(&tgt_key).ok_or_else(|| {
                CliError::InvalidInput(format!("link target '{tgt_key}' not found in nodes"))
            })?;
            pairs.push((u, v));
            if let Some(w) = link.weight {
                has_any_weight = true;
                weights.push(w);
            } else {
                weights.push(1.0);
            }
        }

        let weight_slice = if has_any_weight {
            Some(weights.as_slice())
        } else {
            None
        };
        let graph = CsrGraph::from_edges(num_nodes, &pairs, weight_slice, directed, true)?;
        let layout = if all_have_coords {
            Some(Layout2d::from_xy(xs, ys)?)
        } else {
            None
        };
        return Ok((graph, layout));
    }

    Err(CliError::InvalidInput(
        "unrecognized JSON graph schema; expected {num_nodes, indptr, indices}, {num_nodes, edges}, or {nodes, links}"
            .to_string(),
    ))
}

/// Parses Matrix Market (`.mtx`) coordinate format into a [`CsrGraph<f64>`].
pub fn parse_mtx_graph(content: &str, directed: bool) -> CliResult<CsrGraph<f64>> {
    let mut is_pattern = false;
    let mut header_seen = false;
    let mut dims: Option<(usize, usize, usize)> = None;
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    let mut saw_weights = false;

    for (line_no, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("%%MatrixMarket") {
            header_seen = true;
            let tokens: Vec<&str> = line.split_ascii_whitespace().collect();
            if tokens.len() < 4 {
                return Err(CliError::InvalidInput(format!(
                    "malformed MatrixMarket banner at line {}",
                    line_no + 1
                )));
            }
            if !tokens[1].eq_ignore_ascii_case("matrix")
                || !tokens[2].eq_ignore_ascii_case("coordinate")
            {
                return Err(CliError::InvalidInput(format!(
                    "unsupported MatrixMarket format '{} {}'; expected 'matrix coordinate'",
                    tokens[1], tokens[2]
                )));
            }
            is_pattern = tokens[3].eq_ignore_ascii_case("pattern");
            continue;
        }
        if line.starts_with('%') {
            continue;
        }

        let parts: Vec<&str> = line.split_ascii_whitespace().collect();
        if dims.is_none() {
            if parts.len() < 3 {
                return Err(CliError::InvalidInput(format!(
                    "expected 'nrows ncols nnz' header at line {}",
                    line_no + 1
                )));
            }
            let nrows = parts[0].parse::<usize>().map_err(|err| {
                CliError::InvalidInput(format!("invalid nrows at line {}: {err}", line_no + 1))
            })?;
            let ncols = parts[1].parse::<usize>().map_err(|err| {
                CliError::InvalidInput(format!("invalid ncols at line {}: {err}", line_no + 1))
            })?;
            let nnz = parts[2].parse::<usize>().map_err(|err| {
                CliError::InvalidInput(format!("invalid nnz at line {}: {err}", line_no + 1))
            })?;
            dims = Some((nrows, ncols, nnz));
            edges.reserve(nnz);
            continue;
        }

        if parts.len() < 2 {
            return Err(CliError::InvalidInput(format!(
                "expected at least 2 indices at MTX line {}",
                line_no + 1
            )));
        }
        let row_1based = parts[0].parse::<usize>().map_err(|err| {
            CliError::InvalidInput(format!("invalid row index at line {}: {err}", line_no + 1))
        })?;
        let col_1based = parts[1].parse::<usize>().map_err(|err| {
            CliError::InvalidInput(format!("invalid col index at line {}: {err}", line_no + 1))
        })?;
        if row_1based == 0 || col_1based == 0 {
            return Err(CliError::InvalidInput(format!(
                "MatrixMarket indices must be 1-based (>= 1) at line {}",
                line_no + 1
            )));
        }
        let u = u32::try_from(row_1based - 1).map_err(|_| {
            CliError::InvalidInput(format!("row index overflows u32 at line {}", line_no + 1))
        })?;
        let v = u32::try_from(col_1based - 1).map_err(|_| {
            CliError::InvalidInput(format!("col index overflows u32 at line {}", line_no + 1))
        })?;
        edges.push((u, v));

        if !is_pattern && parts.len() >= 3 {
            let w = parts[2].parse::<f64>().map_err(|err| {
                CliError::InvalidInput(format!(
                    "invalid edge weight at line {}: {err}",
                    line_no + 1
                ))
            })?;
            saw_weights = true;
            weights.push(w);
        } else {
            weights.push(1.0);
        }
    }

    let _ = header_seen;
    let (nrows, ncols, _nnz) = dims.ok_or_else(|| {
        CliError::InvalidInput("MatrixMarket file missing size header".to_string())
    })?;
    let num_nodes = nrows.max(ncols);
    let weight_slice = if saw_weights && !is_pattern {
        Some(weights.as_slice())
    } else {
        None
    };

    Ok(CsrGraph::from_edges(
        num_nodes,
        &edges,
        weight_slice,
        directed,
        true,
    )?)
}

/// Loads 2D coordinates from a JSON layout file (`{coords: [[x,y],...]}`, `{x:[...], y:[...]}`,
/// `{nodes: [{x, y},...]}`, or `[[x, y],...]`).
pub fn load_layout_from_path(path: &Path, expected_nodes: usize) -> CliResult<Layout2d<f64>> {
    let content = fs::read_to_string(path)?;
    let val: serde_json::Value = serde_json::from_str(&content)?;

    let layout = if val.get("coords").is_some() {
        let obj: LayoutCoordsObject = serde_json::from_value(val)?;
        coords_pairs_to_layout(&obj.coords)?
    } else if val.get("x").is_some() && val.get("y").is_some() {
        let obj: LayoutSplitObject = serde_json::from_value(val)?;
        Layout2d::from_xy(obj.x, obj.y)?
    } else if val.get("nodes").is_some() {
        let obj: LayoutNodesObject = serde_json::from_value(val)?;
        let mut xs = Vec::with_capacity(obj.nodes.len());
        let mut ys = Vec::with_capacity(obj.nodes.len());
        for pt in obj.nodes {
            xs.push(pt.x);
            ys.push(pt.y);
        }
        Layout2d::from_xy(xs, ys)?
    } else if val.is_array() {
        let pairs: Vec<[f64; 2]> = serde_json::from_value(val)?;
        coords_pairs_to_layout(&pairs)?
    } else {
        return Err(CliError::InvalidInput(format!(
            "unrecognized layout JSON schema in '{}'",
            path.display()
        )));
    };

    if layout.len() != expected_nodes {
        return Err(CliError::Core(CoreError::dimension_mismatch(
            expected_nodes,
            layout.len(),
            "load_layout_from_path",
        )));
    }
    layout.validate_finite()?;
    Ok(layout)
}

fn coords_pairs_to_layout(pairs: &[[f64; 2]]) -> CliResult<Layout2d<f64>> {
    let mut xs = Vec::with_capacity(pairs.len());
    let mut ys = Vec::with_capacity(pairs.len());
    for &[x, y] in pairs {
        xs.push(x);
        ys.push(y);
    }
    Ok(Layout2d::from_xy(xs, ys)?)
}

/// Writes serializable output as formatted JSON to `output` or to stdout when `output` is `None`.
pub fn write_json_output<T: Serialize>(value: &T, output: Option<&Path>) -> CliResult<()> {
    match output {
        Some(path) => {
            let file = File::create(path)?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, value)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        None => {
            let stdout = io::stdout();
            let mut handle = BufWriter::new(stdout.lock());
            serde_json::to_writer_pretty(&mut handle, value)?;
            handle.write_all(b"\n")?;
            handle.flush()?;
        }
    }
    Ok(())
}
