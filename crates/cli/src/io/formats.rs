use std::collections::HashMap;

use egraph::prelude::{CoreError, CsrGraph, Layout2d};
use serde::Deserialize;

use super::{CliError, CliResult};

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

fn node_id_key(value: &serde_json::Value) -> CliResult<String> {
    match value {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        other => Err(CliError::InvalidInput(format!(
            "node id must be string or number, got {other}"
        ))),
    }
}

pub(super) fn parse_json_graph(
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
