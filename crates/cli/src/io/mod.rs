//! Graph and layout I/O utilities for `egraph-cli`.
//!
//! Supports embedded benchmark datasets, JSON (CSR, edge-list, and node-link
//! coordinates), Matrix Market (`.mtx`), and CSR edge-list CSV (`.csv`).

use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use egraph::prelude::{CoreError, CsrGraph, Layout2d, parse_csr_csv};
use serde::Serialize;

mod formats;
mod layout_json;

pub use formats::parse_mtx_graph;
pub use layout_json::LoadedLayout;

use formats::parse_json_graph;
use layout_json::{parse_2d_layout_value, parse_any_layout_value};

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

/// Loads 2D coordinates from a JSON layout file (`{coords: [[x,y],...]}`, `{x:[...], y:[...]}`,
/// `{nodes: [{x, y},...]}`, or `[[x, y],...]`).
pub fn load_layout_from_path(path: &Path, expected_nodes: usize) -> CliResult<Layout2d<f64>> {
    let content = fs::read_to_string(path)?;
    let val: serde_json::Value = serde_json::from_str(&content)?;
    parse_2d_layout_value(&val, path, expected_nodes)
}

/// Loads a 2D, N-D (`coords_nd` / `dims`), or manifold (`manifold`, `manifold_scale`,
/// `torus_aspect`) layout from a JSON file.
pub fn load_any_layout_from_path(path: &Path, expected_nodes: usize) -> CliResult<LoadedLayout> {
    let content = fs::read_to_string(path)?;
    let val: serde_json::Value = serde_json::from_str(&content)?;
    parse_any_layout_value(&val, path, expected_nodes)
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
