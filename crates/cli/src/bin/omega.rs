//! Applies the Omega SGD layout algorithm to a graph using spectral coordinates.
//!
//! Reads a graph layout from an input JSON file, applies the Omega layout algorithm
//! to update node positions, and writes the resulting layout to an output JSON file.
//!
//! The Omega algorithm differs from standard SGD by using spectral coordinates derived
//! from the graph Laplacian to construct node pairs more intelligently:
//! 1. Computes the smallest d non-zero eigenvalues and eigenvectors of the graph Laplacian
//! 2. Creates d-dimensional coordinates by dividing eigenvectors by sqrt of eigenvalues  
//! 3. Adds edge-based node pairs using Euclidean distances from coordinates
//! 4. Adds k random node pairs per node using Euclidean distances (avoiding duplicates)
//!
//! # Usage
//!
//! `cargo run --bin omega -- <input.json> <output.json> [OPTIONS]`
//!
//! * `<input.json>`: Path to the input JSON file (GraphData format). Initial coordinates are used.
//! * `<output.json>`: Path to the output JSON file for the layout after Omega SGD.
//!
//! # Command-line Options
//!
//! * `--d <value>`: Number of spectral dimensions (default: 2)
//! * `--k <value>`: Number of random pairs per node (default: 30)
//! * `--min-dist <value>`: Minimum distance between node pairs (default: 1e-3)
//! * `--eigenvalue-max-iterations <value>`: Max iterations for eigenvalue computation (default: 1000)
//! * `--cg-max-iterations <value>`: Max iterations for CG method (default: 100)
//! * `--eigenvalue-tolerance <value>`: Convergence tolerance for eigenvalue computation (default: 1e-4)
//! * `--cg-tolerance <value>`: Convergence tolerance for CG method (default: 1e-4)
//! * `--unit-edge-length <value>`: Length value for all edges (default: 30.0)
//! * `--sgd-iterations <value>`: Number of SGD iterations (default: 100)
//! * `--sgd-eps <value>`: Final learning rate for SGD scheduler (default: 0.1)
//!
//! # Examples
//!
//! Basic usage:
//! `cargo run --bin omega -- input.json output.json`
//!
//! With custom parameters:
//! `cargo run --bin omega -- input.json output.json --d 3 --k 50 --sgd-iterations 200`

use argparse::{ArgumentParser, Store, StoreOption};
use egraph_cli::{read_graph, write_pos};
use petgraph::prelude::*;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_omega::OmegaBuilder;
use petgraph_layout_sgd::{Scheduler, SchedulerExponential, Sgd};
use rand::thread_rng;

/// Command-line parameters for the Omega algorithm.
#[derive(Debug)]
struct OmegaParams {
    d: usize,
    k: usize,
    min_dist: f32,
    eigenvalue_max_iterations: usize,
    cg_max_iterations: usize,
    eigenvalue_tolerance: f32,
    cg_tolerance: f32,
    unit_edge_length: f32,
    sgd_iterations: usize,
    sgd_eps: f32,
}

impl Default for OmegaParams {
    fn default() -> Self {
        Self {
            d: 2,
            k: 30,
            min_dist: 1e-3,
            eigenvalue_max_iterations: 1000,
            cg_max_iterations: 100,
            eigenvalue_tolerance: 1e-4,
            cg_tolerance: 1e-4,
            unit_edge_length: 30.0,
            sgd_iterations: 100,
            sgd_eps: 0.1,
        }
    }
}

/// Parses command-line arguments for input and output file paths and algorithm parameters.
///
/// # Arguments
///
/// * `input_path` - Mutable string to store the input file path.
/// * `output_path` - Mutable string to store the output file path.
/// * `params` - Mutable struct to store the algorithm parameters.
fn parse_args(input_path: &mut String, output_path: &mut String, params: &mut OmegaParams) {
    // Create option variables for command line arguments
    let mut d_opt: Option<usize> = None;
    let mut k_opt: Option<usize> = None;
    let mut min_dist_opt: Option<f32> = None;
    let mut eigenvalue_max_iterations_opt: Option<usize> = None;
    let mut cg_max_iterations_opt: Option<usize> = None;
    let mut eigenvalue_tolerance_opt: Option<f32> = None;
    let mut cg_tolerance_opt: Option<f32> = None;
    let mut unit_edge_length_opt: Option<f32> = None;
    let mut sgd_iterations_opt: Option<usize> = None;
    let mut sgd_eps_opt: Option<f32> = None;

    // Parse arguments in a separate scope to release borrows
    {
        let mut parser = ArgumentParser::new();
        parser.set_description(
            "Applies the Omega SGD layout algorithm to a graph using spectral coordinates.",
        );

        parser
            .refer(input_path)
            .add_argument("input", Store, "input file path")
            .required();
        parser
            .refer(output_path)
            .add_argument("output", Store, "output file path")
            .required();

        parser.refer(&mut d_opt).add_option(
            &["--d"],
            StoreOption,
            "number of spectral dimensions (default: 2)",
        );
        parser.refer(&mut k_opt).add_option(
            &["--k"],
            StoreOption,
            "number of random pairs per node (default: 30)",
        );
        parser.refer(&mut min_dist_opt).add_option(
            &["--min-dist"],
            StoreOption,
            "minimum distance between node pairs (default: 1e-3)",
        );
        parser.refer(&mut eigenvalue_max_iterations_opt).add_option(
            &["--eigenvalue-max-iterations"],
            StoreOption,
            "max iterations for eigenvalue computation (default: 1000)",
        );
        parser.refer(&mut cg_max_iterations_opt).add_option(
            &["--cg-max-iterations"],
            StoreOption,
            "max iterations for CG method (default: 100)",
        );
        parser.refer(&mut eigenvalue_tolerance_opt).add_option(
            &["--eigenvalue-tolerance"],
            StoreOption,
            "convergence tolerance for eigenvalue computation (default: 1e-4)",
        );
        parser.refer(&mut cg_tolerance_opt).add_option(
            &["--cg-tolerance"],
            StoreOption,
            "convergence tolerance for CG method (default: 1e-4)",
        );
        parser.refer(&mut unit_edge_length_opt).add_option(
            &["--unit-edge-length"],
            StoreOption,
            "length value for all edges (default: 30.0)",
        );
        parser.refer(&mut sgd_iterations_opt).add_option(
            &["--sgd-iterations"],
            StoreOption,
            "number of SGD iterations (default: 100)",
        );
        parser.refer(&mut sgd_eps_opt).add_option(
            &["--sgd-eps"],
            StoreOption,
            "final learning rate for SGD scheduler (default: 0.1)",
        );

        parser.parse_args_or_exit();
    } // parser is dropped here, releasing all borrows

    // Apply parsed values or keep defaults
    if let Some(d) = d_opt {
        params.d = d;
    }
    if let Some(k) = k_opt {
        params.k = k;
    }
    if let Some(min_dist) = min_dist_opt {
        params.min_dist = min_dist;
    }
    if let Some(eigenvalue_max_iterations) = eigenvalue_max_iterations_opt {
        params.eigenvalue_max_iterations = eigenvalue_max_iterations;
    }
    if let Some(cg_max_iterations) = cg_max_iterations_opt {
        params.cg_max_iterations = cg_max_iterations;
    }
    if let Some(eigenvalue_tolerance) = eigenvalue_tolerance_opt {
        params.eigenvalue_tolerance = eigenvalue_tolerance;
    }
    if let Some(cg_tolerance) = cg_tolerance_opt {
        params.cg_tolerance = cg_tolerance;
    }
    if let Some(unit_edge_length) = unit_edge_length_opt {
        params.unit_edge_length = unit_edge_length;
    }
    if let Some(sgd_iterations) = sgd_iterations_opt {
        params.sgd_iterations = sgd_iterations;
    }
    if let Some(sgd_eps) = sgd_eps_opt {
        params.sgd_eps = sgd_eps;
    }
}

/// Applies the Omega SGD layout algorithm using spectral coordinates.
///
/// Modifies the provided `coordinates` in place using the `Omega` implementation
/// with an exponential learning rate scheduler. The algorithm uses spectral coordinates
/// derived from the graph Laplacian to construct more meaningful node pairs for SGD.
///
/// # Arguments
///
/// * `graph` - The graph to layout (node/edge data ignored).
/// * `drawing` - Mutable `DrawingEuclidean2d` containing the initial and resulting node positions.
/// * `params` - Algorithm parameters for Omega and SGD configuration.
fn layout(
    graph: &Graph<Option<()>, Option<()>, Undirected>,
    drawing: &mut DrawingEuclidean2d<NodeIndex, f32>,
    params: &OmegaParams,
) {
    let mut rng = thread_rng();

    // Create Omega instance using builder pattern with command-line parameters
    let mut omega = OmegaBuilder::new()
        .d(params.d)
        .k(params.k)
        .min_dist(params.min_dist)
        .eigenvalue_max_iterations(params.eigenvalue_max_iterations)
        .cg_max_iterations(params.cg_max_iterations)
        .eigenvalue_tolerance(params.eigenvalue_tolerance)
        .cg_tolerance(params.cg_tolerance)
        .build(graph, |_| params.unit_edge_length, &mut rng);

    // Use SGD parameters from command line
    let mut scheduler =
        omega.scheduler::<SchedulerExponential<f32>>(params.sgd_iterations, params.sgd_eps);

    scheduler.run(&mut |eta| {
        omega.shuffle(&mut rng);
        omega.apply(drawing, eta);
    });
}

/// Main entry point: parses args, reads graph, applies Omega layout, writes result.
fn main() {
    let mut input_path = "".to_string();
    let mut output_path = "".to_string();
    let mut params = OmegaParams::default();
    parse_args(&mut input_path, &mut output_path, &mut params);
    let (input_graph, node_ids, mut drawing) = read_graph(&input_path);
    layout(&input_graph, &mut drawing, &params);
    write_pos(&node_ids, &drawing, &output_path);
}
