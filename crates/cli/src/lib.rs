//! Single-threaded command-line interface binary crate for `egraph`.
//!
//! Parses process arguments and executes `layout`, `metrics`, `cluster`, and
//! `bench` subcommands backed by the `egraph` facade crate.

mod args;
mod cluster_bench;
mod io;
mod layout_exec;
mod layout_methods;
mod layout_nd_manifold;
mod metrics_exec;
mod metrics_targets;
mod output;

#[cfg(test)]
mod tests;

use clap::Parser;

use crate::args::{Cli, Commands};
use crate::cluster_bench::{execute_bench, execute_cluster};
use crate::io::{CliResult, write_json_output};
use crate::layout_exec::execute_layout;
use crate::metrics_exec::execute_metrics;

fn run_cli_inner() -> CliResult<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Layout(args) => {
            let out = execute_layout(&args)?;
            write_json_output(&out, args.output.as_deref())
        }
        Commands::Metrics(args) => {
            let out = execute_metrics(&args)?;
            write_json_output(&out, args.output.as_deref())
        }
        Commands::Cluster(args) => {
            let out = execute_cluster(&args)?;
            write_json_output(&out, args.output.as_deref())
        }
        Commands::Bench(args) => {
            let out = execute_bench(&args)?;
            write_json_output(&out, args.output.as_deref())
        }
    }
}

/// Parses and executes the `egraph` CLI from process arguments.
pub fn run_cli() -> Result<(), String> {
    run_cli_inner().map_err(|err| err.to_string())
}
