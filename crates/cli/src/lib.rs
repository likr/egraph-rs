//! Modernized single-threaded command-line interface library for `egraph`.
//!
//! Exposes [`Cli`], [`Commands`], and execution entrypoints for `layout`,
//! `metrics`, `cluster`, and `bench` subcommands backed exclusively by the
//! `egraph` facade crate.

pub mod io;

mod args;
mod cluster_bench;
mod layout_exec;
mod layout_methods;
mod metrics_exec;
mod output;

#[cfg(test)]
mod tests;

use clap::Parser;

use crate::io::{CliResult, write_json_output};

pub use args::{
    BenchArgs, Cli, ClusterArgs, ClusterMethod, ClusterRefineArg, Commands, InitMethod, LayoutArgs,
    LayoutMethod, MetricsArgs, OverlapRemovalArg,
};
pub use cluster_bench::{execute_bench, execute_cluster};
pub use layout_exec::execute_layout;
pub use layout_methods::{
    build_initial_layout, run_layout_method, run_layout_method_with_communities,
};
pub use metrics_exec::{execute_metrics, load_community_labels_from_path, parse_metric_targets};
pub use output::{BenchOutput, ClusterOutput, GcafDetailOutput, LayoutOutput, MetricsOutput};

/// Parses and executes the CLI from process arguments.
pub fn run_cli() -> CliResult<()> {
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
