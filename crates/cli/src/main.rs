//! Entrypoint for the `egraph` command-line tool.

use std::process::ExitCode;

fn main() -> ExitCode {
    match egraph_cli::run_cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
