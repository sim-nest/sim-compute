#![forbid(unsafe_code)]
//! Repository automation wrapper for generated documentation and policy checks.

mod file_sizes;
mod recipes;
mod simdoc;

use std::sync::Arc;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let program = args.first().map(String::as_str).unwrap_or("xtask");
    let result = match args.get(1).map(String::as_str) {
        Some("simdoc") => simdoc::run(args),
        Some("crate-catalog") => simdoc::run_repo_tool(args, "crate-catalog"),
        Some("check-recipes") => recipes::run(),
        Some("check-file-sizes") => file_sizes::run(),
        Some("acceptance") => acceptance(args),
        _ => Err(format!(
            "usage: {program} simdoc [--check] | crate-catalog [--check] | check-file-sizes | check-recipes | acceptance <capture|verify|import> ..."
        )),
    };

    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn acceptance(args: Vec<String>) -> Result<(), String> {
    let command_args = std::iter::once("compute".to_owned())
        .chain(args.into_iter().skip(1))
        .collect::<Vec<_>>();
    let command = sim_lib_compute_cli::parse_compute_args(&command_args)
        .map_err(|error| error.to_string())?;
    let (mut cx, seat) = sim_kernel::Cx::new_seated(
        Arc::new(sim_kernel::EagerPolicy),
        Arc::new(sim_kernel::DefaultFactory),
    );
    seat.grant(
        &mut cx,
        sim_lib_compute_cli::compute_acceptance_capability(),
    )
    .map_err(|error| error.to_string())?;
    let output = sim_lib_compute_cli::run_command(&mut cx, None, &command)
        .map_err(|error| error.to_string())?;
    print!("{output}");
    Ok(())
}
