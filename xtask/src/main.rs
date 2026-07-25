#![forbid(unsafe_code)]
//! Repository automation wrapper for generated documentation and policy checks.

mod file_sizes;
mod recipes;
mod simdoc;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let program = args.first().map(String::as_str).unwrap_or("xtask");
    let result = match args.get(1).map(String::as_str) {
        Some("simdoc") => simdoc::run(args),
        Some("crate-catalog") => simdoc::run_repo_tool(args, "crate-catalog"),
        Some("check-recipes") => recipes::run(),
        Some("check-file-sizes") => file_sizes::run(),
        _ => Err(format!(
            "usage: {program} simdoc [--check] | crate-catalog [--check] | check-file-sizes | check-recipes"
        )),
    };

    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
