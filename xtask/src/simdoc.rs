// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Thin launchers for the documentation tools.
//!
//! `simdoc` runs the isolated simdoc encoder. `crate-catalog` runs through
//! the sim-tooling workspace manifest.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn run(args: Vec<String>) -> Result<(), String> {
    let program = args.first().map(String::as_str).unwrap_or("xtask");
    if args.get(1).map(String::as_str) != Some("simdoc") {
        return Err(format!("usage: {program} simdoc [--check]"));
    }

    crate::recipes::run()?;
    let root = env::current_dir().map_err(|err| format!("current dir: {err}"))?;
    let manifest = locate_simdoc_manifest(&root)?;
    let mut command = encoder_command(&root, &manifest, &args[2..]);
    let status = command
        .status()
        .map_err(|err| format!("run isolated simdoc encoder: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "isolated simdoc encoder failed with status {status}"
        ))
    }
}

fn encoder_command(root: &Path, manifest: &Path, extra: &[String]) -> Command {
    let mut command = Command::new("cargo");
    command.args(["run", "--locked", "--offline", "--manifest-path"]);
    command.arg(manifest);
    command.args(["--quiet", "--", "simdoc", "--repo-root"]);
    command.arg(root);
    command.args(extra);
    command
}

fn locate_simdoc_manifest(repo_root: &Path) -> Result<PathBuf, String> {
    if let Ok(path) = env::var("SIMDOC_TOOLING_MANIFEST") {
        // CI sets this to the tooling workspace manifest. That manifest
        // refuses the simdoc verb, so resolve the isolated encoder.
        return isolated_manifest_from_hint(Path::new(&path));
    }
    locate_sibling_manifest(repo_root)
}

fn locate_sibling_manifest(repo_root: &Path) -> Result<PathBuf, String> {
    let isolated = repo_root
        .parent()
        .unwrap_or(repo_root)
        .join("sim-tooling")
        .join("crates")
        .join("simdoc")
        .join("Cargo.toml");
    if isolated.is_file() {
        return Ok(isolated);
    }
    Err(
        "set SIMDOC_TOOLING_MANIFEST to the sim-tooling checkout or crates/simdoc/Cargo.toml"
            .to_owned(),
    )
}

fn isolated_manifest_from_hint(hint: &Path) -> Result<PathBuf, String> {
    let isolated_rel = Path::new("crates").join("simdoc").join("Cargo.toml");
    let candidates = [
        hint.to_path_buf(),
        hint.join("Cargo.toml"),
        hint.join(&isolated_rel),
        hint.parent().unwrap_or(hint).join(&isolated_rel),
    ];
    for candidate in candidates {
        if candidate.is_file() && candidate.ends_with(&isolated_rel) {
            return Ok(candidate);
        }
    }
    Err(format!(
        "SIMDOC_TOOLING_MANIFEST={} does not locate crates/simdoc/Cargo.toml",
        hint.display()
    ))
}

pub fn run_repo_tool(args: Vec<String>, command_name: &str) -> Result<(), String> {
    let program = args.first().map(String::as_str).unwrap_or("xtask");
    if args.get(1).map(String::as_str) != Some(command_name) {
        return Err(format!("usage: {program} {command_name} [--check]"));
    }

    let root = env::current_dir().map_err(|err| format!("current dir: {err}"))?;
    let manifest = locate_sim_tooling_manifest(&root)?;
    let mut command = Command::new("cargo");
    command.args(["run", "--manifest-path"]);
    command.arg(manifest);
    command.args(["--quiet", "--", command_name]);
    if command_name == "simdoc" {
        command.arg("--repo-root");
        command.arg(&root);
    } else if !has_repo_arg(&args) {
        command.arg("--repo");
        command.arg(&root);
    }
    for arg in args.iter().skip(2) {
        command.arg(arg);
    }

    let status = command
        .status()
        .map_err(|err| format!("run shared {command_name} tool: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "shared {command_name} tool failed with status {status}"
        ))
    }
}

fn has_repo_arg(args: &[String]) -> bool {
    args.iter()
        .skip(2)
        .any(|arg| arg == "--repo" || arg.starts_with("--repo="))
}

fn locate_sim_tooling_manifest(repo_root: &Path) -> Result<PathBuf, String> {
    if let Ok(path) = env::var("SIMDOC_TOOLING_MANIFEST") {
        return Ok(PathBuf::from(path));
    }
    let sibling = repo_root
        .parent()
        .unwrap_or(repo_root)
        .join("sim-tooling")
        .join("Cargo.toml");
    if sibling.is_file() {
        return Ok(sibling);
    }
    Err("set SIMDOC_TOOLING_MANIFEST to the sim-tooling Cargo.toml".to_owned())
}

#[cfg(test)]
mod tests {
    use super::{encoder_command, isolated_manifest_from_hint, locate_sibling_manifest};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "sim-compute-simdoc-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos()
            ));
            let simdoc = root.join("sim-tooling/crates/simdoc");
            fs::create_dir_all(&simdoc).expect("fixture directory");
            fs::write(simdoc.join("Cargo.toml"), "[package]\nname = \"simdoc\"\n")
                .expect("fixture manifest");
            fs::write(simdoc.join("Cargo.lock"), "# fixture\n").expect("fixture lock");
            fs::create_dir_all(root.join("repo")).expect("fixture repo");
            Self { root }
        }

        fn manifest(&self) -> PathBuf {
            self.root.join("sim-tooling/crates/simdoc/Cargo.toml")
        }

        fn repo(&self) -> PathBuf {
            self.root.join("repo")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn tooling_hints_resolve_to_the_isolated_encoder() {
        let fixture = Fixture::new();
        let selected = locate_sibling_manifest(&fixture.repo()).expect("isolated encoder");
        assert_eq!(selected, fixture.manifest());
        assert!(selected.with_file_name("Cargo.lock").is_file());

        let tooling = fixture.root.join("sim-tooling");
        assert_eq!(
            isolated_manifest_from_hint(&tooling.join("Cargo.toml")).expect("workspace hint"),
            selected
        );
        assert_eq!(
            isolated_manifest_from_hint(&selected).expect("isolated hint"),
            selected
        );
        assert_eq!(
            isolated_manifest_from_hint(&tooling).expect("checkout hint"),
            selected
        );
    }

    #[test]
    fn missing_tooling_hint_is_an_error() {
        let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("no-such-tooling")
            .join("Cargo.toml");
        let error = isolated_manifest_from_hint(&missing).expect_err("missing hint");
        assert!(error.contains("does not locate crates/simdoc/Cargo.toml"));
    }

    #[test]
    fn encoder_command_uses_the_isolated_manifest_offline() {
        let manifest = Path::new("sim-tooling/crates/simdoc/Cargo.toml");
        let root = Path::new("sim-compute");
        let command = encoder_command(root, manifest, &["--check".to_owned()]);
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--locked", "--offline"])
        );
        assert!(args.iter().any(|arg| Path::new(arg) == manifest));
        assert!(args.iter().any(|arg| arg == "--check"));
        assert!(args.iter().any(|arg| arg == "--repo-root"));
        assert!(!args.iter().any(|arg| arg == "--repo"));
    }
}
