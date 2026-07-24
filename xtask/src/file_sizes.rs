//! Repository source-size gate.

use std::process::Command;

pub fn run() -> Result<(), String> {
    let status = Command::new("sh")
        .arg("scripts/check-file-sizes.sh")
        .status()
        .map_err(|err| format!("run file-size checker: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("file-size checker failed with status {status}"))
    }
}
