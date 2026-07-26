//! Host probing and sanitization for physical acceptance.

use std::{path::Path, process::Command};

use crate::ComputeCliError;

const PRIVATE_KEYS: &[&str] = &[
    "hostname",
    "host",
    "user",
    "username",
    "path",
    "cwd",
    "home",
    "serial",
    "uuid",
    "environment",
    "env",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct GpuEvidence {
    pub(super) name: String,
    pub(super) driver: String,
    pub(super) power: String,
    pub(super) temperature: String,
}

pub(super) fn physical_gpu() -> Result<GpuEvidence, ComputeCliError> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|err| ComputeCliError::new(format!("run nvidia-smi: {err}")))?;
    if !output.status.success() {
        return Err(ComputeCliError::new("nvidia-smi did not report a GPU"));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|_| ComputeCliError::new("nvidia-smi output was not utf-8"))?;
    let row = stdout
        .lines()
        .find(|line| line.contains("RTX 5080"))
        .ok_or_else(|| ComputeCliError::new("no RTX 5080 GPU found"))?;
    let cols = row.split(',').map(str::trim).collect::<Vec<_>>();
    if cols.len() != 4 {
        return Err(ComputeCliError::new("unexpected nvidia-smi column count"));
    }
    Ok(GpuEvidence {
        name: cols[0].to_owned(),
        driver: cols[1].to_owned(),
        power: format!("{}W", cols[2]),
        temperature: format!("{}C", cols[3]),
    })
}

pub(super) fn sanitize_adapter(value: &str) -> Result<String, ComputeCliError> {
    let sanitized = sanitize_token(value, "adapter")?;
    if sanitized.contains("serial") || sanitized.contains("uuid") {
        return Err(ComputeCliError::new("adapter identity is not sanitized"));
    }
    Ok(sanitized)
}

pub(super) fn sanitize_token(value: &str, name: &str) -> Result<String, ComputeCliError> {
    if value.is_empty() || value.len() > 96 {
        return Err(ComputeCliError::new(format!("{name} is outside policy")));
    }
    if !value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ' ' | '-' | '_' | '.' | '/'))
    {
        return Err(ComputeCliError::new(format!(
            "{name} contains unsupported characters"
        )));
    }
    reject_private_value(value)?;
    Ok(value.to_owned())
}

pub(super) fn sanitize_measurement(value: &str, name: &str) -> Result<String, ComputeCliError> {
    if value.is_empty()
        || value.len() > 24
        || !value
            .chars()
            .all(|ch| ch.is_ascii_digit() || matches!(ch, '.' | 'W' | 'C'))
    {
        return Err(ComputeCliError::new(format!("{name} is outside policy")));
    }
    Ok(value.to_owned())
}

pub(super) fn reject_private_text(text: &str) -> Result<(), ComputeCliError> {
    for token in PRIVATE_KEYS {
        if text.to_ascii_lowercase().contains(token) {
            return Err(ComputeCliError::new(
                "private host data in acceptance artifact",
            ));
        }
    }
    Ok(())
}

pub(super) fn reject_private_value(value: &str) -> Result<(), ComputeCliError> {
    if value.contains('\\') || Path::new(value).is_absolute() || value.contains("..") {
        return Err(ComputeCliError::new(
            "private path data in acceptance artifact",
        ));
    }
    reject_private_text(value)
}
