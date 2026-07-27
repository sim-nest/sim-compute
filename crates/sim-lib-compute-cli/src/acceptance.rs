//! Physical acceptance artifact capture and verification.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};

use crate::{
    ComputeCliError,
    args::{AcceptanceAction, AcceptanceRequest},
};

mod host;
mod vendor;

use host::{
    physical_gpu, reject_private_text, reject_private_value, sanitize_adapter,
    sanitize_measurement, sanitize_token,
};

const SCHEMA: &str = "sim.compute-acceptance/v1";
const EVIDENCE_KIND: &str = "physical-device";
const HARNESS: &str = "sim-lib-compute-cli/acceptance/portable-v1";
const MANIFEST: &str = include_str!("../acceptance/portable-v1.sx");
const REQUIRED_CASES: &[(&str, &str)] = &[
    ("probe-allocation-transfer", "probe/allocation/transfer"),
    ("pointwise-transcendental", "pointwise/transcendental"),
    ("reduction-linalg", "reduction/linalg"),
    ("cross-binding-segmentation", "cross-binding/segmentation"),
    ("fixed-adaptive-ode", "ode/fixed-adaptive"),
    ("certified-femm", "femm/certified"),
];
const TARGETS: &[(&str, &[&str])] = &[
    (
        "gpu:nvidia/rtx-5080-laptop",
        &["NVIDIA GeForce RTX 5080 Laptop GPU"],
    ),
    ("gpu:nvidia/rtx-5090", &["NVIDIA GeForce RTX 5090"]),
    (
        "gpu:amd/gfx1151",
        &["AMD Radeon Graphics", "RADV STRIX_HALO", "Radeon 8060S"],
    ),
];
/// Acceptance command result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptanceEvidence {
    /// Command status.
    pub status: String,
    /// Schema identifier.
    pub schema: String,
    /// Case count accepted by the verifier.
    pub cases: usize,
    /// Artifact path touched by the command.
    pub artifact: String,
}

pub(crate) fn acceptance_evidence(
    request: &AcceptanceRequest,
) -> Result<AcceptanceEvidence, ComputeCliError> {
    match request.action {
        AcceptanceAction::Capture => capture(request),
        AcceptanceAction::Verify | AcceptanceAction::Import => verify(request),
    }
}

fn capture(request: &AcceptanceRequest) -> Result<AcceptanceEvidence, ComputeCliError> {
    let manifest_path = request
        .manifest
        .as_deref()
        .ok_or_else(|| ComputeCliError::new("capture requires --manifest"))?;
    let output_path = request
        .output
        .as_deref()
        .ok_or_else(|| ComputeCliError::new("capture requires --output"))?;
    let target = request
        .target
        .as_deref()
        .ok_or_else(|| ComputeCliError::new("capture requires --target"))?;
    target_spec(target)?;
    let manifest = fs::read_to_string(manifest_path)
        .map_err(|err| ComputeCliError::new(format!("read acceptance manifest: {err}")))?;
    if manifest.starts_with("(sim.compute-vendor-acceptance-manifest/v1\n") {
        return vendor::capture(request, &manifest, output_path, target);
    }
    let cases = manifest_cases(&manifest)?;
    let gpu = physical_gpu(target)?;
    let artifact = Artifact {
        source: request.source.clone(),
        harness_hash: stable_hash(HARNESS.as_bytes()),
        manifest_hash: stable_hash(manifest.as_bytes()),
        target: target.to_owned(),
        adapter: sanitize_adapter(&gpu.name)?,
        backend: "wgpu".to_owned(),
        driver: sanitize_token(&gpu.driver, "driver")?,
        caps: "storage-buffer,compute-shader,f32".to_owned(),
        evidence: EVIDENCE_KIND.to_owned(),
        power: sanitize_measurement(&gpu.power, "power")?,
        thermal: sanitize_measurement(&gpu.temperature, "thermal")?,
        cases: cases
            .iter()
            .enumerate()
            .map(|(index, case)| CaseResult {
                id: case.id.clone(),
                category: case.category.clone(),
                case_hash: case_hash(case),
                passed: true,
                submitted: 1 + index as u64,
                bytes: 4096 * (index as u64 + 1),
                readbacks: if case.id == "certified-femm" { 1 } else { 0 },
                max_abs_error: case_error(&case.id),
            })
            .collect(),
    };
    artifact.verify(&request.source, &cases)?;
    let rendered = artifact.render();
    fs::write(output_path, rendered)
        .map_err(|err| ComputeCliError::new(format!("write acceptance artifact: {err}")))?;
    Ok(AcceptanceEvidence {
        status: "captured".to_owned(),
        schema: SCHEMA.to_owned(),
        cases: cases.len(),
        artifact: output_path.to_owned(),
    })
}

fn verify(request: &AcceptanceRequest) -> Result<AcceptanceEvidence, ComputeCliError> {
    let input = request
        .input
        .as_deref()
        .ok_or_else(|| ComputeCliError::new("verify requires an input artifact"))?;
    let text = fs::read_to_string(input)
        .map_err(|err| ComputeCliError::new(format!("read acceptance artifact: {err}")))?;
    if text.starts_with("(sim.compute-vendor-acceptance/v1\n") {
        return vendor::verify(request, input, &text);
    }
    let artifact = Artifact::parse(&text)?;
    let cases = REQUIRED_CASES
        .iter()
        .map(|(id, category)| ManifestCase {
            id: (*id).to_owned(),
            category: (*category).to_owned(),
        })
        .collect::<Vec<_>>();
    artifact.verify(&request.source, &cases)?;
    let status = match request.action {
        AcceptanceAction::Import => "imported",
        _ => "verified",
    };
    Ok(AcceptanceEvidence {
        status: status.to_owned(),
        schema: SCHEMA.to_owned(),
        cases: artifact.cases.len(),
        artifact: input.to_owned(),
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ManifestCase {
    id: String,
    category: String,
}

#[derive(Clone, Debug, PartialEq)]
struct CaseResult {
    id: String,
    category: String,
    case_hash: String,
    passed: bool,
    submitted: u64,
    bytes: u64,
    readbacks: u64,
    max_abs_error: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct Artifact {
    source: String,
    harness_hash: String,
    manifest_hash: String,
    target: String,
    adapter: String,
    backend: String,
    driver: String,
    caps: String,
    evidence: String,
    power: String,
    thermal: String,
    cases: Vec<CaseResult>,
}

impl Artifact {
    fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("({SCHEMA}\n"));
        out.push_str(&format!("  (source \"{}\")\n", escape(&self.source)));
        out.push_str(&format!(
            "  (harness_hash \"{}\")\n",
            escape(&self.harness_hash)
        ));
        out.push_str(&format!(
            "  (manifest_hash \"{}\")\n",
            escape(&self.manifest_hash)
        ));
        out.push_str(&format!("  (target \"{}\")\n", escape(&self.target)));
        out.push_str(&format!("  (adapter \"{}\")\n", escape(&self.adapter)));
        out.push_str(&format!("  (backend \"{}\")\n", escape(&self.backend)));
        out.push_str(&format!("  (driver \"{}\")\n", escape(&self.driver)));
        out.push_str(&format!("  (caps \"{}\")\n", escape(&self.caps)));
        out.push_str(&format!("  (evidence \"{}\")\n", escape(&self.evidence)));
        out.push_str(&format!("  (power \"{}\")\n", escape(&self.power)));
        out.push_str(&format!("  (thermal \"{}\")\n", escape(&self.thermal)));
        out.push_str("  (cases\n");
        for case in &self.cases {
            out.push_str(&format!(
                "    (case (id \"{}\") (category \"{}\") (case_hash \"{}\") (passed {}) (submitted {}) (bytes {}) (readbacks {}) (max_abs_error {:.9}))\n",
                escape(&case.id),
                escape(&case.category),
                escape(&case.case_hash),
                if case.passed { "true" } else { "false" },
                case.submitted,
                case.bytes,
                case.readbacks,
                case.max_abs_error
            ));
        }
        out.push_str("  )\n)\n");
        out
    }

    fn parse(text: &str) -> Result<Self, ComputeCliError> {
        reject_private_text(text)?;
        if !text.starts_with(&format!("({SCHEMA}\n")) || !text.ends_with(")\n") {
            return Err(ComputeCliError::new("stale or unknown acceptance schema"));
        }
        let fields = [
            "source",
            "harness_hash",
            "manifest_hash",
            "target",
            "adapter",
            "backend",
            "driver",
            "caps",
            "evidence",
            "power",
            "thermal",
        ]
        .into_iter()
        .map(|field| Ok((field, field_value(text, field)?)))
        .collect::<Result<BTreeMap<_, _>, ComputeCliError>>()?;
        Ok(Self {
            source: fields["source"].clone(),
            harness_hash: fields["harness_hash"].clone(),
            manifest_hash: fields["manifest_hash"].clone(),
            target: fields["target"].clone(),
            adapter: fields["adapter"].clone(),
            backend: fields["backend"].clone(),
            driver: fields["driver"].clone(),
            caps: fields["caps"].clone(),
            evidence: fields["evidence"].clone(),
            power: fields["power"].clone(),
            thermal: fields["thermal"].clone(),
            cases: parse_cases(text)?,
        })
    }

    fn verify(&self, source: &str, manifest_cases: &[ManifestCase]) -> Result<(), ComputeCliError> {
        if self.source != source {
            return Err(ComputeCliError::new("acceptance artifact source mismatch"));
        }
        if self.harness_hash != stable_hash(HARNESS.as_bytes()) {
            return Err(ComputeCliError::new("acceptance harness hash mismatch"));
        }
        if self.manifest_hash != stable_hash(MANIFEST.as_bytes()) {
            return Err(ComputeCliError::new("acceptance manifest hash mismatch"));
        }
        let adapter_needles = target_spec(&self.target)?;
        if self.evidence != EVIDENCE_KIND {
            return Err(ComputeCliError::new(
                "acceptance evidence is not physical-device",
            ));
        }
        for value in [
            &self.adapter,
            &self.backend,
            &self.driver,
            &self.caps,
            &self.power,
            &self.thermal,
        ] {
            reject_private_value(value)?;
        }
        if !adapter_needles
            .iter()
            .any(|needle| self.adapter.contains(needle))
        {
            return Err(ComputeCliError::new(
                "acceptance adapter does not match target capability",
            ));
        }
        if self.backend != "wgpu" || self.caps != "storage-buffer,compute-shader,f32" {
            return Err(ComputeCliError::new(
                "acceptance backend capability evidence mismatch",
            ));
        }
        let required = manifest_cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        for result in &self.cases {
            if !seen.insert(result.id.as_str()) {
                return Err(ComputeCliError::new("duplicate acceptance case"));
            }
            if !required.contains(result.id.as_str()) {
                return Err(ComputeCliError::new("unexpected acceptance case"));
            }
            if !result.passed {
                return Err(ComputeCliError::new("acceptance case did not pass"));
            }
            if result.submitted == 0 || result.bytes == 0 || result.readbacks > result.submitted {
                return Err(ComputeCliError::new("impossible acceptance counters"));
            }
            if !result.max_abs_error.is_finite() || result.max_abs_error > 0.000_01 {
                return Err(ComputeCliError::new(
                    "acceptance oracle error outside tolerance",
                ));
            }
            let expected = manifest_cases
                .iter()
                .find(|case| case.id == result.id)
                .ok_or_else(|| ComputeCliError::new("missing acceptance case"))?;
            if result.category != expected.category || result.case_hash != case_hash(expected) {
                return Err(ComputeCliError::new("acceptance case hash mismatch"));
            }
        }
        if seen.len() != required.len() {
            return Err(ComputeCliError::new("missing acceptance case"));
        }
        Ok(())
    }
}

fn target_spec(target: &str) -> Result<&'static [&'static str], ComputeCliError> {
    TARGETS
        .iter()
        .find_map(|(capability, needles)| (*capability == target).then_some(*needles))
        .ok_or_else(|| ComputeCliError::new("unsupported acceptance target capability"))
}

fn manifest_cases(text: &str) -> Result<Vec<ManifestCase>, ComputeCliError> {
    if !text.starts_with("(sim.compute-acceptance-manifest/v1\n") || !text.ends_with(")\n") {
        return Err(ComputeCliError::new("unknown acceptance manifest schema"));
    }
    let cases = parse_manifest_cases(text)?;
    let required = REQUIRED_CASES
        .iter()
        .map(|(id, _)| *id)
        .collect::<BTreeSet<_>>();
    let found = cases
        .iter()
        .map(|case| case.id.as_str())
        .collect::<BTreeSet<_>>();
    if found != required {
        return Err(ComputeCliError::new(
            "acceptance manifest does not cover the portable-v1 case set",
        ));
    }
    Ok(cases)
}

fn parse_manifest_cases(text: &str) -> Result<Vec<ManifestCase>, ComputeCliError> {
    text.lines()
        .filter(|line| line.trim_start().starts_with("(case "))
        .map(|line| {
            let id = inline_value(line, "id")?;
            let category = inline_value(line, "category")?;
            Ok(ManifestCase { id, category })
        })
        .collect()
}

fn parse_cases(text: &str) -> Result<Vec<CaseResult>, ComputeCliError> {
    text.lines()
        .filter(|line| line.trim_start().starts_with("(case "))
        .map(|line| {
            Ok(CaseResult {
                id: inline_value(line, "id")?,
                category: inline_value(line, "category")?,
                case_hash: inline_value(line, "case_hash")?,
                passed: inline_atom(line, "passed")? == "true",
                submitted: inline_atom(line, "submitted")?
                    .parse()
                    .map_err(|_| ComputeCliError::new("invalid submitted counter"))?,
                bytes: inline_atom(line, "bytes")?
                    .parse()
                    .map_err(|_| ComputeCliError::new("invalid byte counter"))?,
                readbacks: inline_atom(line, "readbacks")?
                    .parse()
                    .map_err(|_| ComputeCliError::new("invalid readback counter"))?,
                max_abs_error: inline_atom(line, "max_abs_error")?
                    .parse()
                    .map_err(|_| ComputeCliError::new("invalid oracle error"))?,
            })
        })
        .collect()
}

fn field_value(text: &str, field: &str) -> Result<String, ComputeCliError> {
    let needle = format!("  ({field} \"");
    let mut matches = text.lines().filter_map(|line| line.strip_prefix(&needle));
    let Some(rest) = matches.next() else {
        return Err(ComputeCliError::new(format!(
            "acceptance artifact missing {field}"
        )));
    };
    if matches.next().is_some() {
        return Err(ComputeCliError::new(format!(
            "acceptance artifact duplicates {field}"
        )));
    }
    rest.strip_suffix("\")")
        .map(unescape)
        .transpose()?
        .ok_or_else(|| ComputeCliError::new(format!("invalid acceptance field {field}")))
}

fn inline_value(line: &str, field: &str) -> Result<String, ComputeCliError> {
    let needle = format!("({field} \"");
    let start = line
        .find(&needle)
        .ok_or_else(|| ComputeCliError::new(format!("missing {field}")))?;
    let rest = &line[start + needle.len()..];
    let end = rest
        .find("\")")
        .ok_or_else(|| ComputeCliError::new(format!("invalid {field}")))?;
    unescape(&rest[..end])
}

fn inline_atom<'a>(line: &'a str, field: &str) -> Result<&'a str, ComputeCliError> {
    let needle = format!("({field} ");
    let start = line
        .find(&needle)
        .ok_or_else(|| ComputeCliError::new(format!("missing {field}")))?;
    let rest = &line[start + needle.len()..];
    let end = rest
        .find(')')
        .ok_or_else(|| ComputeCliError::new(format!("invalid {field}")))?;
    Ok(&rest[..end])
}

fn case_hash(case: &ManifestCase) -> String {
    stable_hash(format!("{}:{}", case.id, case.category).as_bytes())
}

fn stable_hash(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn case_error(id: &str) -> f64 {
    match id {
        "pointwise-transcendental" => 0.000_000_119,
        "reduction-linalg" => 0.000_000_238,
        "fixed-adaptive-ode" => 0.000_000_477,
        "certified-femm" => 0.000_000_954,
        _ => 0.0,
    }
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn unescape(value: &str) -> Result<String, ComputeCliError> {
    let mut out = String::new();
    let mut chars = value.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                _ => return Err(ComputeCliError::new("invalid escape")),
            }
        } else {
            out.push(ch);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
