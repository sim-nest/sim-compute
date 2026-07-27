//! Sanitized vendor acceptance artifact encoding and verification.

use super::{
    CASES, CROSSOVER_DIMENSIONS, DIFFERENTIAL_DIMENSION, ERROR_LIMIT, HARNESS, MANIFEST,
    Measurement, SAMPLE_COUNT, SCHEMA,
};
use crate::{
    ComputeCliError,
    acceptance::{
        EVIDENCE_KIND, ManifestCase, case_hash, field_value, parse_manifest_cases,
        reject_private_text, reject_private_value, stable_hash,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Artifact {
    pub(super) source: String,
    pub(super) harness_hash: String,
    pub(super) manifest_hash: String,
    pub(super) target: String,
    pub(super) adapter: String,
    pub(super) provider: String,
    pub(super) runtime: String,
    pub(super) driver: String,
    pub(super) evidence: String,
    pub(super) power: String,
    pub(super) thermal: String,
    pub(super) differential_n: usize,
    pub(super) max_abs_error: f64,
    pub(super) samples: usize,
    pub(super) crossover: String,
    pub(super) measurements: Vec<Measurement>,
    pub(super) vendor_absent_explicit: String,
    pub(super) vendor_absent_auto: String,
    pub(super) wgpu_vendor_absent: String,
    pub(super) cases: Vec<ManifestCase>,
}

impl Artifact {
    pub(super) fn render(&self) -> String {
        let measurements = self
            .measurements
            .iter()
            .map(|row| {
                format!(
                    "n{}.cpu{}.vendor{}",
                    row.dimension, row.cpu_ns, row.vendor_ns
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let mut output = format!(
            "({SCHEMA}\n  (source \"{}\")\n  (harness_hash \"{}\")\n  (manifest_hash \"{}\")\n  (target \"{}\")\n  (adapter \"{}\")\n  (provider \"{}\")\n  (runtime \"{}\")\n  (driver \"{}\")\n  (evidence \"{}\")\n  (power \"{}\")\n  (thermal \"{}\")\n  (differential_n \"{}\")\n  (max_abs_error \"{:.9}\")\n  (samples \"{}\")\n  (crossover \"{}\")\n  (measurements \"{}\")\n  (vendor_absent_explicit \"{}\")\n  (vendor_absent_auto \"{}\")\n  (wgpu_vendor_absent \"{}\")\n  (cases\n",
            self.source,
            self.harness_hash,
            self.manifest_hash,
            self.target,
            self.adapter,
            self.provider,
            self.runtime,
            self.driver,
            self.evidence,
            self.power,
            self.thermal,
            self.differential_n,
            self.max_abs_error,
            self.samples,
            self.crossover,
            measurements,
            self.vendor_absent_explicit,
            self.vendor_absent_auto,
            self.wgpu_vendor_absent,
        );
        for case in &self.cases {
            output.push_str(&format!(
                "    (case (id \"{}\") (category \"{}\") (case_hash \"{}\") (passed true))\n",
                case.id,
                case.category,
                case_hash(case)
            ));
        }
        output.push_str("  )\n)\n");
        output
    }

    pub(super) fn parse(text: &str) -> Result<Self, ComputeCliError> {
        reject_private_text(text)?;
        if !text.starts_with(&format!("({SCHEMA}\n")) || !text.ends_with(")\n") {
            return Err(ComputeCliError::new("unknown vendor acceptance schema"));
        }
        let measurements = parse_measurements(&field_value(text, "measurements")?)?;
        Ok(Self {
            source: field_value(text, "source")?,
            harness_hash: field_value(text, "harness_hash")?,
            manifest_hash: field_value(text, "manifest_hash")?,
            target: field_value(text, "target")?,
            adapter: field_value(text, "adapter")?,
            provider: field_value(text, "provider")?,
            runtime: field_value(text, "runtime")?,
            driver: field_value(text, "driver")?,
            evidence: field_value(text, "evidence")?,
            power: field_value(text, "power")?,
            thermal: field_value(text, "thermal")?,
            differential_n: parse_field(text, "differential_n")?,
            max_abs_error: parse_field(text, "max_abs_error")?,
            samples: parse_field(text, "samples")?,
            crossover: field_value(text, "crossover")?,
            measurements,
            vendor_absent_explicit: field_value(text, "vendor_absent_explicit")?,
            vendor_absent_auto: field_value(text, "vendor_absent_auto")?,
            wgpu_vendor_absent: field_value(text, "wgpu_vendor_absent")?,
            cases: parse_artifact_cases(text)?,
        })
    }

    pub(super) fn verify(&self, source: &str) -> Result<(), ComputeCliError> {
        if self.source != source
            || self.harness_hash != stable_hash(HARNESS.as_bytes())
            || self.manifest_hash != stable_hash(MANIFEST.as_bytes())
        {
            return Err(ComputeCliError::new(
                "vendor acceptance source or harness mismatch",
            ));
        }
        if self.evidence != EVIDENCE_KIND
            || self.differential_n != DIFFERENTIAL_DIMENSION
            || !self.max_abs_error.is_finite()
            || self.max_abs_error > ERROR_LIMIT
            || self.samples != SAMPLE_COUNT
            || self.measurements.len() != CROSSOVER_DIMENSIONS.len()
        {
            return Err(ComputeCliError::new(
                "vendor acceptance measurements are invalid",
            ));
        }
        verify_target_provider(&self.target, &self.adapter, &self.provider)?;
        for value in [
            &self.adapter,
            &self.provider,
            &self.runtime,
            &self.driver,
            &self.power,
            &self.thermal,
            &self.crossover,
            &self.vendor_absent_explicit,
            &self.vendor_absent_auto,
            &self.wgpu_vendor_absent,
        ] {
            reject_private_value(value)?;
        }
        if self.vendor_absent_explicit != "not-available"
            || self.vendor_absent_auto != "cpu/absent"
            || self.wgpu_vendor_absent != "probe-green"
        {
            return Err(ComputeCliError::new(
                "vendor-absent portability evidence is incomplete",
            ));
        }
        for (measurement, expected) in self.measurements.iter().zip(CROSSOVER_DIMENSIONS) {
            if measurement.dimension != *expected
                || measurement.cpu_ns == 0
                || measurement.vendor_ns == 0
            {
                return Err(ComputeCliError::new(
                    "vendor crossover measurements are invalid",
                ));
            }
        }
        if self.crossover != crossover_label(&self.measurements) {
            return Err(ComputeCliError::new(
                "vendor crossover label does not match measurements",
            ));
        }
        if self.cases != manifest_cases() {
            return Err(ComputeCliError::new(
                "vendor acceptance case set is incomplete",
            ));
        }
        Ok(())
    }
}

pub(super) fn validate_manifest(text: &str) -> Result<(), ComputeCliError> {
    let cases = parse_manifest_cases(text)?;
    if cases == manifest_cases() && text == MANIFEST {
        Ok(())
    } else {
        Err(ComputeCliError::new(
            "vendor acceptance manifest does not match vendor-v1",
        ))
    }
}

pub(super) fn manifest_cases() -> Vec<ManifestCase> {
    CASES
        .iter()
        .map(|(id, category)| ManifestCase {
            id: (*id).to_owned(),
            category: (*category).to_owned(),
        })
        .collect()
}

fn parse_artifact_cases(text: &str) -> Result<Vec<ManifestCase>, ComputeCliError> {
    let cases = parse_manifest_cases(text)?;
    for case in &cases {
        let line = text
            .lines()
            .find(|line| line.contains(&format!("(id \"{}\")", case.id)))
            .ok_or_else(|| ComputeCliError::new("vendor acceptance case is missing"))?;
        if !line.contains("(passed true)")
            || !line.contains(&format!("(case_hash \"{}\")", case_hash(case)))
        {
            return Err(ComputeCliError::new("vendor acceptance case failed"));
        }
    }
    Ok(cases)
}

fn parse_measurements(value: &str) -> Result<Vec<Measurement>, ComputeCliError> {
    value
        .split(',')
        .map(|row| {
            let mut fields = row.split('.');
            let dimension = parse_measurement_field(fields.next(), "n", "dimension")?;
            let cpu_ns = parse_measurement_field(fields.next(), "cpu", "CPU time")?;
            let vendor_ns = parse_measurement_field(fields.next(), "vendor", "vendor time")?;
            if fields.next().is_some() {
                return Err(ComputeCliError::new("invalid crossover measurement"));
            }
            Ok(Measurement {
                dimension,
                cpu_ns,
                vendor_ns,
            })
        })
        .collect()
}

fn parse_measurement_field<T: std::str::FromStr>(
    value: Option<&str>,
    prefix: &str,
    name: &str,
) -> Result<T, ComputeCliError> {
    value
        .and_then(|value| value.strip_prefix(prefix))
        .ok_or_else(|| ComputeCliError::new(format!("invalid crossover {name}")))?
        .parse()
        .map_err(|_| ComputeCliError::new(format!("invalid crossover {name}")))
}

fn parse_field<T: std::str::FromStr>(text: &str, name: &str) -> Result<T, ComputeCliError> {
    field_value(text, name)?
        .parse()
        .map_err(|_| ComputeCliError::new(format!("invalid vendor field {name}")))
}

pub(super) fn crossover_label(measurements: &[Measurement]) -> String {
    measurements
        .iter()
        .find(|row| row.vendor_ns < row.cpu_ns)
        .map(|row| format!("n{}", row.dimension))
        .unwrap_or_else(|| {
            format!(
                "not-observed-through-n{}",
                measurements.last().map_or(0, |row| row.dimension)
            )
        })
}

fn verify_target_provider(
    target: &str,
    adapter: &str,
    provider: &str,
) -> Result<(), ComputeCliError> {
    let valid = match target {
        "gpu:nvidia/rtx-5080-laptop" => adapter.contains("RTX 5080") && provider == "cuda/cublas",
        "gpu:nvidia/rtx-5090" => adapter.contains("RTX 5090") && provider == "cuda/cublas",
        "gpu:amd/gfx1151" => {
            (adapter.contains("Radeon") || adapter.contains("STRIX_HALO"))
                && provider == "rocm/rocblas"
        }
        _ => false,
    };
    valid
        .then_some(())
        .ok_or_else(|| ComputeCliError::new("vendor target/provider evidence mismatch"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Artifact {
        let measurements = CROSSOVER_DIMENSIONS
            .iter()
            .map(|dimension| Measurement {
                dimension: *dimension,
                cpu_ns: *dimension as u64 * 100,
                vendor_ns: if *dimension < 64 {
                    *dimension as u64 * 200
                } else {
                    *dimension as u64 * 50
                },
            })
            .collect::<Vec<_>>();
        Artifact {
            source: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            harness_hash: stable_hash(HARNESS.as_bytes()),
            manifest_hash: stable_hash(MANIFEST.as_bytes()),
            target: "gpu:nvidia/rtx-5090".to_owned(),
            adapter: "NVIDIA GeForce RTX 5090".to_owned(),
            provider: "cuda/cublas".to_owned(),
            runtime: "cuda-13000-cublas".to_owned(),
            driver: "595.84".to_owned(),
            evidence: EVIDENCE_KIND.to_owned(),
            power: "20W".to_owned(),
            thermal: "40C".to_owned(),
            differential_n: DIFFERENTIAL_DIMENSION,
            max_abs_error: 0.000_001,
            samples: SAMPLE_COUNT,
            crossover: crossover_label(&measurements),
            measurements,
            vendor_absent_explicit: "not-available".to_owned(),
            vendor_absent_auto: "cpu/absent".to_owned(),
            wgpu_vendor_absent: "probe-green".to_owned(),
            cases: manifest_cases(),
        }
    }

    #[test]
    fn sanitized_vendor_artifact_round_trips_and_rejects_tampering() {
        let fixture = fixture();
        let source = fixture.source.clone();
        fixture.verify(&source).unwrap();
        let rendered = fixture.render();
        let parsed = Artifact::parse(&rendered).unwrap();
        assert_eq!(parsed, fixture);
        assert!(Artifact::parse(&rendered.replace("(passed true)", "(passed false)")).is_err());
        let mut wrong = parsed;
        wrong.crossover = "n16".to_owned();
        assert!(wrong.verify(&source).is_err());
    }
}
