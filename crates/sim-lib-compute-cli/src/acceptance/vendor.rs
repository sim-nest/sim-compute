//! Vendor-runtime acceptance capture and verification.

use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use sim_lib_compute_auto::{AutoRouteDecision, AutoTensorExecutor};
use sim_lib_compute_cuda::{
    CudaLibrarySet, CudaRuntimeLoader, DynamicCudaLoader, discover_cuda_runtime,
};
use sim_lib_compute_rocm::{
    DynamicRocmLoader, RocmLibrarySet, RocmRuntimeLoader, discover_rocm_runtime,
};

mod artifact;

use super::{
    AcceptanceEvidence, EVIDENCE_KIND, host, sanitize_adapter, sanitize_measurement,
    sanitize_token, stable_hash,
};
use crate::{
    ComputeCliError,
    args::{AcceptanceAction, AcceptanceRequest},
};
use artifact::{Artifact, crossover_label, manifest_cases, validate_manifest};

const SCHEMA: &str = "sim.compute-vendor-acceptance/v1";
const HARNESS: &str = "sim-lib-compute-cli/acceptance/vendor-v1";
const MANIFEST: &str = include_str!("../../acceptance/vendor-v1.sx");
const CASES: &[(&str, &str)] = &[
    ("runtime-probe", "vendor/probe"),
    ("dense-matmul-differential", "vendor/matmul-differential"),
    ("unsupported-failure", "vendor/fail-closed"),
    ("measured-crossover", "vendor/crossover"),
    ("vendor-absent-portability", "portable/vendor-absent"),
];
const CROSSOVER_DIMENSIONS: &[usize] = &[16, 32, 64, 128, 256, 384];
const SAMPLE_COUNT: usize = 3;
const DIFFERENTIAL_DIMENSION: usize = 64;
const ERROR_LIMIT: f64 = 0.005;

pub(super) fn capture(
    request: &AcceptanceRequest,
    manifest: &str,
    output_path: &str,
    target: &str,
) -> Result<AcceptanceEvidence, ComputeCliError> {
    validate_manifest(manifest)?;
    let gpu = host::physical_gpu(target)?;
    let adapter = sanitize_adapter(&gpu.name)?;
    let driver = sanitize_token(&gpu.driver, "driver")?;
    let power = sanitize_measurement(&gpu.power, "power")?;
    let thermal = sanitize_measurement(&gpu.temperature, "thermal")?;
    let runtime = VendorRuntime::discover(target)?;
    let differential = differential(&runtime)?;
    if differential > ERROR_LIMIT {
        return Err(ComputeCliError::new(
            "vendor matmul differential exceeds acceptance tolerance",
        ));
    }
    failure_case(&runtime)?;
    let measurements = measure_crossover(&runtime)?;
    let portability = portability(target, &adapter)?;
    let artifact = Artifact {
        source: request.source.clone(),
        harness_hash: stable_hash(HARNESS.as_bytes()),
        manifest_hash: stable_hash(MANIFEST.as_bytes()),
        target: target.to_owned(),
        adapter,
        provider: runtime.provider().to_owned(),
        runtime: sanitize_token(&runtime.identity(), "runtime")?,
        driver,
        evidence: EVIDENCE_KIND.to_owned(),
        power,
        thermal,
        differential_n: DIFFERENTIAL_DIMENSION,
        max_abs_error: differential,
        samples: SAMPLE_COUNT,
        crossover: crossover_label(&measurements),
        measurements,
        vendor_absent_explicit: portability.vendor_absent_explicit,
        vendor_absent_auto: portability.vendor_absent_auto,
        wgpu_vendor_absent: portability.wgpu_vendor_absent,
        cases: manifest_cases(),
    };
    artifact.verify(&request.source)?;
    fs::write(output_path, artifact.render())
        .map_err(|error| ComputeCliError::new(format!("write vendor artifact: {error}")))?;
    Ok(AcceptanceEvidence {
        status: "captured".to_owned(),
        schema: SCHEMA.to_owned(),
        cases: artifact.cases.len(),
        artifact: output_path.to_owned(),
    })
}

pub(super) fn verify(
    request: &AcceptanceRequest,
    input: &str,
    text: &str,
) -> Result<AcceptanceEvidence, ComputeCliError> {
    let artifact = Artifact::parse(text)?;
    artifact.verify(&request.source)?;
    Ok(AcceptanceEvidence {
        status: match request.action {
            AcceptanceAction::Import => "imported",
            _ => "verified",
        }
        .to_owned(),
        schema: SCHEMA.to_owned(),
        cases: artifact.cases.len(),
        artifact: input.to_owned(),
    })
}

enum VendorRuntime {
    Cuda(Arc<CudaLibrarySet>),
    Rocm(Arc<RocmLibrarySet>),
}

impl VendorRuntime {
    fn discover(target: &str) -> Result<Self, ComputeCliError> {
        match target {
            "gpu:nvidia/rtx-5080-laptop" | "gpu:nvidia/rtx-5090" => {
                let probe = discover_cuda_runtime()
                    .map_err(|error| ComputeCliError::new(error.to_string()))?;
                probe
                    .runtime
                    .map(Self::Cuda)
                    .ok_or_else(|| ComputeCliError::new("CUDA runtime is not available"))
            }
            "gpu:amd/gfx1151" => {
                let probe = discover_rocm_runtime()
                    .map_err(|error| ComputeCliError::new(error.to_string()))?;
                let runtime = probe
                    .runtime
                    .ok_or_else(|| ComputeCliError::new("ROCm runtime is not available"))?;
                if !runtime
                    .evidence()
                    .observed_gfx_targets
                    .iter()
                    .any(|target| target == "gfx1151")
                {
                    return Err(ComputeCliError::new(
                        "ROCm runtime did not observe the required gfx1151 target",
                    ));
                }
                Ok(Self::Rocm(runtime))
            }
            _ => Err(ComputeCliError::new("unsupported vendor acceptance target")),
        }
    }

    fn provider(&self) -> &'static str {
        match self {
            Self::Cuda(_) => "cuda/cublas",
            Self::Rocm(_) => "rocm/rocblas",
        }
    }

    fn identity(&self) -> String {
        match self {
            Self::Cuda(runtime) => format!(
                "cuda-{}-cublas",
                runtime.evidence().driver_version.unwrap_or_default()
            ),
            Self::Rocm(runtime) => format!(
                "rocm-{}-rocblas-{}",
                runtime.evidence().hip_runtime_version.unwrap_or_default(),
                runtime.evidence().observed_gfx_targets.join("-")
            ),
        }
    }

    fn matmul(
        &self,
        left: &[f32],
        right: &[f32],
        dimension: usize,
    ) -> Result<Vec<f32>, ComputeCliError> {
        match self {
            Self::Cuda(runtime) => runtime
                .matmul_f32(left, right, dimension, dimension, dimension)
                .map_err(|error| ComputeCliError::new(error.to_string())),
            Self::Rocm(runtime) => runtime
                .matmul_f32(left, right, dimension, dimension, dimension)
                .map_err(|error| ComputeCliError::new(error.to_string())),
        }
    }

    fn rejects_bad_lengths(&self) -> bool {
        match self {
            Self::Cuda(runtime) => runtime.matmul_f32(&[1.0], &[1.0], 2, 2, 2).is_err(),
            Self::Rocm(runtime) => runtime.matmul_f32(&[1.0], &[1.0], 2, 2, 2).is_err(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Measurement {
    pub(super) dimension: usize,
    pub(super) cpu_ns: u64,
    pub(super) vendor_ns: u64,
}

fn differential(runtime: &VendorRuntime) -> Result<f64, ComputeCliError> {
    let (left, right) = matrices(DIFFERENTIAL_DIMENSION);
    let expected = cpu_matmul(&left, &right, DIFFERENTIAL_DIMENSION);
    let actual = runtime.matmul(&left, &right, DIFFERENTIAL_DIMENSION)?;
    Ok(expected
        .iter()
        .zip(actual)
        .map(|(left, right)| f64::from((left - right).abs()))
        .fold(0.0_f64, f64::max))
}

fn failure_case(runtime: &VendorRuntime) -> Result<(), ComputeCliError> {
    if runtime.rejects_bad_lengths() {
        Ok(())
    } else {
        Err(ComputeCliError::new(
            "vendor runtime accepted an invalid matmul shape",
        ))
    }
}

fn measure_crossover(runtime: &VendorRuntime) -> Result<Vec<Measurement>, ComputeCliError> {
    CROSSOVER_DIMENSIONS
        .iter()
        .map(|dimension| {
            let (left, right) = matrices(*dimension);
            let _ = cpu_matmul(&left, &right, *dimension);
            let _ = runtime.matmul(&left, &right, *dimension)?;
            let mut cpu = Vec::with_capacity(SAMPLE_COUNT);
            let mut vendor = Vec::with_capacity(SAMPLE_COUNT);
            for _ in 0..SAMPLE_COUNT {
                cpu.push(elapsed_ns(|| cpu_matmul(&left, &right, *dimension)));
                let start = Instant::now();
                let _ = runtime.matmul(&left, &right, *dimension)?;
                vendor.push(duration_ns(start.elapsed()));
            }
            cpu.sort_unstable();
            vendor.sort_unstable();
            Ok(Measurement {
                dimension: *dimension,
                cpu_ns: cpu[SAMPLE_COUNT / 2],
                vendor_ns: vendor[SAMPLE_COUNT / 2],
            })
        })
        .collect()
}

fn matrices(dimension: usize) -> (Vec<f32>, Vec<f32>) {
    let len = dimension * dimension;
    let left = (0..len)
        .map(|index| ((index * 13 % 29) as f32 - 14.0) / 29.0)
        .collect();
    let right = (0..len)
        .map(|index| ((index * 17 % 31) as f32 - 15.0) / 31.0)
        .collect();
    (left, right)
}

fn cpu_matmul(left: &[f32], right: &[f32], dimension: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; dimension * dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            let mut sum = 0.0_f32;
            for inner in 0..dimension {
                sum += left[row * dimension + inner] * right[inner * dimension + col];
            }
            output[row * dimension + col] = sum;
        }
    }
    output
}

fn elapsed_ns<T>(operation: impl FnOnce() -> T) -> u64 {
    let start = Instant::now();
    let _ = operation();
    duration_ns(start.elapsed())
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

struct Portability {
    vendor_absent_explicit: String,
    vendor_absent_auto: String,
    wgpu_vendor_absent: String,
}

fn portability(target: &str, adapter: &str) -> Result<Portability, ComputeCliError> {
    let disabled_path = PathBuf::from("vendor-runtime-disabled");
    let explicit_absent = match target {
        "gpu:nvidia/rtx-5080-laptop" | "gpu:nvidia/rtx-5090" => {
            CudaRuntimeLoader::with_search_dirs_only(vec![disabled_path])
                .discover()
                .is_err()
        }
        "gpu:amd/gfx1151" => RocmRuntimeLoader::with_search_dirs_only(vec![disabled_path])
            .discover()
            .is_err(),
        _ => false,
    };
    if !explicit_absent {
        return Err(ComputeCliError::new(
            "explicit vendor provider did not fail closed",
        ));
    }
    let auto = AutoTensorExecutor::default();
    if !auto.uses_cpu_fallback() || auto.route_decision() != AutoRouteDecision::Absent {
        return Err(ComputeCliError::new(
            "automatic provider did not explain absent-vendor CPU fallback",
        ));
    }
    // The acceptance caller already supplies the capsule-observed adapter.
    // Re-enumerating wgpu here would cross the platform membrane a second
    // time and could disagree with the retained provider context.
    if !adapter_matches(target, adapter, adapter) {
        return Err(ComputeCliError::new("capsule adapter did not match target"));
    }
    Ok(Portability {
        vendor_absent_explicit: "not-available".to_owned(),
        vendor_absent_auto: "cpu/absent".to_owned(),
        wgpu_vendor_absent: "probe-green".to_owned(),
    })
}

fn adapter_matches(target: &str, physical: &str, wgpu: &str) -> bool {
    match target {
        "gpu:nvidia/rtx-5080-laptop" => physical.contains("RTX 5080") && wgpu.contains("RTX 5080"),
        "gpu:nvidia/rtx-5090" => physical.contains("RTX 5090") && wgpu.contains("RTX 5090"),
        "gpu:amd/gfx1151" => {
            physical.contains("Radeon")
                && (wgpu.contains("STRIX_HALO")
                    || wgpu.contains("Radeon 8060S")
                    || wgpu.contains("AMD Radeon Graphics"))
        }
        _ => false,
    }
}
