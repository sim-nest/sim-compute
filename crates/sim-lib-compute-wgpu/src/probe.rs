//! Portable GPU adapter discovery and raw probe evidence.

use sim_lib_compute_auto::{ComputeDeviceIdentity, ComputeEvidenceKind, ComputePhysicalEvidence};
use wgpu::{Backends, Features, Limits};

/// Requested limits and optional features passed to `wgpu`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestedWgpuProfile {
    /// Requested device limits.
    pub limits: WgpuLimitEvidence,
    /// Whether timestamp queries were requested.
    pub timestamp_query: bool,
    /// Whether shader f16 was requested.
    pub shader_f16: bool,
}

impl RequestedWgpuProfile {
    /// Captures the profile requested by a platform capsule.
    pub fn from_parts(limits: Limits, features: Features) -> Self {
        Self {
            limits: WgpuLimitEvidence::from_limits(&limits),
            timestamp_query: features.contains(Features::TIMESTAMP_QUERY),
            shader_f16: features.contains(Features::SHADER_F16),
        }
    }
}

/// Limits recorded from either the requested or granted device contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuLimitEvidence {
    /// Maximum buffer size.
    pub max_buffer_size: u64,
    /// Maximum storage buffer binding size.
    pub max_storage_buffer_binding_size: u64,
    /// Maximum uniform buffer binding size.
    pub max_uniform_buffer_binding_size: u64,
    /// Minimum storage buffer offset alignment.
    pub min_storage_buffer_offset_alignment: u32,
    /// Minimum uniform buffer offset alignment.
    pub min_uniform_buffer_offset_alignment: u32,
    /// Maximum compute workgroups per dimension.
    pub max_compute_workgroups_per_dimension: u32,
    /// Maximum compute invocations per workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// Maximum compute workgroup size x.
    pub max_compute_workgroup_size_x: u32,
    /// Maximum compute workgroup size y.
    pub max_compute_workgroup_size_y: u32,
    /// Maximum compute workgroup size z.
    pub max_compute_workgroup_size_z: u32,
}

impl WgpuLimitEvidence {
    /// Captures stable limit evidence from a capsule-observed device.
    pub fn from_limits(limits: &Limits) -> Self {
        Self {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
            min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
        }
    }
}

/// Feature evidence recorded from a granted device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuCapabilityEvidence {
    /// Whether timestamp queries are granted.
    pub timestamp_query: bool,
    /// Whether shader f16 is granted.
    pub shader_f16: bool,
    /// Whether primary buffers may be mapped.
    pub mappable_primary_buffers: bool,
}

impl WgpuCapabilityEvidence {
    /// Captures stable feature evidence from a capsule-observed device.
    pub fn from_features(features: Features) -> Self {
        Self {
            timestamp_query: features.contains(Features::TIMESTAMP_QUERY),
            shader_f16: features.contains(Features::SHADER_F16),
            mappable_primary_buffers: features.contains(Features::MAPPABLE_PRIMARY_BUFFERS),
        }
    }
}

/// Adapter identity and requested/granted capability evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuAdapterEvidence {
    /// Deterministic ordinal assigned after sorting adapters.
    pub ordinal: usize,
    /// Diagnostic adapter name from `wgpu`.
    pub name: String,
    /// Diagnostic backend label from `wgpu`.
    pub backend: String,
    /// Diagnostic adapter type from `wgpu`.
    pub adapter_type: String,
    /// Diagnostic vendor id.
    pub vendor: u32,
    /// Diagnostic device id.
    pub device: u32,
    /// Requested profile.
    pub requested: RequestedWgpuProfile,
    /// Granted device limits.
    pub granted_limits: WgpuLimitEvidence,
    /// Granted features.
    pub granted_features: WgpuCapabilityEvidence,
}

impl WgpuAdapterEvidence {
    /// Sort key that keeps enumeration deterministic without treating identity
    /// as product logic.
    pub fn sort_key(&self) -> (&str, &str, &str, u32, u32) {
        (
            self.backend.as_str(),
            self.adapter_type.as_str(),
            self.name.as_str(),
            self.vendor,
            self.device,
        )
    }
}

/// Transfer and mapping probe evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferEvidence {
    /// Bytes written and read back.
    pub bytes: u64,
    /// Whether queue write plus copy completed.
    pub transfer_ok: bool,
    /// Whether map-read completed and matched the payload.
    pub mapping_ok: bool,
}

/// One bounded allocation attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocationAttempt {
    /// Attempted byte size.
    pub bytes: u64,
    /// Whether creating the buffer succeeded.
    pub success: bool,
}

/// Probe evidence required before a site is exported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeEvidence {
    /// Transfer and mapping evidence.
    pub transfer: TransferEvidence,
    /// Bounded allocation attempts.
    pub allocation_attempts: Vec<AllocationAttempt>,
}

impl ProbeEvidence {
    /// Returns true when all required probes succeeded.
    pub fn successful(&self) -> bool {
        self.transfer.transfer_ok
            && self.transfer.mapping_ok
            && self
                .allocation_attempts
                .iter()
                .any(|attempt| attempt.success && attempt.bytes > 0)
    }
}

/// One successful adapter probe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuAdapterProbe {
    /// Whether this evidence came from a real retained device or a synthetic fixture.
    pub evidence_kind: ComputeEvidenceKind,
    /// Claimed adapter identity for physical evidence verification.
    pub claimed_identity: Option<ComputeDeviceIdentity>,
    /// Observed adapter identity captured by the producer.
    pub observed_identity: Option<ComputeDeviceIdentity>,
    /// Adapter and capability evidence.
    pub adapter: WgpuAdapterEvidence,
    /// Raw probe evidence.
    pub probe: ProbeEvidence,
}

/// A successful adapter probe with the retained device context that produced it.
pub struct WgpuAdapterRuntime {
    pub(crate) probe: WgpuAdapterProbe,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

impl WgpuAdapterRuntime {
    /// Joins capsule-observed evidence to the retained device and queue that
    /// produced it. The provider never enumerates the host itself.
    pub fn new(probe: WgpuAdapterProbe, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            probe,
            device,
            queue,
        }
    }

    /// Returns the capsule-supplied probe evidence.
    pub fn probe(&self) -> &WgpuAdapterProbe {
        &self.probe
    }
}

/// Smallest host membrane consumed by the wgpu provider.
pub trait WgpuProbePort {
    /// Returns already-probed adapters with their retained execution handles.
    fn probe_wgpu(
        &self,
        policy: &ProbePolicy,
    ) -> Result<Vec<WgpuAdapterRuntime>, WgpuDiscoveryError>;
}

impl ComputePhysicalEvidence for WgpuAdapterProbe {
    fn evidence_kind(&self) -> ComputeEvidenceKind {
        self.evidence_kind
    }

    fn claimed_identity(&self) -> Option<&ComputeDeviceIdentity> {
        self.claimed_identity.as_ref()
    }

    fn observed_identity(&self) -> Option<&ComputeDeviceIdentity> {
        self.observed_identity.as_ref()
    }
}

/// Complete discovery result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WgpuDiscovery {
    /// Successful adapter probes, in deterministic order.
    pub adapters: Vec<WgpuAdapterProbe>,
    /// Diagnostic errors from adapters that did not become sites.
    pub diagnostics: Vec<String>,
}

impl WgpuDiscovery {
    /// Builds a discovery result and drops unsuccessful adapters from the site
    /// list while preserving their diagnostics.
    pub fn from_probes(probes: Vec<WgpuAdapterProbe>, mut diagnostics: Vec<String>) -> Self {
        let mut adapters = Vec::new();
        for probe in probes {
            if probe.probe.successful() {
                adapters.push(probe);
            } else {
                diagnostics.push(format!(
                    "wgpu adapter {} did not pass required probes",
                    probe.adapter.name
                ));
            }
        }
        adapters.sort_by(|left, right| left.adapter.sort_key().cmp(&right.adapter.sort_key()));
        for (ordinal, probe) in adapters.iter_mut().enumerate() {
            probe.adapter.ordinal = ordinal;
        }
        Self {
            adapters,
            diagnostics,
        }
    }
}

/// Bounded probe policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbePolicy {
    /// Backends to enumerate.
    pub backends: Backends,
    /// Bytes used by the transfer and map probe.
    pub transfer_bytes: u64,
    /// Largest allocation attempt, capped again by granted limits.
    pub max_allocation_probe_bytes: u64,
}

impl Default for ProbePolicy {
    fn default() -> Self {
        Self {
            backends: Backends::all(),
            transfer_bytes: 16,
            max_allocation_probe_bytes: 16 * 1024 * 1024,
        }
    }
}

/// Discovery failure for infrastructure-level probe setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuDiscoveryError {
    message: String,
}

impl WgpuDiscoveryError {
    /// Builds a bounded platform-probe diagnostic.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for WgpuDiscoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for WgpuDiscoveryError {}
