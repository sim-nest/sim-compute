//! Measured compute profiles, bounded persistence, and conservative routing.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use sim_kernel::Symbol;
use sim_lib_compute_model::ModeledComputeProfile;
use sim_lib_numbers_tensor::TensorRequest;

const DEFAULT_STALE_AFTER_TICKS: u64 = 10_000;
const CELL_BYTES: u64 = 8;

/// Adapter, driver, and backend identity for measured compute evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeDeviceIdentity {
    /// Stable adapter label.
    pub adapter: String,
    /// Driver label or version string.
    pub driver: String,
    /// Backend label, such as `wgpu`, `cuda`, `rocm`, or `modeled`.
    pub backend: String,
}

impl ComputeDeviceIdentity {
    /// Builds a device identity.
    pub fn new(
        adapter: impl Into<String>,
        driver: impl Into<String>,
        backend: impl Into<String>,
    ) -> Self {
        Self {
            adapter: adapter.into(),
            driver: driver.into(),
            backend: backend.into(),
        }
    }

    fn compatible_with(&self, other: &Self) -> bool {
        self == other
    }
}

/// Hardware and scheduling limits captured with a measured profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeProfileLimits {
    /// Maximum resident bytes accepted by the provider.
    pub max_resident_bytes: u64,
    /// Maximum storage binding bytes accepted by the provider.
    pub max_storage_binding_bytes: u64,
    /// Maximum queued submissions.
    pub max_queue_depth: usize,
    /// Maximum queued bytes.
    pub max_queue_bytes: u64,
    /// Maximum submission deadline in logical ticks.
    pub submission_deadline_ticks: u64,
}

impl From<&ModeledComputeProfile> for ComputeProfileLimits {
    fn from(profile: &ModeledComputeProfile) -> Self {
        Self {
            max_resident_bytes: profile.max_resident_bytes,
            max_storage_binding_bytes: profile.max_storage_binding_bytes,
            max_queue_depth: profile.max_queue_depth,
            max_queue_bytes: profile.max_queue_bytes,
            submission_deadline_ticks: profile.submission_deadline_ticks,
        }
    }
}

/// Transfer, launch, arithmetic, reduction, and matmul sample distributions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeProfileSamples {
    /// Upload bandwidth samples, in bytes per logical tick.
    pub upload_bytes_per_tick: Vec<u64>,
    /// Download bandwidth samples, in bytes per logical tick.
    pub download_bytes_per_tick: Vec<u64>,
    /// Launch latency samples, in logical ticks.
    pub launch_ticks: Vec<u64>,
    /// Element-wise throughput samples, in elements per logical tick.
    pub element_elements_per_tick: Vec<u64>,
    /// Reduction throughput samples, in elements per logical tick.
    pub reduction_elements_per_tick: Vec<u64>,
    /// Matmul throughput samples, in multiply-adds per logical tick.
    pub matmul_ops_per_tick: Vec<u64>,
}

impl ComputeProfileSamples {
    fn conclusive(&self) -> bool {
        [
            &self.upload_bytes_per_tick,
            &self.download_bytes_per_tick,
            &self.launch_ticks,
            &self.element_elements_per_tick,
            &self.reduction_elements_per_tick,
            &self.matmul_ops_per_tick,
        ]
        .into_iter()
        .all(|samples| samples.iter().any(|sample| *sample > 0))
    }
}

/// Thermal and power context captured beside benchmark samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeThermalPowerContext {
    /// Thermal context label.
    pub thermal: String,
    /// Power context label.
    pub power: String,
}

/// Provenance for a measured profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeProfileProvenance {
    /// Tool or library that produced the profile.
    pub producer: String,
    /// Logical measurement tick.
    pub measured_at_tick: u64,
    /// Profile validity horizon in logical ticks.
    pub stale_after_ticks: u64,
}

/// Checked measured profile used by automatic routing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasuredComputeProfile {
    /// Device identity.
    pub identity: ComputeDeviceIdentity,
    /// Provider limits.
    pub limits: ComputeProfileLimits,
    /// Transfer and operation sample distributions.
    pub samples: ComputeProfileSamples,
    /// Thermal and power measurement context.
    pub context: ComputeThermalPowerContext,
    /// Selected tile byte size.
    pub tile_bytes: u64,
    /// Allocation probe byte sizes that succeeded.
    pub allocation_bytes: Vec<u64>,
    /// Provenance for this record.
    pub provenance: ComputeProfileProvenance,
    /// Modeled profile to use when the evidence is accepted.
    pub modeled: ModeledComputeProfile,
}

impl sim_citizen::Citizen for MeasuredComputeProfile {
    fn citizen_symbol() -> Symbol {
        measured_compute_profile_citizen_symbol()
    }

    fn citizen_version() -> u32 {
        0
    }

    fn citizen_arity() -> usize {
        9
    }

    fn citizen_fields() -> &'static [&'static str] {
        &[
            "identity",
            "limits",
            "samples",
            "context",
            "tile_bytes",
            "allocation_bytes",
            "provenance",
            "modeled_provider",
            "shape",
        ]
    }
}

impl MeasuredComputeProfile {
    /// Returns true when the profile has usable, bounded measurement evidence.
    pub fn is_conclusive(&self) -> bool {
        self.tile_bytes > 0
            && self.allocation_bytes.iter().any(|bytes| *bytes > 0)
            && self.samples.conclusive()
            && self.modeled.fault.is_none()
    }

    /// Returns true when this profile still applies to `identity` at `now_tick`.
    pub fn is_compatible(&self, identity: &ComputeDeviceIdentity, now_tick: u64) -> bool {
        self.identity.compatible_with(identity)
            && now_tick.saturating_sub(self.provenance.measured_at_tick)
                <= self.provenance.stale_after_ticks
            && self.limits.max_resident_bytes > 0
            && self.limits.max_storage_binding_bytes > 0
    }

    /// Returns the modeled executor profile selected by this measured evidence.
    pub fn modeled_profile(&self) -> ModeledComputeProfile {
        self.modeled.clone()
    }

    pub(crate) fn estimated_profile_bytes(&self) -> usize {
        self.identity.adapter.len()
            + self.identity.driver.len()
            + self.identity.backend.len()
            + self.context.thermal.len()
            + self.context.power.len()
            + self.provenance.producer.len()
            + (self.allocation_bytes.len()
                + self.samples.upload_bytes_per_tick.len()
                + self.samples.download_bytes_per_tick.len()
                + self.samples.launch_ticks.len()
                + self.samples.element_elements_per_tick.len()
                + self.samples.reduction_elements_per_tick.len()
                + self.samples.matmul_ops_per_tick.len())
                * std::mem::size_of::<u64>()
            + 256
    }
}

/// Citizen class symbol for measured compute profile read-construct records.
pub fn measured_compute_profile_citizen_symbol() -> Symbol {
    Symbol::qualified("compute-profile", "MeasuredProfile")
}

/// Shape symbol for checked measured compute profile records.
pub fn measured_compute_profile_shape_symbol() -> Symbol {
    Symbol::qualified("compute-profile", "MeasuredProfileShape")
}

/// Bounded synthetic benchmark inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkBounds {
    /// Transfer byte sizes to sample.
    pub transfer_bytes: Vec<u64>,
    /// Element counts to sample.
    pub element_counts: Vec<u64>,
    /// Square matrix edges to sample.
    pub matrix_edges: Vec<u64>,
    /// Maximum accepted byte size in any sample.
    pub max_sample_bytes: u64,
}

impl Default for BenchmarkBounds {
    fn default() -> Self {
        Self {
            transfer_bytes: vec![4096, 16 * 1024, 64 * 1024],
            element_counts: vec![256, 1024, 4096],
            matrix_edges: vec![8, 16, 32],
            max_sample_bytes: 256 * 1024,
        }
    }
}

/// Runs a deterministic bounded profile harness from supplied provider facts.
pub fn measure_bounded_profile(
    identity: ComputeDeviceIdentity,
    modeled: ModeledComputeProfile,
    context: ComputeThermalPowerContext,
    producer: impl Into<String>,
    now_tick: u64,
    bounds: BenchmarkBounds,
) -> MeasuredComputeProfile {
    let bounded_transfers = bounded_nonzero(bounds.transfer_bytes, bounds.max_sample_bytes);
    let bounded_elements = bounded_nonzero(
        bounds.element_counts,
        bounds.max_sample_bytes.saturating_div(CELL_BYTES).max(1),
    );
    let bounded_edges = bounded_nonzero(bounds.matrix_edges, 512);
    let tile_bytes = modeled
        .segment_tile_bytes
        .min(modeled.max_storage_binding_bytes)
        .max(CELL_BYTES);
    let allocation_bytes = bounded_transfers
        .iter()
        .copied()
        .filter(|bytes| *bytes <= modeled.max_resident_bytes)
        .collect::<Vec<_>>();
    MeasuredComputeProfile {
        identity,
        limits: ComputeProfileLimits::from(&modeled),
        samples: ComputeProfileSamples {
            upload_bytes_per_tick: bounded_transfers.clone(),
            download_bytes_per_tick: bounded_transfers
                .iter()
                .map(|bytes| (*bytes).saturating_mul(9) / 10)
                .collect(),
            launch_ticks: bounded_transfers
                .iter()
                .enumerate()
                .map(|(index, _)| (index as u64) + 1)
                .collect(),
            element_elements_per_tick: bounded_elements.clone(),
            reduction_elements_per_tick: bounded_elements.iter().map(|count| count / 2).collect(),
            matmul_ops_per_tick: bounded_edges.iter().map(|edge| edge.pow(3)).collect(),
        },
        context,
        tile_bytes,
        allocation_bytes,
        provenance: ComputeProfileProvenance {
            producer: producer.into(),
            measured_at_tick: now_tick,
            stale_after_ticks: DEFAULT_STALE_AFTER_TICKS,
        },
        modeled,
    }
}

/// Reason an automatic route selected CPU or device placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AutoRouteDecision {
    /// No measured profile was supplied or loaded.
    Absent,
    /// The measured profile is stale for the current logical tick.
    Stale,
    /// Adapter, driver, or backend identity does not match.
    Incompatible,
    /// The profile lacks required bounded samples or limits.
    Inconclusive,
    /// Device evidence was accepted.
    Device,
}

/// Auto router that accepts device placement only with fresh compatible evidence.
#[derive(Clone, Debug)]
pub struct AutoComputeRouter {
    expected: ComputeDeviceIdentity,
    now_tick: u64,
}

impl AutoComputeRouter {
    /// Builds a router for `expected` device identity at `now_tick`.
    pub fn new(expected: ComputeDeviceIdentity, now_tick: u64) -> Self {
        Self { expected, now_tick }
    }

    /// Returns the decision and modeled profile when device placement is proven.
    pub fn choose(
        &self,
        profile: Option<&MeasuredComputeProfile>,
    ) -> (AutoRouteDecision, Option<ModeledComputeProfile>) {
        let Some(profile) = profile else {
            return (AutoRouteDecision::Absent, None);
        };
        if !profile.identity.compatible_with(&self.expected) {
            return (AutoRouteDecision::Incompatible, None);
        }
        if self
            .now_tick
            .saturating_sub(profile.provenance.measured_at_tick)
            > profile.provenance.stale_after_ticks
        {
            return (AutoRouteDecision::Stale, None);
        }
        if !profile.is_conclusive() {
            return (AutoRouteDecision::Inconclusive, None);
        }
        (AutoRouteDecision::Device, Some(profile.modeled_profile()))
    }
}

/// One provider-selection ledger row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoRoutingEvent {
    /// Provider label chosen for the request or flush.
    pub provider: String,
    /// Routing decision.
    pub decision: AutoRouteDecision,
    /// Estimated materialization bytes.
    pub materialization_bytes: u64,
    /// Synchronization count represented by the event.
    pub synchronizations: usize,
}

/// In-memory routing ledger for explainable placement decisions.
#[derive(Clone, Debug, Default)]
pub struct AutoRoutingLedger {
    events: Arc<Mutex<Vec<AutoRoutingEvent>>>,
}

impl AutoRoutingLedger {
    /// Records one routing event.
    pub fn record(&self, event: AutoRoutingEvent) {
        self.events
            .lock()
            .expect("auto routing ledger poisoned")
            .push(event);
    }

    /// Returns recorded routing events.
    pub fn events(&self) -> Vec<AutoRoutingEvent> {
        self.events
            .lock()
            .expect("auto routing ledger poisoned")
            .clone()
    }
}

pub(crate) fn request_materialization_bytes(request: &TensorRequest) -> u64 {
    request
        .inputs
        .iter()
        .map(|tensor| tensor.shape().iter().copied().product::<usize>() as u64 * CELL_BYTES)
        .chain(std::iter::once(
            request.output.shape().iter().copied().product::<usize>() as u64 * CELL_BYTES,
        ))
        .sum()
}

fn bounded_nonzero(values: Vec<u64>, max: u64) -> Vec<u64> {
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .filter(|value| *value > 0 && *value <= max)
        .filter(|value| seen.insert(*value))
        .collect()
}
