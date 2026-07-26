//! Physical evidence discriminators and acceptance checks.

use sim_lib_compute_model::ModeledComputeProfile;

use crate::profile::ComputeDeviceIdentity;

/// Discriminator for compute evidence that might otherwise look physical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeEvidenceKind {
    /// Deterministic model evidence; never proves a physical device.
    Modeled,
    /// Host-side emulation of a device-shaped provider; never proves physical execution.
    HostEmulated,
    /// Evidence captured from a physical device path.
    PhysicalDevice,
}

impl ComputeEvidenceKind {
    /// Stable profile encoding label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Modeled => "modeled",
            Self::HostEmulated => "host-emulated",
            Self::PhysicalDevice => "physical-device",
        }
    }
}

impl TryFrom<&str> for ComputeEvidenceKind {
    type Error = PhysicalEvidenceError;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        match value {
            "modeled" => Ok(Self::Modeled),
            "host-emulated" => Ok(Self::HostEmulated),
            "physical-device" => Ok(Self::PhysicalDevice),
            other => Err(PhysicalEvidenceError::new(format!(
                "unknown compute evidence kind: {other}"
            ))),
        }
    }
}

/// Evidence that can be checked before accepting a physical-device claim.
pub trait ComputePhysicalEvidence {
    /// Reported evidence kind.
    fn evidence_kind(&self) -> ComputeEvidenceKind;

    /// Claimed device identity, when the record carries one.
    fn claimed_identity(&self) -> Option<&ComputeDeviceIdentity> {
        None
    }

    /// Observed device identity captured by the producer, when available.
    fn observed_identity(&self) -> Option<&ComputeDeviceIdentity> {
        None
    }
}

impl ComputePhysicalEvidence for ModeledComputeProfile {
    fn evidence_kind(&self) -> ComputeEvidenceKind {
        ComputeEvidenceKind::Modeled
    }
}

/// Failure returned when evidence is not acceptable as physical-device proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalEvidenceError {
    message: String,
}

impl PhysicalEvidenceError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for PhysicalEvidenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PhysicalEvidenceError {}

/// Verifies that evidence can satisfy a physical-device acceptance boundary.
pub fn verify_physical(
    evidence: &(impl ComputePhysicalEvidence + ?Sized),
) -> std::result::Result<(), PhysicalEvidenceError> {
    if evidence.evidence_kind() != ComputeEvidenceKind::PhysicalDevice {
        return Err(PhysicalEvidenceError::new(format!(
            "compute evidence is {}, not physical-device",
            evidence.evidence_kind().as_str()
        )));
    }
    match (evidence.claimed_identity(), evidence.observed_identity()) {
        (Some(claimed), Some(observed)) if claimed == observed => Ok(()),
        (Some(_), Some(_)) => Err(PhysicalEvidenceError::new(
            "compute evidence identity was renamed after capture",
        )),
        _ => Err(PhysicalEvidenceError::new(
            "compute physical evidence is missing observed identity",
        )),
    }
}
