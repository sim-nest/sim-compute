#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Automatic tensor compute-site selection.

mod evidence;
mod profile;
mod store;

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, DefaultFactory, Export, Factory, Lib, LibManifest, LibTarget, Linker, Result,
    Symbol, Version,
};
use sim_lib_compute_model::{ModeledComputeProfile, ModeledTensorExecutor};
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, SubmissionEvidence, TensorExecError, TensorExecution, TensorExecutor,
    TensorExecutorCard, TensorRequest, TensorSite,
};

pub use evidence::{
    ComputeEvidenceKind, ComputePhysicalEvidence, PhysicalEvidenceError, verify_physical,
};
pub use profile::{
    AutoComputeRouter, AutoRouteDecision, AutoRoutingEvent, AutoRoutingLedger, BenchmarkBounds,
    ComputeDeviceIdentity, ComputeProfileLimits, ComputeProfileProvenance, ComputeProfileSamples,
    ComputeThermalPowerContext, MeasuredComputeProfile, measure_bounded_profile,
    measured_compute_profile_citizen_symbol, measured_compute_profile_shape_symbol,
};
pub use store::{ProfileStore, ProfileStorePolicy};

/// Stable symbol for the automatic tensor executor.
pub fn auto_executor_symbol() -> Symbol {
    Symbol::qualified("compute", "executor/auto")
}

/// Site symbol exported by the automatic compute provider.
pub fn compute_auto_site_symbol() -> Symbol {
    Symbol::new("site/compute/auto")
}

/// Runtime library symbol for the automatic compute provider.
pub fn compute_auto_lib_symbol() -> Symbol {
    Symbol::qualified("compute", "auto-lib")
}

/// Automatic executor profile.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AutoComputeProfile {
    /// Optional legacy modeled profile. When absent, auto uses CPU unless
    /// measured evidence below proves a compatible device route.
    pub modeled: Option<ModeledComputeProfile>,
    /// Optional measured profile used by the conservative router.
    pub measured: Option<MeasuredComputeProfile>,
    /// Expected adapter/driver/backend identity for measured routing.
    pub expected: Option<ComputeDeviceIdentity>,
    /// Current logical tick used for staleness checks.
    pub now_tick: u64,
}

/// Tensor executor that picks the best compatible provider and falls back to CPU.
#[derive(Clone)]
pub struct AutoTensorExecutor {
    modeled: Option<ModeledTensorExecutor>,
    decision: AutoRouteDecision,
    ledger: AutoRoutingLedger,
}

impl AutoTensorExecutor {
    /// Builds an automatic executor from a profile.
    pub fn new(profile: AutoComputeProfile) -> Self {
        let (decision, measured_modeled) = match profile.expected.clone() {
            Some(expected) => {
                AutoComputeRouter::new(expected, profile.now_tick).choose(profile.measured.as_ref())
            }
            None if profile.measured.is_some() => (AutoRouteDecision::Incompatible, None),
            None => (AutoRouteDecision::Absent, None),
        };
        let modeled = measured_modeled.or(profile.modeled);
        Self {
            modeled: modeled.map(ModeledTensorExecutor::new),
            decision,
            ledger: AutoRoutingLedger::default(),
        }
    }

    /// Returns true when the selector is using the CPU fallback.
    pub fn uses_cpu_fallback(&self) -> bool {
        self.modeled.is_none()
    }

    /// Returns the router decision that produced this executor.
    pub fn route_decision(&self) -> AutoRouteDecision {
        self.decision.clone()
    }

    /// Returns routing ledger events recorded by execute and flush calls.
    pub fn routing_events(&self) -> Vec<AutoRoutingEvent> {
        self.ledger.events()
    }

    fn selected(&self) -> Arc<dyn TensorExecutor> {
        self.modeled
            .clone()
            .map(|executor| Arc::new(executor) as Arc<dyn TensorExecutor>)
            .unwrap_or_else(|| Arc::new(CpuTensorExecutor::new()))
    }
}

impl TensorExecutor for AutoTensorExecutor {
    fn card(&self) -> TensorExecutorCard {
        let selected = self.selected().card();
        TensorExecutorCard::new(
            auto_executor_symbol(),
            if self.uses_cpu_fallback() {
                "auto/cpu"
            } else {
                "auto/modeled"
            },
            Symbol::qualified("compute", "auto"),
            selected.operations.to_vec(),
            selected.device_capability,
        )
    }

    fn execute(
        &self,
        cx: &mut sim_kernel::Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        let materialization_bytes = profile::request_materialization_bytes(&request);
        self.ledger.record(AutoRoutingEvent {
            provider: if self.uses_cpu_fallback() {
                "auto/cpu".to_owned()
            } else {
                "auto/modeled".to_owned()
            },
            decision: self.decision.clone(),
            materialization_bytes,
            synchronizations: 0,
        });
        self.selected().execute(cx, request)
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        let evidence = self.selected().flush()?;
        self.ledger.record(AutoRoutingEvent {
            provider: if self.uses_cpu_fallback() {
                "auto/cpu".to_owned()
            } else {
                "auto/modeled".to_owned()
            },
            decision: self.decision.clone(),
            materialization_bytes: 0,
            synchronizations: 1,
        });
        Ok(evidence)
    }
}

impl Default for AutoTensorExecutor {
    fn default() -> Self {
        Self::new(AutoComputeProfile::default())
    }
}

/// Loadable library that registers the automatic compute site.
#[derive(Clone, Debug, Default)]
pub struct ComputeAutoLib {
    profile: AutoComputeProfile,
}

impl ComputeAutoLib {
    /// Builds an automatic compute library from a profile.
    pub fn new(profile: AutoComputeProfile) -> Self {
        Self { profile }
    }
}

impl Lib for ComputeAutoLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: compute_auto_lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Site {
                symbol: compute_auto_site_symbol(),
                runtime_id: None,
            }],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        let executor = Arc::new(AutoTensorExecutor::new(self.profile.clone()));
        let site = TensorSite::new(compute_auto_site_symbol(), executor, Vec::new());
        linker.site_value(
            compute_auto_site_symbol(),
            DefaultFactory.opaque(Arc::new(site))?,
        )?;
        Ok(())
    }
}

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
