#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Automatic tensor compute-site selection.

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
    /// Optional compatible modeled profile. When absent, auto uses CPU.
    pub modeled: Option<ModeledComputeProfile>,
}

/// Tensor executor that picks the best compatible provider and falls back to CPU.
#[derive(Clone, Default)]
pub struct AutoTensorExecutor {
    modeled: Option<ModeledTensorExecutor>,
}

impl AutoTensorExecutor {
    /// Builds an automatic executor from a profile.
    pub fn new(profile: AutoComputeProfile) -> Self {
        Self {
            modeled: profile.modeled.map(ModeledTensorExecutor::new),
        }
    }

    /// Returns true when the selector is using the CPU fallback.
    pub fn uses_cpu_fallback(&self) -> bool {
        self.modeled.is_none()
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
        self.selected().execute(cx, request)
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        self.selected().flush()
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
