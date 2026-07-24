//! Loadable modeled compute site exports.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, DefaultFactory, Export, Factory, Lib, LibManifest, LibTarget, Linker, Result,
    Symbol, Version,
};
use sim_lib_numbers_tensor::TensorSite;

use crate::model::{ModeledComputeProfile, ModeledTensorExecutor};

/// Runtime library symbol for the modeled compute provider.
pub fn compute_model_lib_symbol() -> Symbol {
    Symbol::qualified("compute", "model-lib")
}

/// Site symbol exported by the modeled compute provider.
pub fn compute_model_site_symbol() -> Symbol {
    Symbol::new("site/compute/model")
}

/// Loadable library that registers the modeled compute site.
#[derive(Clone, Debug, Default)]
pub struct ComputeModelLib {
    profile: ModeledComputeProfile,
}

impl ComputeModelLib {
    /// Builds a modeled compute library from a profile.
    pub fn new(profile: ModeledComputeProfile) -> Self {
        Self { profile }
    }
}

impl Lib for ComputeModelLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: compute_model_lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Site {
                symbol: compute_model_site_symbol(),
                runtime_id: None,
            }],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        let executor = Arc::new(ModeledTensorExecutor::new(self.profile.clone()));
        let site = TensorSite::new(compute_model_site_symbol(), executor, Vec::new());
        linker.site_value(
            compute_model_site_symbol(),
            DefaultFactory.opaque(Arc::new(site))?,
        )?;
        Ok(())
    }
}
