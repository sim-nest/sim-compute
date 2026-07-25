//! Loadable wgpu compute site exports.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, CapabilityName, DefaultFactory, Export, Factory, Lib, LibManifest, LibTarget,
    Linker, Result, Symbol, Version,
};
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, SubmissionEvidence, TensorExecError, TensorExecution, TensorExecutor,
    TensorExecutorCard, TensorRequest, TensorSite,
};

use crate::{WgpuAdapterProbe, WgpuDiscovery, discover_wgpu_adapters};

/// Stable symbol for the wgpu runtime library.
pub fn compute_wgpu_lib_symbol() -> Symbol {
    Symbol::qualified("compute", "wgpu-lib")
}

/// Stable symbol for a wgpu tensor executor.
pub fn wgpu_executor_symbol(ordinal: usize) -> Symbol {
    Symbol::qualified("compute", format!("executor/wgpu/{ordinal}"))
}

/// Site symbol exported for a successful wgpu adapter.
pub fn compute_wgpu_site_symbol(ordinal: usize) -> Symbol {
    Symbol::new(format!("site/compute/wgpu/{ordinal}"))
}

/// Capability required before a wgpu hardware tensor site can be realized.
pub fn compute_wgpu_capability() -> CapabilityName {
    CapabilityName::new("device.gpu.wgpu")
}

/// Tensor executor descriptor backed by a successful wgpu probe.
#[derive(Clone)]
pub struct WgpuTensorExecutor {
    probe: WgpuAdapterProbe,
}

impl WgpuTensorExecutor {
    /// Builds an executor from successful probe evidence.
    pub fn new(probe: WgpuAdapterProbe) -> Self {
        Self { probe }
    }

    /// Returns the underlying probe evidence.
    pub fn probe(&self) -> &WgpuAdapterProbe {
        &self.probe
    }
}

impl TensorExecutor for WgpuTensorExecutor {
    fn card(&self) -> TensorExecutorCard {
        let cpu = CpuTensorExecutor::new().card();
        TensorExecutorCard::new(
            wgpu_executor_symbol(self.probe.adapter.ordinal),
            format!(
                "wgpu/{}/{}",
                self.probe.adapter.backend, self.probe.adapter.name
            ),
            Symbol::qualified("compute", "wgpu"),
            cpu.operations.to_vec(),
            Some(compute_wgpu_capability()),
        )
    }

    fn execute(
        &self,
        _cx: &mut sim_kernel::Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        Err(TensorExecError::Unsupported {
            operation: request.operation.symbol,
            reason: Arc::from("wgpu tensor kernels are not installed for this adapter"),
        })
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        Ok(SubmissionEvidence::new(
            wgpu_executor_symbol(self.probe.adapter.ordinal),
            0,
        ))
    }
}

/// Loadable library that registers only successful wgpu adapter sites.
#[derive(Clone, Debug, Default)]
pub struct ComputeWgpuLib {
    discovery: WgpuDiscovery,
}

impl ComputeWgpuLib {
    /// Probes local wgpu adapters and builds a library from successful sites.
    pub fn probe() -> Result<Self> {
        let discovery = discover_wgpu_adapters(&Default::default())
            .map_err(|err| sim_kernel::Error::Eval(err.to_string()))?;
        Ok(Self { discovery })
    }

    /// Builds a library from precomputed discovery evidence.
    pub fn from_discovery(discovery: WgpuDiscovery) -> Self {
        Self { discovery }
    }

    /// Returns discovery evidence, including failed-adapter diagnostics.
    pub fn discovery(&self) -> &WgpuDiscovery {
        &self.discovery
    }
}

impl Lib for ComputeWgpuLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: compute_wgpu_lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: vec![compute_wgpu_capability()],
            exports: self
                .discovery
                .adapters
                .iter()
                .map(|probe| Export::Site {
                    symbol: compute_wgpu_site_symbol(probe.adapter.ordinal),
                    runtime_id: None,
                })
                .collect(),
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        for probe in &self.discovery.adapters {
            let symbol = compute_wgpu_site_symbol(probe.adapter.ordinal);
            let executor = Arc::new(WgpuTensorExecutor::new(probe.clone()));
            let site = TensorSite::new(symbol.clone(), executor, vec![compute_wgpu_capability()]);
            linker.site_value(symbol, DefaultFactory.opaque(Arc::new(site))?)?;
        }
        Ok(())
    }
}
