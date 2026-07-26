use std::sync::Arc;

use sim_kernel::{
    AbiVersion, DefaultFactory, Export, Factory, Lib, LibManifest, LibTarget, Linker,
    Result as KernelResult, Symbol, Version,
};
use sim_lib_femm_solve::{LinearSolverValue, linear_solver_symbol};

use crate::solver::{ResidentCsrConfig, ResidentCsrSolver};

/// Runtime library symbol for the resident FEMM compute solver.
pub fn compute_femm_lib_symbol() -> Symbol {
    Symbol::qualified("compute", "femm-lib")
}

/// Loadable library that registers the resident CSR FEMM solver value.
#[derive(Clone, Debug, Default)]
pub struct ComputeFemmLib {
    config: ResidentCsrConfig,
}

impl ComputeFemmLib {
    /// Builds a FEMM compute library from a solver configuration.
    pub fn new(config: ResidentCsrConfig) -> Self {
        Self { config }
    }
}

impl Lib for ComputeFemmLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: compute_femm_lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: linear_solver_symbol(),
            }],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        let solver = Arc::new(ResidentCsrSolver::new(self.config.clone()));
        let value = LinearSolverValue::new(solver);
        linker.value(
            linear_solver_symbol(),
            DefaultFactory.opaque(Arc::new(value))?,
        )?;
        Ok(())
    }
}
