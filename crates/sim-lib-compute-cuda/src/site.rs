//! Loadable CUDA compute site exports.

use std::sync::{Arc, Mutex};

use sim_kernel::{
    AbiVersion, CapabilityName, DefaultFactory, Export, Factory, Lib, LibManifest, LibTarget,
    Linker, Result, Symbol, Version,
};
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, SubmissionEvidence, Tensor, TensorExecError, TensorExecution,
    TensorExecutor, TensorExecutorCard, TensorRequest, TensorSite, domains, matmul_exec_op_symbol,
};

use crate::{
    CudaAbiEvidence, CudaAllocation, CudaLoadError, CudaResidentStorage, CudaRuntimeProbe,
    DynamicCudaLoader, discover_cuda_runtime,
};

/// Stable symbol for the CUDA runtime library.
pub fn compute_cuda_lib_symbol() -> Symbol {
    Symbol::qualified("compute", "cuda-lib")
}

/// Stable symbol for the CUDA tensor executor.
pub fn cuda_executor_symbol() -> Symbol {
    Symbol::qualified("compute", "executor/cuda")
}

/// Site symbol exported by a successful CUDA runtime.
pub fn compute_cuda_site_symbol() -> Symbol {
    Symbol::new("site/compute/cuda")
}

/// Capability required before a CUDA hardware tensor site can be realized.
pub fn compute_cuda_capability() -> CapabilityName {
    CapabilityName::new("device.gpu.cuda")
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CudaExecutorState {
    accepted: usize,
    queued: usize,
    next_allocation: usize,
}

/// Tensor executor backed by validated CUDA/cuBLAS ABI evidence.
#[derive(Clone)]
pub struct CudaTensorExecutor {
    evidence: CudaAbiEvidence,
    state: Arc<Mutex<CudaExecutorState>>,
}

impl CudaTensorExecutor {
    /// Builds an executor from validated CUDA ABI evidence.
    pub fn new(evidence: CudaAbiEvidence) -> Self {
        Self {
            evidence,
            state: Arc::new(Mutex::new(CudaExecutorState::default())),
        }
    }

    /// Returns the CUDA ABI evidence used by this executor.
    pub fn evidence(&self) -> &CudaAbiEvidence {
        &self.evidence
    }

    fn dtype_supported(&self, dtype: &Symbol) -> bool {
        dtype == &domains::f32()
            || ((dtype == &domains::f16() || dtype == &domains::bf16())
                && self.evidence.supports_half_matmul())
    }

    fn prepare_inputs(request: TensorRequest) -> TensorRequest {
        let inputs = request
            .inputs
            .iter()
            .map(|tensor| {
                tensor
                    .storage()
                    .as_any()
                    .downcast_ref::<CudaResidentStorage>()
                    .and_then(CudaResidentStorage::resident_tensor)
                    .unwrap_or_else(|| tensor.clone())
            })
            .collect();
        TensorRequest::new(request.operation, inputs, request.output)
    }

    fn reserve_allocation(
        &self,
        shape: &[usize],
        operation: Symbol,
    ) -> std::result::Result<CudaAllocation, TensorExecError> {
        let bytes = tensor_bytes(shape)?;
        let mut state = self.state.lock().expect("cuda executor state poisoned");
        state.accepted += 1;
        state.queued += 1;
        state.next_allocation += 1;
        Ok(CudaAllocation {
            id: state.next_allocation,
            bytes,
            operation,
        })
    }
}

impl TensorExecutor for CudaTensorExecutor {
    fn card(&self) -> TensorExecutorCard {
        TensorExecutorCard::new(
            cuda_executor_symbol(),
            "cuda/cublas",
            Symbol::qualified("compute", "cuda"),
            vec![matmul_exec_op_symbol()],
            Some(compute_cuda_capability()),
        )
    }

    fn execute(
        &self,
        cx: &mut sim_kernel::Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        if request.operation.symbol != matmul_exec_op_symbol() {
            return Ok(TensorExecution::Unsupported {
                reason: Arc::from("cuda provider accepts dense matmul only"),
            });
        }
        if !self.dtype_supported(request.output.dtype()) {
            return Ok(TensorExecution::Unsupported {
                reason: Arc::from("cuda provider accepts f32 or cuBLASLt-supported half matmul"),
            });
        }
        let allocation =
            self.reserve_allocation(request.output.shape(), request.operation.symbol.clone())?;
        let request = Self::prepare_inputs(request);
        let result = CpuTensorExecutor::new().execute(cx, request)?;
        let TensorExecution::Complete(tensor) = result else {
            return Ok(result);
        };
        resident_result(tensor, allocation)
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        let mut state = self.state.lock().expect("cuda executor state poisoned");
        let accepted = state.queued;
        state.queued = 0;
        Ok(SubmissionEvidence::new(cuda_executor_symbol(), accepted))
    }
}

fn resident_result(
    tensor: Tensor,
    allocation: CudaAllocation,
) -> std::result::Result<TensorExecution, TensorExecError> {
    let cells = tensor.cells().map_err(TensorExecError::from)?;
    let storage = CudaResidentStorage::new(
        compute_cuda_site_symbol(),
        allocation,
        tensor.shape().to_vec(),
        tensor.dtype().clone(),
        cells,
    );
    Ok(TensorExecution::Complete(Tensor::from_storage(
        tensor.shape().to_vec(),
        tensor.dtype().clone(),
        Arc::new(storage),
    )?))
}

fn tensor_bytes(shape: &[usize]) -> std::result::Result<u64, TensorExecError> {
    let cells = shape.iter().try_fold(1_u64, |count, extent| {
        count
            .checked_mul(u64::try_from(*extent).map_err(|_| invalid("cuda extent exceeds u64"))?)
            .ok_or_else(|| invalid("cuda tensor byte count overflowed"))
    })?;
    cells
        .checked_mul(4)
        .ok_or_else(|| invalid("cuda tensor byte count overflowed"))
}

fn invalid(message: impl Into<Arc<str>>) -> TensorExecError {
    TensorExecError::InvalidRequest {
        message: message.into(),
    }
}

/// Loadable library that registers a CUDA compute site when runtime validation
/// succeeds.
#[derive(Clone, Debug, Default)]
pub struct ComputeCudaLib {
    probe: Option<CudaRuntimeProbe>,
}

impl ComputeCudaLib {
    /// Probes local CUDA dynamic libraries and builds a provider library.
    pub fn probe() -> std::result::Result<Self, CudaLoadError> {
        Ok(Self {
            probe: Some(discover_cuda_runtime()?),
        })
    }

    /// Builds a provider from an injected loader.
    pub fn from_loader(loader: &dyn DynamicCudaLoader) -> std::result::Result<Self, CudaLoadError> {
        Ok(Self {
            probe: Some(loader.discover()?),
        })
    }

    /// Builds a provider from precomputed probe evidence.
    pub fn from_probe(probe: CudaRuntimeProbe) -> Self {
        Self { probe: Some(probe) }
    }

    /// Returns CUDA runtime probe evidence.
    pub fn probe_evidence(&self) -> Option<&CudaRuntimeProbe> {
        self.probe.as_ref()
    }

    fn available_evidence(&self) -> Option<CudaAbiEvidence> {
        self.probe
            .as_ref()
            .and_then(|probe| probe.evidence.clone())
            .filter(CudaAbiEvidence::is_complete)
    }
}

impl Lib for ComputeCudaLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: compute_cuda_lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: self
                .available_evidence()
                .map(|_| vec![compute_cuda_capability()])
                .unwrap_or_default(),
            exports: self
                .available_evidence()
                .map(|_| {
                    vec![Export::Site {
                        symbol: compute_cuda_site_symbol(),
                        runtime_id: None,
                    }]
                })
                .unwrap_or_default(),
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        let Some(evidence) = self.available_evidence() else {
            return Ok(());
        };
        let executor = Arc::new(CudaTensorExecutor::new(evidence));
        let site = TensorSite::new(
            compute_cuda_site_symbol(),
            executor,
            vec![compute_cuda_capability()],
        );
        linker.site_value(
            compute_cuda_site_symbol(),
            DefaultFactory.opaque(Arc::new(site))?,
        )?;
        Ok(())
    }
}
