//! Loadable wgpu compute site exports.

use std::sync::{Arc, Mutex};

use sim_kernel::{
    AbiVersion, CapabilityName, DefaultFactory, Export, Factory, Lib, LibManifest, LibTarget,
    Linker, Result, Symbol, Version,
};
use sim_lib_numbers_tensor::{
    SubmissionEvidence, TensorExecError, TensorExecution, TensorExecutor, TensorExecutorCard,
    TensorRequest, TensorSite, domains,
};

use crate::{
    WgpuAdapterProbe, WgpuDiscovery, WgpuKernelDType, WgpuPipelineCache, WgpuQueueLimits,
    WgpuResidentArena, WgpuResidentStorage, WgpuSegmentPlan, WgpuTileProfile,
    dispatch::{execute_pointwise_dispatch, is_pointwise_dispatch},
    kernels::{execute_portable_kernel, kernel_op},
    probe::discover_wgpu_adapter_runtimes,
};

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
    pub(crate) probe: WgpuAdapterProbe,
    pub(crate) state: Arc<Mutex<WgpuExecutorState>>,
    pub(crate) context: Option<WgpuExecutionContext>,
}

#[derive(Debug)]
pub(crate) struct WgpuExecutorState {
    pub(crate) pipelines: WgpuPipelineCache,
    arena: WgpuResidentArena,
    queued: usize,
    queued_bytes: u64,
    accepted: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct WgpuExecutionContext {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
}

impl WgpuTensorExecutor {
    /// Builds an executor from successful probe evidence.
    pub fn new(probe: WgpuAdapterProbe) -> Self {
        Self::from_parts(probe, None)
    }

    pub(crate) fn from_parts(
        probe: WgpuAdapterProbe,
        context: Option<WgpuExecutionContext>,
    ) -> Self {
        let arena_bytes = probe.adapter.granted_limits.max_buffer_size.max(4);
        Self {
            probe,
            state: Arc::new(Mutex::new(WgpuExecutorState {
                pipelines: WgpuPipelineCache::default(),
                arena: WgpuResidentArena::new(arena_bytes),
                queued: 0,
                queued_bytes: 0,
                accepted: 0,
            })),
            context,
        }
    }

    /// Returns the underlying probe evidence.
    pub fn probe(&self) -> &WgpuAdapterProbe {
        &self.probe
    }

    /// Returns the pipeline cache snapshot.
    pub fn pipeline_cache_snapshot(&self) -> crate::WgpuPipelineCacheSnapshot {
        self.state
            .lock()
            .expect("wgpu executor state poisoned")
            .pipelines
            .snapshot()
    }

    fn dtype_for(
        &self,
        request: &TensorRequest,
    ) -> std::result::Result<WgpuKernelDType, TensorExecError> {
        let dtype = request.output.dtype();
        if dtype == &domains::f32() || dtype == &domains::f64() {
            Ok(WgpuKernelDType::F32)
        } else if dtype == &domains::f16() {
            if self.probe.adapter.granted_features.shader_f16 {
                Ok(WgpuKernelDType::F16Native)
            } else {
                Ok(WgpuKernelDType::Bf16WidenedToF32)
            }
        } else if dtype == &domains::bf16() {
            Ok(WgpuKernelDType::Bf16WidenedToF32)
        } else {
            Err(unsupported(
                request.operation.symbol.clone(),
                "wgpu portable kernels accept f32/f64/half-family tensor dtypes",
            ))
        }
    }

    fn prepare_inputs(&self, request: TensorRequest) -> TensorRequest {
        let inputs = request
            .inputs
            .iter()
            .map(|tensor| {
                tensor
                    .storage()
                    .as_any()
                    .downcast_ref::<WgpuResidentStorage>()
                    .and_then(WgpuResidentStorage::resident_tensor)
                    .unwrap_or_else(|| tensor.clone())
            })
            .collect();
        TensorRequest::new(request.operation, inputs, request.output)
    }

    fn check_submission_limits(&self, bytes: u64) -> std::result::Result<(), TensorExecError> {
        let state = self.state.lock().expect("wgpu executor state poisoned");
        let tile = WgpuTileProfile::from_probe(&self.probe);
        let limits = WgpuQueueLimits {
            max_nodes: 64,
            max_bytes: tile.max_dispatch_bytes,
            deadline_tick: u64::MAX,
        };
        if state.queued >= limits.max_nodes {
            return Err(invalid("wgpu submission queue node limit reached"));
        }
        if state.queued_bytes.saturating_add(bytes) > limits.max_bytes {
            return Err(invalid("wgpu submission queue byte limit reached"));
        }
        Ok(())
    }
}

impl TensorExecutor for WgpuTensorExecutor {
    fn card(&self) -> TensorExecutorCard {
        TensorExecutorCard::new(
            wgpu_executor_symbol(self.probe.adapter.ordinal),
            format!(
                "wgpu/{}/{}",
                self.probe.adapter.backend, self.probe.adapter.name
            ),
            Symbol::qualified("compute", "wgpu"),
            vec![
                sim_lib_numbers_tensor::add_op_symbol(),
                sim_lib_numbers_tensor::sub_op_symbol(),
                sim_lib_numbers_tensor::mul_op_symbol(),
                sim_lib_numbers_tensor::div_op_symbol(),
                sim_lib_numbers_tensor::neg_op_symbol(),
                sim_lib_numbers_tensor::sqrt_op_symbol(),
                sim_lib_numbers_tensor::exp_op_symbol(),
                Symbol::qualified("tensor", "op/log"),
                sim_lib_numbers_tensor::sin_op_symbol(),
                sim_lib_numbers_tensor::cos_op_symbol(),
                sim_lib_numbers_tensor::sum_op_symbol(),
                sim_lib_numbers_tensor::min_op_symbol(),
                sim_lib_numbers_tensor::max_op_symbol(),
                sim_lib_numbers_tensor::norm_op_symbol(),
                sim_lib_numbers_tensor::transpose_exec_op_symbol(),
                sim_lib_numbers_tensor::dot_op_symbol(),
                sim_lib_numbers_tensor::matmul_exec_op_symbol(),
            ],
            Some(compute_wgpu_capability()),
        )
    }

    fn execute(
        &self,
        cx: &mut sim_kernel::Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        let Some(op) = kernel_op(&request.operation.symbol) else {
            return Ok(TensorExecution::Unsupported {
                reason: Arc::from("operation is outside the portable wgpu kernel set"),
            });
        };
        let dtype = self.dtype_for(&request)?;
        let request = self.prepare_inputs(request);
        let bytes = tensor_bytes(request.output.shape())?;
        self.check_submission_limits(bytes)?;
        let dispatched = if is_pointwise_dispatch(op) {
            Some(execute_pointwise_dispatch(
                self, cx, &request, op, dtype, bytes,
            )?)
        } else {
            None
        };
        let (tensor, pipeline_symbol) = if let Some((tensor, symbol)) = dispatched {
            (tensor, Some(symbol))
        } else {
            let tensor = execute_portable_kernel(cx, &request, dtype)?;
            (tensor, None)
        };
        let boundary = self
            .probe
            .adapter
            .granted_limits
            .max_storage_buffer_binding_size
            .max(4);
        let segments = WgpuSegmentPlan::new(bytes, boundary, boundary);
        let pipeline = {
            let mut state = self.state.lock().expect("wgpu executor state poisoned");
            let allocation = state.arena.allocate(bytes.max(4)).map_err(invalid)?;
            state.queued += 1;
            state.queued_bytes += bytes;
            state.accepted += 1;
            let pipeline = if let Some(pipeline_symbol) = pipeline_symbol {
                pipeline_symbol
            } else {
                state
                    .pipelines
                    .get_or_insert(&self.probe, op, dtype, tensor.shape().len())
                    .symbol
            };
            (allocation, pipeline)
        };
        let cells = tensor.cells().map_err(TensorExecError::from)?;
        let storage = WgpuResidentStorage::new(
            compute_wgpu_site_symbol(self.probe.adapter.ordinal),
            pipeline.0,
            pipeline.1,
            segments.segments,
            tensor.shape().to_vec(),
            tensor.dtype().clone(),
            cells,
        );
        Ok(TensorExecution::Complete(
            sim_lib_numbers_tensor::Tensor::from_storage(
                tensor.shape().to_vec(),
                tensor.dtype().clone(),
                Arc::new(storage),
            )?,
        ))
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        let mut state = self.state.lock().expect("wgpu executor state poisoned");
        let accepted = state.queued;
        state.queued = 0;
        state.queued_bytes = 0;
        Ok(SubmissionEvidence::new(
            wgpu_executor_symbol(self.probe.adapter.ordinal),
            accepted,
        ))
    }
}

fn tensor_bytes(shape: &[usize]) -> std::result::Result<u64, TensorExecError> {
    let cells = shape.iter().try_fold(1_u64, |count, extent| {
        count
            .checked_mul(
                u64::try_from(*extent).map_err(|_| invalid("wgpu tensor extent exceeds u64"))?,
            )
            .ok_or_else(|| invalid("wgpu tensor byte count overflowed"))
    })?;
    cells
        .checked_mul(4)
        .ok_or_else(|| invalid("wgpu tensor byte count overflowed"))
}

fn invalid(message: impl Into<Arc<str>>) -> TensorExecError {
    TensorExecError::InvalidRequest {
        message: message.into(),
    }
}

fn unsupported(operation: Symbol, reason: impl Into<Arc<str>>) -> TensorExecError {
    TensorExecError::Unsupported {
        operation,
        reason: reason.into(),
    }
}

/// Loadable library that registers only successful wgpu adapter sites.
#[derive(Clone, Debug, Default)]
pub struct ComputeWgpuLib {
    discovery: WgpuDiscovery,
    contexts: Vec<WgpuExecutionContext>,
}

impl ComputeWgpuLib {
    /// Probes local wgpu adapters and builds a library from successful sites.
    pub fn probe() -> Result<Self> {
        let runtimes = discover_wgpu_adapter_runtimes(&Default::default())
            .map_err(|err| sim_kernel::Error::Eval(err.to_string()))?;
        let mut probes = Vec::with_capacity(runtimes.len());
        let mut contexts = Vec::with_capacity(runtimes.len());
        for runtime in runtimes {
            probes.push(runtime.probe);
            contexts.push(WgpuExecutionContext {
                device: Arc::new(runtime.device),
                queue: Arc::new(runtime.queue),
            });
        }
        let discovery = WgpuDiscovery::from_probes(probes, Vec::new());
        Ok(Self {
            discovery,
            contexts,
        })
    }

    /// Builds a library from precomputed discovery evidence.
    pub fn from_discovery(discovery: WgpuDiscovery) -> Self {
        Self {
            discovery,
            contexts: Vec::new(),
        }
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
            let executor = if let Some(context) = self.contexts.get(probe.adapter.ordinal) {
                Arc::new(WgpuTensorExecutor::from_parts(
                    probe.clone(),
                    Some(context.clone()),
                ))
            } else {
                Arc::new(WgpuTensorExecutor::new(probe.clone()))
            };
            let site = TensorSite::new(symbol.clone(), executor, vec![compute_wgpu_capability()]);
            linker.site_value(symbol, DefaultFactory.opaque(Arc::new(site))?)?;
        }
        Ok(())
    }
}
