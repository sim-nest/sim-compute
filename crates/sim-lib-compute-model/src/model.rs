//! Modeled tensor executor with bounded queues, segmented residency, and faults.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use sim_kernel::Symbol;
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, SubmissionEvidence, Tensor, TensorExecError, TensorExecution,
    TensorExecutor, TensorExecutorCard, TensorRequest,
};

use crate::storage::{ModeledResidentDescriptor, ModeledResidentStorage, ResidentHandle};

const DEFAULT_SEGMENT_TILE_BYTES: u64 = 256 * 1024;
const DEFAULT_STORAGE_BINDING_BYTES: u64 = 1024 * 1024;
const DEFAULT_RESIDENT_BYTES: u64 = 8 * 1024 * 1024;
const DEFAULT_QUEUE_BYTES: u64 = 2 * 1024 * 1024;
const DEFAULT_DEADLINE_TICKS: u64 = 8;
const MODELED_CELL_BYTES: u64 = 8;

/// Stable symbol for the modeled tensor executor.
pub fn modeled_executor_symbol() -> Symbol {
    Symbol::qualified("compute", "executor/model")
}

/// Fault injected into the modeled executor or resident storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModeledComputeFault {
    /// Refuse before accepting a submission, modeling out-of-memory.
    OomBeforeSubmit,
    /// Fail after acceptance, modeling device loss during execution.
    DeviceLostDuringExecute,
    /// Resident storage fails when materialized.
    ReadbackFailure,
}

/// Configuration for a modeled compute site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModeledComputeProfile {
    /// Stable provider label.
    pub provider: String,
    /// Maximum accepted submissions waiting for flush evidence.
    pub max_queue_depth: usize,
    /// Maximum bytes accepted into the submission queue.
    pub max_queue_bytes: u64,
    /// Maximum modeled resident bytes retained by the site.
    pub max_resident_bytes: u64,
    /// Largest tile used before a tensor is segmented.
    pub segment_tile_bytes: u64,
    /// Checked storage binding boundary used by the modeled layout.
    pub max_storage_binding_bytes: u64,
    /// Maximum modeled ticks a submission may wait before rejection.
    pub submission_deadline_ticks: u64,
    /// Optional deterministic fault.
    pub fault: Option<ModeledComputeFault>,
}

impl Default for ModeledComputeProfile {
    fn default() -> Self {
        Self {
            provider: "modeled-compute".to_owned(),
            max_queue_depth: 8,
            max_queue_bytes: DEFAULT_QUEUE_BYTES,
            max_resident_bytes: DEFAULT_RESIDENT_BYTES,
            segment_tile_bytes: DEFAULT_SEGMENT_TILE_BYTES,
            max_storage_binding_bytes: DEFAULT_STORAGE_BINDING_BYTES,
            submission_deadline_ticks: DEFAULT_DEADLINE_TICKS,
            fault: None,
        }
    }
}

/// One resident segment in the modeled storage arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModeledResidentSegment {
    /// Segment ordinal inside the allocation.
    pub index: usize,
    /// Byte offset from the start of the tensor payload.
    pub offset: u64,
    /// Segment length in bytes.
    pub bytes: u64,
}

/// Snapshot of modeled executor counters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModeledComputeSnapshot {
    /// Submissions accepted by the modeled executor.
    pub accepted: usize,
    /// Submissions completed by the modeled executor.
    pub completed: usize,
    /// Submissions waiting for flush evidence.
    pub queued: usize,
    /// Bytes currently waiting for flush evidence.
    pub queued_bytes: u64,
    /// Resident allocations produced by this executor.
    pub resident_allocations: usize,
    /// Resident allocations still present in the modeled bounded pool.
    pub live_allocations: usize,
    /// Resident bytes still present in the modeled bounded pool.
    pub resident_bytes: u64,
    /// Resident allocations evicted by the bounded pool.
    pub evictions: usize,
    /// Total resident segments allocated.
    pub segments: usize,
    /// Resident readbacks observed through modeled storage.
    pub readbacks: usize,
    /// Stable resident materialization failures.
    pub materialization_failures: usize,
}

#[derive(Default)]
pub(crate) struct ModeledCounters {
    accepted: usize,
    completed: usize,
    queued: usize,
    queued_bytes: u64,
    resident_allocations: usize,
    live_allocations: VecDeque<ResidentAllocation>,
    resident_bytes: u64,
    evictions: usize,
    segments: usize,
    readbacks: usize,
    materialization_failures: usize,
    tick: u64,
}

#[derive(Clone)]
struct ResidentAllocation {
    handle: ResidentHandle,
    bytes: u64,
    active: bool,
}

/// Deterministic tensor executor that models resident compute placement.
#[derive(Clone)]
pub struct ModeledTensorExecutor {
    profile: ModeledComputeProfile,
    counters: Arc<Mutex<ModeledCounters>>,
}

impl ModeledTensorExecutor {
    /// Builds a modeled executor from a profile.
    pub fn new(profile: ModeledComputeProfile) -> Self {
        Self {
            profile,
            counters: Arc::new(Mutex::new(ModeledCounters::default())),
        }
    }

    /// Builds a modeled executor with the default profile.
    pub fn default_profile() -> Self {
        Self::new(ModeledComputeProfile::default())
    }

    /// Returns the current counter snapshot.
    pub fn snapshot(&self) -> ModeledComputeSnapshot {
        let counters = self.counters.lock().expect("modeled counters poisoned");
        ModeledComputeSnapshot {
            accepted: counters.accepted,
            completed: counters.completed,
            queued: counters.queued,
            queued_bytes: counters.queued_bytes,
            resident_allocations: counters.resident_allocations,
            live_allocations: counters
                .live_allocations
                .iter()
                .filter(|allocation| allocation.active)
                .count(),
            resident_bytes: counters.resident_bytes,
            evictions: counters.evictions,
            segments: counters.segments,
            readbacks: counters.readbacks,
            materialization_failures: counters.materialization_failures,
        }
    }

    pub(crate) fn increment_readbacks(&self) {
        let mut counters = self.counters.lock().expect("modeled counters poisoned");
        counters.readbacks += 1;
    }

    pub(crate) fn increment_materialization_failures(&self) {
        let mut counters = self.counters.lock().expect("modeled counters poisoned");
        counters.materialization_failures += 1;
    }

    pub(crate) fn is_resident_active(&self, handle: &ResidentHandle) -> bool {
        let counters = self.counters.lock().expect("modeled counters poisoned");
        counters
            .live_allocations
            .iter()
            .any(|allocation| allocation.active && allocation.handle == *handle)
    }

    fn prepare_inputs(&self, request: TensorRequest) -> TensorRequest {
        let inputs = request
            .inputs
            .iter()
            .map(|tensor| {
                tensor
                    .storage()
                    .as_any()
                    .downcast_ref::<ModeledResidentStorage>()
                    .and_then(ModeledResidentStorage::resident_tensor)
                    .unwrap_or_else(|| tensor.clone())
            })
            .collect();
        TensorRequest::new(request.operation, inputs, request.output)
    }

    fn request_bytes(request: &TensorRequest) -> std::result::Result<u64, TensorExecError> {
        let output_bytes = tensor_bytes(request.output.shape())?;
        request
            .inputs
            .iter()
            .try_fold(output_bytes, |bytes, tensor| {
                Ok(bytes.saturating_add(tensor_bytes(tensor.shape())?))
            })
    }

    fn segment_layout(&self, bytes: u64) -> Vec<ModeledResidentSegment> {
        let boundary = self
            .profile
            .segment_tile_bytes
            .min(self.profile.max_storage_binding_bytes)
            .max(MODELED_CELL_BYTES);
        let mut segments = Vec::new();
        let mut offset = 0;
        while offset < bytes {
            let segment_bytes = (bytes - offset).min(boundary);
            segments.push(ModeledResidentSegment {
                index: segments.len(),
                offset,
                bytes: segment_bytes,
            });
            offset += segment_bytes;
        }
        segments
    }

    fn reserve_submission(&self, bytes: u64) -> std::result::Result<(), TensorExecError> {
        let mut counters = self.counters.lock().expect("modeled counters poisoned");
        counters.tick = counters.tick.saturating_add(1);
        if counters.queued >= self.profile.max_queue_depth {
            return Err(TensorExecError::InvalidRequest {
                message: Arc::from("modeled compute queue is full"),
            });
        }
        if counters.queued_bytes.saturating_add(bytes) > self.profile.max_queue_bytes {
            return Err(TensorExecError::InvalidRequest {
                message: Arc::from("modeled compute queue byte budget is full"),
            });
        }
        if self.profile.submission_deadline_ticks == 0 {
            return Err(TensorExecError::InvalidRequest {
                message: Arc::from("modeled compute submission deadline expired"),
            });
        }
        counters.accepted += 1;
        counters.queued += 1;
        counters.queued_bytes += bytes;
        Ok(())
    }

    fn release_submission(&self, bytes: u64) {
        let mut counters = self.counters.lock().expect("modeled counters poisoned");
        counters.queued = counters.queued.saturating_sub(1);
        counters.queued_bytes = counters.queued_bytes.saturating_sub(bytes);
    }

    fn allocate_resident(
        &self,
        bytes: u64,
        segments: usize,
    ) -> std::result::Result<ResidentHandle, TensorExecError> {
        if bytes > self.profile.max_resident_bytes {
            return Err(TensorExecError::InvalidRequest {
                message: Arc::from("modeled compute resident allocation exceeds pool"),
            });
        }
        let mut counters = self.counters.lock().expect("modeled counters poisoned");
        while counters.resident_bytes.saturating_add(bytes) > self.profile.max_resident_bytes {
            let Some(mut allocation) = counters.live_allocations.pop_front() else {
                break;
            };
            if allocation.active {
                allocation.active = false;
                counters.resident_bytes = counters.resident_bytes.saturating_sub(allocation.bytes);
                counters.evictions += 1;
            }
            counters.live_allocations.push_back(allocation);
        }
        counters.completed += 1;
        counters.resident_allocations += 1;
        counters.resident_bytes += bytes;
        counters.segments += segments;
        let handle = ResidentHandle::new(counters.resident_allocations);
        counters.live_allocations.push_back(ResidentAllocation {
            handle: handle.clone(),
            bytes,
            active: true,
        });
        Ok(handle)
    }
}

fn tensor_bytes(shape: &[usize]) -> std::result::Result<u64, TensorExecError> {
    let cells = shape.iter().try_fold(1_u64, |count, extent| {
        count
            .checked_mul(
                u64::try_from(*extent).map_err(|_| TensorExecError::InvalidRequest {
                    message: Arc::from("modeled compute tensor extent exceeds u64"),
                })?,
            )
            .ok_or_else(|| TensorExecError::InvalidRequest {
                message: Arc::from("modeled compute tensor byte count overflowed"),
            })
    })?;
    cells
        .checked_mul(MODELED_CELL_BYTES)
        .ok_or_else(|| TensorExecError::InvalidRequest {
            message: Arc::from("modeled compute tensor byte count overflowed"),
        })
}

impl Default for ModeledTensorExecutor {
    fn default() -> Self {
        Self::default_profile()
    }
}

impl TensorExecutor for ModeledTensorExecutor {
    fn card(&self) -> TensorExecutorCard {
        let cpu = CpuTensorExecutor::new().card();
        TensorExecutorCard::new(
            modeled_executor_symbol(),
            self.profile.provider.clone(),
            Symbol::qualified("compute", "modeled-resident"),
            cpu.operations.to_vec(),
            None,
        )
    }

    fn execute(
        &self,
        cx: &mut sim_kernel::Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        let request_bytes = Self::request_bytes(&request)?;
        if self.profile.fault == Some(ModeledComputeFault::OomBeforeSubmit) {
            return Err(TensorExecError::InvalidRequest {
                message: Arc::from("modeled compute out of memory before submission"),
            });
        }
        self.reserve_submission(request_bytes)?;
        if self.profile.fault == Some(ModeledComputeFault::DeviceLostDuringExecute) {
            self.release_submission(request_bytes);
            return Err(TensorExecError::Eval {
                message: Arc::from("modeled compute device lost during execution"),
            });
        }

        let request = self.prepare_inputs(request);
        let result = match CpuTensorExecutor::new().execute(cx, request) {
            Ok(result) => result,
            Err(error) => {
                self.release_submission(request_bytes);
                return Err(error);
            }
        };
        let TensorExecution::Complete(tensor) = result else {
            return Ok(result);
        };
        let host_cells = tensor.cells().map_err(TensorExecError::from)?;
        let resident_bytes = tensor_bytes(tensor.shape())?;
        let segments = self.segment_layout(resident_bytes);
        let allocation = self.allocate_resident(resident_bytes, segments.len())?;
        let storage = ModeledResidentStorage::new(
            ModeledResidentDescriptor {
                site: Symbol::new("site/compute/model"),
                allocation,
                segments,
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype().clone(),
            },
            host_cells,
            self.clone(),
            self.profile.fault.clone(),
        );
        Ok(TensorExecution::Complete(Tensor::from_storage(
            tensor.shape().to_vec(),
            tensor.dtype().clone(),
            Arc::new(storage),
        )?))
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        let mut counters = self.counters.lock().expect("modeled counters poisoned");
        let accepted = counters.queued;
        counters.queued = 0;
        counters.queued_bytes = 0;
        Ok(SubmissionEvidence::new(modeled_executor_symbol(), accepted))
    }
}
