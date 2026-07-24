//! Modeled tensor executor with bounded queues and injected faults.

use std::sync::{Arc, Mutex};

use sim_kernel::Symbol;
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, SubmissionEvidence, Tensor, TensorExecError, TensorExecution,
    TensorExecutor, TensorExecutorCard, TensorRequest,
};

use crate::storage::{ModeledResidentStorage, ResidentHandle};

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
    /// Optional deterministic fault.
    pub fault: Option<ModeledComputeFault>,
}

impl Default for ModeledComputeProfile {
    fn default() -> Self {
        Self {
            provider: "modeled-compute".to_owned(),
            max_queue_depth: 8,
            fault: None,
        }
    }
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
    /// Resident allocations produced by this executor.
    pub resident_allocations: usize,
    /// Resident readbacks observed through modeled storage.
    pub readbacks: usize,
}

#[derive(Default)]
pub(crate) struct ModeledCounters {
    accepted: usize,
    completed: usize,
    queued: usize,
    resident_allocations: usize,
    readbacks: usize,
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
            resident_allocations: counters.resident_allocations,
            readbacks: counters.readbacks,
        }
    }

    pub(crate) fn increment_readbacks(&self) {
        let mut counters = self.counters.lock().expect("modeled counters poisoned");
        counters.readbacks += 1;
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
        if self.profile.fault == Some(ModeledComputeFault::OomBeforeSubmit) {
            return Err(TensorExecError::InvalidRequest {
                message: Arc::from("modeled compute out of memory before submission"),
            });
        }
        {
            let mut counters = self.counters.lock().expect("modeled counters poisoned");
            if counters.queued >= self.profile.max_queue_depth {
                return Err(TensorExecError::InvalidRequest {
                    message: Arc::from("modeled compute queue is full"),
                });
            }
            counters.accepted += 1;
            counters.queued += 1;
        }
        if self.profile.fault == Some(ModeledComputeFault::DeviceLostDuringExecute) {
            return Err(TensorExecError::Eval {
                message: Arc::from("modeled compute device lost during execution"),
            });
        }

        let request = self.prepare_inputs(request);
        let result = CpuTensorExecutor::new().execute(cx, request)?;
        let TensorExecution::Complete(tensor) = result else {
            return Ok(result);
        };
        let host_cells = tensor.cells().map_err(TensorExecError::from)?;
        let allocation = {
            let mut counters = self.counters.lock().expect("modeled counters poisoned");
            counters.completed += 1;
            counters.resident_allocations += 1;
            ResidentHandle::new(counters.resident_allocations)
        };
        let storage = ModeledResidentStorage::new(
            Symbol::new("site/compute/model"),
            allocation,
            tensor.shape().to_vec(),
            tensor.dtype().clone(),
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
        Ok(SubmissionEvidence::new(modeled_executor_symbol(), accepted))
    }
}
