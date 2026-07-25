//! Resident tensor ODE execution over the canonical numeric/RK pipeline.

use std::sync::Arc;

use sim_kernel::{
    Consistency, Cx, Error, EvalFabric, EvalMode, EvalReply, EvalRequest, Expr, Result, Symbol,
};
use sim_lib_numbers_tensor::{TensorExecutor, TensorSite, tensor_value_ref};

use crate::model::{ModeledComputeProfile, ModeledComputeSnapshot, ModeledTensorExecutor};
use crate::site::compute_model_site_symbol;

/// ODE execution family requested by the resident adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentOdeKind {
    /// Fixed-step RK methods.
    Fixed,
    /// Adaptive RK methods with scalar error decisions.
    Adaptive,
}

/// Provider-side lowering classification for a tensor RHS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentRhsLowering {
    /// RHS is a lowerable tensor expression.
    TensorExpression,
    /// RHS is host-only or effectful and must not be accepted by resident ODE.
    NonLowerable,
}

/// Checked resident ODE plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentOdePlan {
    kind: ResidentOdeKind,
    state_shape: Vec<usize>,
    dtype: Symbol,
}

impl ResidentOdePlan {
    /// Builds a fixed-step resident ODE plan for tensor state.
    pub fn fixed(state_shape: Vec<usize>, dtype: Symbol, rhs: ResidentRhsLowering) -> Result<Self> {
        Self::new(ResidentOdeKind::Fixed, state_shape, dtype, rhs)
    }

    /// Builds an adaptive resident ODE plan for tensor state.
    pub fn adaptive(
        state_shape: Vec<usize>,
        dtype: Symbol,
        rhs: ResidentRhsLowering,
    ) -> Result<Self> {
        Self::new(ResidentOdeKind::Adaptive, state_shape, dtype, rhs)
    }

    fn new(
        kind: ResidentOdeKind,
        state_shape: Vec<usize>,
        dtype: Symbol,
        rhs: ResidentRhsLowering,
    ) -> Result<Self> {
        if state_shape.iter().product::<usize>() < 2 {
            return Err(Error::Eval(
                "resident ODE declines scalar or too-small tensor state".to_owned(),
            ));
        }
        if rhs != ResidentRhsLowering::TensorExpression {
            return Err(Error::Eval(
                "resident ODE requires a lowerable tensor RHS expression".to_owned(),
            ));
        }
        Ok(Self {
            kind,
            state_shape,
            dtype,
        })
    }

    /// Returns the execution family.
    pub fn kind(&self) -> ResidentOdeKind {
        self.kind
    }

    /// Returns the planned tensor state shape.
    pub fn state_shape(&self) -> &[usize] {
        &self.state_shape
    }

    /// Returns the planned tensor dtype.
    pub fn dtype(&self) -> &Symbol {
        &self.dtype
    }
}

/// Result plus provider counters for a resident ODE execution.
#[derive(Clone)]
pub struct ResidentOdeExecution {
    /// Eval reply produced by the canonical numeric pipeline.
    pub reply: EvalReply,
    /// Counter snapshot after final synchronization.
    pub snapshot: ModeledComputeSnapshot,
    /// Resident readbacks performed while executing the request.
    pub readbacks: usize,
    /// Accepted tensor submissions represented by the final flush.
    pub final_flush_accepted: usize,
}

/// Executes checked tensor ODE plans through the modeled resident site.
#[derive(Clone)]
pub struct ResidentOdeExecutor {
    executor: ModeledTensorExecutor,
}

impl ResidentOdeExecutor {
    /// Builds an ODE executor that auto-flushes bounded tensor submission batches.
    pub fn new(mut profile: ModeledComputeProfile) -> Self {
        profile.auto_flush_batches = true;
        Self {
            executor: ModeledTensorExecutor::new(profile),
        }
    }

    /// Returns the underlying modeled tensor executor.
    pub fn tensor_executor(&self) -> &ModeledTensorExecutor {
        &self.executor
    }

    /// Executes an ODE expression through the resident tensor site.
    pub fn execute(
        &self,
        cx: &mut Cx,
        plan: &ResidentOdePlan,
        expr: Expr,
    ) -> Result<ResidentOdeExecution> {
        let before = self.executor.snapshot();
        let site = TensorSite::new(
            compute_model_site_symbol(),
            Arc::new(self.executor.clone()) as Arc<dyn TensorExecutor>,
            Vec::new(),
        );
        let request = eval_request(expr);
        let reply = if plan.kind() == ResidentOdeKind::Fixed {
            self.executor.begin_internal_materialization();
            let reply = site.realize(cx, request);
            self.executor.end_internal_materialization();
            reply?
        } else {
            site.realize(cx, request)?
        };
        let value = if let Some(table) = reply.value.object().as_table_impl() {
            table.get(cx, Symbol::new("value"))?
        } else {
            reply.value.clone()
        };
        let tensor = tensor_value_ref(&value).ok_or_else(|| {
            Error::Eval("resident ODE result did not produce tensor state".to_owned())
        })?;
        if tensor.shape() != plan.state_shape() || tensor.dtype() != plan.dtype() {
            return Err(Error::Eval(format!(
                "resident ODE result shape/dtype {:?}/{} did not match plan {:?}/{}",
                tensor.shape(),
                tensor.dtype(),
                plan.state_shape(),
                plan.dtype()
            )));
        }
        let final_flush_accepted = self.executor.flush().map_err(Error::from)?.accepted;
        let snapshot = self.executor.snapshot();
        Ok(ResidentOdeExecution {
            reply,
            readbacks: snapshot.readbacks.saturating_sub(before.readbacks),
            snapshot,
            final_flush_accepted,
        })
    }
}

impl Default for ResidentOdeExecutor {
    fn default() -> Self {
        Self::new(ModeledComputeProfile::default())
    }
}

fn eval_request(expr: Expr) -> EvalRequest {
    EvalRequest {
        expr,
        result_shape: None,
        required_capabilities: Vec::new(),
        deadline: None,
        consistency: Consistency::LocalFirst,
        mode: EvalMode::Eval,
        answer_limit: None,
        stream_buffer: None,
        stream: false,
        trace: false,
    }
}
