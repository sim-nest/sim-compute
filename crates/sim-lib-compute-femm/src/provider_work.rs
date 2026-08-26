use std::sync::Arc;

use sim_kernel::{Cx, DefaultFactory, EagerPolicy, Symbol, Value};
use sim_lib_femm_core::{FemmError, FemmResult};
use sim_lib_numbers_tensor::{
    Tensor, TensorExecution, TensorMeta, TensorOp, TensorRequest, add_op_symbol,
    build_tensor_value, dot_op_symbol, mul_op_symbol, norm_op_symbol, sub_op_symbol,
    tensor_value_ref,
};

use crate::solver::ResidentCsrSolver;

pub(crate) struct ProviderWork<'a> {
    solver: &'a ResidentCsrSolver,
    cx: Cx,
}

impl<'a> ProviderWork<'a> {
    pub(crate) fn new(solver: &'a ResidentCsrSolver) -> Self {
        let mut cx = Cx::new(
            Arc::new(EagerPolicy),
            Arc::new(DefaultFactory),
            sim_kernel::HandleSeed::new(0x4645_4d4d),
        );
        cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
            .expect("numbers arithmetic lib loads");
        cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
            .expect("numbers f64 lib loads");
        cx.load_lib(&sim_lib_numbers_float::F32NumbersLib::new())
            .expect("numbers f32 lib loads");
        cx.load_lib(&sim_lib_numbers_tensor::TensorNumbersLib::new())
            .expect("numbers tensor lib loads");
        Self { solver, cx }
    }

    pub(crate) fn upload_tensor(
        &mut self,
        shape: Vec<usize>,
        values: &[f32],
    ) -> FemmResult<Tensor> {
        let tensor = self.host_tensor(shape.clone(), values)?;
        let zero_values = vec![0.0; values.len()];
        let zero = self.host_tensor(shape.clone(), &zero_values)?;
        self.execute(
            add_op_symbol(),
            vec![tensor, zero],
            shape,
            "resident tensor upload",
        )
    }

    pub(crate) fn execute(
        &mut self,
        symbol: Symbol,
        inputs: Vec<Tensor>,
        shape: Vec<usize>,
        context: &str,
    ) -> FemmResult<Tensor> {
        let op = TensorOp::without_attributes(&mut self.cx, symbol.clone())
            .map_err(|err| FemmError::SolveDidNotConverge(err.to_string()))?;
        let request = TensorRequest::new(
            op,
            inputs,
            TensorMeta::new(shape, Symbol::qualified("numbers", "f32")),
        );
        match self.solver.executor.execute(&mut self.cx, request) {
            Ok(TensorExecution::Complete(tensor)) => Ok(tensor),
            Ok(TensorExecution::Unsupported { reason }) => {
                self.solver.refuse(FemmError::SolveDidNotConverge(format!(
                    "{context} unsupported by provider: {reason}"
                )))
            }
            Err(err) => self.solver.refuse(FemmError::SolveDidNotConverge(format!(
                "{context} provider failure: {err}"
            ))),
        }
    }

    pub(crate) fn add(
        &mut self,
        left: &Tensor,
        right: &Tensor,
        context: &str,
    ) -> FemmResult<Tensor> {
        self.execute(
            add_op_symbol(),
            vec![left.clone(), right.clone()],
            left.shape().to_vec(),
            context,
        )
    }

    pub(crate) fn sub(
        &mut self,
        left: &Tensor,
        right: &Tensor,
        context: &str,
    ) -> FemmResult<Tensor> {
        self.execute(
            sub_op_symbol(),
            vec![left.clone(), right.clone()],
            left.shape().to_vec(),
            context,
        )
    }

    pub(crate) fn scale(&mut self, tensor: &Tensor, scalar: f32) -> FemmResult<Tensor> {
        let scalar = self.host_tensor(Vec::new(), &[scalar])?;
        self.execute(
            mul_op_symbol(),
            vec![tensor.clone(), scalar],
            tensor.shape().to_vec(),
            "resident vector scale",
        )
    }

    pub(crate) fn dot(&mut self, left: &Tensor, right: &Tensor, context: &str) -> FemmResult<f32> {
        let tensor = self.execute(
            dot_op_symbol(),
            vec![left.clone(), right.clone()],
            Vec::new(),
            context,
        )?;
        self.scalar_value(&tensor, context)
    }

    pub(crate) fn norm(&mut self, tensor: &Tensor, context: &str) -> FemmResult<f32> {
        let tensor = self.execute(norm_op_symbol(), vec![tensor.clone()], Vec::new(), context)?;
        self.scalar_value(&tensor, context)
    }

    pub(crate) fn finish(&mut self) -> FemmResult<()> {
        let evidence = self.solver.executor.flush().map_err(|err| {
            FemmError::SolveDidNotConverge(format!("provider flush failed: {err}"))
        })?;
        let mut state = self
            .solver
            .state
            .lock()
            .expect("resident CSR state poisoned");
        state.snapshot.provider_submissions = state
            .snapshot
            .provider_submissions
            .saturating_add(evidence.accepted);
        if let Some(snapshot) = self.solver.modeled_snapshot() {
            state.snapshot.provider_readbacks = snapshot.readbacks;
        }
        Ok(())
    }

    pub(crate) fn tensor_values(&mut self, tensor: &Tensor) -> FemmResult<Vec<f32>> {
        tensor
            .cells()
            .map_err(|err| FemmError::SolveDidNotConverge(err.to_string()))?
            .iter()
            .map(|value| value_to_f32(&mut self.cx, value))
            .collect()
    }

    fn host_tensor(&mut self, shape: Vec<usize>, values: &[f32]) -> FemmResult<Tensor> {
        let cells = values
            .iter()
            .map(|value| {
                self.cx
                    .factory()
                    .number_literal(Symbol::qualified("numbers", "f32"), value.to_string())
                    .map_err(|err| FemmError::SolveDidNotConverge(err.to_string()))
            })
            .collect::<FemmResult<Vec<_>>>()?;
        tensor_value_ref(
            &build_tensor_value(
                &mut self.cx,
                shape,
                Some(Symbol::qualified("numbers", "f32")),
                cells,
            )
            .map_err(|err| FemmError::SolveDidNotConverge(err.to_string()))?,
        )
        .cloned()
        .ok_or_else(|| {
            FemmError::SolveDidNotConverge(
                "provider tensor upload produced a non-tensor value".to_owned(),
            )
        })
    }

    fn scalar_value(&mut self, tensor: &Tensor, context: &str) -> FemmResult<f32> {
        if !tensor.shape().is_empty() || tensor.len() != 1 {
            return Err(FemmError::SolveDidNotConverge(format!(
                "{context} did not produce a scalar"
            )));
        }
        finite_f32(self.tensor_values(tensor)?[0], context)
    }
}

fn value_to_f32(cx: &mut Cx, value: &Value) -> FemmResult<f32> {
    let number = cx
        .number_value_ref(value.clone())
        .map_err(|err| FemmError::SolveDidNotConverge(err.to_string()))?
        .ok_or_else(|| {
            FemmError::SolveDidNotConverge("provider tensor cell was not numeric".to_owned())
        })?;
    let literal = number.literal.ok_or_else(|| {
        FemmError::SolveDidNotConverge("provider tensor cell had no literal".to_owned())
    })?;
    literal.canonical.parse::<f32>().map_err(|_| {
        FemmError::SolveDidNotConverge("provider tensor cell was not f32-parsable".to_owned())
    })
}

fn finite_f32(value: f32, context: &str) -> FemmResult<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FemmError::SolveDidNotConverge(context.to_owned()))
    }
}
