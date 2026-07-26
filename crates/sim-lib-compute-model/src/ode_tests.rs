use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use sim_kernel::{DefaultFactory, EagerPolicy, Expr, NumberLiteral, Symbol, Value};
use sim_lib_numbers_tensor::{
    SubmissionEvidence, TensorExecError, TensorExecution, TensorExecutor, TensorExecutorCard,
    TensorLocation, TensorRequest, tensor_value_ref,
};

use crate::{
    ModeledComputeFault, ModeledComputeProfile, ModeledTensorExecutor, ResidentOdeExecutor,
    ResidentOdePlan, ResidentRhsLowering, compute_model_site_symbol,
};

fn test_cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_tensor::TensorNumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_tensor_bcast::TensorBroadcastLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_numeric::NumericNumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_rk::RkNumbersLib::new())
        .unwrap();
    cx
}

fn f64_expr(canonical: &str) -> Expr {
    Expr::Number(NumberLiteral {
        domain: Symbol::qualified("numbers", "f64"),
        canonical: canonical.to_owned(),
    })
}

fn symbol_expr(symbol: impl Into<Symbol>) -> Expr {
    Expr::Symbol(symbol.into())
}

fn call_expr(symbol: impl Into<Symbol>, args: Vec<Expr>) -> Expr {
    Expr::Call {
        operator: Box::new(symbol_expr(symbol)),
        args,
    }
}

fn vec_expr(values: &[&str]) -> Expr {
    call_expr(
        Symbol::new("vec"),
        values.iter().map(|value| f64_expr(value)).collect(),
    )
}

fn tensor_identity_rhs(cx: &mut sim_kernel::Cx, name: &str) {
    let rhs = cx
        .factory()
        .opaque(Arc::new(sim_lib_numbers_func::Func::native(
            vec![Symbol::new("x"), Symbol::new("y")],
            Arc::new(|_cx, args| {
                let [_, y] = args else {
                    return Err(sim_kernel::Error::Eval("expected x and y".to_owned()));
                };
                Ok(y.clone())
            }),
        )))
        .unwrap();
    cx.env_mut().define(Symbol::new(name), rhs);
}

fn tensor_pipeline_expr(method: &str, h: &str, y0: Expr) -> Expr {
    call_expr(
        Symbol::qualified("numeric", "run-composed"),
        vec![
            call_expr(
                Symbol::qualified("numeric", "compose"),
                vec![
                    symbol_expr(Symbol::new("tensor-rhs")),
                    symbol_expr(Symbol::new("ode-solve")),
                    symbol_expr(Symbol::new(method)),
                    symbol_expr(Symbol::new("tensor")),
                ],
            ),
            f64_expr("0.0"),
            f64_expr("0.2"),
            y0,
            f64_expr(h),
        ],
    )
}

fn run_cpu_ode(cx: &mut sim_kernel::Cx, method: &str, h: &str) -> Value {
    cx.eval_expr(tensor_pipeline_expr(
        method,
        h,
        vec_expr(&["1.0", "2.0", "-0.5", "4.0"]),
    ))
    .unwrap()
    .object()
    .as_table_impl()
    .unwrap()
    .get(cx, Symbol::new("value"))
    .unwrap()
}

fn tensor_f64s(cx: &mut sim_kernel::Cx, value: &Value) -> Vec<f64> {
    tensor_value_ref(value)
        .unwrap()
        .cells()
        .unwrap()
        .iter()
        .map(|cell| cell.object().display(cx).unwrap().parse::<f64>().unwrap())
        .collect()
}

#[derive(Clone)]
struct CountingExecutor {
    inner: ModeledTensorExecutor,
    accepted: Arc<AtomicUsize>,
    flushes: Arc<AtomicUsize>,
}

impl CountingExecutor {
    fn new() -> Self {
        Self {
            inner: ModeledTensorExecutor::new(ModeledComputeProfile {
                auto_flush_batches: true,
                max_queue_depth: 4,
                ..ModeledComputeProfile::default()
            }),
            accepted: Arc::new(AtomicUsize::new(0)),
            flushes: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn snapshot(&self) -> (usize, usize, crate::ModeledComputeSnapshot) {
        (
            self.accepted.load(Ordering::SeqCst),
            self.flushes.load(Ordering::SeqCst),
            self.inner.snapshot(),
        )
    }
}

impl TensorExecutor for CountingExecutor {
    fn card(&self) -> TensorExecutorCard {
        TensorExecutorCard::new(
            Symbol::qualified("test", "executor/resident-ode"),
            "test-resident-ode",
            Symbol::qualified("test", "resident"),
            self.inner.card().operations.to_vec(),
            None,
        )
    }

    fn execute(
        &self,
        cx: &mut sim_kernel::Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        let result = self.inner.execute(cx, request);
        if matches!(result, Ok(TensorExecution::Complete(_))) {
            self.accepted.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        self.flushes.fetch_add(1, Ordering::SeqCst);
        self.inner.flush()
    }
}

#[derive(Clone)]
struct LostDeviceExecutor;

impl TensorExecutor for LostDeviceExecutor {
    fn card(&self) -> TensorExecutorCard {
        TensorExecutorCard::new(
            Symbol::qualified("test", "executor/lost"),
            "test-lost-device",
            Symbol::qualified("test", "device"),
            ModeledTensorExecutor::default().card().operations.to_vec(),
            None,
        )
    }

    fn execute(
        &self,
        _cx: &mut sim_kernel::Cx,
        _request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        Err(TensorExecError::Eval {
            message: Arc::from("test device lost during tensor execution"),
        })
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        Ok(SubmissionEvidence::new(self.card().symbol, 0))
    }
}

#[test]
fn resident_fixed_ode_batches_stages_without_readback_and_matches_cpu() {
    let mut cpu_cx = test_cx();
    tensor_identity_rhs(&mut cpu_cx, "tensor-rhs");
    let expected = run_cpu_ode(&mut cpu_cx, "rk4", "0.02");

    let mut cx = test_cx();
    tensor_identity_rhs(&mut cx, "tensor-rhs");
    let executor = ResidentOdeExecutor::new(ModeledComputeProfile {
        max_queue_depth: 4,
        ..ModeledComputeProfile::default()
    });
    let plan = ResidentOdePlan::fixed(
        vec![4],
        Symbol::qualified("numbers", "f64"),
        ResidentRhsLowering::TensorExpression,
    )
    .unwrap();

    let execution = executor
        .execute(
            &mut cx,
            &plan,
            tensor_pipeline_expr("rk4", "0.02", vec_expr(&["1.0", "2.0", "-0.5", "4.0"])),
        )
        .unwrap();
    let value = execution
        .reply
        .value
        .object()
        .as_table_impl()
        .unwrap()
        .get(&mut cx, Symbol::new("value"))
        .unwrap();

    assert_eq!(execution.modeled_readbacks, Some(0));
    let snapshot = execution.modeled_snapshot.as_ref().unwrap();
    assert!(snapshot.batch_flushes > 0);
    assert!(execution.final_flush.accepted <= 4);
    assert!(matches!(
        tensor_value_ref(&value).unwrap().location(),
        TensorLocation::Resident { site, .. } if site == compute_model_site_symbol()
    ));
    let actual = tensor_f64s(&mut cx, &value);
    let expected = tensor_f64s(&mut cpu_cx, &expected);
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-9);
    }
}

#[test]
fn resident_adaptive_ode_keeps_candidate_resident_with_scalar_decisions() {
    let mut cx = test_cx();
    tensor_identity_rhs(&mut cx, "tensor-rhs");
    let executor = ResidentOdeExecutor::new(ModeledComputeProfile {
        max_queue_depth: 8,
        ..ModeledComputeProfile::default()
    });
    let plan = ResidentOdePlan::adaptive(
        vec![4],
        Symbol::qualified("numbers", "f64"),
        ResidentRhsLowering::TensorExpression,
    )
    .unwrap();

    let execution = executor
        .execute(
            &mut cx,
            &plan,
            tensor_pipeline_expr("rkf45", "0.1", vec_expr(&["1.0", "2.0", "-0.5", "4.0"])),
        )
        .unwrap();
    let value = execution
        .reply
        .value
        .object()
        .as_table_impl()
        .unwrap()
        .get(&mut cx, Symbol::new("value"))
        .unwrap();

    let readbacks = execution.modeled_readbacks.unwrap();
    let snapshot = execution.modeled_snapshot.as_ref().unwrap();
    assert!(readbacks > 0);
    assert!(readbacks < snapshot.accepted);
    assert!(matches!(
        tensor_value_ref(&value).unwrap().location(),
        TensorLocation::Resident { site, .. } if site == compute_model_site_symbol()
    ));
}

#[test]
fn resident_ode_declines_small_and_nonlowerable_work_before_execution() {
    let small = ResidentOdePlan::fixed(
        vec![1],
        Symbol::qualified("numbers", "f64"),
        ResidentRhsLowering::TensorExpression,
    )
    .unwrap_err();
    assert!(small.to_string().contains("too-small"));

    let nonlowerable = ResidentOdePlan::adaptive(
        vec![4],
        Symbol::qualified("numbers", "f64"),
        ResidentRhsLowering::NonLowerable,
    )
    .unwrap_err();
    assert!(nonlowerable.to_string().contains("lowerable"));
}

#[test]
fn resident_ode_runs_over_dynamic_tensor_executor_without_adapter_readback() {
    let mut cpu_cx = test_cx();
    tensor_identity_rhs(&mut cpu_cx, "tensor-rhs");
    let expected = run_cpu_ode(&mut cpu_cx, "rk4", "0.02");

    let executor = CountingExecutor::new();
    let observed = executor.clone();
    let mut cx = test_cx();
    tensor_identity_rhs(&mut cx, "tensor-rhs");
    let resident = ResidentOdeExecutor::with_executor(
        Symbol::qualified("test", "site/resident-ode"),
        Arc::new(executor),
    );
    let plan = ResidentOdePlan::fixed(
        vec![4],
        Symbol::qualified("numbers", "f64"),
        ResidentRhsLowering::TensorExpression,
    )
    .unwrap();

    let execution = resident
        .execute(
            &mut cx,
            &plan,
            tensor_pipeline_expr("rk4", "0.02", vec_expr(&["1.0", "2.0", "-0.5", "4.0"])),
        )
        .unwrap();
    let value = execution
        .reply
        .value
        .object()
        .as_table_impl()
        .unwrap()
        .get(&mut cx, Symbol::new("value"))
        .unwrap();
    let (accepted, flushes, _snapshot) = observed.snapshot();

    assert_eq!(
        execution.executor.symbol,
        Symbol::qualified("test", "executor/resident-ode")
    );
    assert_eq!(execution.modeled_snapshot, None);
    assert_eq!(execution.modeled_readbacks, None);
    assert!(accepted > 0);
    assert_eq!(flushes, 1);
    assert!(execution.final_flush.accepted <= 4);

    let actual = tensor_f64s(&mut cx, &value);
    let expected = tensor_f64s(&mut cpu_cx, &expected);
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-9);
    }
}

#[test]
fn resident_ode_reports_device_loss_from_active_tensor_executor() {
    let mut cx = test_cx();
    tensor_identity_rhs(&mut cx, "tensor-rhs");
    let resident = ResidentOdeExecutor::with_executor(
        Symbol::qualified("test", "site/lost-device"),
        Arc::new(LostDeviceExecutor),
    );
    let plan = ResidentOdePlan::fixed(
        vec![4],
        Symbol::qualified("numbers", "f64"),
        ResidentRhsLowering::TensorExpression,
    )
    .unwrap();

    let error = match resident.execute(
        &mut cx,
        &plan,
        tensor_pipeline_expr("rk4", "0.02", vec_expr(&["1.0", "2.0", "-0.5", "4.0"])),
    ) {
        Ok(_) => panic!("resident ODE must propagate device loss"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("device lost"), "{error}");
}

#[test]
fn resident_ode_propagates_modeled_device_loss_without_rk_changes() {
    let mut cx = test_cx();
    tensor_identity_rhs(&mut cx, "tensor-rhs");
    let executor = ResidentOdeExecutor::new(ModeledComputeProfile {
        fault: Some(ModeledComputeFault::DeviceLostDuringExecute),
        ..ModeledComputeProfile::default()
    });
    let plan = ResidentOdePlan::fixed(
        vec![4],
        Symbol::qualified("numbers", "f64"),
        ResidentRhsLowering::TensorExpression,
    )
    .unwrap();

    let error = match executor.execute(
        &mut cx,
        &plan,
        tensor_pipeline_expr("rk4", "0.02", vec_expr(&["1.0", "2.0", "-0.5", "4.0"])),
    ) {
        Ok(_) => panic!("resident ODE must propagate modeled device loss"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("device lost"), "{error}");
}
