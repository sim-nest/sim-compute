use std::sync::Arc;

use sim_kernel::{DefaultFactory, EagerPolicy, Symbol};
use sim_lib_numbers_tensor::{
    TensorExecution, TensorExecutor, TensorLocation, TensorMeta, TensorOp, TensorRequest,
    add_op_symbol, build_tensor_value, tensor_value_ref,
};

use crate::{
    ComputeModelLib, ModeledComputeFault, ModeledComputeProfile, ModeledResidentStorage,
    ModeledTensorExecutor, compute_model_site_symbol,
};

// conformance: modeled compute provider returns resident Tensor storage, bounded flush evidence, and deterministic faults.

fn test_cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_float::F32NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_i64::I64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_tensor::TensorNumbersLib::new())
        .unwrap();
    cx
}

fn i64(cx: &mut sim_kernel::Cx, canonical: &str) -> sim_kernel::Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "i64"), canonical.to_owned())
        .unwrap()
}

fn vector(cx: &mut sim_kernel::Cx, cells: &[&str]) -> sim_lib_numbers_tensor::Tensor {
    let values = cells.iter().map(|cell| i64(cx, cell)).collect();
    tensor_value_ref(
        &build_tensor_value(
            cx,
            vec![cells.len()],
            Some(Symbol::qualified("numbers", "i64")),
            values,
        )
        .unwrap(),
    )
    .unwrap()
    .clone()
}

#[test]
fn modeled_executor_returns_resident_tensor_and_flush_evidence() {
    let mut cx = test_cx();
    let executor = ModeledTensorExecutor::default();
    let op = TensorOp::without_attributes(&mut cx, add_op_symbol()).unwrap();
    let request = TensorRequest::new(
        op,
        vec![vector(&mut cx, &["1", "2"]), vector(&mut cx, &["10", "20"])],
        TensorMeta::new(vec![2], Symbol::qualified("numbers", "i64")),
    );

    let result = executor.execute(&mut cx, request).unwrap();
    let TensorExecution::Complete(tensor) = result else {
        panic!("modeled executor must complete add");
    };
    assert!(matches!(
        tensor.location(),
        TensorLocation::Resident { site, .. } if site == compute_model_site_symbol()
    ));
    assert_eq!(executor.snapshot().accepted, 1);
    assert_eq!(executor.flush().unwrap().accepted, 1);
    assert_eq!(executor.snapshot().queued, 0);

    let storage = tensor
        .storage()
        .as_any()
        .downcast_ref::<ModeledResidentStorage>()
        .expect("modeled resident storage");
    assert!(
        storage
            .allocation()
            .symbol()
            .to_string()
            .contains("compute.alloc")
    );
    let cells = tensor.cells().unwrap();
    assert_eq!(cells.len(), 2);
    assert_eq!(executor.snapshot().readbacks, 1);
    let _ = tensor.cells().unwrap();
    assert_eq!(executor.snapshot().readbacks, 1);
}

#[test]
fn resident_chaining_avoids_readback_until_materialization() {
    let mut cx = test_cx();
    let executor = ModeledTensorExecutor::default();
    let op = TensorOp::without_attributes(&mut cx, add_op_symbol()).unwrap();
    let first_left = vector(&mut cx, &["1", "2"]);
    let first_right = vector(&mut cx, &["3", "4"]);
    let first_request = TensorRequest::new(
        op.clone(),
        vec![first_left, first_right],
        TensorMeta::new(vec![2], Symbol::qualified("numbers", "i64")),
    );
    let first = match executor.execute(&mut cx, first_request).unwrap() {
        TensorExecution::Complete(tensor) => tensor,
        TensorExecution::Unsupported { reason } => panic!("{reason}"),
    };
    let addend = vector(&mut cx, &["10", "20"]);
    let second_request = TensorRequest::new(
        op,
        vec![first, addend],
        TensorMeta::new(vec![2], Symbol::qualified("numbers", "i64")),
    );
    let second = match executor.execute(&mut cx, second_request).unwrap() {
        TensorExecution::Complete(tensor) => tensor,
        TensorExecution::Unsupported { reason } => panic!("{reason}"),
    };
    assert_eq!(executor.snapshot().readbacks, 0);
    assert_eq!(second.cells().unwrap().len(), 2);
    assert_eq!(executor.snapshot().readbacks, 1);
}

#[test]
fn modeled_faults_are_injected_at_submission_execution_and_readback() {
    let mut cx = test_cx();
    let oom = ModeledTensorExecutor::new(ModeledComputeProfile {
        fault: Some(ModeledComputeFault::OomBeforeSubmit),
        ..ModeledComputeProfile::default()
    });
    let op = TensorOp::without_attributes(&mut cx, add_op_symbol()).unwrap();
    let request = TensorRequest::new(
        op.clone(),
        vec![vector(&mut cx, &["1"]), vector(&mut cx, &["2"])],
        TensorMeta::new(vec![1], Symbol::qualified("numbers", "i64")),
    );
    let oom_error = match oom.execute(&mut cx, request) {
        Ok(_) => panic!("OOM fault must reject submission"),
        Err(error) => error,
    };
    assert!(oom_error.to_string().contains("out of memory"));

    let lost = ModeledTensorExecutor::new(ModeledComputeProfile {
        fault: Some(ModeledComputeFault::DeviceLostDuringExecute),
        ..ModeledComputeProfile::default()
    });
    let request = TensorRequest::new(
        op.clone(),
        vec![vector(&mut cx, &["1"]), vector(&mut cx, &["2"])],
        TensorMeta::new(vec![1], Symbol::qualified("numbers", "i64")),
    );
    let lost_error = match lost.execute(&mut cx, request) {
        Ok(_) => panic!("device-loss fault must fail execution"),
        Err(error) => error,
    };
    assert!(lost_error.to_string().contains("device lost"));

    let readback = ModeledTensorExecutor::new(ModeledComputeProfile {
        fault: Some(ModeledComputeFault::ReadbackFailure),
        ..ModeledComputeProfile::default()
    });
    let request = TensorRequest::new(
        op,
        vec![vector(&mut cx, &["1"]), vector(&mut cx, &["2"])],
        TensorMeta::new(vec![1], Symbol::qualified("numbers", "i64")),
    );
    let tensor = match readback.execute(&mut cx, request).unwrap() {
        TensorExecution::Complete(tensor) => tensor,
        TensorExecution::Unsupported { reason } => panic!("{reason}"),
    };
    assert!(
        tensor
            .cells()
            .unwrap_err()
            .to_string()
            .contains("readback failed")
    );
}

#[test]
fn modeled_lib_exports_site() {
    let mut cx = test_cx();
    cx.load_lib(&ComputeModelLib::default()).unwrap();
    let site = cx
        .registry()
        .site_by_symbol(&compute_model_site_symbol())
        .expect("modeled compute site");
    assert!(site.object().as_eval_fabric().is_some());
}
