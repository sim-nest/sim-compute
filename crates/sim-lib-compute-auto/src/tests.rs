use std::sync::Arc;

use sim_kernel::{DefaultFactory, EagerPolicy, Symbol};
use sim_lib_compute_model::ModeledComputeProfile;
use sim_lib_numbers_tensor::{
    TensorExecution, TensorExecutor, TensorLocation, TensorMeta, TensorOp, TensorRequest,
    add_op_symbol, build_tensor_value, tensor_value_ref,
};

use crate::{AutoComputeProfile, AutoTensorExecutor, ComputeAutoLib, compute_auto_site_symbol};

// conformance: auto compute site selects modeled providers and falls back to CPU without a compatible profile.

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

fn vector(cx: &mut sim_kernel::Cx, cells: &[&str]) -> sim_lib_numbers_tensor::Tensor {
    let values = cells
        .iter()
        .map(|cell| {
            cx.factory()
                .number_literal(Symbol::qualified("numbers", "i64"), (*cell).to_owned())
                .unwrap()
        })
        .collect();
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
fn auto_without_profile_uses_cpu_fallback() {
    let mut cx = test_cx();
    let executor = AutoTensorExecutor::default();
    assert!(executor.uses_cpu_fallback());
    let op = TensorOp::without_attributes(&mut cx, add_op_symbol()).unwrap();
    let left = vector(&mut cx, &["1"]);
    let right = vector(&mut cx, &["2"]);
    let request = TensorRequest::new(
        op,
        vec![left, right],
        TensorMeta::new(vec![1], Symbol::qualified("numbers", "i64")),
    );
    let tensor = match executor.execute(&mut cx, request).unwrap() {
        TensorExecution::Complete(tensor) => tensor,
        TensorExecution::Unsupported { reason } => panic!("{reason}"),
    };
    assert_eq!(tensor.location(), TensorLocation::Host);
}

#[test]
fn auto_with_modeled_profile_returns_resident_storage() {
    let mut cx = test_cx();
    let executor = AutoTensorExecutor::new(AutoComputeProfile {
        modeled: Some(ModeledComputeProfile::default()),
    });
    let op = TensorOp::without_attributes(&mut cx, add_op_symbol()).unwrap();
    let left = vector(&mut cx, &["1"]);
    let right = vector(&mut cx, &["2"]);
    let request = TensorRequest::new(
        op,
        vec![left, right],
        TensorMeta::new(vec![1], Symbol::qualified("numbers", "i64")),
    );
    let tensor = match executor.execute(&mut cx, request).unwrap() {
        TensorExecution::Complete(tensor) => tensor,
        TensorExecution::Unsupported { reason } => panic!("{reason}"),
    };
    assert!(matches!(tensor.location(), TensorLocation::Resident { .. }));
}

#[test]
fn auto_lib_exports_site() {
    let mut cx = test_cx();
    cx.load_lib(&ComputeAutoLib::default()).unwrap();
    let site = cx
        .registry()
        .site_by_symbol(&compute_auto_site_symbol())
        .expect("auto compute site");
    assert!(site.object().as_eval_fabric().is_some());
}
