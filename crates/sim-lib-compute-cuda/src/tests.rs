use std::sync::Arc;

use sim_kernel::{DefaultFactory, EagerPolicy, Lib, Symbol};
use sim_lib_numbers_tensor::{
    Tensor, TensorExecution, TensorExecutor, TensorLocation, TensorMeta, TensorOp, TensorRequest,
    TensorStorage, add_op_symbol, build_tensor_value, domains, matmul_exec_op_symbol,
    parse_f32_literal_cell, tensor_value_ref,
};

use crate::{
    ComputeCudaLib, CudaResidentStorage, CudaTensorExecutor, FakeCudaLoader,
    compute_cuda_capability, compute_cuda_site_symbol,
};

// conformance: CUDA discovery records ABI evidence, exports only validated sites, and accepts dense matmul while declining unsupported requests before acceptance.

fn test_cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(
        Arc::new(EagerPolicy),
        Arc::new(DefaultFactory),
        sim_kernel::HandleSeed::new(0x7eb3_b7d0_272c_a1a3),
    );
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_float::F32NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_tensor::TensorNumbersLib::new())
        .unwrap();
    cx
}

fn f32_value(cx: &mut sim_kernel::Cx, canonical: &str) -> sim_kernel::Value {
    cx.factory()
        .number_literal(domains::f32(), canonical.to_owned())
        .unwrap()
}

fn tensor(cx: &mut sim_kernel::Cx, shape: Vec<usize>, cells: &[&str]) -> Tensor {
    let values = cells.iter().map(|cell| f32_value(cx, cell)).collect();
    tensor_value_ref(&build_tensor_value(cx, shape, Some(domains::f32()), values).unwrap())
        .unwrap()
        .clone()
}

fn f32_cells(tensor: &Tensor) -> Vec<f32> {
    tensor
        .cells()
        .unwrap()
        .iter()
        .map(|cell| parse_f32_literal_cell(cell).expect("f32 tensor cell"))
        .collect()
}

fn execute_cuda(
    cx: &mut sim_kernel::Cx,
    executor: &CudaTensorExecutor,
    symbol: Symbol,
    inputs: Vec<Tensor>,
    shape: Vec<usize>,
    dtype: Symbol,
) -> TensorExecution {
    let op = TensorOp::without_attributes(cx, symbol).unwrap();
    executor
        .execute(
            cx,
            TensorRequest::new(op, inputs, TensorMeta::new(shape, dtype)),
        )
        .unwrap()
}

#[test]
fn fake_loader_controls_site_exports_without_cuda_installed() {
    let available = ComputeCudaLib::from_probe_port(&FakeCudaLoader::available()).unwrap();
    let manifest = available.manifest();
    assert!(manifest.exports.is_empty());
    assert!(manifest.capabilities.is_empty());

    let incomplete = ComputeCudaLib::from_probe_port(&FakeCudaLoader::incomplete()).unwrap();
    assert!(incomplete.manifest().exports.is_empty());

    assert!(ComputeCudaLib::from_probe_port(&FakeCudaLoader::absent()).is_err());
}

#[test]
fn cuda_lib_does_not_register_site_without_live_runtime_handles() {
    let lib = ComputeCudaLib::from_probe_port(&FakeCudaLoader::available()).unwrap();
    let mut cx = sim_kernel::Cx::new(
        Arc::new(EagerPolicy),
        Arc::new(DefaultFactory),
        sim_kernel::HandleSeed::new(0xdb81_a03b_e2c0_179d),
    );
    cx.grant(compute_cuda_capability());
    cx.load_lib(&lib).unwrap();
    assert!(
        cx.registry()
            .site_by_symbol(&compute_cuda_site_symbol())
            .is_none()
    );
}

#[test]
fn dense_f32_matmul_returns_cuda_resident_storage() {
    let mut cx = test_cx();
    let evidence = ComputeCudaLib::from_probe_port(&FakeCudaLoader::available())
        .unwrap()
        .probe_evidence()
        .and_then(|probe| probe.evidence.clone())
        .unwrap();
    let executor = CudaTensorExecutor::new(evidence);
    let left = tensor(&mut cx, vec![2, 3], &["1", "2", "3", "4", "5", "6"]);
    let right = tensor(&mut cx, vec![3, 2], &["7", "8", "9", "10", "11", "12"]);
    let TensorExecution::Complete(result) = execute_cuda(
        &mut cx,
        &executor,
        matmul_exec_op_symbol(),
        vec![left, right],
        vec![2, 2],
        domains::f32(),
    ) else {
        panic!("cuda matmul should complete");
    };

    assert_eq!(f32_cells(&result), vec![58.0, 64.0, 139.0, 154.0]);
    let storage = result
        .storage()
        .as_any()
        .downcast_ref::<CudaResidentStorage>()
        .expect("cuda resident storage");
    assert_eq!(
        storage.location(),
        TensorLocation::Resident {
            site: compute_cuda_site_symbol(),
            allocation: Symbol::qualified("compute.alloc.cuda", "1"),
        }
    );
    assert_eq!(executor.flush().unwrap().accepted, 1);
}

#[test]
fn unsupported_operations_are_declined_before_acceptance() {
    let mut cx = test_cx();
    let evidence = ComputeCudaLib::from_probe_port(&FakeCudaLoader::available())
        .unwrap()
        .probe_evidence()
        .and_then(|probe| probe.evidence.clone())
        .unwrap();
    let executor = CudaTensorExecutor::new(evidence);
    let left = tensor(&mut cx, vec![2], &["1", "2"]);
    let right = tensor(&mut cx, vec![2], &["3", "4"]);
    let TensorExecution::Unsupported { reason } = execute_cuda(
        &mut cx,
        &executor,
        add_op_symbol(),
        vec![left, right],
        vec![2],
        domains::f32(),
    ) else {
        panic!("cuda provider must decline non-matmul operations");
    };
    assert!(reason.contains("dense matmul only"));
    assert_eq!(executor.flush().unwrap().accepted, 0);
}

#[test]
fn half_matmul_fails_closed_until_a_real_cublaslt_execution_path_exists() {
    let mut cx = test_cx();
    let evidence = ComputeCudaLib::from_probe_port(&FakeCudaLoader::incomplete())
        .unwrap()
        .probe_evidence()
        .and_then(|probe| probe.evidence.clone())
        .unwrap();
    let executor = CudaTensorExecutor::new(evidence);
    let left = tensor(&mut cx, vec![1, 1], &["1"]);
    let right = tensor(&mut cx, vec![1, 1], &["2"]);
    let TensorExecution::Unsupported { reason } = execute_cuda(
        &mut cx,
        &executor,
        matmul_exec_op_symbol(),
        vec![left, right],
        vec![1, 1],
        domains::f16(),
    ) else {
        panic!("half matmul must require cuBLASLt ABI evidence");
    };
    assert!(reason.contains("dense f32 matmul"));
    assert_eq!(executor.flush().unwrap().accepted, 0);
}
