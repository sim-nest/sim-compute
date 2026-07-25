use std::sync::Arc;

use sim_citizen::Citizen;
use sim_kernel::{DefaultFactory, EagerPolicy, Symbol};
use sim_lib_compute_model::ModeledComputeProfile;
use sim_lib_numbers_tensor::{
    TensorExecution, TensorExecutor, TensorLocation, TensorMeta, TensorOp, TensorRequest,
    add_op_symbol, build_tensor_value, tensor_value_ref,
};

use crate::{
    AutoComputeProfile, AutoRouteDecision, AutoTensorExecutor, BenchmarkBounds, ComputeAutoLib,
    ComputeDeviceIdentity, ComputeThermalPowerContext, ProfileStore, ProfileStorePolicy,
    compute_auto_site_symbol, measure_bounded_profile, measured_compute_profile_citizen_symbol,
    measured_compute_profile_shape_symbol,
};

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
        measured: None,
        expected: None,
        now_tick: 0,
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
fn measured_profile_round_trips_through_supplied_table() {
    let mut cx = test_cx();
    let identity = ComputeDeviceIdentity::new("adapter-a", "driver-1", "modeled");
    let measured = measure_bounded_profile(
        identity.clone(),
        ModeledComputeProfile::default(),
        ComputeThermalPowerContext {
            thermal: "steady".to_owned(),
            power: "plugged".to_owned(),
        },
        "unit-test",
        7,
        BenchmarkBounds::default(),
    );
    let table = cx.factory().table(Vec::new()).unwrap();
    let store = ProfileStore::new(table, ProfileStorePolicy::default()).unwrap();
    let key = Symbol::qualified("compute-profile", "adapter-a");

    store.save(&mut cx, key.clone(), &measured).unwrap();
    let loaded = store.load(&mut cx, key).unwrap().unwrap();

    assert_eq!(loaded.identity, identity);
    assert!(loaded.is_conclusive());
    assert_eq!(store.keys(&mut cx).unwrap().len(), 1);
}

#[test]
fn measured_profile_exposes_citizen_and_shape_records() {
    assert_eq!(
        measured_compute_profile_citizen_symbol().to_string(),
        "compute-profile/MeasuredProfile"
    );
    assert_eq!(
        measured_compute_profile_shape_symbol().to_string(),
        "compute-profile/MeasuredProfileShape"
    );
    assert_eq!(crate::MeasuredComputeProfile::citizen_version(), 0);
    assert_eq!(crate::MeasuredComputeProfile::citizen_arity(), 9);
}

#[test]
fn measured_profile_routes_device_only_when_fresh_compatible_and_conclusive() {
    let mut cx = test_cx();
    let identity = ComputeDeviceIdentity::new("adapter-a", "driver-1", "modeled");
    let measured = measure_bounded_profile(
        identity.clone(),
        ModeledComputeProfile::default(),
        ComputeThermalPowerContext {
            thermal: "steady".to_owned(),
            power: "plugged".to_owned(),
        },
        "unit-test",
        7,
        BenchmarkBounds::default(),
    );
    let executor = AutoTensorExecutor::new(AutoComputeProfile {
        modeled: None,
        measured: Some(measured),
        expected: Some(identity),
        now_tick: 8,
    });
    assert_eq!(executor.route_decision(), AutoRouteDecision::Device);

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
    executor.flush().unwrap();
    let events = executor.routing_events();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].decision, AutoRouteDecision::Device);
    assert!(events[0].materialization_bytes > 0);
    assert_eq!(events[1].synchronizations, 1);
}

#[test]
fn stale_or_incompatible_measured_profile_uses_cpu() {
    let mut cx = test_cx();
    let identity = ComputeDeviceIdentity::new("adapter-a", "driver-1", "modeled");
    let measured = measure_bounded_profile(
        identity,
        ModeledComputeProfile::default(),
        ComputeThermalPowerContext {
            thermal: "steady".to_owned(),
            power: "plugged".to_owned(),
        },
        "unit-test",
        7,
        BenchmarkBounds::default(),
    );
    let executor = AutoTensorExecutor::new(AutoComputeProfile {
        modeled: None,
        measured: Some(measured),
        expected: Some(ComputeDeviceIdentity::new(
            "adapter-b",
            "driver-1",
            "modeled",
        )),
        now_tick: 8,
    });
    assert_eq!(executor.route_decision(), AutoRouteDecision::Incompatible);
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
fn auto_lib_exports_site() {
    let mut cx = test_cx();
    cx.load_lib(&ComputeAutoLib::default()).unwrap();
    let site = cx
        .registry()
        .site_by_symbol(&compute_auto_site_symbol())
        .expect("auto compute site");
    assert!(site.object().as_eval_fabric().is_some());
}
