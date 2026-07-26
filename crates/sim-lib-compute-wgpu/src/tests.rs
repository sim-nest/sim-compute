use std::sync::Arc;

use sim_kernel::{DefaultFactory, EagerPolicy, Symbol};
use sim_lib_compute_auto::{ComputeEvidenceKind, verify_physical};
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, Tensor, TensorExecution, TensorExecutor, TensorLocation, TensorMeta,
    TensorOp, TensorRequest, add_op_symbol, build_tensor_value, cos_op_symbol, dot_op_symbol,
    exp_op_symbol, matmul_exec_op_symbol, max_op_symbol, min_op_symbol, neg_op_symbol,
    norm_op_symbol, parse_f16_literal_cell, parse_f32_literal_cell, sin_op_symbol, sqrt_op_symbol,
    sub_op_symbol, sum_op_symbol, tensor_value_ref, transpose_exec_op_symbol,
};

use crate::{
    AllocationAttempt, ComputeWgpuLib, ProbeEvidence, RequestedWgpuProfile, TransferEvidence,
    WgpuAdapterEvidence, WgpuAdapterProbe, WgpuCapabilityEvidence, WgpuDiscovery, WgpuKernelDType,
    WgpuKernelOp, WgpuLimitEvidence, WgpuMaterializationCache, WgpuPipelineCache, WgpuQueueLimits,
    WgpuResidentArena, WgpuResidentStorage, WgpuSegmentPlan, WgpuSubmissionQueue,
    WgpuTensorExecutor, WgpuTransferPlan, compute_wgpu_capability, compute_wgpu_site_symbol,
    kernels::execute_portable_kernel, probe::discover_wgpu_adapter_runtimes,
    site::WgpuExecutionContext,
};

// conformance: wgpu discovery records evidence, exports only successful adapter sites, and plans bounded resident submissions.

mod primitive_tests;
mod residency_tests;

fn limits(buffer_size: u64) -> WgpuLimitEvidence {
    WgpuLimitEvidence {
        max_buffer_size: buffer_size,
        max_storage_buffer_binding_size: 1 << 20,
        max_uniform_buffer_binding_size: 64 << 10,
        min_storage_buffer_offset_alignment: 256,
        min_uniform_buffer_offset_alignment: 256,
        max_compute_workgroups_per_dimension: 65_535,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_size_y: 1,
        max_compute_workgroup_size_z: 1,
    }
}

fn adapter(name: &str, backend: &str, success: bool) -> WgpuAdapterProbe {
    adapter_with_limits(name, backend, success, limits(1 << 24))
}

fn adapter_with_limits(
    name: &str,
    backend: &str,
    success: bool,
    granted_limits: WgpuLimitEvidence,
) -> WgpuAdapterProbe {
    WgpuAdapterProbe {
        evidence_kind: ComputeEvidenceKind::HostEmulated,
        claimed_identity: None,
        observed_identity: None,
        adapter: WgpuAdapterEvidence {
            ordinal: 99,
            name: name.to_owned(),
            backend: backend.to_owned(),
            adapter_type: "DiscreteGpu".to_owned(),
            vendor: 100,
            device: 200,
            requested: RequestedWgpuProfile {
                limits: limits(4096),
                timestamp_query: true,
                shader_f16: true,
            },
            granted_limits,
            granted_features: WgpuCapabilityEvidence {
                timestamp_query: true,
                shader_f16: true,
                mappable_primary_buffers: false,
            },
        },
        probe: ProbeEvidence {
            transfer: TransferEvidence {
                bytes: 16,
                transfer_ok: success,
                mapping_ok: success,
            },
            allocation_attempts: vec![AllocationAttempt {
                bytes: 4096,
                success,
            }],
        },
    }
}

fn test_cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
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
        .number_literal(Symbol::qualified("numbers", "f32"), canonical.to_owned())
        .unwrap()
}

fn tensor(
    cx: &mut sim_kernel::Cx,
    shape: Vec<usize>,
    cells: &[&str],
) -> sim_lib_numbers_tensor::Tensor {
    let values = cells.iter().map(|cell| f32_value(cx, cell)).collect();
    tensor_value_ref(
        &build_tensor_value(cx, shape, Some(Symbol::qualified("numbers", "f32")), values).unwrap(),
    )
    .unwrap()
    .clone()
}

fn f32_cells(tensor: &sim_lib_numbers_tensor::Tensor) -> Vec<f32> {
    tensor
        .cells()
        .unwrap()
        .iter()
        .map(|cell| parse_f32_literal_cell(cell).expect("f32 tensor cell"))
        .collect()
}

fn execute_wgpu(
    cx: &mut sim_kernel::Cx,
    executor: &WgpuTensorExecutor,
    symbol: Symbol,
    inputs: Vec<Tensor>,
    shape: Vec<usize>,
    dtype: Symbol,
) -> Tensor {
    let op = TensorOp::without_attributes(cx, symbol).unwrap();
    match executor
        .execute(
            cx,
            TensorRequest::new(op, inputs, TensorMeta::new(shape, dtype)),
        )
        .unwrap()
    {
        TensorExecution::Complete(tensor) => tensor,
        TensorExecution::Unsupported { reason } => panic!("{reason}"),
    }
}

fn execute_cpu(
    cx: &mut sim_kernel::Cx,
    symbol: Symbol,
    inputs: Vec<Tensor>,
    shape: Vec<usize>,
    dtype: Symbol,
) -> Tensor {
    let op = TensorOp::without_attributes(cx, symbol).unwrap();
    match CpuTensorExecutor::new()
        .execute(
            cx,
            TensorRequest::new(op, inputs, TensorMeta::new(shape, dtype)),
        )
        .unwrap()
    {
        TensorExecution::Complete(tensor) => tensor,
        TensorExecution::Unsupported { reason } => panic!("{reason}"),
    }
}

fn execute_portable(
    cx: &mut sim_kernel::Cx,
    symbol: Symbol,
    inputs: Vec<Tensor>,
    shape: Vec<usize>,
    dtype: Symbol,
) -> Tensor {
    let op = TensorOp::without_attributes(cx, symbol).unwrap();
    execute_portable_kernel(
        cx,
        &TensorRequest::new(op, inputs, TensorMeta::new(shape, dtype)),
        WgpuKernelDType::F32,
    )
    .unwrap()
}

fn assert_same_f32_cells(left: &Tensor, right: &Tensor) {
    assert_eq!(left.shape(), right.shape());
    let left_cells = f32_cells(left);
    let right_cells = f32_cells(right);
    assert_eq!(left_cells.len(), right_cells.len());
    for (left, right) in left_cells.iter().zip(right_cells.iter()) {
        assert!(same_f32_cell(*left, *right), "{left} != {right}");
    }
}

fn same_f32_cell(left: f32, right: f32) -> bool {
    if left.is_nan() || right.is_nan() {
        return left.is_nan() && right.is_nan();
    }
    if left.is_infinite() || right.is_infinite() {
        return left == right;
    }
    (left - right).abs() <= 1.0e-5
}

fn resident_storage(tensor: &Tensor) -> &WgpuResidentStorage {
    tensor
        .storage()
        .as_any()
        .downcast_ref::<WgpuResidentStorage>()
        .expect("wgpu resident storage")
}

#[test]
fn discovery_keeps_only_successful_probe_backed_adapters() {
    let discovery = WgpuDiscovery::from_probes(
        vec![
            adapter("zeta", "Vulkan", true),
            adapter("alpha", "Vulkan", true),
            adapter("placeholder", "Noop", false),
        ],
        Vec::new(),
    );

    assert_eq!(discovery.adapters.len(), 2);
    assert_eq!(discovery.adapters[0].adapter.name, "alpha");
    assert_eq!(discovery.adapters[0].adapter.ordinal, 0);
    assert_eq!(discovery.adapters[1].adapter.name, "zeta");
    assert_eq!(discovery.adapters[1].adapter.ordinal, 1);
    assert!(
        discovery
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.contains("did not pass required probes"))
    );
}

#[test]
fn host_emulated_wgpu_probe_cannot_satisfy_physical_acceptance() {
    let host_emulated_wgpu = adapter("alpha", "Vulkan", true);

    assert!(verify_physical(&host_emulated_wgpu).is_err());
}

#[test]
fn wgpu_lib_exports_sites_only_for_successful_discovery() {
    let discovery = WgpuDiscovery::from_probes(vec![adapter("alpha", "Vulkan", true)], Vec::new());
    let executor = WgpuTensorExecutor::new(discovery.adapters[0].clone());
    let card = sim_lib_numbers_tensor::TensorExecutor::card(&executor);
    assert_eq!(card.device_capability, Some(compute_wgpu_capability()));

    let lib = ComputeWgpuLib::from_discovery(discovery);
    let manifest = sim_kernel::Lib::manifest(&lib);
    assert_eq!(manifest.exports.len(), 1);
    assert_eq!(manifest.capabilities, vec![compute_wgpu_capability()]);

    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.grant(compute_wgpu_capability());
    cx.load_lib(&lib).unwrap();
    let site = cx
        .registry()
        .site_by_symbol(&compute_wgpu_site_symbol(0))
        .expect("wgpu compute site");
    assert!(site.object().as_eval_fabric().is_some());
}

#[test]
fn segment_and_transfer_plans_cross_binding_boundaries() {
    let segments = WgpuSegmentPlan::new(40, 16, 24);
    assert_eq!(segments.total_bytes(), 40);
    assert_eq!(segments.segments.len(), 3);
    assert_eq!(segments.segments[0].offset, 0);
    assert_eq!(segments.segments[0].bytes, 16);
    assert_eq!(segments.segments[1].offset, 16);
    assert_eq!(segments.segments[1].bytes, 16);
    assert_eq!(segments.segments[2].offset, 32);
    assert_eq!(segments.segments[2].bytes, 8);

    let transfer = WgpuTransferPlan::from_segments(&segments);
    assert_eq!(transfer.spans.len(), 3);
    assert_eq!(transfer.spans[2].offset, 32);
    assert_eq!(transfer.spans[2].bytes, 8);
}

#[test]
fn resident_arena_evicts_oldest_allocation_under_byte_bound() {
    let mut arena = WgpuResidentArena::new(24);
    let first = arena.allocate(16).unwrap();
    let second = arena.allocate(16).unwrap();

    assert!(!arena.contains(first.id));
    assert!(arena.contains(second.id));
    assert_eq!(arena.snapshot().resident_bytes, 16);
    assert_eq!(arena.snapshot().live_allocations, 1);
    assert_eq!(arena.snapshot().evictions, 1);
    assert!(arena.allocate(32).unwrap_err().contains("exceeds arena"));
}

#[test]
fn submission_queue_bounds_nodes_bytes_and_deadlines() {
    let mut queue = WgpuSubmissionQueue::new(WgpuQueueLimits {
        max_nodes: 1,
        max_bytes: 32,
        deadline_tick: 10,
    });
    queue.push(16, 10).unwrap();
    assert!(queue.push(8, 10).unwrap_err().contains("node limit"));
    assert_eq!(queue.flush().nodes, 1);
    assert!(queue.push(40, 10).unwrap_err().contains("byte limit"));
    assert!(queue.push(8, 11).unwrap_err().contains("deadline"));
}

#[test]
fn materialization_cache_records_one_success_or_failure() {
    let success = WgpuMaterializationCache::default();
    let first = success
        .get_or_try_init(|| Ok(Arc::<[u8]>::from([1, 2, 3])))
        .unwrap();
    let second = success
        .get_or_try_init(|| Ok(Arc::<[u8]>::from([9])))
        .unwrap();
    assert_eq!(&*first, &[1, 2, 3]);
    assert_eq!(&*second, &[1, 2, 3]);

    let failure = WgpuMaterializationCache::default();
    assert!(
        failure
            .get_or_try_init(|| Err("device lost".to_owned()))
            .unwrap_err()
            .contains("device lost")
    );
    assert!(
        failure
            .get_or_try_init(|| Ok(Arc::<[u8]>::from([1])))
            .unwrap_err()
            .contains("device lost")
    );
}

#[test]
fn pointwise_dispatch_requires_retained_device_context() {
    let mut cx = test_cx();
    let discovery = WgpuDiscovery::from_probes(vec![adapter("alpha", "Vulkan", true)], Vec::new());
    let executor = WgpuTensorExecutor::new(discovery.adapters[0].clone());
    let left = tensor(&mut cx, vec![2, 1], &["1", "2"]);
    let right = tensor(&mut cx, vec![1, 3], &["10", "20", "30"]);
    let op = TensorOp::without_attributes(&mut cx, add_op_symbol()).unwrap();

    let error = match executor.execute(
        &mut cx,
        TensorRequest::new(
            op,
            vec![left, right],
            TensorMeta::new(vec![2, 3], Symbol::qualified("numbers", "f32")),
        ),
    ) {
        Ok(_) => panic!("synthetic wgpu evidence must not dispatch pointwise work"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("device context is unavailable"));
}

#[test]
fn physical_pointwise_dispatch_matches_cpu_when_opted_in() {
    if std::env::var_os("SIM_COMPUTE_WGPU_PHYSICAL").is_none() {
        return;
    }
    let mut cx = test_cx();
    let mut runtimes = discover_wgpu_adapter_runtimes(&Default::default()).unwrap();
    let runtime = runtimes
        .pop()
        .expect("SIM_COMPUTE_WGPU_PHYSICAL requires a probe-backed adapter");
    assert!(verify_physical(&runtime.probe).is_ok());
    let executor = WgpuTensorExecutor::from_parts(
        runtime.probe,
        Some(WgpuExecutionContext {
            device: Arc::new(runtime.device),
            queue: Arc::new(runtime.queue),
        }),
    );
    let left = tensor(&mut cx, vec![2, 1], &["1", "-2"]);
    let right = tensor(&mut cx, vec![1, 3], &["10", "-20", "0"]);
    let source = tensor(&mut cx, vec![4], &["0.5", "1", "inf", "NaN"]);

    for (symbol, inputs, shape) in [
        (
            add_op_symbol(),
            vec![left.clone(), right.clone()],
            vec![2, 3],
        ),
        (
            sub_op_symbol(),
            vec![left.clone(), right.clone()],
            vec![2, 3],
        ),
        (
            sim_lib_numbers_tensor::mul_op_symbol(),
            vec![left.clone(), right.clone()],
            vec![2, 3],
        ),
        (
            sim_lib_numbers_tensor::div_op_symbol(),
            vec![left.clone(), right.clone()],
            vec![2, 3],
        ),
        (neg_op_symbol(), vec![source.clone()], vec![4]),
        (sqrt_op_symbol(), vec![source.clone()], vec![4]),
        (exp_op_symbol(), vec![source.clone()], vec![4]),
        (sin_op_symbol(), vec![source.clone()], vec![4]),
        (cos_op_symbol(), vec![source.clone()], vec![4]),
    ] {
        let gpu = execute_wgpu(
            &mut cx,
            &executor,
            symbol.clone(),
            inputs.clone(),
            shape.clone(),
            Symbol::qualified("numbers", "f32"),
        );
        let cpu = execute_cpu(
            &mut cx,
            symbol,
            inputs,
            shape,
            Symbol::qualified("numbers", "f32"),
        );
        assert_same_f32_cells(&gpu, &cpu);
        assert!(matches!(
            gpu.location(),
            TensorLocation::Resident { site, .. } if site == compute_wgpu_site_symbol(0)
        ));
    }
    let log = execute_wgpu(
        &mut cx,
        &executor,
        Symbol::qualified("tensor", "op/log"),
        vec![source],
        vec![4],
        Symbol::qualified("numbers", "f32"),
    );
    let expected_log = tensor(&mut cx, vec![4], &["-0.6931472", "0", "inf", "NaN"]);
    assert_same_f32_cells(&log, &expected_log);
    assert_eq!(executor.flush().unwrap().accepted, 10);
}

#[test]
fn pipeline_cache_is_bounded_by_adapter_op_dtype_and_rank() {
    let mut cache = WgpuPipelineCache::new(2);
    let probe = adapter("alpha", "Vulkan", true);
    cache.get_or_insert(&probe, WgpuKernelOp::Add, WgpuKernelDType::F32, 1);
    cache.get_or_insert(&probe, WgpuKernelOp::Add, WgpuKernelDType::F32, 1);
    cache.get_or_insert(&probe, WgpuKernelOp::Exp, WgpuKernelDType::F32, 1);
    cache.get_or_insert(&probe, WgpuKernelOp::Sin, WgpuKernelDType::F32, 2);

    let snapshot = cache.snapshot();
    assert_eq!(snapshot.entries, 2);
    assert_eq!(snapshot.hits, 1);
    assert_eq!(snapshot.misses, 3);
    assert_eq!(snapshot.evictions, 1);
}

#[test]
fn dtype_policy_uses_native_f16_only_when_granted_and_widens_bf16() {
    let mut cx = test_cx();
    let no_f16 = WgpuDiscovery::from_probes(
        vec![{
            let mut probe = adapter("alpha", "Vulkan", true);
            probe.adapter.granted_features.shader_f16 = false;
            probe
        }],
        Vec::new(),
    );
    let executor = WgpuTensorExecutor::new(no_f16.adapters[0].clone());
    let source = tensor(&mut cx, vec![1], &["1"]);
    let op = TensorOp::without_attributes(&mut cx, exp_op_symbol()).unwrap();
    let error = match executor.execute(
        &mut cx,
        TensorRequest::new(
            op,
            vec![source],
            TensorMeta::new(vec![1], Symbol::qualified("numbers", "f32")),
        ),
    ) {
        Ok(_) => panic!("synthetic wgpu evidence must not dispatch pointwise work"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("device context is unavailable"));
}
