use super::*;

#[test]
fn adapter_evidence_records_requested_granted_and_failed_allocation_ceiling() {
    let mut probe = adapter_with_limits("alpha", "Vulkan", true, limits(8192));
    probe.probe.allocation_attempts.push(AllocationAttempt {
        bytes: probe.adapter.granted_limits.max_buffer_size + 1,
        success: false,
    });

    assert_eq!(probe.adapter.requested.limits.max_buffer_size, 4096);
    assert_eq!(probe.adapter.granted_limits.max_buffer_size, 8192);
    assert!(
        probe
            .probe
            .allocation_attempts
            .iter()
            .any(|attempt| attempt.success && attempt.bytes <= 8192)
    );
    assert!(
        probe
            .probe
            .allocation_attempts
            .iter()
            .any(|attempt| !attempt.success && attempt.bytes > 8192)
    );
}

#[test]
fn physical_submission_evidence_defaults_to_queue_observed_zeroes() {
    let executor = WgpuTensorExecutor::new(adapter("alpha", "Vulkan", true));
    let evidence = executor.physical_evidence();

    assert_eq!(evidence.submissions, 0);
    assert_eq!(evidence.uploaded_bytes, 0);
    assert_eq!(evidence.full_readbacks, 0);
    assert_eq!(evidence.scalar_syncs, 0);
    assert!(evidence.segments_touched.is_empty());
}

#[test]
#[cfg(any())]
fn physical_chained_pointwise_records_only_terminal_materialization() {
    if std::env::var_os("SIM_COMPUTE_WGPU_PHYSICAL").is_none() {
        return;
    }
    let mut cx = test_cx();
    let mut runtimes = discover_wgpu_adapter_runtimes(&Default::default()).unwrap();
    let runtime = runtimes
        .pop()
        .expect("SIM_COMPUTE_WGPU_PHYSICAL requires a probe-backed adapter");
    assert!(verify_physical(&runtime.probe).is_ok());
    let mut probe = runtime.probe;
    probe.adapter.granted_limits.max_storage_buffer_binding_size = 8;
    let executor = WgpuTensorExecutor::from_parts(
        probe,
        Some(WgpuExecutionContext {
            device: Arc::new(runtime.device),
            queue: Arc::new(runtime.queue),
        }),
    );
    let dtype = Symbol::qualified("numbers", "f32");
    let left = tensor(&mut cx, vec![4], &["1", "2", "3", "4"]);
    let right = tensor(&mut cx, vec![4], &["10", "20", "30", "40"]);
    let first = execute_wgpu(
        &mut cx,
        &executor,
        add_op_symbol(),
        vec![left, right],
        vec![4],
        dtype.clone(),
    );
    let before = executor.physical_evidence();
    let second = execute_wgpu(
        &mut cx,
        &executor,
        neg_op_symbol(),
        vec![first],
        vec![4],
        dtype,
    );
    let chained = executor.physical_evidence();
    assert_eq!(chained.full_readbacks, before.full_readbacks);
    assert!(chained.submissions > before.submissions);
    assert!(
        chained
            .segments_touched
            .iter()
            .filter(|range| range.bytes == 8)
            .count()
            >= 4
    );

    assert_eq!(f32_cells(&second), vec![-11.0, -22.0, -33.0, -44.0]);
    let terminal = executor.physical_evidence();
    assert_eq!(terminal.full_readbacks, chained.full_readbacks + 1);
}
