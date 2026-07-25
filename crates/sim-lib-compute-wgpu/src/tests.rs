use std::sync::Arc;

use sim_kernel::{DefaultFactory, EagerPolicy};

use crate::{
    AllocationAttempt, ComputeWgpuLib, ProbeEvidence, RequestedWgpuProfile, TransferEvidence,
    WgpuAdapterEvidence, WgpuAdapterProbe, WgpuCapabilityEvidence, WgpuDiscovery,
    WgpuLimitEvidence, WgpuMaterializationCache, WgpuQueueLimits, WgpuResidentArena,
    WgpuSegmentPlan, WgpuSubmissionQueue, WgpuTensorExecutor, WgpuTransferPlan,
    compute_wgpu_capability, compute_wgpu_site_symbol,
};

// conformance: wgpu discovery records evidence, exports only successful adapter sites, and plans bounded resident submissions.

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
    WgpuAdapterProbe {
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
            granted_limits: limits(1 << 24),
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
