use std::sync::Arc;

use sim_kernel::{DefaultFactory, EagerPolicy};

use crate::{
    AllocationAttempt, ComputeWgpuLib, ProbeEvidence, RequestedWgpuProfile, TransferEvidence,
    WgpuAdapterEvidence, WgpuAdapterProbe, WgpuCapabilityEvidence, WgpuDiscovery,
    WgpuLimitEvidence, WgpuTensorExecutor, compute_wgpu_capability, compute_wgpu_site_symbol,
};

// conformance: wgpu discovery records evidence and exports only successful adapter sites.

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
