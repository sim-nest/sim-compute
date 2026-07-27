use super::*;

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
fn empty_wgpu_discovery_loads_without_device_authority() {
    let lib = ComputeWgpuLib::from_discovery(WgpuDiscovery::from_probes(Vec::new(), Vec::new()));
    let manifest = sim_kernel::Lib::manifest(&lib);

    assert!(manifest.exports.is_empty());
    assert!(manifest.capabilities.is_empty());

    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&lib).unwrap();
}
