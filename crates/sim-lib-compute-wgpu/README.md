# sim-lib-compute-wgpu

`sim-lib-compute-wgpu` discovers `wgpu` adapters and registers SIM tensor sites
only for adapters that pass device, transfer, mapping, and bounded allocation
probes. Adapter names and device ids are diagnostic evidence, not stable SIM
identity.

The exported site currently reports hardware capabilities and flush evidence.
GPU tensor execution is intentionally declined until later roadmap phases add
portable kernels.

The crate also exposes reusable arena, segment, transfer, queue, and
materialization planning types so later kernels share the same bounded resident
submission contract.
