# sim-lib-compute-wgpu

`sim-lib-compute-wgpu` discovers `wgpu` adapters and registers SIM tensor sites
only for adapters that pass device, transfer, mapping, and bounded allocation
probes. Adapter names and device ids are diagnostic evidence, not stable SIM
identity.

The exported site runs portable f32 element-wise arithmetic and
transcendental kernels (`sqrt`, `exp`, `sin`, and `cos`) over broadcast output
segments. Native f16 is selected only when shader f16 was granted by the adapter;
bf16 and unsupported half paths widen to f32. Validated pipelines are cached by
adapter, operation, dtype strategy, and rank.

The crate also exposes reusable arena, segment, transfer, queue, pipeline-cache,
and materialization planning types so later kernels share the same bounded
resident submission contract.
