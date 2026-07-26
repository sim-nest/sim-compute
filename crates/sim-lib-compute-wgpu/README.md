# sim-lib-compute-wgpu

`sim-lib-compute-wgpu` discovers `wgpu` adapters and registers SIM tensor sites
only for adapters that pass device, transfer, mapping, and bounded allocation
probes. Adapter names and device ids are diagnostic evidence, not stable SIM
identity. Retained physical probes keep the selected `wgpu::Device` and
`wgpu::Queue` and dispatch pointwise tensor kernels on that device. Synthetic
evidence-only fixtures stay host-emulated and fail explicitly if asked to
dispatch pointwise work without a retained device context.

The exported site runs f32 add, subtract, multiply, divide, negation, `sqrt`,
`exp`, `log`, `sin`, and `cos` through real `wgpu` compute passes, then returns bounded
resident tensors. Fixed-tree `sum`/`min`/`max`/`norm` reductions, `transpose`,
`dot`, and tiled `matmul` remain on the portable resident path. Native f16 is
selected only when shader f16 was granted by the adapter; bf16 and unsupported
half paths widen to f32. Validated pipelines are cached by adapter, operation,
dtype strategy, rank, and the tile profile selected from granted limits.

The crate also exposes reusable arena, segment, transfer, queue, pipeline-cache,
and materialization planning types so later kernels share the same bounded
resident submission contract.
