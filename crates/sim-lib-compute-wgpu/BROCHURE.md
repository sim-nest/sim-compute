# SIM Compute WGPU

In one line: `sim-lib-compute-wgpu` discovers portable GPU adapters and exports
only probe-backed compute sites.

Use it when SIM needs a hardware tensor-placement candidate with raw evidence:
adapter identity, requested and granted limits, transfer and map checks, bounded
allocation attempts, timestamp support, and f16 support. The crate does not turn
an absent adapter into a placeholder site.

Pair it with `sim-lib-compute-model` for deterministic tests and with later
kernel phases for real GPU tensor kernels.
