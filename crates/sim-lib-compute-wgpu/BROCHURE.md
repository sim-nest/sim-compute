# SIM Compute WGPU

In one line: `sim-lib-compute-wgpu` discovers portable GPU adapters and exports
only probe-backed compute sites.

Use it when SIM needs a hardware tensor-placement candidate with raw evidence:
adapter identity, requested and granted limits, transfer and map checks, bounded
allocation attempts, timestamp support, f16 support, and portable f32
element-wise arithmetic/transcendental kernel execution. The crate does not turn
an absent adapter into a placeholder site.

Pair it with `sim-lib-compute-model` for deterministic tests and with later
kernel phases for reductions and linear algebra.
