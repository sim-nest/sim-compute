# SIM Compute WGPU

In one line: `sim-lib-compute-wgpu` discovers portable GPU adapters and exports probe-backed, currently host-emulated compute sites.

## What it gives you

It checks adapter identity, requested and granted limits, transfer and map
behavior, bounded allocation attempts, timestamp support, f16 support, portable
f32 element-wise arithmetic, transcendentals, fixed-tree reductions, transpose,
dot, and tiled matrix multiplication. A site appears only when those probes
succeed, so placement receives bounded adapter evidence instead of a vague
hardware promise. The current kernel execution and resident records remain on the
host and are marked host-emulated rather than physical-device proof.

## Why you will be glad

- Use one portable wgpu-shaped provider across desktop adapters and browser-shaped APIs.
- Keep tensor placement honest with recorded limits and capability probes.
- Exercise host-retained resident tensor materialization without vendor-specific setup.
- Fall back cleanly when a machine has no adapter that meets the contract.

## Where it fits

This is the portable GPU provider in `sim-compute`. Pair it with
`sim-lib-compute-model` for deterministic tests and place it under
`sim-lib-compute-auto` when callers should ask for compute without naming a
vendor backend.
