# ROCm Dense Tensor Provider

In one line: `sim-lib-compute-rocm` gives SIM an AMD-backed dense tensor site that is admitted only after HIP and rocBLAS prove the device contract.

## What it gives you

It discovers compatible ROCm devices at runtime, loads HIP and rocBLAS through
checked bindings, records `gfx*` target evidence, and exposes dense matrix
execution as a SIM compute site. The provider keeps unsupported machines on the
ordinary CPU or portable GPU path and reports why ROCm was unavailable instead
of silently changing tensor semantics.

## Why you will be glad

- Add AMD accelerator placement without changing the caller's tensor program.
- Carry device, target, and ABI evidence with every admitted provider.
- Keep failure behavior explicit when the ROCm stack is missing or mismatched.
- Test the same placement logic on machines that do not have AMD hardware.

## Where it fits

This is the AMD specialization in the `sim-compute` family. Use it below
`sim-lib-compute-auto` when automatic placement should consider ROCm, and beside
the model provider when tests need stable comparison between resident and CPU
execution paths.
