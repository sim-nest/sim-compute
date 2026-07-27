# SIM Compute

In one line: `sim-compute` gives SIM portable tensor execution sites that can prove placement, residency, hardware discovery, kernel coverage, and failure behavior.

## What it gives you

`sim-compute` lets a tensor expression run at a modeled compute site without
changing the canonical tensor value or the kernel. It provides a deterministic
modeled provider, segmented resident tensor handles, bounded submission
evidence, eviction-safe materialization, an automatic placement surface that
persists bounded synthetic profiles through Table/Dir backends and falls back
cleanly when evidence is absent, stale, incompatible, inconclusive, modeled,
host-emulated, or caller-renamed, and a wgpu provider that records adapter
limits, features, transfer, mapping, and bounded-allocation evidence before
exporting retained-device pointwise dispatch plus portable reduction,
transpose, dot, and matmul kernels, plus an optional
CUDA provider that runtime-loads the NVIDIA driver, runtime, and cuBLAS symbols
before executing dense `f32` matmul into device-resident storage, plus a ROCm
provider that runtime-loads HIP and rocBLAS, records observed AMD `gfx*` target
evidence, and executes dense `f32` matmul into device-resident storage. Both
vendor sites fail closed for half-family requests until their Lt execution
paths exist.

## Why you will be glad

Compute placement is easiest to trust when the control contract can be tested
without a real accelerator, while hardware discovery is easiest to trust when it
keeps raw evidence. This repository gives SIM stable offline fixtures for
modeled residency, readback, flush evidence, counters, injected failures,
successful wgpu adapter probes with retained pointwise dispatch, CUDA ABI
validation, ROCm HIP/rocBLAS/gfx validation, and CPU-matched portable matrix
primitives. The reusable wgpu arena, queue, segment, transfer, materialization,
and CUDA/ROCm resident-storage planning types keep provider kernels on the same
bounded contract.

## Where it fits

Use `sim-compute` beside `sim-numbers`: numbers owns tensor semantics, compute
owns where the tensor work runs. The kernel carries the loadable site records
as data and does not learn a device enum.
