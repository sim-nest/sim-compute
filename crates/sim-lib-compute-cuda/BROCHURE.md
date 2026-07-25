# CUDA Dense Tensor Provider

In one line: `sim-lib-compute-cuda` gives SIM an NVIDIA-backed dense tensor site that is loaded only when the local CUDA stack proves it can run.

## What it gives you

It discovers CUDA at runtime, opens the driver and cuBLAS through checked ABI
bindings, records device and library evidence, and exports a dense matrix
provider for SIM tensor placement. The crate keeps every hardware claim tied to
probe results, so machines without a usable NVIDIA stack fall back to the CPU or
portable provider path without inventing a fake accelerator.

## Why you will be glad

- Run dense tensor work on CUDA when the local driver is real and compatible.
- Keep hardware evidence beside placement decisions for repeatable diagnosis.
- Preserve the same tensor program when the CUDA provider is absent.
- Avoid build-time CUDA requirements for users who only need portable execution.

## Where it fits

This is the NVIDIA specialization in the `sim-compute` family. Use it below
`sim-lib-compute-auto` when automatic placement should consider CUDA, and beside
`sim-lib-compute-model` when tests need deterministic proof of the non-CUDA
fallback path.
