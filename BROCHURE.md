# SIM Compute

In one line: `sim-compute` gives SIM portable tensor execution sites that can prove placement, residency, hardware discovery, and failure behavior before kernels are enabled.

## What it gives you

`sim-compute` lets a tensor expression run at a modeled compute site without
changing the canonical tensor value or the kernel. It provides a deterministic
modeled provider, resident tensor handles, bounded submission evidence, an
automatic placement surface that falls back cleanly when a modeled profile is
not available, and a wgpu provider that records adapter limits, features,
transfer, mapping, and bounded-allocation evidence before exporting a hardware
site.

## Why you will be glad

Compute placement is easiest to trust when the control contract can be tested
without a real accelerator, while hardware discovery is easiest to trust when it
keeps raw evidence. This repository gives SIM stable offline fixtures for
residency, readback, flush evidence, counters, injected failures, and successful
wgpu adapter probes before GPU or remote-provider kernels arrive.

## Where it fits

Use `sim-compute` beside `sim-numbers`: numbers owns tensor semantics, compute
owns where the tensor work runs. The kernel carries the loadable site records
as data and does not learn a device enum.
