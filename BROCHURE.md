# SIM Compute

In one line: `sim-compute` gives SIM portable tensor execution sites that can prove placement, residency, and failure behavior before hardware providers are enabled.

## What it gives you

`sim-compute` lets a tensor expression run at a modeled compute site without
changing the canonical tensor value or the kernel. It provides a deterministic
modeled provider, resident tensor handles, bounded submission evidence, and an
automatic placement surface that falls back cleanly when a modeled profile is
not available.

## Why you will be glad

Compute placement is easiest to trust when the control contract can be tested
without a real accelerator. This repository gives SIM stable offline fixtures
for residency, readback, flush evidence, counters, and injected failures before
GPU or remote-provider crates arrive.

## Where it fits

Use `sim-compute` beside `sim-numbers`: numbers owns tensor semantics, compute
owns where the tensor work runs. The kernel carries the loadable site records
as data and does not learn a device enum.
