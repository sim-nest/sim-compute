# Modeled Compute

In one line: `sim-lib-compute-model` gives SIM a deterministic tensor execution site for proving provider behavior before real hardware is involved.

## What it gives you

`sim-lib-compute-model` models resident tensor handles, bounded submissions,
flush evidence, counters, and explicit failure cases such as out-of-memory,
device-loss, and readback errors. It does that without owning tensor semantics:
the actual tensor value, shape, dtype, and arithmetic rules remain in
`sim-numbers`.

The crate is useful for recipes, conformance, and provider development because a
host can exercise placement, residency, counters, and failure reporting with
stable offline evidence.

## Why you will be glad

Real accelerators are noisy dependencies for a language/runtime test suite. This
crate lets the project prove the control contract first: what gets submitted,
what remains resident, what must be materialized, and which errors are surfaced
instead of hidden.

## Where it fits

Use this crate when you need deterministic compute-site evidence or a reference
provider for `site/compute/auto`. Real GPU or remote-provider crates can then
match the same placement contract while keeping their hardware-specific behavior
outside the kernel.
