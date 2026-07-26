# Auto Compute

In one line: `sim-lib-compute-auto` gives SIM a stable automatic tensor placement surface that chooses an available compute site without changing the caller's tensor program.

## What it gives you

`sim-lib-compute-auto` provides `site/compute/auto`, the automatic placement
entry point for tensor execution. A caller can supply bounded profiles through
any Table/Dir backend, let the library select the modeled provider only when the
evidence is fresh, compatible, conclusive, `physical-device`, and not
caller-renamed, and fall back to local CPU behavior for absent, stale,
incompatible, inconclusive, modeled, host-emulated, or renamed evidence.

The result is a durable handoff point for higher-level numeric code: tensor
semantics stay in `sim-numbers`, provider fixtures stay in compute libraries,
and the runtime sees one ordinary loadable site.

## Why you will be glad

Automatic placement keeps examples and applications from hard-coding a device
decision too early. Development, conformance, and demos can save synthetic
profiles without turning them into physical proof, while production hosts can add
real providers later through the same site contract. Every route records provider
choice, materialization bytes, and synchronization counts so placement remains
explainable.

## Where it fits

Use this crate when a workflow wants "the best available compute site" rather
than a specific modeled, CPU, or hardware-backed target. Pair it with
`sim-lib-compute-model` for deterministic provider evidence and with
`sim-numbers` for the tensor operations being placed.
