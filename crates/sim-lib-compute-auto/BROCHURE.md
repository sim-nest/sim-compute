# Auto Compute

In one line: `sim-lib-compute-auto` gives SIM a stable automatic tensor placement surface that chooses an available compute site without changing the caller's tensor program.

## What it gives you

`sim-lib-compute-auto` provides `site/compute/auto`, the automatic placement
entry point for tensor execution. A caller can ask for an automatic site and let
the library select the modeled provider when a compatible profile is installed,
falling back to local CPU behavior when no modeled or resident provider is
available.

The result is a durable handoff point for higher-level numeric code: tensor
semantics stay in `sim-numbers`, provider fixtures stay in compute libraries,
and the runtime sees one ordinary loadable site.

## Why you will be glad

Automatic placement keeps examples and applications from hard-coding a device
decision too early. Development, conformance, and demos can run on the modeled
provider while production hosts can add real providers later through the same
site contract.

## Where it fits

Use this crate when a workflow wants "the best available compute site" rather
than a specific modeled, CPU, or hardware-backed target. Pair it with
`sim-lib-compute-model` for deterministic provider evidence and with
`sim-numbers` for the tensor operations being placed.
