# sim-lib-compute-cli

In one line: `sim-lib-compute-cli` gives SIM a loadable `compute` command for inspecting modeled, automatic, and hardware providers without adding another product binary.

## What it gives you

The crate exports the `cli/main/compute` surface used by bootloaded SIM command
sessions. It reports installed compute sites, renders machine-readable provider
evidence, exposes modeled fallback behavior, and validates profile operations
through caller-supplied Table storage instead of raw host paths.

## Why you will be glad

Provider discovery is useful only when it is explainable. This command keeps
hardware, profile, and fallback decisions visible while preserving the same
capability checks as library callers. A headless install can ask what compute
would do and receive structured evidence without needing CUDA, ROCm, wgpu
hardware, or a separate diagnostic executable.

## Where it fits

Use it as the command surface over `sim-lib-compute-model` and
`sim-lib-compute-auto`, with optional provider libraries loaded by the runtime.
`sim-run` can route the verb through the standard bootloader while SDK and
automation clients consume the same text or machine output.
