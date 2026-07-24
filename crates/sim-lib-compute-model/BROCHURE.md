# Modeled Compute

`sim-lib-compute-model` gives SIM a deterministic tensor execution site for
provider development. It models resident handles, bounded submissions, flush
evidence, counters, and injected OOM, device-loss, and readback failures while
leaving tensor semantics in `sim-numbers`.

