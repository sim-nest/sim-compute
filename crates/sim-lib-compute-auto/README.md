# sim-lib-compute-auto

Automatic compute-site selection for SIM tensors.

`AutoTensorExecutor` accepts measured profile records captured through a bounded
benchmark harness. Profiles record adapter, driver, backend, limits, transfer
and allocation evidence, tile sizes, operation sample distributions, thermal and
power context, and provenance. They persist through a caller-supplied Table or
Dir value with bounded keys and record sizes; the auto library never opens a
host path.

Automatic placement stays conservative: absent, stale, incompatible, or
inconclusive measured evidence routes to the published CPU tensor executor from
`sim-numbers`. Fresh compatible evidence can select the modeled resident
provider, and each execute/flush records provider choice, materialization bytes,
and synchronization counts in the routing ledger.
