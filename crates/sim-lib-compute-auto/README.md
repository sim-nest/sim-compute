# sim-lib-compute-auto

Automatic compute-site selection for SIM tensors.

`AutoTensorExecutor` accepts bounded profile records captured through the current
synthetic harness or later physical-device producers. Profiles record adapter,
driver, backend, limits, transfer and allocation evidence, tile sizes, operation
sample distributions, thermal and power context, evidence kind, observed
identity, and provenance. They persist through a caller-supplied Table or Dir
value with bounded keys and record sizes; the auto library never opens a host
path.

Automatic placement stays conservative: absent, stale, incompatible, or
inconclusive evidence routes to the published CPU tensor executor from
`sim-numbers`. Fresh compatible evidence can select the modeled resident
provider only when `verify_physical` accepts a `physical-device` profile whose
claimed identity still matches the observed identity. Current synthetic and
host-emulated records route to CPU, and each execute/flush records provider
choice, materialization bytes, and synchronization counts in the routing ledger.
