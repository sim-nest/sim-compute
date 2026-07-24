# sim-lib-compute-model

Modeled tensor compute provider for SIM.

The crate exports `site/compute/model`, an `EvalFabric` site backed by
`ModeledTensorExecutor`. Accepted tensor requests run through the canonical
`sim-lib-numbers-tensor` CPU semantics and return resident tensor storage owned
by the modeled site. Resident storage supports deterministic readback faults and
counters so recipes and tests can prove placement behavior without hardware.

