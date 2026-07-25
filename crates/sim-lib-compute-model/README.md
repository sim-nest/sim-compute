# sim-lib-compute-model

Modeled tensor compute provider for SIM.

The crate exports `site/compute/model`, an `EvalFabric` site backed by
`ModeledTensorExecutor`. Accepted tensor requests run through the canonical
`sim-lib-numbers-tensor` CPU semantics and return resident tensor storage owned
by the modeled site. Resident storage supports deterministic readback faults and
counters so recipes and tests can prove placement behavior without hardware.
The resident pool is segmented by tile and binding limits, bounded by queue and
arena budgets, and caches one synchronized materialization result.

The resident ODE adapter composes the published `sim-numbers` tensor ODE
pipeline with the modeled tensor site. It batches fixed RK stages through
bounded auto-flushed submissions, keeps adaptive candidates resident, and limits
adaptive host observation to scalar error decisions.
