# sim-compute

`sim-compute` provides loadable tensor execution sites outside the SIM kernel.
The first providers are a modeled resident executor and an automatic selector
that falls back to the published `sim-numbers` CPU tensor executor when no
compatible compute profile is available.

The repo owns provider behavior only. Tensor identity, operation validation, and
CPU semantics stay in `sim-lib-numbers-tensor`; Table and Dir contracts stay in
the existing SIM storage stack.

