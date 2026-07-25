# sim-compute

`sim-compute` provides loadable tensor execution sites outside the SIM kernel.
The first providers are a modeled resident executor, an automatic selector that
loads measured profiles through caller-supplied Table/Dir values and falls back
to the published `sim-numbers` CPU tensor executor when no fresh compatible
evidence is available, a probe-backed `wgpu` discovery library that exports
hardware sites only after successful evidence capture, and an optional
runtime-loaded CUDA/cuBLAS provider for dense matmul when NVIDIA libraries are
present, and a Linux-only ROCm/rocBLAS provider for dense matmul when compatible
AMD HIP runtime libraries and `gfx*` target evidence are present.

The repo owns provider behavior only. Tensor identity, operation validation, and
CPU semantics stay in `sim-lib-numbers-tensor`; Table and Dir contracts stay in
the existing SIM storage stack.
