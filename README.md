# sim-compute

`sim-compute` provides loadable tensor execution sites outside the SIM kernel.
The first providers are a modeled resident executor, an automatic selector that
loads bounded synthetic profiles through caller-supplied Table/Dir values and
falls back to the published `sim-numbers` CPU tensor executor unless explicit
`physical-device` evidence is present, a probe-backed `wgpu` discovery library
whose retained device path dispatches pointwise tensor work on `wgpu`, an
optional runtime-loaded CUDA/cuBLAS provider for dense matmul when NVIDIA
libraries are present, and a Linux-only ROCm/rocBLAS provider for dense matmul
when compatible AMD HIP runtime libraries and `gfx*` target evidence are
present.

The repo owns provider behavior only. Tensor identity, operation validation, and
CPU semantics stay in `sim-lib-numbers-tensor`; Table and Dir contracts stay in
the existing SIM storage stack.
