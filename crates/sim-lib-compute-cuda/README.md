# sim-lib-compute-cuda

Runtime-loaded CUDA/cuBLAS tensor compute site for SIM.

This crate has no CUDA toolkit or link-time dependency. It validates the CUDA
driver and cuBLAS/cuBLASLt ABI through dynamic symbols, exports a site only when
the runtime is present, and accepts only dense matmul requests that the CUDA
provider can execute semantically.
