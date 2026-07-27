# sim-lib-compute-cuda

Runtime-loaded CUDA/cuBLAS tensor compute site for SIM.

This crate has no CUDA toolkit or link-time dependency. It validates the CUDA
driver, runtime, and cuBLAS/cuBLASLt ABI through dynamic symbols, exports a site
only while live runtime handles are retained, and executes dense `f32` matmul
through cuBLAS with device-resident results. Half-family requests fail closed
until a real cuBLASLt execution path is implemented.
