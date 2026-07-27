# sim-lib-compute-rocm

Runtime-loaded ROCm/rocBLAS tensor compute site for SIM.

This crate has no ROCm toolkit or link-time dependency. On Linux it validates
the HIP runtime, rocBLAS ABI, and observed AMD `gfx*` target through runtime
evidence, exports a site only when the runtime is present, and accepts only dense
`f32` matmul requests that the ROCm provider executes through rocBLAS with
device-resident results. Half-family requests fail closed until a real
rocBLASLt execution path is implemented.
