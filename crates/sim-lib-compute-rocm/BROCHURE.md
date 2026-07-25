# ROCm Dense Tensor Provider

`sim-lib-compute-rocm` adds an optional AMD dense-matmul specialization for
SIM tensor execution. It loads HIP and rocBLAS at runtime, records checked ABI
and `gfx*` target evidence, and leaves machines without compatible ROCm on the
ordinary CPU or portable provider path.
