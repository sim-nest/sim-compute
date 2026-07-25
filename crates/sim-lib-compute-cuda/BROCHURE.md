# CUDA Dense Tensor Provider

`sim-lib-compute-cuda` adds an optional NVIDIA dense-matmul specialization for
SIM tensor execution. It loads the driver and cuBLAS at runtime, records checked
ABI evidence, and leaves machines without CUDA on the ordinary CPU or portable
provider path.
