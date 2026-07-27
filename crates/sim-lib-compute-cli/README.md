# sim-lib-compute-cli

`sim-lib-compute-cli` exports the loadable `cli/main/compute` command surface.
It inspects installed compute sites, renders bounded probe and profile evidence,
accepts profile storage only as a caller-supplied Table or Dir value, and
captures or verifies sanitized portable and vendor physical acceptance
artifacts. A capture names one registered target capability with `--target`;
the verifier binds NVIDIA RTX 5080 Laptop, NVIDIA RTX 5090, and AMD gfx1151
artifacts to their matching physical adapter identities.

The committed portable-v1 evidence set records verifier-clean wgpu captures
for all three targets against
`504255b534e69a37d7163ba06048bc23ebdb7cb4`. The vendor-v1 set records
runtime probe, dense f32 matmul differential, invalid-shape refusal, three-sample
crossover, explicit absent-runtime refusal, automatic CPU/Absent explanation,
and retained wgpu probe evidence against
`12cb08ae6572946d5c11d06635ad079eecf8533c`. These are bounded observations,
not universal performance claims:

- RTX 5080 Laptop: CUDA 13.2/cuBLAS, maximum error `0.000000477`,
  observed crossover `n=128`.
- RTX 5090: CUDA 13.2/cuBLAS, maximum error `0.000000477`, observed
  crossover `n=128`.
- AMD gfx1151: HIP/rocBLAS, maximum error `0.000000358`, observed crossover
  `n=128`.
