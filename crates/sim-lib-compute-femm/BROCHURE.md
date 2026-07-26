# Resident FEMM Compute

In one line: `sim-lib-compute-femm` gives SIM a provider-neutral resident sparse
linear solver that still has to pass f64 FEMM certificate checks.

## What it gives you

The crate implements a loadable FEMM `LinearSolver` over CSR matrices. It models
factor upload reuse by fingerprint, resident f32 Krylov work vectors, explicit
convergence synchronizations, and bounded f64 refinement.

## Why you will be glad

GPU-backed FEMM acceleration should not be allowed to bypass solve quality. This
crate makes the acceptance rule explicit: resident f32 work may propose the
solution, but CPU f64 residual recomputation and the existing FEMM
`SolveCertificate` path decide whether the solve is usable.

## Where it fits

Use it behind `sim-lib-femm-solve` when a compute provider can hold CSR data and
work vectors resident. Vendor crates can replace the modeled storage while
preserving the same solver and certification contract.
