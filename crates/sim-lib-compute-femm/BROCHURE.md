# Resident FEMM Compute

In one line: `sim-lib-compute-femm` gives SIM a provider-neutral resident sparse linear solver that still has to pass f64 FEMM certificate checks.

## What it gives you

The crate implements a loadable FEMM `LinearSolver` over CSR matrices. It routes
factor upload reuse, resident f32 Krylov work vectors, sparse matvecs, dot/norm
reductions, and vector updates through the selected compute provider, then keeps
the bounded CPU f64 refinement and certificate checks.

## Why you will be glad

GPU-backed FEMM acceleration should not be allowed to bypass solve quality. This
crate makes the acceptance rule explicit: resident f32 work may propose the
solution, but CPU f64 residual recomputation and the existing FEMM
`SolveCertificate` path decide whether the solve is usable.

## Where it fits

Use it behind `sim-lib-femm-solve` when a compute provider can hold CSR data and
work vectors resident. Vendor crates can provide the tensor executor while
preserving the same solver and certification contract.
