# sim-lib-compute-femm

Resident CSR Krylov solver for FEMM.

The crate exports a loadable `femm/linear-solver` value backed by
`ResidentCsrSolver`. The solver uploads a validated CSR matrix once, keys reuse
by the published matrix fingerprint, keeps CSR and work vectors in a modeled
resident arena, and runs f32 CG or BiCGSTAB iterations in the CPU-backed modeled
path without per-vector readback.

Convergence synchronization is intentionally narrow: the solver synchronizes one
f32 residual scalar per iteration, then recomputes the accepted residual on CPU
in f64. If the f64 residual is still above the certificate tolerance, it applies
a bounded iterative correction using the resident factor. Nonconvergence,
Krylov breakdown, stale factor payloads, transpose requests, non-finite values,
and f64 certification failure are rejected.
