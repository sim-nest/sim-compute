#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Provider-neutral resident CSR FEMM linear solver.
//!
//! This crate composes the published `sim-lib-femm-solve` linear-solver seam.
//! It keeps CSR data and Krylov work vectors in a modeled resident arena, uses
//! f32 CG or BiCGSTAB for device-like iteration, synchronizes only scalar
//! convergence evidence during the Krylov loop, and accepts a solve only after
//! recomputing the residual on the CPU in f64.
//!
//! The existing FEMM steady solve remains the certificate authority: this crate
//! exports a `femm/linear-solver` value, and `sim-lib-femm-solve` builds the
//! `SolveCertificate` after its own f64 residual acceptance.

mod kernels;
mod runtime;
mod solver;

pub use runtime::{ComputeFemmLib, compute_femm_lib_symbol};
pub use solver::{
    ResidentCsrConfig, ResidentCsrSnapshot, ResidentCsrSolver, ResidentKrylovMethod,
    resident_csr_method_symbol,
};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

// conformance: resident FEMM CSR solves require f64 certificate acceptance.
#[cfg(test)]
mod tests;
