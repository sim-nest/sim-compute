#![allow(unsafe_code)]
#![deny(missing_docs)]
//! Optional runtime-loaded ROCm/rocBLAS tensor compute provider.
//!
//! The ROCm provider has no link-time ROCm dependency. It validates HIP and
//! rocBLAS entry points plus an observed AMD `gfx*` target at runtime and
//! exports a tensor site only while the live library handles remain retained.
//! Dense `f32` matmul executes through rocBLAS and retains the result in device
//! storage. Half-family requests fail closed until a rocBLASLt execution path
//! exists.

mod loader;
mod runtime;
mod site;
mod storage;

pub use loader::{
    DynamicRocmLoader, FakeRocmLoader, RocmAbiEvidence, RocmLibrarySet, RocmLoadError,
    RocmRuntimeLoader, RocmRuntimeProbe, RocmSymbolEvidence, discover_rocm_runtime,
};
pub use site::{
    ComputeRocmLib, RocmTensorExecutor, compute_rocm_capability, compute_rocm_lib_symbol,
    compute_rocm_site_symbol, rocm_executor_symbol,
};
pub use storage::{RocmAllocation, RocmResidentStorage};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
