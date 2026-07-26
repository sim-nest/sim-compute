#![allow(unsafe_code)]
#![deny(missing_docs)]
//! Optional runtime-loaded ROCm/rocBLAS tensor compute provider.
//!
//! The ROCm provider has no link-time ROCm dependency. It validates HIP and
//! rocBLAS entry points plus an observed AMD `gfx*` target at runtime and
//! exports a tensor site only after that ABI evidence exists. The executor
//! accepts dense matmul for `f32` plus half-family dtypes when rocBLASLt evidence
//! is present.

mod loader;
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
