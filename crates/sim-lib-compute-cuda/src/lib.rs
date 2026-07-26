#![allow(unsafe_code)]
#![deny(missing_docs)]
//! Optional runtime-loaded CUDA/cuBLAS tensor compute provider.
//!
//! The CUDA provider has no link-time CUDA dependency. It validates driver,
//! cuBLAS, and cuBLASLt entry points at runtime and exports a tensor site only
//! after that ABI evidence exists. The executor currently accepts dense matmul
//! for `f32` plus half-family dtypes backed by the validated cuBLASLt path.

mod loader;
mod site;
mod storage;

pub use loader::{
    CudaAbiEvidence, CudaLibrarySet, CudaLoadError, CudaRuntimeLoader, CudaRuntimeProbe,
    CudaSymbolEvidence, DynamicCudaLoader, FakeCudaLoader, discover_cuda_runtime,
};
pub use site::{
    ComputeCudaLib, CudaTensorExecutor, compute_cuda_capability, compute_cuda_lib_symbol,
    compute_cuda_site_symbol, cuda_executor_symbol,
};
pub use storage::{CudaAllocation, CudaResidentStorage};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
