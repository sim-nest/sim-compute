#![allow(unsafe_code)]
#![deny(missing_docs)]
//! Optional runtime-loaded CUDA/cuBLAS tensor compute provider.
//!
//! The CUDA provider has no link-time CUDA dependency. It validates driver,
//! runtime, cuBLAS, and cuBLASLt entry points at runtime and exports a tensor
//! site only while the live library handles remain retained. Dense `f32`
//! matmul executes through cuBLAS and retains the result in device storage.
//! Half-family requests fail closed until a cuBLASLt execution path exists.

mod loader;
mod runtime;
mod site;
mod storage;

pub use loader::{
    CudaAbiEvidence, CudaLibrarySet, CudaLoadError, CudaProbePort, CudaRuntimeLoader,
    CudaRuntimeProbe, CudaSymbolEvidence, DynamicCudaLoader, FakeCudaLoader, discover_cuda_runtime,
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
