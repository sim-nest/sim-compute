#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Evidence-based `wgpu` tensor compute site discovery.
//!
//! This crate discovers physical adapters deterministically, requests a bounded
//! device profile, records granted limits and features, and exports SIM tensor
//! sites only for adapters that pass transfer, mapping, and allocation probes.

mod arena;
mod dispatch;
mod kernel_elementwise;
mod kernel_linalg;
mod kernel_reductions;
mod kernel_support;
mod kernel_wgsl;
mod kernels;
mod materialization;
mod pipeline;
mod probe;
mod queue;
mod segments;
mod site;
mod storage;
mod transfer;

pub use arena::{WgpuAllocationId, WgpuArenaAllocation, WgpuArenaSnapshot, WgpuResidentArena};
pub use materialization::WgpuMaterializationCache;
pub use pipeline::{
    WgpuKernelDType, WgpuKernelOp, WgpuPipelineCache, WgpuPipelineCacheSnapshot, WgpuPipelineKey,
    WgpuPipelineRecord, WgpuTileProfile,
};
pub use probe::{
    AllocationAttempt, ProbeEvidence, ProbePolicy, RequestedWgpuProfile, TransferEvidence,
    WgpuAdapterEvidence, WgpuAdapterProbe, WgpuCapabilityEvidence, WgpuDiscovery,
    WgpuDiscoveryError, WgpuLimitEvidence, discover_wgpu_adapters,
};
pub use queue::{WgpuQueueLimits, WgpuQueueSnapshot, WgpuSubmissionQueue};
pub use segments::{WgpuResidentSegment, WgpuSegmentPlan};
pub use site::{
    ComputeWgpuLib, WgpuTensorExecutor, compute_wgpu_capability, compute_wgpu_lib_symbol,
    compute_wgpu_site_symbol, wgpu_executor_symbol,
};
pub use storage::WgpuResidentStorage;
pub use transfer::{WgpuTransferPlan, WgpuTransferSpan};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
