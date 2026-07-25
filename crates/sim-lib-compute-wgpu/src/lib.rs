#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Evidence-based `wgpu` tensor compute site discovery.
//!
//! This crate discovers physical adapters deterministically, requests a bounded
//! device profile, records granted limits and features, and exports SIM tensor
//! sites only for adapters that pass transfer, mapping, and allocation probes.

mod probe;
mod site;

pub use probe::{
    AllocationAttempt, ProbeEvidence, ProbePolicy, RequestedWgpuProfile, TransferEvidence,
    WgpuAdapterEvidence, WgpuAdapterProbe, WgpuCapabilityEvidence, WgpuDiscovery,
    WgpuDiscoveryError, WgpuLimitEvidence, discover_wgpu_adapters,
};
pub use site::{
    ComputeWgpuLib, WgpuTensorExecutor, compute_wgpu_capability, compute_wgpu_lib_symbol,
    compute_wgpu_site_symbol, wgpu_executor_symbol,
};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
