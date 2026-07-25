#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Modeled resident tensor compute provider.
//!
//! The modeled provider is a deterministic `TensorExecutor` for tests,
//! recipes, and future hardware-provider conformance. It executes through the
//! published `sim-lib-numbers-tensor` CPU semantics, then wraps successful
//! results in resident storage owned by `site/compute/model`.

mod model;
mod site;
mod storage;

pub use model::{
    ModeledComputeFault, ModeledComputeProfile, ModeledComputeSnapshot, ModeledResidentSegment,
    ModeledTensorExecutor, modeled_executor_symbol,
};
pub use site::{ComputeModelLib, compute_model_lib_symbol, compute_model_site_symbol};
pub use storage::{ModeledResidentStorage, ResidentHandle};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
