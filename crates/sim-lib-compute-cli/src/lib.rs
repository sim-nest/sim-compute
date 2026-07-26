#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Loadable compute command surface.
//!
//! The crate exports `cli/main/compute` as a host-registered callable. It
//! inspects installed compute sites, profile storage supplied as a Table/Dir
//! value by the embedding host, and physical acceptance artifacts; it creates no
//! separate bootstrap.

mod acceptance;
mod args;
mod envelope;
mod evidence;
mod render;

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, Args, Callable, CapabilityName, Cx, Error, Export, Lib, LibManifest, LibTarget,
    Linker, LoadCx, Object, ObjectCompat, Result, Symbol, Value, Version,
};

pub use args::{
    AcceptanceAction, AcceptanceRequest, ComputeCommand, OutputMode, ProfileAction,
    parse_compute_args,
};
pub use render::help;

use crate::{
    acceptance::acceptance_evidence,
    envelope::envelope_args,
    evidence::{profile_evidence, provider_rows, recipe_evidence},
    render::{render_acceptance, render_profile, render_providers, render_recipe},
};

/// Capability required for device inspection and probe evidence.
pub fn compute_device_capability() -> CapabilityName {
    CapabilityName::new("compute.device")
}

/// Capability required for profile reads.
pub fn compute_profile_read_capability() -> CapabilityName {
    CapabilityName::new("compute.profile.read")
}

/// Capability required for profile writes.
pub fn compute_profile_write_capability() -> CapabilityName {
    CapabilityName::new("compute.profile.write")
}

/// Capability required for physical acceptance capture.
pub fn compute_acceptance_capability() -> CapabilityName {
    CapabilityName::new("compute.acceptance")
}

/// Runtime library symbol for the compute CLI.
pub fn compute_cli_lib_symbol() -> Symbol {
    Symbol::qualified("lib", "compute-cli")
}

/// Symbol exported by the compute command entrypoint.
pub fn compute_entrypoint_symbol() -> Symbol {
    Symbol::qualified("cli", "main/compute")
}

/// Error returned by compute command parsing and rendering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeCliError {
    message: String,
}

impl ComputeCliError {
    /// Builds an error from a user-facing message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn from_kernel(error: Error) -> Self {
        Self::new(error.to_string())
    }
}

impl std::fmt::Display for ComputeCliError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ComputeCliError {}

/// Host-registered library that exports the compute command entrypoint.
#[derive(Clone, Default)]
pub struct ComputeCliLib {
    profile_store: Option<Value>,
}

impl ComputeCliLib {
    /// Builds the library without profile storage.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds the library with caller-supplied Table/Dir profile storage.
    pub fn with_profile_store(profile_store: Value) -> Self {
        Self {
            profile_store: Some(profile_store),
        }
    }
}

impl Lib for ComputeCliLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: compute_cli_lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Function {
                symbol: compute_entrypoint_symbol(),
                function_id: None,
            }],
        }
    }

    fn load(&self, cx: &mut LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.function_value(
            compute_entrypoint_symbol(),
            cx.factory().opaque(Arc::new(ComputeEntrypoint {
                profile_store: self.profile_store.clone(),
            }))?,
        )?;
        Ok(())
    }
}

#[derive(Clone)]
struct ComputeEntrypoint {
    profile_store: Option<Value>,
}

impl Object for ComputeEntrypoint {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok("cli/main/compute".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ObjectCompat for ComputeEntrypoint {
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for ComputeEntrypoint {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let Some(envelope) = args.values().first() else {
            return Err(Error::Eval("missing compute envelope".to_owned()));
        };
        let args = envelope_args(cx, envelope)?;
        let command = parse_compute_args(&args).map_err(|err| Error::Eval(err.to_string()))?;
        let output = run_command(cx, self.profile_store.as_ref(), &command)
            .map_err(|err| Error::Eval(err.to_string()))?;
        print!("{output}");
        cx.factory().bool(true)
    }
}

/// Runs a parsed compute command and returns rendered output.
pub fn run_command(
    cx: &mut Cx,
    profile_store: Option<&Value>,
    command: &ComputeCommand,
) -> std::result::Result<String, ComputeCliError> {
    match command {
        ComputeCommand::Help => Ok(help().to_owned()),
        ComputeCommand::Devices(selection) | ComputeCommand::Probe(selection) => {
            cx.require(&compute_device_capability())
                .map_err(ComputeCliError::from_kernel)?;
            Ok(render_providers(command, &provider_rows(cx, selection)))
        }
        ComputeCommand::Profile(request) => {
            let capability = match request.action {
                ProfileAction::Save => compute_profile_write_capability(),
                ProfileAction::List | ProfileAction::Read => compute_profile_read_capability(),
            };
            cx.require(&capability)
                .map_err(ComputeCliError::from_kernel)?;
            let evidence = profile_evidence(cx, profile_store, request)?;
            Ok(render_profile(command, &evidence))
        }
        ComputeCommand::Explain(request) => {
            cx.require(&compute_device_capability())
                .map_err(ComputeCliError::from_kernel)?;
            cx.require(&compute_profile_read_capability())
                .map_err(ComputeCliError::from_kernel)?;
            let evidence = profile_evidence(cx, profile_store, request)?;
            Ok(render_profile(command, &evidence))
        }
        ComputeCommand::Recipe(_) => {
            cx.require(&compute_device_capability())
                .map_err(ComputeCliError::from_kernel)?;
            Ok(render_recipe(command, &recipe_evidence()))
        }
        ComputeCommand::Acceptance(request) => {
            if matches!(request.action, AcceptanceAction::Capture) {
                cx.require(&compute_acceptance_capability())
                    .map_err(ComputeCliError::from_kernel)?;
            }
            Ok(render_acceptance(&acceptance_evidence(request)?))
        }
    }
}

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
