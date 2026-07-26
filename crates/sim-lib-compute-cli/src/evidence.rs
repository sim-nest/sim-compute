//! Structured compute command evidence.

use sim_kernel::{Cx, Symbol, Value};
use sim_lib_compute_auto::{
    AutoComputeRouter, AutoRouteDecision, BenchmarkBounds, ComputeDeviceIdentity,
    ComputeThermalPowerContext, ProfileStore, ProfileStorePolicy, measure_bounded_profile,
};
use sim_lib_compute_model::ModeledComputeProfile;

use crate::{
    ComputeCliError,
    args::{ProfileAction, ProfileRequest, Selection},
};

/// Provider row rendered by devices and probe commands.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderEvidence {
    /// Provider selector.
    pub selector: String,
    /// Exported site symbol.
    pub site: String,
    /// Whether the site is installed in the active registry.
    pub installed: bool,
    /// Capability required by hardware providers.
    pub capability: Option<String>,
    /// Bounded status evidence.
    pub status: String,
}

/// Profile command evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfileEvidence {
    /// Command status.
    pub status: String,
    /// Profile keys involved in the result.
    pub keys: Vec<String>,
    /// Routing decision when one profile was checked.
    pub decision: Option<String>,
}

/// Recipe evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecipeEvidence {
    /// Recipe id.
    pub id: String,
    /// Required capabilities.
    pub capabilities: Vec<String>,
    /// Checked route id.
    pub route: String,
}

pub(crate) fn provider_rows(cx: &Cx, selection: &Selection) -> Vec<ProviderEvidence> {
    let mut rows = fixed_provider_rows(cx);
    for ordinal in 0..selection.max_devices {
        let site = format!("site/compute/wgpu/{ordinal}");
        if cx
            .registry()
            .site_by_symbol(&Symbol::new(site.clone()))
            .is_some()
        {
            rows.push(ProviderEvidence {
                selector: "wgpu".to_owned(),
                site,
                installed: true,
                capability: Some("device.gpu.wgpu".to_owned()),
                status: "probe-backed".to_owned(),
            });
        }
    }
    if let Some(selector) = &selection.selector {
        rows.retain(|row| row.selector == *selector || row.site == *selector);
        if rows.is_empty() {
            rows.push(ProviderEvidence {
                selector: selector.clone(),
                site: selector.clone(),
                installed: false,
                capability: None,
                status: "unknown-selector".to_owned(),
            });
        }
    }
    rows.truncate(selection.max_devices);
    rows
}

pub(crate) fn profile_evidence(
    cx: &mut Cx,
    store: Option<&Value>,
    request: &ProfileRequest,
) -> Result<ProfileEvidence, ComputeCliError> {
    let Some(table) = store else {
        return Ok(ProfileEvidence {
            status: "no-profile-store".to_owned(),
            keys: Vec::new(),
            decision: None,
        });
    };
    let store = ProfileStore::new(
        table.clone(),
        ProfileStorePolicy {
            max_key_bytes: 96,
            max_profile_bytes: request.max_profile_bytes,
        },
    )
    .map_err(ComputeCliError::from_kernel)?;
    match request.action {
        ProfileAction::List => list_profiles(cx, &store),
        ProfileAction::Save => save_profile(cx, &store, request),
        ProfileAction::Read => read_profile(cx, &store, request),
    }
}

pub(crate) fn recipe_evidence() -> RecipeEvidence {
    RecipeEvidence {
        id: "inspect-compute-device".to_owned(),
        capabilities: vec!["compute.device".to_owned()],
        route: "route/inspect-compute-device".to_owned(),
    }
}

fn fixed_provider_rows(cx: &Cx) -> Vec<ProviderEvidence> {
    [
        ("model", "site/compute/model", None),
        ("auto", "site/compute/auto", None),
        ("cuda", "site/compute/cuda", Some("device.gpu.cuda")),
        ("rocm", "site/compute/rocm", Some("device.gpu.rocm")),
    ]
    .into_iter()
    .map(|(selector, site, capability)| {
        let installed = cx.registry().site_by_symbol(&Symbol::new(site)).is_some();
        ProviderEvidence {
            selector: selector.to_owned(),
            site: site.to_owned(),
            installed,
            capability: capability.map(str::to_owned),
            status: if installed {
                "installed".to_owned()
            } else if selector == "model" {
                "model/no-adapter".to_owned()
            } else {
                "no-adapter".to_owned()
            },
        }
    })
    .collect()
}

fn list_profiles(cx: &mut Cx, store: &ProfileStore) -> Result<ProfileEvidence, ComputeCliError> {
    let keys = store
        .keys(cx)
        .map_err(ComputeCliError::from_kernel)?
        .into_iter()
        .map(|key| key.to_string())
        .collect();
    Ok(ProfileEvidence {
        status: "profiles".to_owned(),
        keys,
        decision: None,
    })
}

fn save_profile(
    cx: &mut Cx,
    store: &ProfileStore,
    request: &ProfileRequest,
) -> Result<ProfileEvidence, ComputeCliError> {
    let key = profile_key(cx, store, request)?;
    let identity = identity(request);
    let measured = measure_bounded_profile(
        identity,
        ModeledComputeProfile::default(),
        ComputeThermalPowerContext {
            thermal: "bounded".to_owned(),
            power: "host".to_owned(),
        },
        "sim-lib-compute-cli",
        request.now_tick,
        BenchmarkBounds::default(),
    );
    store
        .save(cx, Symbol::new(key.clone()), &measured)
        .map_err(ComputeCliError::from_kernel)?;
    Ok(ProfileEvidence {
        status: "saved".to_owned(),
        keys: vec![key],
        decision: None,
    })
}

fn read_profile(
    cx: &mut Cx,
    store: &ProfileStore,
    request: &ProfileRequest,
) -> Result<ProfileEvidence, ComputeCliError> {
    let key = profile_key(cx, store, request)?;
    let loaded = store
        .load(cx, Symbol::new(key.clone()))
        .map_err(ComputeCliError::from_kernel)?;
    let Some(profile) = loaded else {
        return Ok(ProfileEvidence {
            status: "missing".to_owned(),
            keys: vec![key],
            decision: None,
        });
    };
    let (decision, _) =
        AutoComputeRouter::new(identity(request), request.now_tick).choose(Some(&profile));
    Ok(ProfileEvidence {
        status: "checked".to_owned(),
        keys: vec![key],
        decision: Some(decision_label(&decision).to_owned()),
    })
}

fn profile_key(
    cx: &mut Cx,
    store: &ProfileStore,
    request: &ProfileRequest,
) -> Result<String, ComputeCliError> {
    if let Some(key) = &request.key {
        return Ok(key.clone());
    }
    let keys = store.keys(cx).map_err(ComputeCliError::from_kernel)?;
    match keys.as_slice() {
        [key] => Ok(key.to_string()),
        [] => Ok(format!("compute-profile/{}", request.adapter)),
        _ => Err(ComputeCliError::new("ambiguous compute profiles")),
    }
}

fn identity(request: &ProfileRequest) -> ComputeDeviceIdentity {
    ComputeDeviceIdentity::new(
        request.adapter.clone(),
        request.driver.clone(),
        request.backend.clone(),
    )
}

fn decision_label(decision: &AutoRouteDecision) -> &'static str {
    match decision {
        AutoRouteDecision::Absent => "absent",
        AutoRouteDecision::Stale => "stale",
        AutoRouteDecision::Incompatible => "incompatible",
        AutoRouteDecision::Inconclusive => "inconclusive",
        AutoRouteDecision::NonPhysical => "non-physical",
        AutoRouteDecision::Device => "device",
    }
}
