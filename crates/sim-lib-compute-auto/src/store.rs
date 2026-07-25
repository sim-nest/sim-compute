//! Table-backed measured profile persistence.

use sim_kernel::{Cx, Error, Expr, NumberLiteral, Result, Symbol, Value};
use sim_lib_compute_model::ModeledComputeProfile;

use crate::{
    ComputeDeviceIdentity, ComputeProfileLimits, ComputeProfileProvenance, ComputeProfileSamples,
    ComputeThermalPowerContext, MeasuredComputeProfile,
};

const PROFILE_SCHEMA: &str = "sim.compute.auto.profile.v1";
const DEFAULT_MAX_KEY_BYTES: usize = 96;
const DEFAULT_MAX_PROFILE_BYTES: usize = 8192;

/// Policy bounding profile persistence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileStorePolicy {
    /// Maximum UTF-8 bytes accepted in a profile key.
    pub max_key_bytes: usize,
    /// Maximum estimated bytes accepted for one profile record.
    pub max_profile_bytes: usize,
}

impl Default for ProfileStorePolicy {
    fn default() -> Self {
        Self {
            max_key_bytes: DEFAULT_MAX_KEY_BYTES,
            max_profile_bytes: DEFAULT_MAX_PROFILE_BYTES,
        }
    }
}

/// Table-backed measured profile store.
pub struct ProfileStore {
    table: Value,
    policy: ProfileStorePolicy,
}

impl ProfileStore {
    /// Builds a store from a caller-supplied Table or Dir value.
    pub fn new(table: Value, policy: ProfileStorePolicy) -> Result<Self> {
        if table.object().as_table_impl().is_none() {
            return Err(Error::TypeMismatch {
                expected: "Table or Dir",
                found: "non-table",
            });
        }
        Ok(Self { table, policy })
    }

    /// Stores a profile with one bounded, atomic table update.
    pub fn save(&self, cx: &mut Cx, key: Symbol, profile: &MeasuredComputeProfile) -> Result<()> {
        self.validate_key(&key)?;
        if profile.estimated_profile_bytes() > self.policy.max_profile_bytes {
            return Err(Error::Eval(
                "compute profile exceeds byte policy".to_owned(),
            ));
        }
        let value = profile_to_value(cx, profile)?;
        self.table
            .object()
            .as_table_impl()
            .expect("profile store table")
            .set(cx, key, value)
    }

    /// Loads a checked profile from a bounded key.
    pub fn load(&self, cx: &mut Cx, key: Symbol) -> Result<Option<MeasuredComputeProfile>> {
        self.validate_key(&key)?;
        let table = self
            .table
            .object()
            .as_table_impl()
            .expect("profile store table");
        if !table.has(cx, key.clone())? {
            return Ok(None);
        }
        let value = table.get(cx, key)?;
        let profile = profile_from_value(cx, &value)?;
        if profile.estimated_profile_bytes() > self.policy.max_profile_bytes {
            return Err(Error::Eval(
                "stored compute profile exceeds byte policy".to_owned(),
            ));
        }
        Ok(Some(profile))
    }

    /// Lists bounded profile keys.
    pub fn keys(&self, cx: &mut Cx) -> Result<Vec<Symbol>> {
        let keys = self
            .table
            .object()
            .as_table_impl()
            .expect("profile store table")
            .keys(cx)?;
        for key in &keys {
            self.validate_key(key)?;
        }
        Ok(keys)
    }

    fn validate_key(&self, key: &Symbol) -> Result<()> {
        if key.to_string().len() > self.policy.max_key_bytes {
            return Err(Error::Eval(
                "compute profile key exceeds byte policy".to_owned(),
            ));
        }
        Ok(())
    }
}

fn profile_to_value(cx: &mut Cx, profile: &MeasuredComputeProfile) -> Result<Value> {
    let entries = vec![
        (sym("schema"), string(cx, PROFILE_SCHEMA)?),
        (sym("adapter"), string(cx, &profile.identity.adapter)?),
        (sym("driver"), string(cx, &profile.identity.driver)?),
        (sym("backend"), string(cx, &profile.identity.backend)?),
        (
            sym("max-resident-bytes"),
            number(cx, profile.limits.max_resident_bytes)?,
        ),
        (
            sym("max-storage-binding-bytes"),
            number(cx, profile.limits.max_storage_binding_bytes)?,
        ),
        (
            sym("max-queue-depth"),
            number(cx, profile.limits.max_queue_depth as u64)?,
        ),
        (
            sym("max-queue-bytes"),
            number(cx, profile.limits.max_queue_bytes)?,
        ),
        (
            sym("submission-deadline-ticks"),
            number(cx, profile.limits.submission_deadline_ticks)?,
        ),
        (
            sym("upload-bytes-per-tick"),
            number_list(cx, &profile.samples.upload_bytes_per_tick)?,
        ),
        (
            sym("download-bytes-per-tick"),
            number_list(cx, &profile.samples.download_bytes_per_tick)?,
        ),
        (
            sym("launch-ticks"),
            number_list(cx, &profile.samples.launch_ticks)?,
        ),
        (
            sym("element-elements-per-tick"),
            number_list(cx, &profile.samples.element_elements_per_tick)?,
        ),
        (
            sym("reduction-elements-per-tick"),
            number_list(cx, &profile.samples.reduction_elements_per_tick)?,
        ),
        (
            sym("matmul-ops-per-tick"),
            number_list(cx, &profile.samples.matmul_ops_per_tick)?,
        ),
        (sym("thermal"), string(cx, &profile.context.thermal)?),
        (sym("power"), string(cx, &profile.context.power)?),
        (sym("tile-bytes"), number(cx, profile.tile_bytes)?),
        (
            sym("allocation-bytes"),
            number_list(cx, &profile.allocation_bytes)?,
        ),
        (sym("producer"), string(cx, &profile.provenance.producer)?),
        (
            sym("measured-at-tick"),
            number(cx, profile.provenance.measured_at_tick)?,
        ),
        (
            sym("stale-after-ticks"),
            number(cx, profile.provenance.stale_after_ticks)?,
        ),
        (
            sym("modeled-provider"),
            string(cx, &profile.modeled.provider)?,
        ),
    ];
    cx.factory().table(entries)
}

fn profile_from_value(cx: &mut Cx, value: &Value) -> Result<MeasuredComputeProfile> {
    let entries = value
        .object()
        .as_table_impl()
        .ok_or(Error::TypeMismatch {
            expected: "profile table",
            found: "non-table",
        })?
        .entries(cx)?;
    let get = |name: &str| -> Result<Value> {
        entries
            .iter()
            .find(|(key, _)| *key == sym(name))
            .map(|(_, value)| value.clone())
            .ok_or_else(|| Error::Eval(format!("compute profile missing {name}")))
    };
    if string_value(cx, &get("schema")?)? != PROFILE_SCHEMA {
        return Err(Error::Eval("unsupported compute profile schema".to_owned()));
    }
    let max_queue_depth = number_value(cx, &get("max-queue-depth")?)?;
    Ok(MeasuredComputeProfile {
        identity: ComputeDeviceIdentity {
            adapter: string_value(cx, &get("adapter")?)?,
            driver: string_value(cx, &get("driver")?)?,
            backend: string_value(cx, &get("backend")?)?,
        },
        limits: ComputeProfileLimits {
            max_resident_bytes: number_value(cx, &get("max-resident-bytes")?)?,
            max_storage_binding_bytes: number_value(cx, &get("max-storage-binding-bytes")?)?,
            max_queue_depth: usize::try_from(max_queue_depth)
                .map_err(|_| Error::Eval("max-queue-depth exceeds usize".to_owned()))?,
            max_queue_bytes: number_value(cx, &get("max-queue-bytes")?)?,
            submission_deadline_ticks: number_value(cx, &get("submission-deadline-ticks")?)?,
        },
        samples: ComputeProfileSamples {
            upload_bytes_per_tick: number_vec(cx, &get("upload-bytes-per-tick")?)?,
            download_bytes_per_tick: number_vec(cx, &get("download-bytes-per-tick")?)?,
            launch_ticks: number_vec(cx, &get("launch-ticks")?)?,
            element_elements_per_tick: number_vec(cx, &get("element-elements-per-tick")?)?,
            reduction_elements_per_tick: number_vec(cx, &get("reduction-elements-per-tick")?)?,
            matmul_ops_per_tick: number_vec(cx, &get("matmul-ops-per-tick")?)?,
        },
        context: ComputeThermalPowerContext {
            thermal: string_value(cx, &get("thermal")?)?,
            power: string_value(cx, &get("power")?)?,
        },
        tile_bytes: number_value(cx, &get("tile-bytes")?)?,
        allocation_bytes: number_vec(cx, &get("allocation-bytes")?)?,
        provenance: ComputeProfileProvenance {
            producer: string_value(cx, &get("producer")?)?,
            measured_at_tick: number_value(cx, &get("measured-at-tick")?)?,
            stale_after_ticks: number_value(cx, &get("stale-after-ticks")?)?,
        },
        modeled: ModeledComputeProfile {
            provider: string_value(cx, &get("modeled-provider")?)?,
            max_queue_depth: usize::try_from(max_queue_depth)
                .map_err(|_| Error::Eval("max-queue-depth exceeds usize".to_owned()))?,
            max_queue_bytes: number_value(cx, &get("max-queue-bytes")?)?,
            max_resident_bytes: number_value(cx, &get("max-resident-bytes")?)?,
            segment_tile_bytes: number_value(cx, &get("tile-bytes")?)?,
            max_storage_binding_bytes: number_value(cx, &get("max-storage-binding-bytes")?)?,
            submission_deadline_ticks: number_value(cx, &get("submission-deadline-ticks")?)?,
            fault: None,
            auto_flush_batches: false,
        },
    })
}

fn number(cx: &mut Cx, value: u64) -> Result<Value> {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "u64"), value.to_string())
}

fn string(cx: &mut Cx, value: &str) -> Result<Value> {
    cx.factory().string(value.to_owned())
}

fn number_list(cx: &mut Cx, values: &[u64]) -> Result<Value> {
    let values = values
        .iter()
        .map(|value| number(cx, *value))
        .collect::<Result<Vec<_>>>()?;
    cx.factory().list(values)
}

fn string_value(cx: &mut Cx, value: &Value) -> Result<String> {
    match value.object().as_expr(cx)? {
        Expr::String(value) => Ok(value),
        _ => Err(Error::TypeMismatch {
            expected: "string",
            found: "non-string",
        }),
    }
}

fn number_value(cx: &mut Cx, value: &Value) -> Result<u64> {
    match value.object().as_expr(cx)? {
        Expr::Number(NumberLiteral { canonical, .. }) => canonical
            .parse::<u64>()
            .map_err(|_| Error::Eval("invalid u64 profile value".to_owned())),
        _ => Err(Error::TypeMismatch {
            expected: "number",
            found: "non-number",
        }),
    }
}

fn number_vec(cx: &mut Cx, value: &Value) -> Result<Vec<u64>> {
    match value.object().as_expr(cx)? {
        Expr::List(values) => values
            .iter()
            .map(|expr| match expr {
                Expr::Number(NumberLiteral { canonical, .. }) => canonical
                    .parse::<u64>()
                    .map_err(|_| Error::Eval("invalid u64 profile sample".to_owned())),
                _ => Err(Error::TypeMismatch {
                    expected: "number list",
                    found: "non-number list",
                }),
            })
            .collect(),
        _ => Err(Error::TypeMismatch {
            expected: "list",
            found: "non-list",
        }),
    }
}

fn sym(name: &str) -> Symbol {
    Symbol::qualified("compute-profile", name)
}
