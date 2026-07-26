//! Argument parsing for the loadable compute command.

use crate::ComputeCliError;

const MAX_SELECTOR_BYTES: usize = 64;
const MAX_DEVICES: usize = 64;
const MAX_PROFILE_BYTES: usize = 64 * 1024;

/// Output encoding requested by the command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputMode {
    /// Stable human-readable text.
    Text,
    /// Stable machine-readable JSON.
    Json,
}

/// Profile store action.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileAction {
    /// List stored profile keys.
    List,
    /// Read and check one stored profile.
    Read,
    /// Save a bounded synthetic profile.
    Save,
}

/// Parsed compute CLI command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeCommand {
    /// Render command help.
    Help,
    /// List installed compute providers.
    Devices(Selection),
    /// Render bounded probe evidence for selected providers.
    Probe(Selection),
    /// Read, write, or list injected profile storage.
    Profile(ProfileRequest),
    /// Explain profile routing for one selected device identity.
    Explain(ProfileRequest),
    /// Render a checked recipe descriptor.
    Recipe(RecipeRequest),
}

/// Provider selection and common output options.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Selection {
    /// Optional provider selector.
    pub selector: Option<String>,
    /// Output encoding.
    pub output: OutputMode,
    /// Maximum provider rows to inspect.
    pub max_devices: usize,
}

/// Profile command options.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfileRequest {
    /// Requested action.
    pub action: ProfileAction,
    /// Optional profile key.
    pub key: Option<String>,
    /// Expected adapter identity.
    pub adapter: String,
    /// Expected driver identity.
    pub driver: String,
    /// Expected backend identity.
    pub backend: String,
    /// Logical tick for stale checks.
    pub now_tick: u64,
    /// Output encoding.
    pub output: OutputMode,
    /// Maximum accepted profile bytes.
    pub max_profile_bytes: usize,
}

/// Recipe render options.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecipeRequest {
    /// Recipe id.
    pub id: String,
    /// Output encoding.
    pub output: OutputMode,
}

/// Parses a `sim compute` payload argument list.
pub fn parse_compute_args(args: &[String]) -> Result<ComputeCommand, ComputeCliError> {
    let args = strip_verb(args);
    if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(ComputeCommand::Help);
    }
    match args[0].as_str() {
        "devices" => Ok(ComputeCommand::Devices(parse_selection(
            "devices",
            &args[1..],
        )?)),
        "probe" => Ok(ComputeCommand::Probe(parse_selection("probe", &args[1..])?)),
        "profile" => Ok(ComputeCommand::Profile(parse_profile(&args[1..])?)),
        "explain" => {
            let mut request = parse_profile(&args[1..])?;
            if request.action == ProfileAction::List {
                request.action = ProfileAction::Read;
            }
            Ok(ComputeCommand::Explain(request))
        }
        "recipe" => Ok(ComputeCommand::Recipe(parse_recipe(&args[1..])?)),
        other => Err(ComputeCliError::new(format!(
            "unknown compute verb: {other}"
        ))),
    }
}

fn strip_verb(args: &[String]) -> &[String] {
    if args.first().is_some_and(|arg| arg == "compute") {
        &args[1..]
    } else {
        args
    }
}

fn parse_selection(verb: &str, args: &[String]) -> Result<Selection, ComputeCliError> {
    let mut output = OutputMode::Text;
    let mut selector = None;
    let mut max_devices = 16;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => output = OutputMode::Json,
            "--max-devices" => {
                max_devices = bounded_usize(take_value(args, &mut i, "--max-devices")?)?;
                if max_devices == 0 || max_devices > MAX_DEVICES {
                    return Err(ComputeCliError::new("max-devices is outside policy"));
                }
            }
            value if value.starts_with('-') => {
                return Err(ComputeCliError::new(format!(
                    "unknown {verb} option: {value}"
                )));
            }
            value => {
                checked_selector(value)?;
                if selector.replace(value.to_owned()).is_some() {
                    return Err(ComputeCliError::new(format!(
                        "{verb} accepts at most one selector"
                    )));
                }
            }
        }
        i += 1;
    }
    Ok(Selection {
        selector,
        output,
        max_devices,
    })
}

fn parse_profile(args: &[String]) -> Result<ProfileRequest, ComputeCliError> {
    let mut request = ProfileRequest {
        action: ProfileAction::List,
        key: None,
        adapter: "modeled-adapter".to_owned(),
        driver: "modeled-driver".to_owned(),
        backend: "modeled".to_owned(),
        now_tick: 0,
        output: OutputMode::Text,
        max_profile_bytes: 8192,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "list" => request.action = ProfileAction::List,
            "read" => request.action = ProfileAction::Read,
            "save" => request.action = ProfileAction::Save,
            "--json" => request.output = OutputMode::Json,
            "--key" => request.key = Some(checked_value(take_value(args, &mut i, "--key")?)?),
            "--adapter" => request.adapter = checked_value(take_value(args, &mut i, "--adapter")?)?,
            "--driver" => request.driver = checked_value(take_value(args, &mut i, "--driver")?)?,
            "--backend" => request.backend = checked_value(take_value(args, &mut i, "--backend")?)?,
            "--now" => request.now_tick = bounded_u64(take_value(args, &mut i, "--now")?)?,
            "--max-profile-bytes" => {
                request.max_profile_bytes =
                    bounded_usize(take_value(args, &mut i, "--max-profile-bytes")?)?;
                if request.max_profile_bytes == 0 || request.max_profile_bytes > MAX_PROFILE_BYTES {
                    return Err(ComputeCliError::new("max-profile-bytes is outside policy"));
                }
            }
            value if value.starts_with('-') => {
                return Err(ComputeCliError::new(format!(
                    "unknown profile option: {value}"
                )));
            }
            value => {
                return Err(ComputeCliError::new(format!(
                    "unknown profile action: {value}"
                )));
            }
        }
        i += 1;
    }
    Ok(request)
}

fn parse_recipe(args: &[String]) -> Result<RecipeRequest, ComputeCliError> {
    let mut output = OutputMode::Text;
    let mut id = "inspect-compute-device".to_owned();
    for arg in args {
        match arg.as_str() {
            "--json" => output = OutputMode::Json,
            value if value.starts_with('-') => {
                return Err(ComputeCliError::new(format!(
                    "unknown recipe option: {value}"
                )));
            }
            value => id = checked_value(value.to_owned())?,
        }
    }
    if id != "inspect-compute-device" {
        return Err(ComputeCliError::new(format!(
            "unknown compute recipe: {id}"
        )));
    }
    Ok(RecipeRequest { id, output })
}

fn take_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, ComputeCliError> {
    *i += 1;
    args.get(*i)
        .filter(|value| !value.starts_with('-'))
        .cloned()
        .ok_or_else(|| ComputeCliError::new(format!("{flag} requires a value")))
}

fn checked_value(value: String) -> Result<String, ComputeCliError> {
    checked_selector(&value)?;
    Ok(value)
}

fn checked_selector(value: &str) -> Result<(), ComputeCliError> {
    if value.is_empty() || value.len() > MAX_SELECTOR_BYTES || value.contains("..") {
        return Err(ComputeCliError::new("selector is outside policy"));
    }
    if !value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '/' | '.'))
    {
        return Err(ComputeCliError::new(
            "selector contains unsupported characters",
        ));
    }
    Ok(())
}

fn bounded_usize(value: String) -> Result<usize, ComputeCliError> {
    value
        .parse::<usize>()
        .map_err(|_| ComputeCliError::new("numeric ceiling must be an unsigned integer"))
}

fn bounded_u64(value: String) -> Result<u64, ComputeCliError> {
    value
        .parse::<u64>()
        .map_err(|_| ComputeCliError::new("tick must be an unsigned integer"))
}
