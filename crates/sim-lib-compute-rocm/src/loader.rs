//! Runtime ROCm/rocBLAS symbol discovery.

use std::{
    ffi::c_void,
    fmt,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

use libloading::Library;

const HIP_NAMES: &[&str] = &["libamdhip64.so.6", "libamdhip64.so"];
const ROCBLAS_NAMES: &[&str] = &["librocblas.so.4", "librocblas.so.0", "librocblas.so"];
const ROCBLASLT_NAMES: &[&str] = &["librocblaslt.so.0", "librocblaslt.so"];

const HIP_SYMBOLS: &[&str] = &["hipInit", "hipRuntimeGetVersion", "hipGetDeviceCount"];
const ROCBLAS_SYMBOLS: &[&str] = &[
    "rocblas_create_handle",
    "rocblas_destroy_handle",
    "rocblas_sgemm",
    "rocblas_gemm_ex",
];
const ROCBLASLT_SYMBOLS: &[&str] = &["rocblaslt_create_handle", "rocblaslt_destroy_handle"];

/// One validated runtime symbol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RocmSymbolEvidence {
    /// Symbol name.
    pub name: String,
    /// Whether the dynamic library exported the symbol.
    pub present: bool,
}

/// Dynamic-library ABI evidence required by the ROCm provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RocmAbiEvidence {
    /// Loaded HIP runtime library path or platform name.
    pub hip_library: String,
    /// Loaded rocBLAS library path or platform name.
    pub rocblas_library: String,
    /// Loaded rocBLASLt library path or platform name, when present.
    pub rocblaslt_library: Option<String>,
    /// HIP runtime version when the runtime can report it.
    pub hip_runtime_version: Option<i32>,
    /// Observed AMD GPU ISA targets such as `gfx1103`.
    pub observed_gfx_targets: Vec<String>,
    /// Checked HIP runtime symbols.
    pub hip_symbols: Vec<RocmSymbolEvidence>,
    /// Checked rocBLAS symbols.
    pub rocblas_symbols: Vec<RocmSymbolEvidence>,
    /// Checked rocBLASLt symbols.
    pub rocblaslt_symbols: Vec<RocmSymbolEvidence>,
}

impl RocmAbiEvidence {
    /// Returns true when Linux, HIP, rocBLAS, and a concrete gfx target exist.
    pub fn is_complete(&self) -> bool {
        cfg!(target_os = "linux")
            && !self.observed_gfx_targets.is_empty()
            && self.hip_symbols.iter().all(|symbol| symbol.present)
            && self.rocblas_symbols.iter().all(|symbol| symbol.present)
    }

    /// Returns true when half-family matmul may use the validated rocBLASLt path.
    pub fn supports_half_matmul(&self) -> bool {
        self.rocblas_symbols
            .iter()
            .any(|symbol| symbol.name == "rocblas_gemm_ex" && symbol.present)
            && !self.rocblaslt_symbols.is_empty()
            && self.rocblaslt_symbols.iter().all(|symbol| symbol.present)
    }
}

/// Loaded ROCm runtime libraries kept alive for function-pointer validity.
pub struct RocmLibrarySet {
    evidence: RocmAbiEvidence,
    hip: Library,
    rocblas: Library,
    rocblaslt: Option<Library>,
}

impl RocmLibrarySet {
    fn new(
        evidence: RocmAbiEvidence,
        hip: Library,
        rocblas: Library,
        rocblaslt: Option<Library>,
    ) -> Self {
        Self {
            evidence,
            hip,
            rocblas,
            rocblaslt,
        }
    }

    /// Returns checked ABI evidence.
    pub fn evidence(&self) -> &RocmAbiEvidence {
        &self.evidence
    }

    /// Returns loaded library handles to keep symbols alive.
    pub fn handles(&self) -> (&Library, &Library, Option<&Library>) {
        (&self.hip, &self.rocblas, self.rocblaslt.as_ref())
    }
}

impl fmt::Debug for RocmLibrarySet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RocmLibrarySet")
            .field("evidence", &self.evidence)
            .finish_non_exhaustive()
    }
}

/// Result of ROCm runtime discovery.
#[derive(Clone, Debug)]
pub struct RocmRuntimeProbe {
    /// Validated loaded runtime, when discovery succeeded.
    pub runtime: Option<Arc<RocmLibrarySet>>,
    /// ABI evidence from the successful runtime or the best failed probe.
    pub evidence: Option<RocmAbiEvidence>,
    /// Diagnostics collected while searching dynamic libraries.
    pub diagnostics: Vec<String>,
}

impl RocmRuntimeProbe {
    /// Builds a successful probe from validated evidence without library
    /// handles. This is intended for deterministic fake-loader tests.
    pub fn fake_present(evidence: RocmAbiEvidence) -> Self {
        Self {
            runtime: None,
            evidence: Some(evidence),
            diagnostics: Vec::new(),
        }
    }

    /// Returns true when discovery validated a usable ROCm provider.
    pub fn is_available(&self) -> bool {
        self.evidence
            .as_ref()
            .is_some_and(RocmAbiEvidence::is_complete)
    }
}

/// ROCm dynamic-loading failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RocmLoadError {
    /// Human-readable failure message.
    pub message: String,
}

impl fmt::Display for RocmLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for RocmLoadError {}

/// Loader abstraction used by real and fake ROCm discovery.
pub trait DynamicRocmLoader {
    /// Performs ROCm runtime discovery.
    fn discover(&self) -> Result<RocmRuntimeProbe, RocmLoadError>;
}

/// Real dynamic loader using platform ROCm shared libraries.
#[derive(Clone, Debug, Default)]
pub struct RocmRuntimeLoader {
    search_dirs: Vec<PathBuf>,
}

impl RocmRuntimeLoader {
    /// Builds a loader that searches platform library paths.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a loader that first searches explicit directories.
    pub fn with_search_dirs(search_dirs: Vec<PathBuf>) -> Self {
        Self { search_dirs }
    }
}

impl DynamicRocmLoader for RocmRuntimeLoader {
    fn discover(&self) -> Result<RocmRuntimeProbe, RocmLoadError> {
        let mut diagnostics = Vec::new();
        if !cfg!(target_os = "linux") {
            return Err(RocmLoadError {
                message: "ROCm provider is supported only on Linux".to_owned(),
            });
        }
        let (hip_name, hip) = self.open_first(HIP_NAMES, &mut diagnostics)?;
        let (rocblas_name, rocblas) = self.open_first(ROCBLAS_NAMES, &mut diagnostics)?;
        let rocblaslt = self.open_first(ROCBLASLT_NAMES, &mut diagnostics).ok();

        let hip_symbols = symbol_evidence(&hip, HIP_SYMBOLS);
        let rocblas_symbols = symbol_evidence(&rocblas, ROCBLAS_SYMBOLS);
        let rocblaslt_symbols = rocblaslt
            .as_ref()
            .map(|(_, library)| symbol_evidence(library, ROCBLASLT_SYMBOLS))
            .unwrap_or_default();
        let hip_runtime_version = hip_runtime_version(&hip).ok();
        let observed_gfx_targets = observed_gfx_targets();
        let evidence = RocmAbiEvidence {
            hip_library: hip_name,
            rocblas_library: rocblas_name,
            rocblaslt_library: rocblaslt.as_ref().map(|(name, _)| name.clone()),
            hip_runtime_version,
            observed_gfx_targets,
            hip_symbols,
            rocblas_symbols,
            rocblaslt_symbols,
        };
        if !evidence.is_complete() {
            return Ok(RocmRuntimeProbe {
                runtime: None,
                evidence: Some(evidence),
                diagnostics,
            });
        }
        let runtime = Arc::new(RocmLibrarySet::new(
            evidence.clone(),
            hip,
            rocblas,
            rocblaslt.map(|(_, library)| library),
        ));
        Ok(RocmRuntimeProbe {
            runtime: Some(runtime),
            evidence: Some(evidence),
            diagnostics,
        })
    }
}

impl RocmRuntimeLoader {
    fn open_first(
        &self,
        names: &[&str],
        diagnostics: &mut Vec<String>,
    ) -> Result<(String, Library), RocmLoadError> {
        for name in candidate_paths(&self.search_dirs, names) {
            match open_library(&name) {
                Ok(library) => return Ok((name.display().to_string(), library)),
                Err(error) => diagnostics.push(format!("{}: {error}", name.display())),
            }
        }
        Err(RocmLoadError {
            message: format!("ROCm library was not found; tried {}", names.join(", ")),
        })
    }
}

/// Fake loader for deterministic tests.
#[derive(Clone, Debug)]
pub struct FakeRocmLoader {
    probe: Result<RocmRuntimeProbe, RocmLoadError>,
}

impl FakeRocmLoader {
    /// Builds a fake loader that returns validated ROCm evidence.
    pub fn available() -> Self {
        Self {
            probe: Ok(RocmRuntimeProbe::fake_present(complete_fake_evidence())),
        }
    }

    /// Builds a fake loader with incomplete core HIP/rocBLAS ABI evidence.
    pub fn incomplete() -> Self {
        let mut evidence = complete_fake_evidence();
        if let Some(symbol) = evidence
            .rocblas_symbols
            .iter_mut()
            .find(|symbol| symbol.name == "rocblas_sgemm")
        {
            symbol.present = false;
        }
        Self {
            probe: Ok(RocmRuntimeProbe {
                runtime: None,
                evidence: Some(evidence),
                diagnostics: vec!["missing rocblas_sgemm".to_owned()],
            }),
        }
    }

    /// Builds a fake loader with missing optional rocBLASLt evidence.
    pub fn without_rocblaslt() -> Self {
        let mut evidence = complete_fake_evidence();
        if let Some(symbol) = evidence
            .rocblaslt_symbols
            .iter_mut()
            .find(|symbol| symbol.name == "rocblaslt_create_handle")
        {
            symbol.present = false;
        }
        Self {
            probe: Ok(RocmRuntimeProbe {
                runtime: None,
                evidence: Some(evidence),
                diagnostics: vec!["missing rocblaslt_create_handle".to_owned()],
            }),
        }
    }

    /// Builds a fake loader that reports ROCm as absent.
    pub fn absent() -> Self {
        Self {
            probe: Err(RocmLoadError {
                message: "ROCm runtime absent".to_owned(),
            }),
        }
    }
}

impl DynamicRocmLoader for FakeRocmLoader {
    fn discover(&self) -> Result<RocmRuntimeProbe, RocmLoadError> {
        self.probe.clone()
    }
}

/// Discovers ROCm using the real platform dynamic loader.
pub fn discover_rocm_runtime() -> Result<RocmRuntimeProbe, RocmLoadError> {
    RocmRuntimeLoader::new().discover()
}

fn complete_fake_evidence() -> RocmAbiEvidence {
    RocmAbiEvidence {
        hip_library: "fake-libamdhip64".to_owned(),
        rocblas_library: "fake-librocblas".to_owned(),
        rocblaslt_library: Some("fake-librocblaslt".to_owned()),
        hip_runtime_version: Some(6_300_000),
        observed_gfx_targets: vec!["gfx1103".to_owned()],
        hip_symbols: HIP_SYMBOLS
            .iter()
            .map(|name| RocmSymbolEvidence {
                name: (*name).to_owned(),
                present: true,
            })
            .collect(),
        rocblas_symbols: ROCBLAS_SYMBOLS
            .iter()
            .map(|name| RocmSymbolEvidence {
                name: (*name).to_owned(),
                present: true,
            })
            .collect(),
        rocblaslt_symbols: ROCBLASLT_SYMBOLS
            .iter()
            .map(|name| RocmSymbolEvidence {
                name: (*name).to_owned(),
                present: true,
            })
            .collect(),
    }
}

fn candidate_paths(search_dirs: &[PathBuf], names: &[&str]) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for directory in search_dirs {
        for name in names {
            candidates.push(directory.join(name));
        }
    }
    candidates.extend(names.iter().map(PathBuf::from));
    candidates
}

fn symbol_evidence(library: &Library, names: &[&str]) -> Vec<RocmSymbolEvidence> {
    names
        .iter()
        .map(|name| RocmSymbolEvidence {
            name: (*name).to_owned(),
            present: symbol_present(library, name),
        })
        .collect()
}

fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    // SAFETY: Loading a ROCm shared library is the intended boundary of this
    // crate. The handle is stored in RocmLibrarySet for at least as long as any
    // validated symbol evidence derived from it is used.
    unsafe { Library::new(path) }
}

fn symbol_present(library: &Library, name: &str) -> bool {
    let mut bytes = name.as_bytes().to_vec();
    bytes.push(0);
    // SAFETY: The lookup only checks whether the library exports the named
    // symbol as an opaque address. The address is not called or dereferenced.
    unsafe { library.get::<*mut c_void>(&bytes).is_ok() }
}

fn hip_runtime_version(library: &Library) -> Result<i32, RocmLoadError> {
    type HipInit = unsafe extern "C" fn(u32) -> i32;
    type HipRuntimeGetVersion = unsafe extern "C" fn(*mut i32) -> i32;
    type HipGetDeviceCount = unsafe extern "C" fn(*mut i32) -> i32;
    // SAFETY: Symbols were loaded from the HIP runtime library by their
    // official C ABI names. The calls use documented signatures, pass initialized
    // pointers, and only accept status 0.
    unsafe {
        let hip_init = library
            .get::<HipInit>(b"hipInit\0")
            .map_err(|error| RocmLoadError {
                message: error.to_string(),
            })?;
        let get_version = library
            .get::<HipRuntimeGetVersion>(b"hipRuntimeGetVersion\0")
            .map_err(|error| RocmLoadError {
                message: error.to_string(),
            })?;
        let get_device_count = library
            .get::<HipGetDeviceCount>(b"hipGetDeviceCount\0")
            .map_err(|error| RocmLoadError {
                message: error.to_string(),
            })?;
        let init_status = hip_init(0);
        if init_status != 0 {
            return Err(RocmLoadError {
                message: format!("hipInit failed with status {init_status}"),
            });
        }
        let mut device_count = 0;
        let count_status = get_device_count(&mut device_count);
        if count_status != 0 || device_count <= 0 {
            return Err(RocmLoadError {
                message: format!("hipGetDeviceCount failed with status {count_status}"),
            });
        }
        let mut version = 0;
        let version_status = get_version(&mut version);
        if version_status != 0 {
            return Err(RocmLoadError {
                message: format!("hipRuntimeGetVersion failed with status {version_status}"),
            });
        }
        Ok(version)
    }
}

fn observed_gfx_targets() -> Vec<String> {
    let Ok(output) = Command::new("rocm_agent_enumerator").output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let Ok(stdout) = String::from_utf8(output.stdout) else {
        return Vec::new();
    };
    let mut targets = stdout
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("gfx") && *line != "gfx000")
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    targets.sort();
    targets.dedup();
    targets
}
