//! Runtime CUDA/cuBLAS symbol discovery.

use std::{
    ffi::c_void,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

use libloading::Library;

const DRIVER_NAMES: &[&str] = &["libcuda.so.1", "libcuda.so", "nvcuda.dll"];
const RUNTIME_NAMES: &[&str] = &[
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "libcudart.so",
    "cudart64_130.dll",
    "cudart64_12.dll",
];
const CUBLAS_NAMES: &[&str] = &["libcublas.so.13", "libcublas.so.12", "libcublas.so"];
const CUBLASLT_NAMES: &[&str] = &["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"];

const DRIVER_SYMBOLS: &[&str] = &["cuInit", "cuDriverGetVersion"];
const RUNTIME_SYMBOLS: &[&str] = &[
    "cudaMalloc",
    "cudaFree",
    "cudaMemcpy",
    "cudaDeviceSynchronize",
];
const CUBLAS_SYMBOLS: &[&str] = &[
    "cublasCreate_v2",
    "cublasDestroy_v2",
    "cublasSgemm_v2",
    "cublasGemmEx",
];
const CUBLASLT_SYMBOLS: &[&str] = &["cublasLtCreate", "cublasLtDestroy", "cublasLtMatmul"];

/// One validated runtime symbol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaSymbolEvidence {
    /// Symbol name.
    pub name: String,
    /// Whether the dynamic library exported the symbol.
    pub present: bool,
}

/// Dynamic-library ABI evidence required by the CUDA provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaAbiEvidence {
    /// Loaded CUDA driver library path or platform name.
    pub driver_library: String,
    /// Loaded CUDA runtime library path or platform name.
    pub runtime_library: String,
    /// Loaded cuBLAS library path or platform name.
    pub cublas_library: String,
    /// Loaded cuBLASLt library path or platform name.
    pub cublaslt_library: String,
    /// CUDA driver version when the runtime can report it.
    pub driver_version: Option<i32>,
    /// Checked driver symbols.
    pub driver_symbols: Vec<CudaSymbolEvidence>,
    /// Checked CUDA runtime symbols.
    pub runtime_symbols: Vec<CudaSymbolEvidence>,
    /// Checked cuBLAS symbols.
    pub cublas_symbols: Vec<CudaSymbolEvidence>,
    /// Checked cuBLASLt symbols.
    pub cublaslt_symbols: Vec<CudaSymbolEvidence>,
}

impl CudaAbiEvidence {
    /// Returns true when all required driver/cuBLAS/cuBLASLt symbols exist.
    pub fn is_complete(&self) -> bool {
        self.driver_symbols.iter().all(|symbol| symbol.present)
            && self.runtime_symbols.iter().all(|symbol| symbol.present)
            && self.cublas_symbols.iter().all(|symbol| symbol.present)
            && self.cublaslt_symbols.iter().all(|symbol| symbol.present)
    }

    /// Returns true when half-family matmul may use the validated cuBLASLt path.
    pub fn supports_half_matmul(&self) -> bool {
        self.cublas_symbols
            .iter()
            .any(|symbol| symbol.name == "cublasGemmEx" && symbol.present)
            && self
                .cublaslt_symbols
                .iter()
                .any(|symbol| symbol.name == "cublasLtMatmul" && symbol.present)
    }
}

/// Loaded CUDA runtime libraries kept alive for function-pointer validity.
pub struct CudaLibrarySet {
    evidence: CudaAbiEvidence,
    driver: Library,
    runtime: Library,
    cublas: Library,
    cublaslt: Library,
}

impl CudaLibrarySet {
    /// Joins capsule-loaded libraries to their validated ABI evidence.
    pub fn new(
        evidence: CudaAbiEvidence,
        driver: Library,
        runtime: Library,
        cublas: Library,
        cublaslt: Library,
    ) -> Self {
        Self {
            evidence,
            driver,
            runtime,
            cublas,
            cublaslt,
        }
    }

    /// Returns checked ABI evidence.
    pub fn evidence(&self) -> &CudaAbiEvidence {
        &self.evidence
    }

    /// Returns loaded library handles to keep symbols alive.
    pub fn handles(&self) -> (&Library, &Library, &Library) {
        (&self.driver, &self.cublas, &self.cublaslt)
    }

    pub(crate) fn execution_handles(&self) -> (&Library, &Library) {
        (&self.runtime, &self.cublas)
    }
}

impl fmt::Debug for CudaLibrarySet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaLibrarySet")
            .field("evidence", &self.evidence)
            .finish_non_exhaustive()
    }
}

/// Result of CUDA runtime discovery.
#[derive(Clone, Debug)]
pub struct CudaRuntimeProbe {
    /// Validated loaded runtime, when discovery succeeded.
    pub runtime: Option<Arc<CudaLibrarySet>>,
    /// ABI evidence from the successful runtime or the best failed probe.
    pub evidence: Option<CudaAbiEvidence>,
    /// Diagnostics collected while searching dynamic libraries.
    pub diagnostics: Vec<String>,
}

impl CudaRuntimeProbe {
    /// Builds a successful probe from validated evidence without library
    /// handles. This is intended for deterministic fake-loader tests.
    pub fn fake_present(evidence: CudaAbiEvidence) -> Self {
        Self {
            runtime: None,
            evidence: Some(evidence),
            diagnostics: Vec::new(),
        }
    }

    /// Returns true when discovery validated a usable CUDA provider.
    pub fn is_available(&self) -> bool {
        self.evidence
            .as_ref()
            .is_some_and(CudaAbiEvidence::is_complete)
    }
}

/// CUDA dynamic-loading failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaLoadError {
    /// Human-readable failure message.
    pub message: String,
}

impl fmt::Display for CudaLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CudaLoadError {}

/// Loader abstraction used by real and fake CUDA discovery.
pub trait DynamicCudaLoader {
    /// Performs CUDA runtime discovery.
    fn discover(&self) -> Result<CudaRuntimeProbe, CudaLoadError>;
}

/// Capsule membrane used by the provider to receive an explicit probe.
pub trait CudaProbePort {
    /// Returns a capsule-owned CUDA probe without ambient rediscovery.
    fn probe_cuda(&self) -> Result<CudaRuntimeProbe, CudaLoadError>;
}

impl<T: DynamicCudaLoader + ?Sized> CudaProbePort for T {
    fn probe_cuda(&self) -> Result<CudaRuntimeProbe, CudaLoadError> {
        self.discover()
    }
}

/// Real dynamic loader using platform CUDA shared libraries.
#[derive(Clone, Debug)]
pub struct CudaRuntimeLoader {
    search_dirs: Vec<PathBuf>,
    search_system: bool,
}

impl Default for CudaRuntimeLoader {
    fn default() -> Self {
        Self {
            search_dirs: Vec::new(),
            search_system: true,
        }
    }
}

impl CudaRuntimeLoader {
    /// Builds a loader that searches platform library paths.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a loader that first searches explicit directories.
    pub fn with_search_dirs(search_dirs: Vec<PathBuf>) -> Self {
        Self {
            search_dirs,
            search_system: true,
        }
    }

    /// Builds a loader restricted to explicit directories. This is used to
    /// prove fail-closed behavior when vendor libraries are unavailable.
    pub fn with_search_dirs_only(search_dirs: Vec<PathBuf>) -> Self {
        Self {
            search_dirs,
            search_system: false,
        }
    }
}

impl DynamicCudaLoader for CudaRuntimeLoader {
    fn discover(&self) -> Result<CudaRuntimeProbe, CudaLoadError> {
        let mut diagnostics = Vec::new();
        let (driver_name, driver) = self.open_first(DRIVER_NAMES, &mut diagnostics)?;
        let (runtime_name, runtime) = self.open_first(RUNTIME_NAMES, &mut diagnostics)?;
        let (cublas_name, cublas) = self.open_first(CUBLAS_NAMES, &mut diagnostics)?;
        let (cublaslt_name, cublaslt) = self.open_first(CUBLASLT_NAMES, &mut diagnostics)?;

        let driver_symbols = symbol_evidence(&driver, DRIVER_SYMBOLS);
        let runtime_symbols = symbol_evidence(&runtime, RUNTIME_SYMBOLS);
        let cublas_symbols = symbol_evidence(&cublas, CUBLAS_SYMBOLS);
        let cublaslt_symbols = symbol_evidence(&cublaslt, CUBLASLT_SYMBOLS);
        let driver_version = driver_version(&driver).ok();
        let evidence = CudaAbiEvidence {
            driver_library: driver_name,
            runtime_library: runtime_name,
            cublas_library: cublas_name,
            cublaslt_library: cublaslt_name,
            driver_version,
            driver_symbols,
            runtime_symbols,
            cublas_symbols,
            cublaslt_symbols,
        };
        if !evidence.is_complete() {
            return Ok(CudaRuntimeProbe {
                runtime: None,
                evidence: Some(evidence),
                diagnostics,
            });
        }
        let runtime = Arc::new(CudaLibrarySet::new(
            evidence.clone(),
            driver,
            runtime,
            cublas,
            cublaslt,
        ));
        Ok(CudaRuntimeProbe {
            runtime: Some(runtime),
            evidence: Some(evidence),
            diagnostics,
        })
    }
}

impl CudaRuntimeLoader {
    fn open_first(
        &self,
        names: &[&str],
        diagnostics: &mut Vec<String>,
    ) -> Result<(String, Library), CudaLoadError> {
        for name in candidate_paths(&self.search_dirs, names, self.search_system) {
            match open_library(&name) {
                Ok(library) => return Ok((name.display().to_string(), library)),
                Err(error) => diagnostics.push(format!("{}: {error}", name.display())),
            }
        }
        Err(CudaLoadError {
            message: format!("CUDA library was not found; tried {}", names.join(", ")),
        })
    }
}

/// Fake loader for deterministic tests.
#[derive(Clone, Debug)]
pub struct FakeCudaLoader {
    probe: Result<CudaRuntimeProbe, CudaLoadError>,
}

impl FakeCudaLoader {
    /// Builds a fake loader that returns validated CUDA evidence.
    pub fn available() -> Self {
        Self {
            probe: Ok(CudaRuntimeProbe::fake_present(complete_fake_evidence())),
        }
    }

    /// Builds a fake loader with incomplete ABI evidence.
    pub fn incomplete() -> Self {
        let mut evidence = complete_fake_evidence();
        if let Some(symbol) = evidence
            .cublaslt_symbols
            .iter_mut()
            .find(|symbol| symbol.name == "cublasLtMatmul")
        {
            symbol.present = false;
        }
        Self {
            probe: Ok(CudaRuntimeProbe {
                runtime: None,
                evidence: Some(evidence),
                diagnostics: vec!["missing cublasLtMatmul".to_owned()],
            }),
        }
    }

    /// Builds a fake loader that reports CUDA as absent.
    pub fn absent() -> Self {
        Self {
            probe: Err(CudaLoadError {
                message: "CUDA runtime absent".to_owned(),
            }),
        }
    }
}

impl DynamicCudaLoader for FakeCudaLoader {
    fn discover(&self) -> Result<CudaRuntimeProbe, CudaLoadError> {
        self.probe.clone()
    }
}

/// Discovers CUDA using the real platform dynamic loader.
pub fn discover_cuda_runtime() -> Result<CudaRuntimeProbe, CudaLoadError> {
    CudaRuntimeLoader::new().discover()
}

fn complete_fake_evidence() -> CudaAbiEvidence {
    CudaAbiEvidence {
        driver_library: "fake-libcuda".to_owned(),
        runtime_library: "fake-libcudart".to_owned(),
        cublas_library: "fake-libcublas".to_owned(),
        cublaslt_library: "fake-libcublasLt".to_owned(),
        driver_version: Some(12_000),
        driver_symbols: DRIVER_SYMBOLS
            .iter()
            .map(|name| CudaSymbolEvidence {
                name: (*name).to_owned(),
                present: true,
            })
            .collect(),
        runtime_symbols: RUNTIME_SYMBOLS
            .iter()
            .map(|name| CudaSymbolEvidence {
                name: (*name).to_owned(),
                present: true,
            })
            .collect(),
        cublas_symbols: CUBLAS_SYMBOLS
            .iter()
            .map(|name| CudaSymbolEvidence {
                name: (*name).to_owned(),
                present: true,
            })
            .collect(),
        cublaslt_symbols: CUBLASLT_SYMBOLS
            .iter()
            .map(|name| CudaSymbolEvidence {
                name: (*name).to_owned(),
                present: true,
            })
            .collect(),
    }
}

fn candidate_paths(search_dirs: &[PathBuf], names: &[&str], search_system: bool) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for directory in search_dirs {
        for name in names {
            candidates.push(directory.join(name));
        }
    }
    if search_system {
        candidates.extend(names.iter().map(PathBuf::from));
    }
    candidates
}

fn symbol_evidence(library: &Library, names: &[&str]) -> Vec<CudaSymbolEvidence> {
    names
        .iter()
        .map(|name| CudaSymbolEvidence {
            name: (*name).to_owned(),
            present: symbol_present(library, name),
        })
        .collect()
}

fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    // SAFETY: Loading a CUDA shared library is the intended boundary of this
    // crate. The handle is stored in CudaLibrarySet for at least as long as any
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

fn driver_version(library: &Library) -> Result<i32, CudaLoadError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDriverGetVersion = unsafe extern "C" fn(*mut i32) -> i32;
    // SAFETY: Symbols were loaded from the CUDA driver library by their official
    // C ABI names. The calls use the documented signatures for cuInit and
    // cuDriverGetVersion and pass initialized pointers.
    unsafe {
        let cu_init = library
            .get::<CuInit>(b"cuInit\0")
            .map_err(|error| CudaLoadError {
                message: error.to_string(),
            })?;
        let get_version = library
            .get::<CuDriverGetVersion>(b"cuDriverGetVersion\0")
            .map_err(|error| CudaLoadError {
                message: error.to_string(),
            })?;
        let init_status = cu_init(0);
        if init_status != 0 {
            return Err(CudaLoadError {
                message: format!("cuInit failed with status {init_status}"),
            });
        }
        let mut version = 0;
        let version_status = get_version(&mut version);
        if version_status != 0 {
            return Err(CudaLoadError {
                message: format!("cuDriverGetVersion failed with status {version_status}"),
            });
        }
        Ok(version)
    }
}
