use std::any::Any;
use std::sync::{Arc, Mutex};

use sim_kernel::{DefaultFactory, Factory, Object, Result as KernelResult, Symbol};
use sim_lib_compute_model::{ModeledComputeProfile, ModeledComputeSnapshot, ModeledTensorExecutor};
use sim_lib_femm_core::{CsrMatrix, FemmError, FemmResult, StableId};
use sim_lib_femm_solve::{
    DenseFallbackSolver, FactorHandle, FactorLifecycle, LinearFactor, LinearMethod, LinearSolver,
    TransposeSupport,
};
use sim_lib_numbers_tensor::TensorExecutor;

use crate::kernels::{ResidentCsrMatrix, add_assign_f64, f64_residual, solve_f32, validate_rhs};

const DEFAULT_TOL_F32: f32 = 1.0e-5;
const DEFAULT_CERT_TOL_F64: f64 = 1.0e-8;
const DEFAULT_MAX_ITERS: usize = 256;
const DEFAULT_MAX_REFINEMENTS: usize = 2;

/// Solver method symbol used in FEMM certificates.
pub fn resident_csr_method_symbol(method: ResidentKrylovMethod) -> Symbol {
    match method {
        ResidentKrylovMethod::Cg => Symbol::qualified("compute-femm", "resident-csr-cg"),
        ResidentKrylovMethod::Bicgstab => {
            Symbol::qualified("compute-femm", "resident-csr-bicgstab")
        }
    }
}

/// Resident Krylov method selected by the solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentKrylovMethod {
    /// Conjugate gradient for symmetric positive-definite systems.
    Cg,
    /// Stabilized BiCG for general nonsymmetric systems.
    Bicgstab,
}

/// Configuration for [`ResidentCsrSolver`].
#[derive(Clone, Debug, PartialEq)]
pub struct ResidentCsrConfig {
    /// Krylov method used for resident f32 iterations.
    pub method: ResidentKrylovMethod,
    /// f32 residual tolerance used inside the resident loop.
    pub tol_f32: f32,
    /// f64 residual tolerance required before accepting the FEMM solve.
    pub certificate_tol_f64: f64,
    /// Maximum resident Krylov iterations for each solve or correction.
    pub max_iters: usize,
    /// Maximum bounded f64 iterative-refinement corrections.
    pub max_refinements: usize,
    /// Number of iterations between scalar residual synchronizations.
    pub scalar_sync_cadence: usize,
    /// Modeled maximum bytes retained in the resident CSR arena.
    pub max_resident_bytes: u64,
}

impl Default for ResidentCsrConfig {
    fn default() -> Self {
        Self {
            method: ResidentKrylovMethod::Cg,
            tol_f32: DEFAULT_TOL_F32,
            certificate_tol_f64: DEFAULT_CERT_TOL_F64,
            max_iters: DEFAULT_MAX_ITERS,
            max_refinements: DEFAULT_MAX_REFINEMENTS,
            scalar_sync_cadence: 1,
            max_resident_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Snapshot of modeled resident sparse-solver evidence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResidentCsrSnapshot {
    /// Distinct CSR uploads accepted by the solver.
    pub uploads: usize,
    /// Uploaded bytes currently modeled as resident.
    pub resident_bytes: u64,
    /// Solver calls accepted.
    pub solves: usize,
    /// Resident SpMV dispatches performed by Krylov/refinement loops.
    pub spmv_dispatches: usize,
    /// Resident work-vector updates performed by Krylov/refinement loops.
    pub vector_dispatches: usize,
    /// Scalar residual synchronizations performed during Krylov loops.
    pub scalar_synchronizations: usize,
    /// CPU f64 residual recomputations performed for certification/refinement.
    pub f64_residual_checks: usize,
    /// Iterative-refinement corrections applied.
    pub corrections: usize,
    /// Factor handles reused by fingerprint.
    pub reused_factors: usize,
    /// Rejected solves or factors.
    pub refusals: usize,
    /// Provider submissions accepted by final flushes.
    pub provider_submissions: usize,
    /// Provider resident readbacks observed during Krylov work.
    pub provider_readbacks: usize,
}

#[derive(Default)]
pub(crate) struct ResidentCsrState {
    pub(crate) next_upload: usize,
    pub(crate) snapshot: ResidentCsrSnapshot,
}

/// Provider-neutral resident CSR FEMM solver.
#[derive(Clone)]
pub struct ResidentCsrSolver {
    pub(crate) config: ResidentCsrConfig,
    pub(crate) executor: Arc<dyn TensorExecutor>,
    modeled: Option<ModeledTensorExecutor>,
    pub(crate) state: Arc<Mutex<ResidentCsrState>>,
}

impl ResidentCsrSolver {
    /// Builds a resident CSR solver from a configuration over modeled compute.
    pub fn new(config: ResidentCsrConfig) -> Self {
        Self::modeled(config, ModeledComputeProfile::default())
    }

    /// Builds a resident CSR solver over an already selected tensor executor.
    pub fn with_executor(config: ResidentCsrConfig, executor: Arc<dyn TensorExecutor>) -> Self {
        Self {
            config,
            executor,
            modeled: None,
            state: Arc::new(Mutex::new(ResidentCsrState::default())),
        }
    }

    /// Builds a resident CSR solver over the modeled tensor provider.
    pub fn modeled(config: ResidentCsrConfig, mut profile: ModeledComputeProfile) -> Self {
        profile.auto_flush_batches = true;
        let executor = ModeledTensorExecutor::new(profile);
        Self {
            config,
            executor: Arc::new(executor.clone()) as Arc<dyn TensorExecutor>,
            modeled: Some(executor),
            state: Arc::new(Mutex::new(ResidentCsrState::default())),
        }
    }

    /// Returns the modeled provider snapshot when this solver owns one.
    pub fn modeled_snapshot(&self) -> Option<ModeledComputeSnapshot> {
        self.modeled.as_ref().map(ModeledTensorExecutor::snapshot)
    }

    /// Returns the current evidence snapshot.
    pub fn snapshot(&self) -> ResidentCsrSnapshot {
        self.state
            .lock()
            .expect("resident CSR state poisoned")
            .snapshot
            .clone()
    }

    /// Returns this solver's configuration.
    pub fn config(&self) -> &ResidentCsrConfig {
        &self.config
    }

    pub(crate) fn refuse<T>(&self, err: FemmError) -> FemmResult<T> {
        self.state
            .lock()
            .expect("resident CSR state poisoned")
            .snapshot
            .refusals += 1;
        Err(err)
    }

    fn factor_payload<'a>(&self, factor: &'a FactorHandle) -> FemmResult<&'a ResidentCsrFactor> {
        let payload = factor
            .payload()
            .object()
            .downcast_ref::<ResidentCsrFactor>()
            .ok_or_else(|| {
                FemmError::SolveDidNotConverge(
                    "resident CSR factor payload has the wrong provider type".to_owned(),
                )
            })?;
        if payload.fingerprint != factor.matrix_fingerprint() {
            return self.refuse(FemmError::SolveDidNotConverge(
                "resident CSR factor fingerprint is stale".to_owned(),
            ));
        }
        if payload.method != self.config.method {
            return self.refuse(FemmError::SolveDidNotConverge(
                "resident CSR factor method does not match solver".to_owned(),
            ));
        }
        Ok(payload)
    }

    fn solve_forward(&self, matrix: &ResidentCsrMatrix, rhs: &[f64]) -> FemmResult<Vec<f64>> {
        validate_rhs(matrix.rows, rhs)?;
        {
            let mut state = self.state.lock().expect("resident CSR state poisoned");
            state.snapshot.solves += 1;
        }
        let mut x = solve_f32(self, matrix, rhs)?;
        let mut refinements = 0;
        loop {
            let residual = f64_residual(matrix, &x, rhs)?;
            self.state
                .lock()
                .expect("resident CSR state poisoned")
                .snapshot
                .f64_residual_checks += 1;
            if residual.norm <= self.config.certificate_tol_f64 {
                return Ok(x);
            }
            if refinements >= self.config.max_refinements {
                break;
            }
            let correction = DenseFallbackSolver::dense_solve(
                &matrix.to_dense_f64()?,
                residual.values.as_slice(),
            )?;
            add_assign_f64(&mut x, &correction)?;
            refinements += 1;
            self.state
                .lock()
                .expect("resident CSR state poisoned")
                .snapshot
                .corrections += 1;
        }
        self.refuse(FemmError::SolveDidNotConverge(
            "resident CSR f64 certification failed".to_owned(),
        ))
    }
}

impl Default for ResidentCsrSolver {
    fn default() -> Self {
        Self::new(ResidentCsrConfig::default())
    }
}

impl LinearSolver for ResidentCsrSolver {
    fn factor(&self, k: &CsrMatrix) -> FemmResult<FactorHandle> {
        k.validate()?;
        let matrix = ResidentCsrMatrix::from_csr(self, k)?;
        let bytes = matrix.resident_bytes()?;
        if bytes > self.config.max_resident_bytes {
            return self.refuse(FemmError::BudgetExceeded(format!(
                "resident CSR upload requires {bytes} bytes"
            )));
        }
        let upload_id = {
            let mut state = self.state.lock().expect("resident CSR state poisoned");
            state.next_upload += 1;
            state.snapshot.uploads += 1;
            state.snapshot.resident_bytes = state.snapshot.resident_bytes.saturating_add(bytes);
            state.next_upload
        };
        let payload = DefaultFactory
            .opaque(Arc::new(ResidentCsrFactor {
                upload_id,
                fingerprint: k.fingerprint(),
                method: self.config.method,
                matrix,
            }))
            .map_err(|err| FemmError::SolveDidNotConverge(err.to_string()))?;
        Ok(FactorHandle::new(LinearFactor {
            method: LinearMethod::Provider(resident_csr_method_symbol(self.config.method)),
            matrix_fingerprint: k.fingerprint(),
            transpose: TransposeSupport::ForwardOnly,
            lifecycle: FactorLifecycle::Fresh,
            payload,
        }))
    }

    fn can_reuse(&self, factor: &FactorHandle) -> bool {
        let reusable = self.factor_payload(factor).is_ok()
            && factor.transpose() == TransposeSupport::ForwardOnly
            && matches!(factor.method(), LinearMethod::Provider(symbol) if symbol == &resident_csr_method_symbol(self.config.method));
        if reusable {
            self.state
                .lock()
                .expect("resident CSR state poisoned")
                .snapshot
                .reused_factors += 1;
        }
        reusable
    }

    fn solve(&self, f: &FactorHandle, b: &[f64]) -> FemmResult<Vec<f64>> {
        let payload = self.factor_payload(f)?;
        self.solve_forward(&payload.matrix, b)
    }

    fn solve_transpose(&self, _f: &FactorHandle, _b: &[f64]) -> FemmResult<Vec<f64>> {
        self.refuse(FemmError::SolveDidNotConverge(
            "resident CSR factors are forward-only; transpose solve refused".to_owned(),
        ))
    }
}

#[derive(Clone)]
struct ResidentCsrFactor {
    upload_id: usize,
    fingerprint: StableId,
    method: ResidentKrylovMethod,
    matrix: ResidentCsrMatrix,
}

impl Object for ResidentCsrFactor {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> KernelResult<String> {
        Ok(format!("#<compute-femm-resident-csr {}>", self.upload_id))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for ResidentCsrFactor {}
