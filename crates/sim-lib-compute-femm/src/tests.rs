use std::sync::Arc;

use sim_kernel::{Cx, DefaultFactory, EagerPolicy, Export, Lib};
use sim_lib_compute_model::{ModeledComputeFault, ModeledComputeProfile};
use sim_lib_femm_core::{CsrMatrix, FemmError, StableId};
use sim_lib_femm_solve::{
    FactorHandle, FactorLifecycle, LinearFactor, LinearMethod, LinearSolver, LinearSolverValue,
    TransposeSupport, linear_solver_symbol,
};
use sim_lib_numbers_tensor::CpuTensorExecutor;

use crate::{
    ComputeFemmLib, ResidentCsrConfig, ResidentCsrSolver, ResidentKrylovMethod,
    resident_csr_method_symbol,
};

fn spd_matrix() -> CsrMatrix {
    CsrMatrix::new(
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0],
    )
    .unwrap()
}

fn nonsymmetric_matrix() -> CsrMatrix {
    CsrMatrix::new(vec![0, 2, 4], vec![0, 1, 0, 1], vec![4.0, 1.0, 2.0, 3.0]).unwrap()
}

#[test]
fn cg_solves_resident_csr_with_f64_certification() {
    let solver = ResidentCsrSolver::default();
    let factor = solver.factor(&spd_matrix()).unwrap();
    let x = solver.solve(&factor, &[15.0, 10.0, 10.0]).unwrap();

    assert!((x[0] - 5.0).abs() < 1.0e-6);
    let snapshot = solver.snapshot();
    assert_eq!(snapshot.uploads, 1);
    assert_eq!(snapshot.solves, 1);
    assert!(snapshot.spmv_dispatches > 0);
    assert!(snapshot.vector_dispatches > 0);
    assert!(snapshot.scalar_synchronizations > 0);
    assert!(snapshot.f64_residual_checks > 0);
}

#[test]
fn bicgstab_solves_nonsymmetric_system() {
    let solver = ResidentCsrSolver::new(ResidentCsrConfig {
        method: ResidentKrylovMethod::Bicgstab,
        ..ResidentCsrConfig::default()
    });
    let matrix = nonsymmetric_matrix();
    let factor = solver.factor(&matrix).unwrap();
    let x = solver.solve(&factor, &[1.0, 1.0]).unwrap();

    assert!((4.0 * x[0] + x[1] - 1.0).abs() < 1.0e-7);
    assert!((2.0 * x[0] + 3.0 * x[1] - 1.0).abs() < 1.0e-7);
}

#[test]
fn factor_reuse_is_fingerprinted_without_reupload() {
    let solver = ResidentCsrSolver::default();
    let factor = solver.factor(&spd_matrix()).unwrap();

    assert!(solver.can_reuse(&factor));
    let reused = factor.clone().reused();
    assert!(solver.can_reuse(&reused));
    assert_eq!(solver.snapshot().uploads, 1);
    assert_eq!(solver.snapshot().reused_factors, 2);
    assert_eq!(reused.lifecycle(), FactorLifecycle::Reused);
}

#[test]
fn modeled_provider_agrees_with_cpu_provider_and_counts_readbacks() {
    let modeled = ResidentCsrSolver::default();
    let modeled_factor = modeled.factor(&spd_matrix()).unwrap();
    let modeled_x = modeled.solve(&modeled_factor, &[15.0, 10.0, 10.0]).unwrap();

    let cpu = ResidentCsrSolver::with_executor(
        ResidentCsrConfig::default(),
        Arc::new(CpuTensorExecutor::new()),
    );
    let cpu_factor = cpu.factor(&spd_matrix()).unwrap();
    let cpu_x = cpu.solve(&cpu_factor, &[15.0, 10.0, 10.0]).unwrap();

    for (modeled, cpu) in modeled_x.iter().zip(&cpu_x) {
        assert!((modeled - cpu).abs() < 1.0e-6);
    }
    let snapshot = modeled.snapshot();
    let modeled_snapshot = modeled.modeled_snapshot().unwrap();
    assert!(modeled_snapshot.accepted > snapshot.spmv_dispatches + snapshot.vector_dispatches);
    assert!(snapshot.provider_readbacks > 0);
    assert!(snapshot.provider_readbacks <= modeled_snapshot.accepted);
    assert_eq!(modeled_snapshot.readbacks, snapshot.provider_readbacks);
}

#[test]
fn provider_loss_fails_closed_before_factor_or_solve_acceptance() {
    let solver = ResidentCsrSolver::modeled(
        ResidentCsrConfig::default(),
        ModeledComputeProfile {
            fault: Some(ModeledComputeFault::DeviceLostDuringExecute),
            ..ModeledComputeProfile::default()
        },
    );
    let err = solver.factor(&spd_matrix()).unwrap_err();

    assert!(
        matches!(err, FemmError::SolveDidNotConverge(message) if message.contains("provider failure"))
    );
    assert_eq!(solver.snapshot().refusals, 1);
}

#[test]
fn refuses_transpose_stale_nonfinite_and_uncertified_solves() {
    let solver = ResidentCsrSolver::default();
    let factor = solver.factor(&spd_matrix()).unwrap();

    let transpose = solver
        .solve_transpose(&factor, &[1.0, 1.0, 1.0])
        .unwrap_err();
    assert!(
        matches!(transpose, FemmError::SolveDidNotConverge(message) if message.contains("transpose"))
    );

    let bad_rhs = solver.solve(&factor, &[f64::NAN, 0.0, 0.0]).unwrap_err();
    assert!(matches!(bad_rhs, FemmError::MalformedMatrix(message) if message.contains("finite")));

    let stale = FactorHandle::new(LinearFactor {
        method: LinearMethod::Provider(resident_csr_method_symbol(ResidentKrylovMethod::Cg)),
        matrix_fingerprint: StableId(0),
        transpose: TransposeSupport::ForwardOnly,
        lifecycle: FactorLifecycle::Fresh,
        payload: factor.payload().clone(),
    });
    let err = solver.solve(&stale, &[1.0, 1.0, 1.0]).unwrap_err();
    assert!(matches!(err, FemmError::SolveDidNotConverge(message) if message.contains("stale")));

    let strict = ResidentCsrSolver::new(ResidentCsrConfig {
        certificate_tol_f64: 1.0e-20,
        max_refinements: 0,
        ..ResidentCsrConfig::default()
    });
    let strict_factor = strict.factor(&spd_matrix()).unwrap();
    let err = strict
        .solve(&strict_factor, &[15.0, 10.0, 10.0])
        .unwrap_err();
    assert!(
        matches!(err, FemmError::SolveDidNotConverge(message) if message.contains("certification"))
    );
}

#[test]
fn loadable_library_exports_femm_linear_solver_value() {
    let mut cx = Cx::new(
        Arc::new(EagerPolicy),
        Arc::new(DefaultFactory),
        sim_kernel::HandleSeed::new(0xc926_8ebd_a432_398a),
    );
    cx.load_lib(&ComputeFemmLib::default()).unwrap();

    let value = cx
        .registry()
        .value_by_symbol(&linear_solver_symbol())
        .expect("linear solver export");
    assert!(value.object().downcast_ref::<LinearSolverValue>().is_some());
    assert!(ComputeFemmLib::default().manifest().exports.iter().any(
        |export| matches!(export, Export::Value { symbol } if *symbol == linear_solver_symbol())
    ));
}
