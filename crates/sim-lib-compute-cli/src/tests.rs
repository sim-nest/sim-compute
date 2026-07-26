use std::sync::Arc;

use sim_kernel::{DefaultFactory, EagerPolicy, Lib};
use sim_lib_compute_auto::ComputeAutoLib;
use sim_lib_compute_model::{ComputeModelLib, ModeledComputeProfile};

use crate::{
    ComputeCliLib, compute_device_capability, compute_entrypoint_symbol,
    compute_profile_read_capability, compute_profile_write_capability, parse_compute_args,
    run_command,
};

// conformance: compute CLI exports bounded devices, probe, profile, explain, and recipe evidence through a loadable command.

fn test_cx() -> (sim_kernel::Cx, sim_kernel::GrantSeat) {
    sim_kernel::Cx::new_seated(Arc::new(EagerPolicy), Arc::new(DefaultFactory))
}

fn grant(cx: &mut sim_kernel::Cx, seat: &sim_kernel::GrantSeat) {
    seat.grant(cx, compute_device_capability()).unwrap();
    seat.grant(cx, compute_profile_read_capability()).unwrap();
    seat.grant(cx, compute_profile_write_capability()).unwrap();
}

#[test]
fn help_parses_without_capabilities() {
    let (mut cx, _) = test_cx();
    let command = parse_compute_args(&["compute".to_owned(), "--help".to_owned()]).unwrap();

    let output = run_command(&mut cx, None, &command).unwrap();

    assert!(output.contains("devices|probe|profile|explain|recipe"));
}

#[test]
fn model_without_adapter_is_structured_evidence() {
    let (mut cx, seat) = test_cx();
    seat.grant(&mut cx, compute_device_capability()).unwrap();
    let command = parse_compute_args(&[
        "compute".to_owned(),
        "devices".to_owned(),
        "model".to_owned(),
        "--json".to_owned(),
    ])
    .unwrap();

    let output = run_command(&mut cx, None, &command).unwrap();

    assert!(output.contains("\"selector\":\"model\""));
    assert!(output.contains("\"status\":\"model/no-adapter\""));
}

#[test]
fn devices_requires_compute_device_capability() {
    let (mut cx, _) = test_cx();
    let command = parse_compute_args(&["devices".to_owned()]).unwrap();

    let err = run_command(&mut cx, None, &command).unwrap_err();

    assert!(err.to_string().contains("compute.device"));
}

#[test]
fn machine_output_lists_installed_model_and_auto_sites() {
    let (mut cx, seat) = test_cx();
    grant(&mut cx, &seat);
    cx.load_lib(&ComputeModelLib::new(ModeledComputeProfile::default()))
        .unwrap();
    cx.load_lib(&ComputeAutoLib::default()).unwrap();
    let command = parse_compute_args(&[
        "devices".to_owned(),
        "--json".to_owned(),
        "--max-devices".to_owned(),
        "8".to_owned(),
    ])
    .unwrap();

    let output = run_command(&mut cx, None, &command).unwrap();

    assert!(output.contains("\"site\":\"site/compute/model\""));
    assert!(output.contains("\"site\":\"site/compute/auto\""));
    assert!(output.contains("\"installed\":true"));
}

#[test]
fn profile_requires_injected_table_not_path() {
    let (mut cx, seat) = test_cx();
    seat.grant(&mut cx, compute_profile_read_capability())
        .unwrap();
    let command = parse_compute_args(&["profile".to_owned(), "list".to_owned()]).unwrap();

    let output = run_command(&mut cx, None, &command).unwrap();

    assert!(output.contains("no-profile-store"));
}

#[test]
fn profile_save_and_stale_explain_use_table_store() {
    let (mut cx, seat) = test_cx();
    grant(&mut cx, &seat);
    let table = cx.factory().table(Vec::new()).unwrap();
    let save = parse_compute_args(&[
        "profile".to_owned(),
        "save".to_owned(),
        "--key".to_owned(),
        "compute-profile/a".to_owned(),
        "--adapter".to_owned(),
        "a".to_owned(),
        "--driver".to_owned(),
        "driver".to_owned(),
        "--backend".to_owned(),
        "modeled".to_owned(),
        "--now".to_owned(),
        "1".to_owned(),
    ])
    .unwrap();
    run_command(&mut cx, Some(&table), &save).unwrap();
    let explain = parse_compute_args(&[
        "explain".to_owned(),
        "--key".to_owned(),
        "compute-profile/a".to_owned(),
        "--adapter".to_owned(),
        "a".to_owned(),
        "--driver".to_owned(),
        "driver".to_owned(),
        "--backend".to_owned(),
        "modeled".to_owned(),
        "--now".to_owned(),
        "100000".to_owned(),
        "--json".to_owned(),
    ])
    .unwrap();

    let output = run_command(&mut cx, Some(&table), &explain).unwrap();

    assert!(output.contains("\"decision\":\"stale\""));
}

#[test]
fn ambiguous_profiles_are_rejected_when_key_is_missing() {
    let (mut cx, seat) = test_cx();
    grant(&mut cx, &seat);
    let table = cx.factory().table(Vec::new()).unwrap();
    for key in ["compute-profile/a", "compute-profile/b"] {
        let command = parse_compute_args(&[
            "profile".to_owned(),
            "save".to_owned(),
            "--key".to_owned(),
            key.to_owned(),
        ])
        .unwrap();
        run_command(&mut cx, Some(&table), &command).unwrap();
    }
    let command = parse_compute_args(&["profile".to_owned(), "read".to_owned()]).unwrap();

    let err = run_command(&mut cx, Some(&table), &command).unwrap_err();

    assert!(err.to_string().contains("ambiguous compute profiles"));
}

#[test]
fn parser_rejects_unknown_and_unbounded_input() {
    assert!(parse_compute_args(&["devices".to_owned(), "../path".to_owned()]).is_err());
    assert!(
        parse_compute_args(&[
            "devices".to_owned(),
            "--max-devices".to_owned(),
            "1000".to_owned()
        ])
        .is_err()
    );
    assert!(parse_compute_args(&["recipe".to_owned(), "unknown".to_owned()]).is_err());
}

#[test]
fn compute_lib_exports_cli_main_compute() {
    let manifest = Lib::manifest(&ComputeCliLib::new());

    assert!(manifest.exports.iter().any(|export| matches!(
        export,
        sim_kernel::Export::Function { symbol, .. } if symbol == &compute_entrypoint_symbol()
    )));
}

#[test]
fn entrypoint_accepts_injected_profile_store() {
    let (mut cx, seat) = test_cx();
    grant(&mut cx, &seat);
    let table = cx.factory().table(Vec::new()).unwrap();
    cx.load_lib(&ComputeCliLib::with_profile_store(table))
        .unwrap();
    let args = cx
        .factory()
        .list(vec![
            cx.factory().string("compute".to_owned()).unwrap(),
            cx.factory().string("profile".to_owned()).unwrap(),
            cx.factory().string("list".to_owned()).unwrap(),
        ])
        .unwrap();
    let envelope = cx
        .factory()
        .table(vec![(sim_kernel::Symbol::new("args"), args)])
        .unwrap();

    let result = cx
        .call_function(
            &compute_entrypoint_symbol(),
            sim_kernel::Args::new(vec![envelope]),
        )
        .unwrap();

    assert_eq!(result.object().display(&mut cx).unwrap(), "true");
}
