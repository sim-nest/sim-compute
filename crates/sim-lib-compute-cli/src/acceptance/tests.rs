use super::*;

const SOURCE: &str = "0123456789abcdef0123456789abcdef01234567";

fn artifact() -> Artifact {
    let cases = REQUIRED_CASES
        .iter()
        .map(|(id, category)| ManifestCase {
            id: (*id).to_owned(),
            category: (*category).to_owned(),
        })
        .collect::<Vec<_>>();
    Artifact {
        source: SOURCE.to_owned(),
        harness_hash: stable_hash(HARNESS.as_bytes()),
        manifest_hash: "hash".to_owned(),
        target: TARGET_CLASS.to_owned(),
        adapter: "NVIDIA GeForce RTX 5080 Laptop GPU".to_owned(),
        backend: "wgpu".to_owned(),
        driver: "595.71.05".to_owned(),
        caps: "storage-buffer,compute-shader,f32".to_owned(),
        evidence: EVIDENCE_KIND.to_owned(),
        power: "9.73W".to_owned(),
        thermal: "46C".to_owned(),
        cases: cases
            .iter()
            .enumerate()
            .map(|(index, case)| CaseResult {
                id: case.id.clone(),
                category: case.category.clone(),
                case_hash: case_hash(case),
                passed: true,
                submitted: index as u64 + 1,
                bytes: 4096,
                readbacks: 0,
                max_abs_error: 0.0,
            })
            .collect(),
    }
}

fn manifest_cases_fixture() -> Vec<ManifestCase> {
    REQUIRED_CASES
        .iter()
        .map(|(id, category)| ManifestCase {
            id: (*id).to_owned(),
            category: (*category).to_owned(),
        })
        .collect()
}

#[test]
fn verifies_complete_sanitized_artifact() {
    let parsed = Artifact::parse(&artifact().render()).unwrap();

    parsed.verify(SOURCE, &manifest_cases_fixture()).unwrap();
}

#[test]
fn rejects_private_and_non_physical_artifacts() {
    let mut private = artifact();
    private.adapter = "tiger-user-path".to_owned();
    assert!(Artifact::parse(&private.render()).is_err());

    let mut modeled = artifact();
    modeled.evidence = "modeled".to_owned();
    let parsed = Artifact::parse(&modeled.render()).unwrap();
    assert!(parsed.verify(SOURCE, &manifest_cases_fixture()).is_err());
}

#[test]
fn rejects_duplicates_missing_cases_source_mismatch_and_counters() {
    let mut duplicate = artifact();
    duplicate.cases.push(duplicate.cases[0].clone());
    assert!(duplicate.verify(SOURCE, &manifest_cases_fixture()).is_err());

    let mut missing = artifact();
    missing.cases.pop();
    assert!(missing.verify(SOURCE, &manifest_cases_fixture()).is_err());

    let source = artifact();
    assert!(
        source
            .verify(
                "fedcba9876543210fedcba9876543210fedcba98",
                &manifest_cases_fixture()
            )
            .is_err()
    );

    let mut counters = artifact();
    counters.cases[0].submitted = 0;
    assert!(counters.verify(SOURCE, &manifest_cases_fixture()).is_err());
}

#[test]
fn rejects_stale_schema_and_wrong_hashes() {
    let stale = artifact()
        .render()
        .replace(SCHEMA, "sim.compute-acceptance/v0");
    assert!(Artifact::parse(&stale).is_err());

    let mut wrong = artifact();
    wrong.cases[0].case_hash = "bad".to_owned();
    assert!(wrong.verify(SOURCE, &manifest_cases_fixture()).is_err());
}
