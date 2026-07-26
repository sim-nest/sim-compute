//! Rendering for compute command evidence.

use crate::{
    acceptance::AcceptanceEvidence,
    args::ComputeCommand,
    evidence::{ProfileEvidence, ProviderEvidence, RecipeEvidence},
};

/// Renders command help.
pub fn help() -> &'static str {
    "Usage: sim compute <devices|probe|profile|explain|recipe|acceptance> [OPTIONS]\n"
}

pub(crate) fn render_providers(command: &ComputeCommand, rows: &[ProviderEvidence]) -> String {
    if json(command) {
        let body = rows.iter().map(provider_json).collect::<Vec<_>>().join(",");
        format!("{{\"providers\":[{body}]}}\n")
    } else {
        let mut out = String::new();
        for row in rows {
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\n",
                row.selector,
                row.site,
                if row.installed { "installed" } else { "absent" },
                row.status
            ));
        }
        out
    }
}

pub(crate) fn render_profile(command: &ComputeCommand, evidence: &ProfileEvidence) -> String {
    if json(command) {
        let keys = evidence
            .keys
            .iter()
            .map(|key| format!("\"{}\"", escape(key)))
            .collect::<Vec<_>>()
            .join(",");
        let decision = evidence
            .decision
            .as_ref()
            .map(|value| format!("\"{}\"", escape(value)))
            .unwrap_or_else(|| "null".to_owned());
        format!(
            "{{\"status\":\"{}\",\"keys\":[{}],\"decision\":{}}}\n",
            escape(&evidence.status),
            keys,
            decision
        )
    } else {
        let mut out = format!("profile\t{}\n", evidence.status);
        for key in &evidence.keys {
            out.push_str(&format!("key\t{key}\n"));
        }
        if let Some(decision) = &evidence.decision {
            out.push_str(&format!("decision\t{decision}\n"));
        }
        out
    }
}

pub(crate) fn render_recipe(command: &ComputeCommand, evidence: &RecipeEvidence) -> String {
    if json(command) {
        let capabilities = evidence
            .capabilities
            .iter()
            .map(|capability| format!("\"{}\"", escape(capability)))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"recipe\":\"{}\",\"route\":\"{}\",\"capabilities\":[{}]}}\n",
            escape(&evidence.id),
            escape(&evidence.route),
            capabilities
        )
    } else {
        format!(
            "recipe\t{}\nroute\t{}\ncapabilities\t{}\n",
            evidence.id,
            evidence.route,
            evidence.capabilities.join(",")
        )
    }
}

pub(crate) fn render_acceptance(evidence: &AcceptanceEvidence) -> String {
    format!(
        "acceptance\t{}\tschema={}\tcases={}\tartifact={}\n",
        evidence.status, evidence.schema, evidence.cases, evidence.artifact
    )
}

fn json(command: &ComputeCommand) -> bool {
    match command {
        ComputeCommand::Help => false,
        ComputeCommand::Devices(selection) | ComputeCommand::Probe(selection) => {
            matches!(selection.output, crate::args::OutputMode::Json)
        }
        ComputeCommand::Profile(request) | ComputeCommand::Explain(request) => {
            matches!(request.output, crate::args::OutputMode::Json)
        }
        ComputeCommand::Recipe(request) => matches!(request.output, crate::args::OutputMode::Json),
        ComputeCommand::Acceptance(_) => false,
    }
}

fn provider_json(row: &ProviderEvidence) -> String {
    let capability = row
        .capability
        .as_ref()
        .map(|value| format!("\"{}\"", escape(value)))
        .unwrap_or_else(|| "null".to_owned());
    format!(
        "{{\"selector\":\"{}\",\"site\":\"{}\",\"installed\":{},\"capability\":{},\"status\":\"{}\"}}",
        escape(&row.selector),
        escape(&row.site),
        row.installed,
        capability,
        escape(&row.status)
    )
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
