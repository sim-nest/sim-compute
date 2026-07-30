//! Checked recipe-contract harness.

use std::fs;
use std::path::Path;

use toml_edit::DocumentMut;

pub fn run() -> Result<(), String> {
    let root = std::env::current_dir().map_err(|err| format!("current dir: {err}"))?;
    let mut checked = 0usize;
    for recipe in recipe_files(&root)? {
        let dir = recipe
            .parent()
            .ok_or_else(|| format!("recipe path has no parent: {}", recipe.display()))?;
        let manifest = fs::read_to_string(&recipe)
            .map_err(|err| format!("read {}: {err}", recipe.display()))?;
        let contract = parse_contract(&recipe, &manifest)?;
        let setup_path = dir.join(&contract.setup);
        let setup = fs::read_to_string(&setup_path)
            .map_err(|err| format!("read {}: {err}", setup_path.display()))?;
        if setup.trim().is_empty() {
            return Err(format!("recipe setup is empty: {}", setup_path.display()));
        }
        let expected_path = dir.join(&contract.expected);
        let expected = fs::read_to_string(&expected_path)
            .map_err(|err| format!("read {}: {err}", expected_path.display()))?;
        if expected.trim() != contract.result {
            return Err(format!(
                "recipe expected fixture disagrees with [[expect]] result: {}",
                dir.display()
            ));
        }
        checked += 1;
    }
    println!("check-recipes: checked {checked} recipe contract(s)");
    Ok(())
}

struct RecipeContract {
    setup: String,
    expected: String,
    result: String,
}

fn parse_contract(path: &Path, source: &str) -> Result<RecipeContract, String> {
    let document = source
        .parse::<DocumentMut>()
        .map_err(|err| format!("parse {}: {err}", path.display()))?;
    let string_field = |name: &str| {
        document
            .get(name)
            .and_then(|item| item.as_str())
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| format!("{} has no non-empty `{name}`", path.display()))
    };
    let expectations = document
        .get("expect")
        .and_then(|item| item.as_array_of_tables())
        .ok_or_else(|| format!("{} has no [[expect]] table", path.display()))?;
    if expectations.len() != 1 {
        return Err(format!(
            "{} must declare exactly one [[expect]] table, found {}",
            path.display(),
            expectations.len()
        ));
    }
    let expectation = expectations
        .get(0)
        .expect("length was checked before reading the expectation");
    let form = expectation
        .get("form")
        .and_then(|item| item.as_integer())
        .ok_or_else(|| format!("{} [[expect]] has no integer `form`", path.display()))?;
    if form != 0 {
        return Err(format!(
            "{} single-form recipe must expect form 0, found {form}",
            path.display()
        ));
    }
    let result = expectation
        .get("result")
        .and_then(|item| item.as_str())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("{} [[expect]] has no non-empty `result`", path.display()))?;
    Ok(RecipeContract {
        setup: string_field("setup")?,
        expected: string_field("expected")?,
        result,
    })
}

fn recipe_files(root: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let mut files = Vec::new();
    collect(&root.join("crates"), &mut files)?;
    files.sort();
    Ok(files)
}

fn collect(path: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(path).map_err(|err| format!("read {}: {err}", path.display()))? {
        let entry = entry.map_err(|err| format!("read {} entry: {err}", path.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect(&path, files)?;
        } else if path.file_name().and_then(|name| name.to_str()) == Some("recipe.toml") {
            files.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_uses_the_declared_output_instead_of_echoing_setup_source() {
        let contract = parse_contract(
            Path::new("recipe.toml"),
            r#"
setup = "setup.siml"
expected = "expected.txt"

[[expect]]
form = 0
result = "(expr:call compute demo)"
"#,
        )
        .unwrap();
        assert_eq!(contract.setup, "setup.siml");
        assert_eq!(contract.expected, "expected.txt");
        assert_eq!(contract.result, "(expr:call compute demo)");
    }

    #[test]
    fn contract_refuses_ambiguous_multiple_results() {
        let error = parse_contract(
            Path::new("recipe.toml"),
            r#"
setup = "setup.siml"
expected = "expected.txt"

[[expect]]
form = 0
result = "one"

[[expect]]
form = 1
result = "two"
"#,
        )
        .err()
        .unwrap();
        assert!(error.contains("exactly one [[expect]]"));
    }
}
