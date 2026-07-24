//! Minimal checked recipe harness for descriptor recipes.

use std::fs;
use std::path::Path;

pub fn run() -> Result<(), String> {
    let root = std::env::current_dir().map_err(|err| format!("current dir: {err}"))?;
    let mut checked = 0usize;
    for recipe in recipe_files(&root)? {
        let dir = recipe
            .parent()
            .ok_or_else(|| format!("recipe path has no parent: {}", recipe.display()))?;
        let setup = fs::read_to_string(dir.join("setup.siml"))
            .map_err(|err| format!("read {} setup.siml: {err}", dir.display()))?;
        let expected = fs::read_to_string(dir.join("expected.txt"))
            .map_err(|err| format!("read {} expected.txt: {err}", dir.display()))?;
        if setup.trim() != expected.trim() {
            return Err(format!("recipe output mismatch: {}", dir.display()));
        }
        checked += 1;
    }
    println!("check-recipes: checked {checked} recipe(s)");
    Ok(())
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
