//! Finding the repository's non-code assets from a process whose working directory is unknown.
//!
//! The cargo workspace lives in `Engine/`, while the frozen golden corpus (`Docs/golden/`) and the
//! helper scripts (`Tools/`) live at the repository root, and `Tools/build-kernels.sh` writes its
//! kernel images into `Engine/target/hip/`. So the two directories a command is plausibly run from
//! are the repo root (`Engine/target/release/sn`, `Tools/ask.sh`) and `Engine/` (`cargo test`).
//! A default path like `Docs/golden/expected.tsv` or `target/hip` cannot be resolved against the
//! working directory alone, and getting it wrong fails silently: no images registered, no expected
//! rows read.
//!
//! Every *default* resource path therefore goes through [`resolve_file`] / [`resolve_dir`]: the path
//! is tried as given first (an explicit path from the command line is never rewritten), then against
//! each ancestor of the working directory, then against each ancestor of the running executable's
//! directory. The first candidate that exists wins. A path handed in by the caller is used verbatim.

use std::path::PathBuf;

/// Directories to try, in order: the working directory and its ancestors, then the directory of the
/// running executable and its ancestors. Deduplicated by string, deepest first, so the closest hit
/// wins. `current_exe()` failing (no permission, exotic platform) only costs the second half.
fn roots() -> Vec<PathBuf> {
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let mut out: Vec<PathBuf> = Vec::new();
    for start in [std::env::current_dir().ok(), exe_dir] {
        let Some(mut p) = start else { continue };
        loop {
            if !out.iter().any(|q| q == &p) {
                out.push(p.clone());
            }
            if !p.pop() {
                break;
            }
        }
    }
    out
}

fn resolve(rel: &str, want_dir: bool) -> Option<PathBuf> {
    let rel = rel.trim();
    if rel.is_empty() {
        return None;
    }
    let hit = |p: &std::path::Path| {
        if want_dir {
            p.is_dir()
        } else {
            p.is_file()
        }
    };
    for root in roots() {
        // `root.join(rel)` keeps `rel` verbatim when it is absolute, which is why an explicit path
        // from the command line resolves on the first iteration and is never rewritten.
        let p = root.join(rel);
        if hit(&p) {
            return Some(p);
        }
    }
    None
}

/// The first existing *file* at `rel`, searched over [`roots`]. `None` means "not in this tree".
pub fn resolve_file(rel: &str) -> Option<PathBuf> {
    resolve(rel, false)
}

/// The first existing *directory* at `rel`, searched over [`roots`].
pub fn resolve_dir(rel: &str) -> Option<PathBuf> {
    resolve(rel, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_next_to_the_test_binary_resolves() {
        // Integration/unit tests run with the package directory as the working directory, so this
        // is the crate's own manifest.
        assert!(resolve_file("Cargo.toml").is_some());
    }

    #[test]
    fn ancestors_of_the_working_directory_are_searched() {
        // From `Engine/crates/sn-core` the workspace root is two levels up: the walk has to climb.
        let hit = resolve_dir("crates/sn-core").expect("the workspace root is an ancestor");
        assert!(hit.join("src/lib.rs").is_file() || hit.join("lib.rs").is_file());
    }

    #[test]
    fn missing_assets_report_none_instead_of_a_phantom_path() {
        assert!(resolve_file("Docs/golden/definitely-not-a-corpus.tsv").is_none());
        assert!(resolve_dir("no/such/directory").is_none());
        assert!(resolve_file("   ").is_none());
    }

    #[test]
    fn the_engine_tree_is_findable_from_a_test() {
        // The whole point: `Docs/` is outside the workspace, so it is *not* reachable from the
        // working directory - only from an ancestor of it.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        if root.join("Docs/golden/expected.tsv").is_file() {
            let hit = resolve_file("Docs/golden/expected.tsv").expect("the corpus is an ancestor");
            assert!(hit.is_absolute() && !hit.to_string_lossy().contains(".."));
            assert_eq!(std::fs::canonicalize(&hit).unwrap(), std::fs::canonicalize(root.join("Docs/golden/expected.tsv")).unwrap());
        }
    }
}
