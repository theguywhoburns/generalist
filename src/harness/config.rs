//! Manifest loading: resolve, merge, parse, validate.
//!
//! Loading is four separable stages, and each has its own failure mode worth
//! distinguishing — a missing file and a schema mismatch used to both surface
//! as an opaque `String`.
//!
//! 1. **Resolve** — follow the `extends` chain, relative to the including
//!    file's own directory, with cycle detection.
//! 2. **Merge** — deep-merge each layer over its parent. Objects merge
//!    key-wise; **arrays replace wholesale**. So overriding
//!    `protocol.k_set` to `[0]` does not leave a stale `[0,1,2,3,5,8]` behind.
//! 3. **Parse** — deserialize the merged document into [`RunConfig`] (or
//!    [`ExperimentPlan`] for a chain file).
//! 4. **Validate** — cross-field checks that no single struct can make,
//!    reported *together* so one run surfaces every problem instead of the
//!    first.
//!
//! ## Why `extends`
//!
//! Manifests are complete specs, not patches: burn's `Config` derive emits no
//! `#[serde(default)]`, so a half-written `model` block fails to parse. Four
//! `stage0-*.json` files were byte-identical in their `model` and `optim`
//! blocks — pure duplication, and a change to either had to be made in five
//! places. `extends` makes a variation cost only its *difference*:
//!
//! ```json
//! { "extends": "stage0-base.json",
//!   "experiment": { "tasks": ["subst-fst-oracle"] },
//!   "train": { "ckpt_dir": "checkpoints-oracle" } }
//! ```
//!
//! The base file is itself a valid manifest, so it can be inspected, diffed,
//! or run directly. Inheritance is by value, not by pointer: the result is
//! one flat `RunConfig` with no runtime notion of "extends", and the same
//! invariants hold whether a run came from one file or five.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::run::{ExperimentPlan, RunConfig};

/// A manifest could not be turned into a run configuration.
#[derive(Debug)]
pub enum ConfigError {
    /// The file could not be read.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The bytes were not valid JSON.
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    /// Valid JSON, wrong shape: a missing or mistyped field.
    Schema {
        path: PathBuf,
        source: serde_json::Error,
    },
    /// `extends` formed a loop.
    Cycle { path: PathBuf, chain: Vec<PathBuf> },
    /// Every problem found by validation, not just the first.
    Invalid {
        path: PathBuf,
        problems: Vec<String>,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            ConfigError::Parse { path, source } => {
                write!(f, "{}: invalid JSON: {source}", path.display())
            }
            ConfigError::Schema { path, source } => {
                write!(
                    f,
                    "{}: does not match the manifest schema: {source}",
                    path.display()
                )
            }
            ConfigError::Cycle { path, chain } => {
                let names: Vec<String> = chain.iter().map(|p| p.display().to_string()).collect();
                write!(
                    f,
                    "{}: extends cycle: {}",
                    path.display(),
                    names.join(" -> ")
                )
            }
            ConfigError::Invalid { path, problems } => {
                writeln!(f, "{}: {} problem(s):", path.display(), problems.len())?;
                for p in problems {
                    writeln!(f, "  - {p}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Deep-merge `layer` over `base`: objects merge key-wise, everything else
/// (including arrays) replaces.
///
/// Replaces rather than unions on arrays is the load-bearing choice. A demo
/// protocol's `k_set` is a *set specification*, and a union would make it
/// impossible to narrow — the exact operation the fixed-map and map-only
/// manifests need (`k_set: [0]` to force zero-shot).
fn merge(base: &mut Value, layer: Value) {
    match (base, layer) {
        (Value::Object(base_map), Value::Object(layer_map)) => {
            for (k, v) in layer_map {
                match base_map.get_mut(&k) {
                    Some(existing) => merge(existing, v),
                    None => {
                        base_map.insert(k, v);
                    }
                }
            }
        }
        (slot, layer) => *slot = layer,
    }
}

/// Load and fully resolve a run manifest.
pub fn load_run(path: &Path) -> Result<RunConfig, ConfigError> {
    let (merged, root) = resolve(path, &mut Vec::new())?;
    // A chain plan loaded as a single run would otherwise fail deep inside the
    // path-variable expander, on the plan's own `$prev` checkpoint sentinel,
    // which reads as a broken manifest rather than as "wrong loader". The two
    // document shapes are disjoint: a plan has `experiments`, a run does not.
    if merged.get("experiments").is_some() {
        return Err(ConfigError::Invalid {
            path: root,
            problems: vec![format!(
                "this is a chain PLAN, not a single-run manifest: it has an \
                 `experiments` array. Use `--example chain`, which takes a plan \
                 and runs its stages in order (`init_from: \"$prev\"` chains each \
                 stage onto the previous one's weights)"
            )],
        });
    }
    let cfg: RunConfig = serde_json::from_value(merged).map_err(|source| ConfigError::Schema {
        path: root.clone(),
        source,
    })?;
    let problems = cfg.validate();
    if problems.is_empty() {
        Ok(cfg)
    } else {
        Err(ConfigError::Invalid {
            path: root,
            problems,
        })
    }
}

/// Load a chain plan. Plans do not participate in `extends` — they are short
/// and a chain that inherits from another chain is a cycle waiting to happen.
pub fn load_plan(path: &Path) -> Result<ExperimentPlan, ConfigError> {
    let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| ConfigError::Schema {
        path: path.to_path_buf(),
        source,
    })
}

/// Write a complete, self-contained manifest: no `extends`, every knob
/// explicit. The round-trip target for authoring a variation — load a base,
/// tweak, save, and the result runs standalone.
pub fn save_run(path: &Path, cfg: &RunConfig) -> Result<(), ConfigError> {
    let value = serde_json::to_value(cfg).map_err(|source| ConfigError::Schema {
        path: path.to_path_buf(),
        source,
    })?;
    // The document is rebuilt from the typed config, so no `_comment` from a
    // source manifest survives into the output.
    let text = format!(
        "{}\n",
        serde_json::to_string_pretty(&value).map_err(|source| ConfigError::Schema {
            path: path.to_path_buf(),
            source
        },)?
    );
    std::fs::write(path, text).map_err(|source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Follow the `extends` chain and return `(merged document, root path)`.
///
/// `stack` carries the chain so far, for cycle detection. Each `extends`
/// resolves relative to the file that names it, so a manifest moved with its
/// directory keeps working.
pub(crate) fn resolve(
    path: &Path,
    stack: &mut Vec<PathBuf>,
) -> Result<(Value, PathBuf), ConfigError> {
    let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut doc: Value = serde_json::from_str(&text).map_err(|source| ConfigError::Parse {
        path: path.to_path_buf(),
        source,
    })?;
    let root = path.to_path_buf();
    // `_comment` is a documentation convention for humans, not a schema field.
    // Dropping it per-file means a child's comment cannot leak into a sibling
    // that extends the same base, and it never reaches the typed parse.
    if let Some(map) = doc.as_object_mut() {
        map.remove(COMMENT_KEY);
    }
    // Path variables are expanded PER FILE, before the parent is read, so
    // `$curdir` means the directory of the manifest that mentions it rather
    // than the directory of whichever file happened to be resolved first.
    expand_path_vars(&mut doc, &root)?;
    if stack.contains(&root) {
        let mut chain = stack.clone();
        chain.push(root.clone());
        return Err(ConfigError::Cycle { path: root, chain });
    }
    stack.push(root.clone());

    // `extends` is optional. Absent => this file is a complete document.
    // Present but not a string => a schema error naming the key, not a panic.
    let parent_spec = match doc.as_object_mut().and_then(|o| o.remove("extends")) {
        None => {
            stack.pop();
            return Ok((doc, root));
        }
        Some(v) => match v.as_str() {
            Some(s) => s.to_string(),
            None => {
                return Err(ConfigError::Invalid {
                    path: root.clone(),
                    problems: vec![format!(
                        "`extends` must be a string path, got {}",
                        kind_of(&v)
                    )],
                });
            }
        },
    };

    let parent_path = resolve_relative(&root, &parent_spec);
    let (mut base, _) = resolve(&parent_path, stack)?;
    merge(&mut base, doc);
    stack.pop();
    Ok((base, root))
}

/// Optional documentation key, stripped during resolution. Any string or
/// array of strings is accepted; it never reaches the typed parse.
pub const COMMENT_KEY: &str = "_comment";

fn resolve_relative(from: &Path, spec: &str) -> PathBuf {
    let p = Path::new(spec);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        from.parent().unwrap_or(Path::new(".")).join(p)
    }
}

/// Directory holding the versioned experiment manifests.
pub const EXPERIMENT_DIR: &str = "research/experiments";

/// Path variables available in every manifest, as `$name` or `${name}`.
///
/// These exist so a manifest under `research/experiments/` can refer to its own
/// location, to the shipped configs, and to the repo root without embedding an
/// absolute path that breaks the moment the repo is cloned somewhere else.
fn path_vars(file: &Path) -> Vec<(&'static str, PathBuf)> {
    let curdir = file
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    vec![
        ("curdir", curdir.clone()),
        (
            "parent_dir",
            curdir
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from(".")),
        ),
        ("repo_root", repo_root.clone()),
        ("experiment_dir", repo_root.join(EXPERIMENT_DIR)),
        ("configs_dir", repo_root.join("configs")),
    ]
}

/// Replace `$name` / `${name}` in every string of `doc`.
///
/// An unknown variable is a **hard error**, not a passthrough. A typo'd
/// `$experimnt_dir` would otherwise survive as a literal path, produce a
/// confusing "no such file" much later, and — worse — could name a real
/// directory by accident and load the wrong file.
fn expand_path_vars(doc: &mut Value, file: &Path) -> Result<(), ConfigError> {
    if !doc.to_string().contains('$') {
        return Ok(());
    }
    let vars = path_vars(file);
    let mut problems = Vec::new();
    walk_strings(doc, &vars, &mut problems);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(ConfigError::Invalid {
            path: file.to_path_buf(),
            problems,
        })
    }
}

fn walk_strings(v: &mut Value, vars: &[(&'static str, PathBuf)], problems: &mut Vec<String>) {
    match v {
        Value::String(s) => {
            if let Some(rep) = substitute(s, vars) {
                match rep {
                    Ok(text) => *s = text,
                    Err(name) => problems.push(format!(
                        "unknown path variable `${name}`; available: {}",
                        vars.iter()
                            .map(|(n, _)| format!("${n}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )),
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| walk_strings(x, vars, problems)),
        Value::Object(m) => m.values_mut().for_each(|x| walk_strings(x, vars, problems)),
        _ => {}
    }
}

/// Substitute path variables in one string, or `None` when there is nothing to
/// do.
fn substitute(s: &str, vars: &[(&'static str, PathBuf)]) -> Option<Result<String, String>> {
    // `$` that is not a variable reference is left alone, so a literal dollar
    // in a prompt or a comment survives.
    let mut out = String::with_capacity(s.len());
    let bytes: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut touched = false;
    while i < bytes.len() {
        if bytes[i] != '$' {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        let mut j = i + 1;
        let braced = bytes.get(j) == Some(&'{');
        if braced {
            j += 1;
        }
        let start = j;
        while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == '_') {
            j += 1;
        }
        // A reference must START with a letter or underscore. Without this,
        // a literal `$5` parses as a variable named "5" and is rejected as
        // unknown -- which breaks any prompt or comment containing a price.
        if start >= bytes.len() || !(bytes[start].is_ascii_alphabetic() || bytes[start] == '_') {
            out.push('$');
            i += 1;
            continue;
        }
        let name: String = bytes[start..j].iter().collect();
        if braced {
            if bytes.get(j) != Some(&'}') {
                // `${` with no closing brace: not a reference, emit literally.
                out.push('$');
                i += 1;
                continue;
            }
            j += 1;
        }
        if name.is_empty() {
            out.push('$');
            i += 1;
            continue;
        }
        touched = true;
        match vars.iter().find(|(n, _)| *n == name) {
            Some((_, val)) => {
                out.push_str(&val.to_string_lossy());
                i = j;
            }
            // Unknown: report it and stop rewriting this string. The caller
            // collects one error per offending string, so a manifest with the
            // same typo in three places reports three identical lines rather
            // than failing three times with different context.
            None => return Some(Err(name)),
        }
    }
    touched.then_some(Ok(out))
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a bool",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Resolve every manifest referenced by a chain plan, returning
/// `(plan, [(experiment name, resolved run config)])`.
///
/// Loading up front means a broken manifest fails before any pool is
/// generated or any weights are touched, rather than after a stage has
/// already trained. The chain runner therefore receives only usable configs.
pub fn load_chain(
    plan_path: &Path,
) -> Result<(ExperimentPlan, Vec<(String, RunConfig)>), ConfigError> {
    let plan = load_plan(plan_path)?;
    let mut out = Vec::with_capacity(plan.experiments.len());
    for exp in &plan.experiments {
        // Manifest paths in a plan are relative to the plan file, matching
        // how `extends` resolves relative to the file that names it.
        let path = resolve_relative(plan_path, &exp.manifest);
        let cfg = load_run(&path)?;
        out.push((exp.name.clone(), cfg));
    }
    Ok((plan, out))
}

/// Names of every run manifest directly under `dir`. Used by the test that
/// keeps checked-in manifests loadable.
pub fn manifest_names(dir: &Path) -> std::io::Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| {
            let name = e.ok()?.file_name().to_string_lossy().into_owned();
            name.ends_with(".json").then_some(name)
        })
        .collect();
    names.sort();
    Ok(names)
}

/// Every `.json` file in `dir` that is a run manifest (as opposed to a chain
/// plan). Chain plans are identified by their `experiments` key.
pub fn is_chain_plan(text: &str) -> bool {
    serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|v| v.get("experiments").cloned())
        .is_some()
}

/// Collects duplicate-key paths across a set of manifests, for tests and for
/// spotting copy-paste drift. Returns paths like `train.ckpt_dir` that appear
/// with differing values in more than one file.
pub fn divergent_keys(values: &[Value]) -> BTreeSet<String> {
    let mut seen: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for v in values {
        collect_leaves(v, String::new(), &mut seen);
    }
    seen.into_iter()
        .filter(|(_, v)| v.len() > 1)
        .map(|(k, _)| k)
        .collect()
}

type BTreeMap<K, V> = std::collections::BTreeMap<K, V>;

fn collect_leaves(v: &Value, prefix: String, out: &mut BTreeMap<String, BTreeSet<String>>) {
    match v {
        Value::Object(map) => {
            for (k, child) in map {
                if k == "extends" {
                    continue;
                }
                let p = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                collect_leaves(child, p, out);
            }
        }
        other => {
            out.entry(prefix).or_default().insert(other.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("generalist-config-test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn write(name: &str, v: Value) -> PathBuf {
        let p = tmp(name);
        std::fs::write(&p, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        p
    }

    #[test]
    fn merge_is_recursive_for_objects() {
        let mut base = json!({"a": 1, "n": {"x": 1, "y": 2}});
        merge(&mut base, json!({"a": 2, "n": {"y": 9, "z": 3}}));
        assert_eq!(base, json!({"a": 2, "n": {"x": 1, "y": 9, "z": 3}}));
    }

    #[test]
    fn merge_replaces_arrays_wholesale() {
        // The load-bearing rule: a narrowed k_set must not keep stale values.
        let mut base = json!({"protocol": {"k_set": [0, 1, 2, 3, 5, 8], "k0_rate": 0.15}});
        merge(&mut base, json!({"protocol": {"k_set": [0]}}));
        assert_eq!(base["protocol"]["k_set"], json!([0]));
        // Sibling keys survive.
        assert_eq!(base["protocol"]["k0_rate"], json!(0.15));
    }

    #[test]
    fn merge_replaces_scalars_and_type_changes() {
        let mut base = json!({"a": {"b": 1}});
        merge(&mut base, json!({"a": 5}));
        assert_eq!(base, json!({"a": 5}));
    }

    #[test]
    fn extends_chain_merges_base_then_overrides() {
        let base = write(
            "merge-base.json",
            json!({
                "model": {"d_model": 256, "n_heads": 4},
                "train": {"batch_size": 6, "steps": 500}
            }),
        );
        let child = write(
            "merge-child.json",
            json!({
                "extends": base.file_name().unwrap().to_str().unwrap(),
                "train": {"steps": 50}
            }),
        );
        let (merged, root) = resolve(&child, &mut Vec::new()).unwrap();
        assert_eq!(root, child);
        // Overridden leaf wins...
        assert_eq!(merged["train"]["steps"], json!(50));
        // ...inherited leaves survive...
        assert_eq!(merged["train"]["batch_size"], json!(6));
        assert_eq!(merged["model"]["d_model"], json!(256));
        // ...and `extends` itself is consumed, not left in the document.
        assert!(merged.get("extends").is_none());
    }

    #[test]
    fn extends_resolves_relative_to_the_including_file() {
        // A manifest in a subdirectory must resolve siblings, not the cwd.
        let dir = tmp("nest");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("base.json"), r#"{"train": {"steps": 7}}"#).unwrap();
        let child = dir.join("child.json");
        std::fs::write(&child, r#"{"extends": "base.json", "train": {"steps": 9}}"#).unwrap();
        let (merged, _) = resolve(&child, &mut Vec::new()).unwrap();
        assert_eq!(merged["train"]["steps"], json!(9));
    }

    #[test]
    fn cycle_is_detected_and_reported_with_the_chain() {
        let a = write("cyc-a.json", json!({"extends": "cyc-b.json"}));
        write("cyc-b.json", json!({"extends": "cyc-a.json"}));
        match resolve(&a, &mut Vec::new()) {
            Err(ConfigError::Cycle { chain, .. }) => {
                assert!(chain.len() >= 3, "chain too short: {chain:?}");
                assert!(chain.iter().any(|p| p.ends_with("cyc-a.json")));
                assert!(chain.iter().any(|p| p.ends_with("cyc-b.json")));
            }
            other => panic!("expected a cycle error, got {other:?}"),
        }
    }

    #[test]
    fn self_extends_is_a_cycle() {
        let p = write("self.json", json!({"extends": "self.json"}));
        assert!(matches!(
            resolve(&p, &mut Vec::new()),
            Err(ConfigError::Cycle { .. })
        ));
    }

    #[test]
    fn non_string_extends_is_a_schema_error_not_a_panic() {
        let p = write("bad-extends.json", json!({"extends": 42}));
        match resolve(&p, &mut Vec::new()) {
            Err(ConfigError::Invalid { problems, .. }) => {
                assert!(problems[0].contains("extends"), "{problems:?}");
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn missing_file_reports_io_with_the_path() {
        let p = tmp("definitely-not-here.json");
        match load_run(&p) {
            Err(ConfigError::Io { path, .. }) => assert_eq!(path, p),
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn malformed_json_reports_parse_and_names_the_file() {
        let p = tmp("garbage.json");
        std::fs::write(&p, "{ not json").unwrap();
        match load_run(&p) {
            Err(ConfigError::Parse { path, .. }) => assert_eq!(path, p),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn saved_manifest_is_standalone_and_reloads_identically() {
        // The contract that makes `save_run` usable: output is a complete
        // spec with no `extends`, so it runs with the base deleted, and
        // re-serializing it is a fixed point.
        let dir = std::env::temp_dir().join("generalist-save-test");
        std::fs::create_dir_all(&dir).unwrap();
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/stage0-fixed.json");
        let cfg = load_run(&src).expect("load source");
        let out = dir.join("standalone.json");
        save_run(&out, &cfg).expect("save");

        let text = std::fs::read_to_string(&out).unwrap();
        assert!(
            !text.contains("extends"),
            "saved output kept an extends key"
        );
        assert!(!text.contains(COMMENT_KEY), "saved output kept a _comment");

        // Reloadable, and the reload is a fixed point under re-serialization.
        let back = load_run(&out).expect("reload standalone");
        assert_eq!(
            serde_json::to_value(&back).unwrap(),
            serde_json::to_value(&cfg).unwrap()
        );
        // And the substantive fields survived.
        assert_eq!(back.model.param_count(), cfg.model.param_count());
        assert_eq!(back.stop, cfg.stop);
        assert_eq!(back.experiment.tasks, cfg.experiment.tasks);
        assert_eq!(back.train.steps, cfg.train.steps);
    }

    #[test]
    fn comment_keys_are_stripped_and_do_not_leak_into_siblings() {
        let dir = std::env::temp_dir().join("generalist-comment-test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("base.json"),
            format!(r#"{{"train": {{"steps": 5}}, "{COMMENT_KEY}": "base note"}}"#),
        )
        .unwrap();
        // A child with its own comment must not adopt the base's, and the base
        // must not acquire the child's.
        std::fs::write(
            dir.join("a.json"),
            format!(r#"{{"extends": "base.json", "{COMMENT_KEY}": "a note"}}"#),
        )
        .unwrap();
        std::fs::write(dir.join("b.json"), r#"{"extends": "base.json"}"#).unwrap();

        for name in ["a.json", "b.json"] {
            let (merged, _) =
                resolve(&dir.join(name), &mut Vec::new()).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(
                merged.get(COMMENT_KEY).is_none(),
                "{name} leaked a comment: {merged}"
            );
        }
        // The base alone still resolves to its own content.
        let (merged, _) = resolve(&dir.join("base.json"), &mut Vec::new()).unwrap();
        assert!(merged.get(COMMENT_KEY).is_none());
        assert_eq!(merged["train"]["steps"], json!(5));
    }

    #[test]
    fn errors_display_with_useful_text() {
        let e = ConfigError::Invalid {
            path: PathBuf::from("x.json"),
            problems: vec!["a".to_string(), "b".to_string()],
        };
        let s = e.to_string();
        assert!(s.contains("x.json"), "{s}");
        assert!(s.contains("2 problem(s)"), "{s}");
        assert!(s.contains("- a") && s.contains("- b"), "{s}");
    }

    #[test]
    fn divergent_keys_finds_copy_paste_drift() {
        let a = json!({"train": {"steps": 1, "seed": 0}});
        let b = json!({"train": {"steps": 2, "seed": 0}});
        let d = divergent_keys(&[a, b]);
        assert!(d.contains("train.steps"), "{d:?}");
        assert!(
            !d.contains("train.seed"),
            "seed agreed, should not be reported: {d:?}"
        );
    }

    #[test]
    fn divergent_keys_ignores_the_extends_key() {
        let a = json!({"extends": "base.json", "train": {"steps": 1}});
        let b = json!({"extends": "other.json", "train": {"steps": 1}});
        assert!(divergent_keys(&[a, b]).is_empty());
    }
    /// An unknown variable must be an error, not a passthrough.
    ///
    /// A typo'd `$experimnt_dir` that survived as a literal would produce a
    /// confusing "no such file" much later, and could — worse — name a real
    /// directory by accident and silently load the wrong file.
    #[test]
    fn unknown_path_variable_is_rejected_with_the_available_set() {
        let dir = std::env::temp_dir().join("generalist-var-unknown");
        std::fs::create_dir_all(&dir).expect("tmpdir");
        let f = dir.join("bad.json");
        std::fs::write(&f, r#"{"train":{"ckpt_dir":"$experimnt_dir/run"}}"#).expect("write");
        let err = resolve(&f, &mut Vec::new()).expect_err("must reject unknown var");
        let text = format!("{err}");
        assert!(text.contains("experimnt_dir"), "{text}");
        // The message must list what IS available, or it is a guessing game.
        for known in ["curdir", "parent_dir", "repo_root", "experiment_dir"] {
            assert!(text.contains(known), "message omits ${known}: {text}");
        }
    }

    /// `$name` and `${name}` both work, and both resolve to the same place.
    #[test]
    fn braced_and_bare_path_variables_resolve_identically() {
        let dir = std::env::temp_dir().join("generalist-var-brace");
        std::fs::create_dir_all(&dir).expect("tmpdir");
        let (doc_bare, root) = {
            let f = dir.join("bare.json");
            std::fs::write(&f, r#"{"a":"$repo_root/x"}"#).expect("write");
            resolve(&f, &mut Vec::new()).expect("bare")
        };
        let (doc_braced, _) = {
            let f = dir.join("braced.json");
            std::fs::write(&f, r#"{"a":"${repo_root}/x"}"#).expect("write");
            resolve(&f, &mut Vec::new()).expect("braced")
        };
        assert_eq!(doc_bare["a"], doc_braced["a"]);
        let want = format!("{}/x", PathBuf::from(env!("CARGO_MANIFEST_DIR")).display());
        assert_eq!(doc_bare["a"].as_str().unwrap(), want);
        let _ = root;
    }

    /// `$curdir` is per-file, not per-run: a child extending a base in another
    /// directory must see its OWN directory, while the base keeps its own.
    #[test]
    fn curdir_is_the_directory_of_the_file_that_mentions_it() {
        let base_dir = std::env::temp_dir().join("generalist-var-cur-base");
        let child_dir = std::env::temp_dir().join("generalist-var-cur-child");
        std::fs::create_dir_all(&base_dir).expect("base dir");
        std::fs::create_dir_all(&child_dir).expect("child dir");
        let base = base_dir.join("base.json");
        std::fs::write(&base, r#"{"train":{"ckpt_dir":"$curdir/from-base"}}"#).expect("base");
        let child = child_dir.join("child.json");
        let body = format!(
            r#"{{"extends":{},"train":{{"eval_max_new":8,"ckpt_dir":"$curdir/from-child"}}}}"#,
            serde_json::to_string(&base.to_string_lossy()).unwrap()
        );
        std::fs::write(&child, body).expect("child");
        let (doc, _) = resolve(&child, &mut Vec::new()).expect("resolve");
        // The child's own value wins, and it is the child's directory.
        assert_eq!(
            doc["train"]["ckpt_dir"].as_str().unwrap(),
            format!("{}/from-child", child_dir.display())
        );
    }

    /// A literal `$` that is not a variable reference must survive untouched, or
    /// prompts and comments containing currency break.
    #[test]
    fn literal_dollars_that_are_not_variables_survive() {
        let dir = std::env::temp_dir().join("generalist-var-literal");
        std::fs::create_dir_all(&dir).expect("tmpdir");
        let f = dir.join("lit.json");
        std::fs::write(
            &f,
            r#"{"a":"costs $5","b":"trailing $","c":"${","d":"$repo_root/x"}"#,
        )
        .expect("write");
        let (doc, _) = resolve(&f, &mut Vec::new()).expect("resolve");
        assert_eq!(doc["a"].as_str().unwrap(), "costs $5");
        assert_eq!(doc["b"].as_str().unwrap(), "trailing $");
        assert_eq!(doc["c"].as_str().unwrap(), "${");
        assert_eq!(
            doc["d"].as_str().unwrap(),
            format!("{}/x", env!("CARGO_MANIFEST_DIR"))
        );

        // Deliberate trade-off, pinned so nobody "fixes" it by accident: a
        // well-formed but unknown `$word` is REJECTED, not passed through. Prompts
        // are generated by tasks rather than written in manifests, so the exposure
        // is comments, and a loud error there beats a silently wrong path. Digits
        // after the first character are fine, so `$x_1` is a reference while `$5`
        // is currency.
        let f2 = dir.join("unknown.json");
        std::fs::write(&f2, r#"{"a":"$x_1"}"#).expect("write");
        let err = resolve(&f2, &mut Vec::new()).expect_err("$x_1 is unknown");
        assert!(format!("{err}").contains("x_1"), "{err}");
    }
}
