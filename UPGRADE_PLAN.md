# PiERN Upgrade Plan

## Scope
This plan focuses on improving stability, data correctness, reproducibility, and operability across the Stage 1-4 pipeline.

## Priority 1: Correctness and Reproducibility

### 1. Strengthen `registry.yaml` as a contract layer
Problem:
- Stage 2 and Stage 3 depend heavily on `configs/text2comp/registry.yaml`.
- There is no strong validation for `output_info.slice`, `observation_config`, units, channel coverage, or scenario overrides.

Actions:
- Add a schema validator for simulator-level and scenario-level registry entries.
- Validate `output_info.slice` against actual HDF5 channel counts.
- Validate `time_modes`, `fixed_channels`, `channel_min/max`, and units.
- Add a CI check that fails when registry entries are incomplete or inconsistent.

### 2. Add dataset lineage and version metadata
Problem:
- Stage outputs do not carry enough provenance to reproduce a dataset build.
- `all_training_data.jsonl` and router outputs are not tied to exact config, prompt, model, seed, or source HDF5 versions.

Actions:
- Record build metadata for each Stage 2/3/4 output:
  - config hash
  - registry hash
  - model/provider
  - prompt/template version
  - seed
  - source HDF5 path and mtime/hash
- Emit a build manifest JSON per run.
- Include manifest references in merged datasets.

### 3. Store structured numeric outputs explicitly in Stage 3
Problem:
- The frontend currently parses numeric arrays back out of `target` text.
- This is fragile and couples visualization to natural-language formatting.

Actions:
- Add explicit structured fields such as `observed_outputs` or `target_arrays` to Stage 3 samples.
- Keep `target` for training, but use structured fields for visualization and validation.
- Refactor frontend charts to consume structured data first.

### 4. Improve Stage 4 negative sample design
Problem:
- Current router negatives are mostly truncated trigger prefixes.
- This is too easy and can overfit superficial formatting cues.

Actions:
- Add harder negatives:
  - wrong trigger prefix from another template
  - wrong simulator/scenario continuation
  - wrong assistant prefix style
  - malformed output structure
  - cross-language mismatches
- Track negative sample type in metadata.
- Evaluate router balance by scenario and negative type.

## Priority 2: Pipeline Architecture and Runtime Stability

### 5. Persist jobs, logs, and task history
Problem:
- Job state is in memory.
- Restarting the API loses logs, history, and running job metadata.

Actions:
- Persist job records and event logs to disk or SQLite.
- Recover recent job history on API startup.
- Store PID/process metadata for managed subprocesses.

### 6. Factor out a shared simulator pipeline skeleton
Problem:
- `modflow`, `simpeg`, `power_flow`, `transient`, and `gcam` pipelines duplicate the same lifecycle:
  - seed generation
  - filtering
  - perturbation/augmentation
  - unified parameter conversion
  - HDF5 save

Actions:
- Extract a shared pipeline base with hooks for domain-specific generation and validation.
- Standardize progress logging and metadata emission.
- Reduce code drift across simulators.

### 7. Unify environment and deployment definitions
Problem:
- Runtime requirements are split across `requirements.txt`, `setup.py`, `start_ui.sh`, and implicit machine state.
- There are stale entry points and compatibility drift.

Actions:
- Add one authoritative environment spec.
- Align Python version with actual dependency constraints.
- Fix stale `setup.py` console scripts.
- Add a documented startup path for backend and frontend.

## Priority 3: Product Workflow and Operator Experience

### 8. Add Stage precondition gates
Problem:
- The UI allows operators to reach later stages with incomplete prerequisites.
- Empty template files and partial registry states are not treated strictly enough.

Actions:
- Gate Stage 2 on valid registry entries.
- Gate Stage 3 on non-empty valid templates.
- Gate Stage 4 on valid Stage 3 samples.
- Surface blocking reasons in the UI.

### 9. Consolidate registration workflows
Problem:
- There are two registration directions in the codebase:
  - interview-driven registration
  - auto-register / direct registry editing
- The product path is not unified.

Actions:
- Define one primary registration flow.
- Use the others as advanced or fallback tools.
- Unify output shape and validation behavior.

### 10. Separate code storage from generated artifacts
Problem:
- The repository tree mixes code and growing data outputs under `data/`.
- This makes deployment, backup, and cleanup harder.

Actions:
- Move generated artifacts to a configured workspace path outside the repo root.
- Keep the repo focused on code, configs, and lightweight examples.
- Add retention and cleanup controls for large intermediate outputs.

### 11. Add end-to-end contract tests
Problem:
- Most risk is at stage boundaries, not inside isolated functions.

Actions:
- Add tests for:
  - HDF5 schema contract
  - registry schema contract
  - template placeholder integrity
  - filled sample structural integrity
  - router label distribution and sample validity
- Run these checks in CI.

### 12. Add semantic quality checks for Stage 2 templates
Problem:
- Format validity is checked, but semantic correctness is not strongly measured.

Actions:
- Check unit consistency, output naming consistency, language/style distribution, and time-mode consistency.
- Flag low-quality template batches before Stage 3 filling.

## Suggested Implementation Order
1. Registry validation + build manifests
2. Structured Stage 3 numeric outputs
3. Persistent jobs/logs
4. Stronger router negatives
5. Shared simulator pipeline base
6. UI gating and registration flow consolidation
7. Storage separation and full contract tests
