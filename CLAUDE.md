# CLAUDE.md

## Role

This file is the implementation playbook for developers and coding agents working in this repository. It records the current real architecture, ownership boundaries, contracts, and sharp edges. User-facing setup belongs in `README.md`; high-level architecture belongs in `PROJECT_OVERVIEW.md`.

## Mental Model

PierNet is one repository and one deployable FastAPI + React app with two product surfaces:

- `/synth`: Stage 1-4 data synthesis
- `/training`: Token Router, Text2Comp, and model assembly workflows

Do not treat the project as an old single Stage 1-4 page. The training surface is broader than Token Router now, but it is still not an unrestricted model-training platform.

## Source Of Truth

Read in this order:

1. `PROJECT_OVERVIEW.md`: system boundary and contracts
2. `README.md`: install, startup, and operational commands
3. `CLAUDE.md`: implementation notes and change constraints
4. current code

Historical plan documents are not authoritative.

## Top-Level Routes

Routes are assembled by `frontend/src/platform/PlatformRouter.tsx`.

- `/`: `frontend/src/platform/LandingPage.tsx`
- `/synth/*`: `frontend/src/synth/SynthApp.tsx`
- `/training/*`: `frontend/src/training/TrainingApp.tsx`
- `/files`: legacy redirect to `/synth/files`

Legacy synth routes still redirect into `/synth/...`. If changing top-level routing, inspect:

- `frontend/src/platform/PlatformRouter.tsx`
- `frontend/src/synth/SynthApp.tsx`
- `frontend/src/training/TrainingApp.tsx`
- `PierNet/shared/api/static.py`

`SPAStaticFiles` must preserve API 404 behavior for both `/api` and `/api/*` while falling back to `index.html` for browser routes.

## Backend Assembly

Runtime entrypoints:

- `PierNet/api/main.py`: real FastAPI app assembly
- `api_server.py`: compatibility re-export of `PierNet.api.main.app`

`PierNet/api/` is assembly only. Business logic belongs under platform namespaces:

```text
PierNet/synth/api/routers/       # synthesis HTTP routers
PierNet/synth/api/schemas/       # synthesis schemas
PierNet/synth/services/          # synthesis services
PierNet/synth/text2comp/         # Stage 2/3 core
PierNet/training/api/routers/    # training, Text2Comp, and assembly HTTP routers
PierNet/training/api/schemas/    # training schemas
PierNet/training/services/       # Token Router job manager
PierNet/training/router/         # Token Router data/model/train core
PierNet/training/text2comp/      # Text2Comp data/model/train core
```

## Frontend Layout

```text
frontend/src/platform/         # landing and top-level routing
frontend/src/shared/           # theme hook and shared frontend infrastructure
frontend/src/lib/              # API client, types, utils, seed context, scroll assist
frontend/src/synth/            # synthesis workbench
frontend/src/training/         # training workbench
frontend/src/files/            # unified file manager
```

`frontend/src/lib/api.ts` is the central frontend API wrapper. `frontend/src/lib/types.ts` is the central API type mirror.

## Synthesis Surface

`frontend/src/synth/SynthApp.tsx` owns these routes:

- `/synth`: `DatasetStats.tsx`
- `/synth/simulate`: `SimulationRunner.tsx`
- `/synth/upload`: `DataUploadPage.tsx`
- `/synth/register`: `RegisterSimulator.tsx`
- `/synth/templates`: `TemplateGenerator.tsx`
- `/synth/fill`: `SampleFiller.tsx`
- `/synth/router`: `RouterDataBuilder.tsx`
- `/synth/template-viewer`: `TemplateViewer.tsx`
- `/synth/samples`: `SampleViewer.tsx`
- `/synth/router-viewer`: `RouterViewer.tsx`
- `/synth/files`: embedded `FileManagerContent`
- `/synth/registry`: `RegistryPage.tsx`
- `/synth/llm-config`: `LLMConfig.tsx`

Synthesis routers in `PierNet/synth/api/routers/`:

- `config.py`: config, LLM config, scenario scans
- `datasets.py`: Stage 3 datasets, samples, dashboard summary
- `simulation.py`: Stage 1 simulation, batch runs, HDF5 upload/list
- `registry.py`: registry CRUD
- `generation.py`: template generation and sample filling jobs
- `router_data.py`: Stage 4 router data status/build/view
- `files.py`: template/sample file operations
- `file_catalog.py`: unified file catalog operations
- `interview.py`: interactive registry assistant
- `jobs.py`: unified synth job status/SSE/stop/delete

Synthesis services in `PierNet/synth/services/`:

- `job_manager.py`: in-memory generation job records, SSE replay, subprocess termination
- `hdf5_data.py`: HDF5 discovery, canonical paths, strict Stage 1 validation
- `file_manager.py`: template/sample/router file operations
- `file_catalog.py`: unified file asset catalog
- `manifest_store.py`: sidecar summaries
- `jsonl_index.py`: sparse byte-offset indexes
- `jsonl_filter_index.py`: sparse filter indexes

## Stage Contracts

Stage 1 HDF5:

```text
data/{simulator}/{simulator}_{scenario}.h5
data/{big_scene}/{big_scene}_{scenario}.h5
```

Required datasets and attrs:

- `timeseries [N,C,T]`, finite numeric
- `params [N,P]`, finite numeric
- `param_names [P]`, string-like
- root attrs `n_samples`, `n_channels`, `n_timesteps`, `n_params` matching shapes

Stage 2:

```text
data/templates/{scenario}_templates.jsonl
configs/text2comp/registry.yaml
configs/text2comp/default.yaml
```

Stage 3:

```text
data/text2comp_parquet/simulator={simulator}/scenario={scenario}/part-*.parquet
```

Legacy JSONL remains readable at `data/text2comp/{scenario}.jsonl` and `data/text2comp/all_training_data.jsonl`.

Stage 4:

```text
data/router_parquet/simulator={simulator}/scenario={scenario}/part-*.parquet
```

Legacy JSONL remains readable at `data/router/by_scenario/{scenario}.jsonl` and `data/router/train.jsonl`.

Training:

```text
artifacts/token_router/{simulator}/prepared/{prepared_name}/
artifacts/token_router/{simulator}/runs/{run_name}/
artifacts/text2comp_models/{simulator}/runs/{job_id}/
artifacts/fno_models/
configs/assembly/models.yaml
configs/assembly/prompt.yaml
.runlogs/training_jobs.sqlite
.runlogs/text2comp/
.runlogs/
```

## Text2Comp Core

Core files:

- `PierNet/synth/text2comp/pipeline.py`
- `PierNet/synth/text2comp/auto_register.py`
- `PierNet/synth/text2comp/interview_agent.py`
- `PierNet/synth/text2comp/generator.py`
- `PierNet/synth/text2comp/template_store.py`
- `scripts/text2comp/generate_templates.py`
- `scripts/text2comp/fill_samples.py`

Important assumptions:

- Stage 2 template generation calls an LLM.
- Stage 3 sample filling is local and should not call an LLM.
- `registry.yaml` is simulator-level with scenario metadata in the current implementation.
- Token Router target templates must end with contiguous output placeholders.

## Training Surface

`frontend/src/training/TrainingApp.tsx` owns:

- `/training`: `TrainingOverviewPage.tsx`
- `/training/new`: `TrainingNewJobPage.tsx`
- `/training/jobs`: `TrainingJobsPage.tsx`
- `/training/jobs/:jobId`: `TrainingJobDetailPage.tsx`
- `/training/text2comp`: `Text2CompPage.tsx`
- `/training/assembly`: `AssemblyPage.tsx`
- `/training/models`: `TrainedModelsPage.tsx`
- `/training/files`: training-scoped `FileManagerContent`

Training API implementation:

- `PierNet/training/api/routers/training.py`
- `PierNet/training/api/routers/text2comp.py`
- `PierNet/training/api/routers/assembly.py`
- `PierNet/training/api/schemas/training.py`
- `PierNet/training/api/schemas/text2comp.py`
- `PierNet/training/services/training_manager.py`
- `PierNet/training/text2comp/text2comp_manager.py`

Current training statuses:

- `queued`
- `starting`
- `running`
- `evaluating`
- `done`
- `error`
- `terminated`
- `external_terminated`
- `stopping`

GPU availability is conservative: free memory at least 2048 MiB and utilization at most 20%, plus process locking.

Text2Comp training artifacts default to `artifacts/text2comp_models/`. Assembly defaults should stay under `/root/data/PierNet` or be explicit config/env overrides; do not hard-code personal development directories as main defaults.

## Token Router Core

Primary files:

- `PierNet/training/router/pretrained_embeddings.py`
- `PierNet/training/router/data.py`
- `PierNet/training/router/model.py`
- `PierNet/training/router/train.py`
- `PierNet/training/router/metrics.py`
- `scripts/router/build_router_data.py`
- `scripts/router/train_token_router.py`

Current assumptions:

- default chat template: Qwen
- default embedding/tokenizer path: `$HOME/Qwen/Qwen2.5-0.5B-Instruct`
- input representation: `embedding` / `pretrained_embeddings`
- prepared data stores source file ids, byte offsets, lengths, labels, scenario ids, and metadata
- training prepares token caches from router Parquet partitions; legacy JSONL and materialized Parquet-to-JSONL caches are compatibility paths
- Router JSONL materialization caches and Token Router prepared caches are reusable caches with explicit `last_used_at` / `last_built_at`; training reuse refreshes `last_used_at`, filesystem scans do not
- model is `FullSeqDilatedConvRouter`, not a Transformer
- split is stable by `build_group_key(context, scenario)`
- only train/test splits exist

## Built-In Simulators

| Namespace | Implementation Notes |
| --- | --- |
| `PierNet/simulators/modflow/` | FloPy + MODFLOW-2005, groundwater scenarios, 18-D unified params |
| `PierNet/simulators/simpeg/` | SimPEG 0.25.x geophysics forward models, `(1,100)` output |
| `PierNet/simulators/power_flow/` | pandapower IEEE-14 steady-state load profiles, `(43,365)` output |
| `PierNet/simulators/transient/` | ANDES transient stability DAE runs, `(5,1000)` output |
| `PierNet/simulators/gcam/` | PyPSA/HiGHS simplified energy-climate LP, `(5,16)` output |

Root `requirements.txt` and `pyproject.toml` are the dependency sources of truth; simulator directories do not carry separate dependency lock-ins. Do not reintroduce `setup.py`; editable installs and console scripts are defined by `pyproject.toml`.

## Read Path And File Catalog

Stage 2 and legacy JSONL reads should prefer manifests/indexes and avoid full scans unless necessary. Stage 3/4 primary Parquet reads should go through `PierNet.shared.storage.portable`, partition manifests, and the file catalog.

Check these together when changing formats, deletion, trimming, or clearing:

- `PierNet/synth/services/manifest_store.py`
- `PierNet/synth/services/jsonl_index.py`
- `PierNet/synth/services/jsonl_filter_index.py`
- `PierNet/synth/services/file_manager.py`
- `PierNet/synth/services/file_catalog.py`
- `scripts/utils/rebuild_manifests.py`
- `scripts/utils/rebuild_indexes.py`

## Frontend Scroll Contract

Nested scroll behavior is mediated by:

- `frontend/src/lib/scrollAssist.ts`
- installation in `frontend/src/main.tsx`

When changing layout or scroll containers, inspect:

- `frontend/src/index.css`
- `.page-content`
- `.workbench-main-scroll`
- `.training-page__body`
- `.training-scroll`
- `.list-scroll-*`
- `.list-table-scroll*`

Do not treat scroll issues as CSS-only.

## Verification Commands

Backend syntax:

```bash
python -m compileall PierNet scripts api_server.py
```

Frontend checks:

```bash
cd frontend
npm run typecheck
npm run lint
npm run test
npm run build
```

Focused Python tests for data-format, API, and training changes:

```bash
pytest tests/test_build_router_data_script.py \
  tests/test_data_browsing_mixed_storage.py \
  tests/test_file_catalog.py \
  tests/test_hdf5_data_validation.py \
  tests/test_router_prepared_inputs.py \
  tests/test_storage_scripts.py \
  tests/test_training_manager_fallbacks.py
```

Docs and structure consistency:

```bash
python scripts/ci/check_consistency.py
```

## Current Sharp Edges

- Service scripts and `start_ui.sh` prefer the repo-local `.conda/env` when present, otherwise default to `$HOME/.conda/envs/PierNet`; use `PierNet_CONDA_ENV` to override.
- `PierNet/training/services/training_manager.py` launches UI-created training jobs with the backend process Python by default; override `PierNet_TRAINING_PYTHON` if a different interpreter is needed.
- `PierNet/core/llm_client.py` has provider-specific paths; test the provider branch you change instead of assuming OpenAI-compatible behavior covers all providers.
- Dependency changes must keep root `requirements.txt` and `pyproject.toml` aligned; simulator-local requirement files should stay absent.
- Frontend has focused unit coverage, but layout and routing changes still need `npm run build` plus a browser smoke check when behavior is visual.

## Change Rules

1. Keep synth and training business code inside their platform namespaces.
2. Keep `PierNet/api/main.py` as assembly, not a business-logic module.
3. Preserve Stage 1 HDF5, Stage 2 template, Stage 3 sample, Stage 4 router, and training artifact contracts unless the whole pipeline is updated.
4. Keep Token Router assumptions explicit: Qwen chat format, dynamic tokenization, frozen embedding table, single GPU.
5. Keep manifests, indexes, and Parquet partition manifests in sync with any Stage 2-4 file lifecycle change.
6. Update docs in the same change when user workflows, platform boundaries, or training assumptions change.

## Development Standards

These standards capture the working rules that should stay stable across future PierNet changes. They are more operational than the architecture notes above.

### Authoritative Workspace

- On the shared server, `/root/data/PierNet` is the authoritative working tree for this project. Do not make project changes in `/root/data/zyx` or in a local shadow copy unless the user explicitly asks for a separate experiment.
- Before editing, check `git status --short` and the current branch. Work with existing user changes; do not reset or revert unrelated changes.
- Keep temporary research pages, screenshots, scratch scripts, and one-off reports outside the repository unless the user explicitly asks to commit them.

### Runtime And Ports

- Keep the standard development ports stable: backend `8000`, frontend Vite `3000`.
- Prefer `scripts/services/start.sh`, `scripts/services/status.sh`, `scripts/services/restart.sh`, and `scripts/services/stop.sh` for persistent server work. These scripts preserve processes after SSH exits and write logs/PIDs under `.runlogs/services/`.
- After service changes, verify at least:
  - backend live/ready health under `http://127.0.0.1:8000/api/health/*`
  - frontend dev or static frontend availability
  - worker state when worker-backed tasks are involved
- If external PierNet web entrypoints are part of the user request, verify them with HTTP status checks after local service health passes.

### API And Contract Changes

- Backend routers stay under their product namespace (`PierNet/synth/...` or `PierNet/training/...`). `PierNet/api/main.py` only assembles the app.
- Frontend API calls go through `frontend/src/lib/api.ts`; shared frontend mirrors live in `frontend/src/lib/types.ts`.
- When backend schemas or routes change, update/export OpenAPI and keep `frontend/src/lib/generated/openapi.json` and `frontend/src/lib/generated/openapi.d.ts` consistent.
- Run the OpenAPI contract check before pushing API changes:

```bash
python scripts/ci/export_openapi.py /tmp/PierNet-openapi-local.json
npm --prefix frontend run openapi:check
```

### Data And Artifact Contracts

- Preserve the Stage 1 HDF5, Stage 2 template, Stage 3 sample, Stage 4 router, and training artifact contracts unless the full producer/consumer chain is updated together.
- Stage 3 sample filling is local deterministic filling and must not call an LLM.
- Any change that creates, deletes, trims, migrates, or renames Stage 2-4 data must consider manifests, sparse indexes, Parquet partition manifests, and the file catalog together.
- Runtime smoke tests that upload or generate temporary expert models, HDF5 files, configs, or registry entries must clean them up and leave `git status --short` clean.
- Automatic cache cleanup is limited to `data/router/.parquet_jsonl_cache/` and `artifacts/token_router/*/prepared/*`. It must never target core Parquet/HDF5 data, training runs/checkpoints, models, `.conda`, or `.git`.
- Before deleting a Token Router prepared cache, check active training statuses (`queued`, `starting`, `running`, `evaluating`, `stopping`) and protect any matching simulator/prepared-name reference; legacy active jobs without `prepared_name` protect all prepared caches for that simulator.

### Training And Assembly

- Treat Token Router, Text2Comp, Assembly, and Uploaded Expert as separate training/product areas even though they share `/training`.
- Keep current Token Router assumptions explicit: Qwen chat format, dynamic tokenization, frozen pretrained embedding table, and single-GPU training.
- Assembly must keep the module chain explicit: LLM, Router, Text2Comp, and an expert executor. Uploaded Expert should remain an independent executor type, not hidden inside FNO.
- Any Assembly or expert-model change must validate dimension compatibility at load or execution time and return clear errors rather than silently falling back to incomplete outputs.
- User-visible placeholder controls should be removed or wired to real backend behavior; do not keep fake upload/configuration UI.

### Frontend And UX

- Build the actual tool surface first. Avoid marketing-style landing pages for internal tools unless explicitly requested.
- Keep the PiERN console-style dark UI consistent, but avoid visually redundant panels, repeated controls, and card nesting.
- Layout or scroll changes must consider `frontend/src/lib/scrollAssist.ts`, `frontend/src/index.css`, `.page-content`, `.workbench-main-scroll`, `.training-page__body`, and `.training-scroll`.
- For visual or workflow questions, test the real UI path with the browser or Playwright instead of relying only on source inspection.
- For pages with scrolling content, inspect enough viewport positions to catch clipped text, hidden controls, and overflow.

### Verification And CI

- Use the local CI wrapper as the default pre-push gate:

```bash
scripts/ci/run_local_ci.sh
```

- For scoped work, use the narrower gates intentionally:

```bash
scripts/ci/run_local_ci.sh --backend
scripts/ci/run_local_ci.sh --frontend --no-e2e
```

- Backend gate includes Ruff, consistency, repository hygiene, migration readiness, OpenAPI export/check, and pytest.
- Frontend gate includes typecheck, lint, format check, tests, build, and optionally Playwright smoke/visual checks.
- After pushing to `main` or updating a PR branch, inspect GitHub Actions and fix failures before declaring the work complete.

### GitHub And Review Flow

- Prefer fixing the active PR branch before merging when a PR has actionable issues.
- Merge only after the branch is reviewed against current `main`, local checks pass, and CI is green.
- When pushing from the shared server, remember that `gh` uses the authenticated account configured on that server.

### Documentation

- Update docs in the same change when ports, commands, public routes, API contracts, data contracts, or training assumptions change.
- `README.md` is for user-facing setup and operations, `PROJECT_OVERVIEW.md` is for system boundaries, and `CLAUDE.md` is for implementation standards and sharp edges.
