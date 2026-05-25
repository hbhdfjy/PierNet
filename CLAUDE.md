# CLAUDE.md

## Role

This file is the implementation playbook for developers and coding agents working in this repository. It records the current real architecture, ownership boundaries, contracts, and sharp edges. User-facing setup belongs in `README.md`; high-level architecture belongs in `PROJECT_OVERVIEW.md`.

## Mental Model

PierNet is one repository and one deployable FastAPI + React app with two product surfaces:

- `/synth`: Stage 1-4 data synthesis
- `/training`: single-GPU Token Router training

Do not treat the project as an old single Stage 1-4 page, and do not treat the training surface as a general model-training platform.

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
PierNet/training/api/routers/    # training HTTP router
PierNet/training/api/schemas/    # training schemas
PierNet/training/services/       # training job manager
PierNet/training/router/         # Token Router data/model/train core
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
.runlogs/training_jobs.sqlite
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
- `/training/files`: training-scoped `FileManagerContent`

Training API implementation:

- `PierNet/training/api/routers/training.py`
- `PierNet/training/api/schemas/training.py`
- `PierNet/training/services/training_manager.py`

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
