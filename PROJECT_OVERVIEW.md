# PierNet Project Overview

## Purpose

This is the maintained high-level overview for PierNet. Read it first when you need the system boundary, platform split, stage ownership, data contracts, and runtime surfaces. Keep implementation notebook details in `CLAUDE.md` and usage commands in `README.md`.

## One-Sentence Summary

PierNet is one FastAPI + React application with two product surfaces: a Stage 1-4 data synthesis workbench and a training workbench for Token Router, Text2Comp, and model assembly.

## Product Surfaces

| Surface | Route | Frontend Namespace | Backend Namespace | Scope |
| --- | --- | --- | --- | --- |
| Landing | `/` | `frontend/src/platform/` | `PierNet/api/main.py` | entry into synth, training, and files |
| Synthesis | `/synth/*` | `frontend/src/synth/` | `PierNet/synth/` | Stage 1-4 data pipeline |
| Training | `/training/*` | `frontend/src/training/` | `PierNet/training/` | Token Router, Text2Comp, model assembly, and training artifacts |
| Files | `/synth/files`, `/training/files`; `/files` redirects to `/synth/files` | `frontend/src/files/` | `PierNet/synth/services/file_catalog.py` | unified data/artifact management |

The surfaces are separated at product and namespace level, but still share one repository, one frontend package, one FastAPI app, one static hosting path, and one startup script.

## Backend Assembly

Runtime entrypoints:

- `PierNet/api/main.py`: real FastAPI app assembly
- `api_server.py`: compatibility entry that re-exports `PierNet.api.main.app`

`PierNet/api/main.py` mounts:

- synthesis routers from `PierNet.synth.api.routers.*`
- training routers from `PierNet.training.api.routers.training`, `text2comp`, and `assembly`
- built frontend assets through `PierNet.shared.api.static.SPAStaticFiles`

`PierNet/api/` is app assembly only. Business routers, schemas, services, and models belong under `PierNet/synth/` or `PierNet/training/`.

## Frontend Assembly

Top-level routing lives in `frontend/src/platform/PlatformRouter.tsx`:

- `/` -> `LandingPage`
- `/synth/*` -> `SynthApp`
- `/training/*` -> `TrainingApp`
- `/files` -> redirect to `/synth/files`
- legacy synthesis routes redirect to `/synth/...`

Synthesis routes are owned by `frontend/src/synth/SynthApp.tsx`; training routes are owned by `frontend/src/training/TrainingApp.tsx`.

## Core Workflow

### Stage 1: Physical Data

Inputs:

- simulator config: `configs/{simulator}/variants/*.yaml`
- optional uploaded HDF5 through `/synth/upload`

Outputs:

```text
data/{simulator}/{simulator}_{scenario}.h5
data/{big_scene}/{big_scene}_{scenario}.h5
```

The HDF5 contract is the hard gate for registration:

- `timeseries [N,C,T]`, numeric and finite
- `params [N,P]`, numeric and finite
- `param_names [P]`, string-like
- root attrs match the dataset shapes

Implementation:

- API: `PierNet/synth/api/routers/simulation.py`
- validation: `PierNet/synth/services/hdf5_data.py`
- built-in simulators: `PierNet/simulators/*`

### Stage 2: Registry And Templates

Inputs:

- Stage 1 HDF5
- registry metadata in `configs/text2comp/registry.yaml`
- LLM config in `configs/text2comp/default.yaml`

Outputs:

```text
data/templates/{scenario}_templates.jsonl
```

Implementation:

- auto registration: `PierNet/synth/text2comp/auto_register.py`
- interactive registration: `PierNet/synth/text2comp/interview_agent.py`
- template generation: `PierNet/synth/text2comp/generator.py`
- template schema/filling helpers: `PierNet/synth/text2comp/template_store.py`
- CLI: `scripts/text2comp/generate_templates.py`

### Stage 3: Sample Filling

Inputs:

- Stage 1 HDF5
- Stage 2 template JSONL

Outputs:

```text
data/text2comp_parquet/simulator={simulator}/scenario={scenario}/part-*.parquet
```

Legacy JSONL remains readable and migratable at `data/text2comp/{scenario}.jsonl` and `data/text2comp/all_training_data.jsonl`.

Stage 3 is local filling. It should not call an LLM.

Implementation:

- CLI: `scripts/text2comp/fill_samples.py`
- file management: `PierNet/synth/services/file_manager.py`

### Stage 4: Router Data

Inputs:

- Stage 3 samples, preferring Parquet partitions and accepting legacy JSONL

Outputs:

```text
data/router_parquet/simulator={simulator}/scenario={scenario}/part-*.parquet
```

Legacy JSONL remains readable and migratable at `data/router/by_scenario/{scenario}.jsonl` and `data/router/train.jsonl`.

Each Stage 3 sample becomes one positive expert-prefix sample plus negative LLM-prefix samples. Stage 4 currently assumes Qwen chat template formatting and records embedding metadata for the Qwen backbone.

Implementation:

- API: `PierNet/synth/api/routers/router_data.py`
- CLI: `scripts/router/build_router_data.py`

## Training Workflow

The training surface now has three runtime modules.

Token Router training consumes Stage 4 router data and writes artifacts under `artifacts/token_router/`.

Current Token Router assumptions:

- training device: single GPU only
- model: `FullSeqDilatedConvRouter`
- split: train/test only
- default `test_ratio`: `0.10`
- input representation: dynamic Qwen tokenization + frozen pretrained embedding lookup
- default embedding backbone: `$HOME/Qwen/Qwen2.5-0.5B-Instruct`
- no offline embedding arrays are written

Text2Comp training consumes JSONL prompt/label data and writes model runs under `artifacts/text2comp_models/{simulator}/runs/{job_id}/`. The UI-selected GPU is exposed to the subprocess through `CUDA_VISIBLE_DEVICES`, so the training process uses `cuda:0` inside that isolated view.

Model assembly scans LLM, Router, Text2Comp, and FNO locations from `configs/assembly/models.yaml`, then exposes loading, prompt management, and smoke-test endpoints under `/api/assembly/*`.

Primary implementation:

- `PierNet/training/services/training_manager.py`
- `PierNet/training/router/pretrained_embeddings.py`
- `PierNet/training/router/data.py`
- `PierNet/training/router/model.py`
- `PierNet/training/router/train.py`
- `PierNet/training/text2comp/text2comp_manager.py`
- `PierNet/training/text2comp/train.py`
- `PierNet/training/api/routers/text2comp.py`
- `PierNet/training/api/routers/assembly.py`
- `scripts/router/train_token_router.py`

Training jobs and artifacts are persisted in:

```text
.runlogs/training_jobs.sqlite
.runlogs/training_jobs/
.runlogs/text2comp/
artifacts/token_router/{simulator}/runs/{run_name}/
artifacts/text2comp_models/{simulator}/runs/{job_id}/
artifacts/fno_models/
```

## Supported Simulators

| Simulator | Domain | Math Type | Output Shape | Scenarios |
| --- | --- | --- | --- | --- |
| `modflow` | Groundwater | Parabolic PDE | `(5, 365)` | 7 |
| `simpeg` | Geophysics | Elliptic PDE | `(1, 100)` | 4 |
| `power_flow` | Steady-state power flow | Nonlinear algebraic system | `(43, 365)` | 5 |
| `transient` | Transient stability | DAE | `(5, 1000)` | 3 |
| `gcam` | Energy-climate planning | Dynamic algebraic / LP | `(5, 16)` | 3 |

## Read Path And File Management

Stage 2 templates are JSONL. Stage 3/4 primary artifacts are portable Parquet partitions, with legacy JSONL support kept for migration and compatibility. Interactive reads use catalog summaries and sidecar acceleration instead of full scans:

- manifests: `data/.manifests/`
- sparse indexes: `data/.indexes/`
- filter indexes: `data/.indexes/`
- Parquet partitions: `data/text2comp_parquet/`, `data/router_parquet/`

Related implementation:

- `PierNet/synth/services/manifest_store.py`
- `PierNet/synth/services/jsonl_index.py`
- `PierNet/synth/services/jsonl_filter_index.py`
- `PierNet/synth/services/file_catalog.py`
- `scripts/utils/rebuild_manifests.py`
- `scripts/utils/rebuild_indexes.py`

The unified file catalog can manage HDF5, template, sample, router, training-job, manifest, and index assets. Protected merged files and indexes are not blindly deleted.

## API Boundaries

Synthesis API prefixes include:

- `/api/dashboard/*`
- `/api/config/*`
- `/api/simulation/*`
- `/api/registry/*`
- `/api/generate/*`
- `/api/files/*`
- `/api/router/*`
- `/api/interview/*`

Training API prefixes:

- `/api/training/*`
- `/api/text2comp/*`
- `/api/assembly/*`

Static frontend serving uses browser-history fallback through `SPAStaticFiles`, while preserving `/api`, `/api/*`, and asset 404 behavior.

## Operational Contracts

- `start_ui.sh` is the main combined backend/frontend development startup path.
- Startup scripts prefer the repo-local `.conda/env` when present, otherwise default to `$HOME/.conda/envs/PierNet`; set `PierNet_CONDA_ENV` before startup to override.
- Frontend nested scroll behavior depends on `frontend/src/lib/scrollAssist.ts`; scroll changes are application behavior, not only CSS.
- Root docs expected by consistency checks are `README.md`, `PROJECT_OVERVIEW.md`, and `CLAUDE.md`.

## Change Checklist

When changing startup or user workflows, update `README.md`.

When changing platform boundaries, routes, stage ownership, or data contracts, update `PROJECT_OVERVIEW.md` and `CLAUDE.md`.

When changing implementation assumptions, known pitfalls, test commands, or agent instructions, update `CLAUDE.md`.

When changing Stage 2-4 artifacts or cleanup logic, check manifests, indexes, file catalog, and router/training consumers together.
