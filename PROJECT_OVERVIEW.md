# PiERN Project Overview

## Purpose

This is the maintained high-level overview for PiERN. Read it first when you need the system boundary, platform split, stage ownership, data contracts, and runtime surfaces. Keep implementation notebook details in `CLAUDE.md` and usage commands in `README.md`.

## One-Sentence Summary

PiERN is one FastAPI + React application with two product surfaces: a Stage 1-4 data synthesis workbench and a single-GPU Token Router training workbench.

## Product Surfaces

| Surface | Route | Frontend Namespace | Backend Namespace | Scope |
| --- | --- | --- | --- | --- |
| Landing | `/` | `frontend/src/platform/` | `piern/api/main.py` | entry into synth, training, and files |
| Synthesis | `/synth/*` | `frontend/src/synth/` | `piern/synth/` | Stage 1-4 data pipeline |
| Training | `/training/*` | `frontend/src/training/` | `piern/training/` | Token Router training jobs |
| Files | `/files` and `/synth/files` | `frontend/src/files/` | `piern/synth/services/file_catalog.py` | unified data/artifact management |

The surfaces are separated at product and namespace level, but still share one repository, one frontend package, one FastAPI app, one static hosting path, and one startup script.

## Backend Assembly

Runtime entrypoints:

- `piern/api/main.py`: real FastAPI app assembly
- `api_server.py`: compatibility entry that re-exports `piern.api.main.app`

`piern/api/main.py` mounts:

- synthesis routers from `piern.synth.api.routers.*`
- training router from `piern.training.api.routers.training`
- built frontend assets through `piern.shared.api.static.SPAStaticFiles`

`piern/api/` is app assembly only. Business routers, schemas, services, and models belong under `piern/synth/` or `piern/training/`.

## Frontend Assembly

Top-level routing lives in `frontend/src/platform/PlatformRouter.tsx`:

- `/` -> `LandingPage`
- `/synth/*` -> `SynthApp`
- `/training/*` -> `TrainingApp`
- `/files` -> standalone `FileManagerPage`
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

- API: `piern/synth/api/routers/simulation.py`
- validation: `piern/synth/services/hdf5_data.py`
- built-in simulators: `piern/simulators/*`

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

- auto registration: `piern/synth/text2comp/auto_register.py`
- interactive registration: `piern/synth/text2comp/interview_agent.py`
- template generation: `piern/synth/text2comp/generator.py`
- template schema/filling helpers: `piern/synth/text2comp/template_store.py`
- CLI: `scripts/text2comp/generate_templates.py`

### Stage 3: Sample Filling

Inputs:

- Stage 1 HDF5
- Stage 2 template JSONL

Outputs:

```text
data/text2comp/{scenario}.jsonl
data/text2comp/all_training_data.jsonl
```

Stage 3 is local filling. It should not call an LLM.

Implementation:

- CLI: `scripts/text2comp/fill_samples.py`
- file management: `piern/synth/services/file_manager.py`

### Stage 4: Router Data

Inputs:

- Stage 3 sample JSONL

Outputs:

```text
data/router/by_scenario/{scenario}.jsonl
data/router/train.jsonl
```

Each Stage 3 sample becomes one positive expert-prefix sample plus negative LLM-prefix samples. Stage 4 currently assumes Qwen chat template formatting and records embedding metadata for the Qwen backbone.

Implementation:

- API: `piern/synth/api/routers/router_data.py`
- CLI: `scripts/router/build_router_data.py`

## Training Workflow

Training consumes Stage 4 router data and writes artifacts under `artifacts/token_router/`.

Current assumptions:

- model family: Token Router only
- training device: single GPU only
- model: `FullSeqDilatedConvRouter`
- split: train/test only
- default `test_ratio`: `0.10`
- input representation: dynamic Qwen tokenization + frozen pretrained embedding lookup
- default embedding backbone: `/home/tpx/Qwen/Qwen2.5-0.5B-Instruct`
- no offline embedding arrays are written

Primary implementation:

- `piern/training/services/training_manager.py`
- `piern/training/router/pretrained_embeddings.py`
- `piern/training/router/data.py`
- `piern/training/router/model.py`
- `piern/training/router/train.py`
- `scripts/router/train_token_router.py`

Training jobs are persisted in:

```text
artifacts/token_router/training_jobs.json
.runlogs/
artifacts/token_router/{simulator}/runs/{run_name}/
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

Stage 2-4 source artifacts are JSONL. Interactive reads use sidecar acceleration:

- manifests: `data/.manifests/`
- sparse indexes: `data/.indexes/`
- filter indexes: `data/.indexes/`

Related implementation:

- `piern/synth/services/manifest_store.py`
- `piern/synth/services/jsonl_index.py`
- `piern/synth/services/jsonl_filter_index.py`
- `piern/synth/services/file_catalog.py`
- `scripts/utils/rebuild_manifests.py`
- `scripts/utils/rebuild_indexes.py`
- `scripts/utils/rebuild_filter_indexes.py`

The unified file catalog can manage HDF5, template, sample, router, training-job, manifest, and index assets. Protected merged files and indexes are not blindly deleted.

## API Boundaries

Synthesis API prefixes include:

- `/api/dashboard/*`
- `/api/config/*`
- `/api/simulation/*`
- `/api/register/*`
- `/api/generate/*`
- `/api/files/*`
- `/api/router/*`
- `/api/interview/*`

Training API prefix:

- `/api/training/*`

Static frontend serving uses browser-history fallback through `SPAStaticFiles`, while preserving `/api/*` and asset 404 behavior.

## Operational Contracts

- `start_ui.sh` is the main combined backend/frontend development startup path.
- If the active conda environment is not the script default, set `PIERN_CONDA_ENV` before startup.
- Frontend nested scroll behavior depends on `frontend/src/lib/scrollAssist.ts`; scroll changes are application behavior, not only CSS.
- Root docs expected by consistency checks are `README.md`, `PROJECT_OVERVIEW.md`, and `CLAUDE.md`.

## Change Checklist

When changing startup or user workflows, update `README.md`.

When changing platform boundaries, routes, stage ownership, or data contracts, update `PROJECT_OVERVIEW.md` and `CLAUDE.md`.

When changing implementation assumptions, known pitfalls, test commands, or agent instructions, update `CLAUDE.md`.

When changing Stage 2-4 artifacts or cleanup logic, check manifests, indexes, file catalog, and router/training consumers together.
