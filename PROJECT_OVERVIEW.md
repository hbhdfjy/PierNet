# PiERN Project Overview

## Role Of This Document

This file is the maintained high-level overview for PiERN.

- Read this first to understand the system boundary.
- Update this file when platform structure, runtime surfaces, stage ownership, or core contracts change.
- Do not turn this file into a changelog or implementation notebook.

## One-Sentence Summary

PiERN is a dual-surface application that combines a Stage 1-4 data synthesis platform with a single-GPU Token Router training platform inside one repository and one deployable FastAPI + React app.

## Product Surfaces

### Landing

- Frontend route: `/`
- Purpose: top-level entry page that routes operators into synth vs training
- Implementation: `frontend/src/platform/LandingPage.tsx`

### Data Synthesis Platform

- Frontend route prefix: `/synth/*`
- Frontend namespace: `frontend/src/synth/`
- Backend namespace: `piern/synth/`
- Scope: Stage 1 simulation, Stage 2 registration/template generation, Stage 3 sample filling, Stage 4 router dataset construction

### Training Platform

- Frontend route prefix: `/training/*`
- Frontend namespace: `frontend/src/training/`
- Backend namespace: `piern/training/`
- Scope: Token Router training data selection, single-GPU training jobs, logs, curves, checkpoints

## Backend Assembly

### Unified Entry Point

- Runtime entry: `piern/api/main.py`
- Compatibility entry: `api_server.py`

`piern/api/main.py` mounts:

- synth routers from `piern.synth.api.routers`
- training router from `piern.training.api.routers.training`
- frontend static hosting through `piern.shared.api.static.SPAStaticFiles`

### API Namespace Boundary

`piern/api/` is now only the unified app assembly namespace. Business routers,
schemas, and services live in the platform namespaces:

- `piern/synth/api/*` and `piern/synth/services/*`
- `piern/training/api/*` and `piern/training/services/*`

## Core Workflow

### Stage 1: Physical Simulation

- Input: simulator-specific YAML under `configs/{simulator}/variants/`
- Execution: `python -m piern.simulators.{simulator}.pipeline`
- Output: `data/{simulator}/{simulator}_{scenario}.h5`

### Stage 2: Registration And Template Generation

- Registry contract: `configs/text2comp/registry.yaml`
- Default config: `configs/text2comp/default.yaml`
- Output: `data/templates/{scenario}_templates.jsonl`
- Read path acceleration: `data/.manifests/templates.json` + `data/.indexes/`

### Stage 3: Sample Filling

- Output: `data/text2comp/{scenario}.jsonl`
- Optional merged artifact: `data/text2comp/all_training_data.jsonl`
- Read path acceleration: `data/.manifests/samples.json` + `data/.indexes/`

### Stage 4: Router Data Construction

- Output: `data/router/train.jsonl` and `data/router/by_scenario/*.jsonl`
- Read path acceleration: `data/.manifests/router.json` + `data/.indexes/`

## Training Workflow

The current training platform consumes Stage 4 outputs and builds prepared caches under `artifacts/token_router/`.

Current training assumptions:

- model family: Token Router only
- device mode: single GPU only
- model: `FullSeqDilatedConvRouter`
- split: `train / test` only
- chat template: Qwen chat format in Stage 4 router data
- embedding backbone: default local Qwen snapshot at `/data/models/Qwen/Qwen2.5-0.5B-Instruct`
- training input: router JSONL context is dynamically tokenized during training and passed through the frozen pretrained embedding table; embeddings are not stored offline

Primary implementation files:

- `piern/training/router/pretrained_embeddings.py`
- `piern/training/router/model.py`
- `piern/training/router/data.py`
- `piern/training/router/train.py`
- `scripts/router/train_token_router.py`

## Supported Simulators

| Simulator | Domain | Math Type | Output Shape | Scenario Count |
| --- | --- | --- | --- | --- |
| `modflow` | Groundwater | Parabolic PDE | `(5, 365)` | 7 |
| `simpeg` | Geophysics | Elliptic PDE | `(1, 100)` | 4 |
| `power_flow` | Steady-state power flow | Nonlinear algebraic system | `(43, 365)` | 5 |
| `transient` | Transient stability | DAE | `(5, 1000)` | 3 |
| `gcam` | Energy-climate planning | Dynamic algebraic / LP | `(5, 16)` | 3 |

## Runtime Surfaces

- `frontend/src/platform/`
  Landing page and top-level platform routing
- `frontend/src/synth/`
  Data synthesis frontend surface
- `frontend/src/training/`
  Training frontend surface
- `frontend/src/shared/`
  Shared frontend theme layer
- `frontend/src/lib/`
  Shared frontend runtime utilities such as API client, types, utils, scroll behavior
- `piern/synth/`
  Synthesis backend routers, schemas, and services
- `piern/training/`
  Training backend routers, schemas, services, and router training core
- `piern/shared/`
  Shared backend infrastructure such as paths and static hosting
- `piern/simulators/`
  Stage 1 simulator implementations
- `piern/synth/text2comp/`
  Stage 2/3 registration, template generation, and sample filling

## Key Project Contracts

- HDF5 naming: `{simulator}_{scenario}.h5`
- Data root: generated data remains under `data/`
- Stage 2 templates: `data/templates/`
- Stage 3 samples: `data/text2comp/`
- Stage 4 router data: `data/router/`
- Read acceleration layer:
  - manifests under `data/.manifests/`
  - sparse indexes under `data/.indexes/`
- Dashboard summary endpoint: `/api/dashboard/summary`
- Training artifacts: `artifacts/token_router/`
- Training API prefix: `/api/training/*`

## Operational Notes

- `start_ui.sh` is the main local startup path for backend + Vite dev server.
- The unified app can serve built frontend assets from `8000`; Vite dev remains on `5173`.
- Frontend nested scroll behavior is currently mediated by `frontend/src/lib/scrollAssist.ts`; scroll changes should be treated as application behavior, not just styling.
- Training and synthesis are separated at the product surface and namespace level, but still share one deployed app and one repository.

## Documentation Set To Maintain

The maintained documentation set is:

1. `PROJECT_OVERVIEW.md`
   High-level system boundary and product overview.
2. `README.md`
   User-facing install, startup, and quick-start guide.
3. `CLAUDE.md`
   Developer and coding-agent implementation context.

Historical plan documents have been removed; durable facts should be kept in
this file, `README.md`, or `CLAUDE.md`.
