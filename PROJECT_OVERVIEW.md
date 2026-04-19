# PiERN Project Overview

## Role Of This Document

This file is the maintained project-level overview for PiERN.

- Read this first when you need a concise understanding of the system.
- Update this file when the architecture, stage boundaries, supported simulators, or runtime surfaces change.
- Do not use this file as a changelog, troubleshooting notebook, or implementation backlog.

## One-Sentence Summary

PiERN is a multi-simulator data generation pipeline that turns heterogeneous physics and engineering simulators into a unified Stage 1-4 workflow for simulation, registration, template generation, sample filling, and router-data construction.

## Core Workflow

### Stage 1: Physical Simulation

- Input: simulator-specific YAML configs under `configs/{simulator}/variants/`
- Execution: `python -m piern.simulators.{simulator}.pipeline`
- Output: `data/{simulator}/{simulator}_{scenario}.h5`

### Stage 2: Registration And Template Generation

- Registration contract lives in `configs/text2comp/registry.yaml`
- Metadata can be produced by auto-registration and refined in the UI
- Template generation writes scenario template libraries into `data/templates/`
- Interactive summary reads use sidecar manifests under `data/.manifests/templates.json`
- Template pagination uses sparse indexes under `data/.indexes/`, with dedicated filter indexes for `language/style`

### Stage 3: Sample Filling

- Filled training samples are generated from Stage 1 numeric outputs plus Stage 2 templates
- Output lives in `data/text2comp/`
- Dataset lists and aggregate stats are served from `data/.manifests/samples.json` in the common read path
- Sample pagination uses sparse indexes under `data/.indexes/`, with dedicated filter indexes for `language/style`

### Stage 4: Router Data Construction

- Router training data is derived from Stage 3 outputs
- Output lives in `data/router/`
- Router status summaries are served from `data/.manifests/router.json` in the common read path
- Router pagination uses sparse indexes under `data/.indexes/`, with dedicated filter indexes for `label`

## Supported Simulators

| Simulator | Domain | Math Type | Output Shape | Scenario Count |
| --- | --- | --- | --- | --- |
| `modflow` | Groundwater | Parabolic PDE | `(5, 365)` | 7 |
| `simpeg` | Geophysics | Elliptic PDE | `(1, 100)` | 4 |
| `power_flow` | Steady-state power flow | Nonlinear algebraic system | `(43, 365)` | 5 |
| `transient` | Transient stability | DAE | `(5, 1000)` | 3 |
| `gcam` | Energy-climate planning | Dynamic algebraic / LP | `(5, 16)` | 3 |

## Runtime Surfaces
- `frontend/src/platform/`: top-level platform switcher for synth vs training UI
- `frontend/src/synth/`: data synthesis frontend surface
- `frontend/src/training/`: training frontend surface
- `frontend/src/shared/`: shared theme and frontend infrastructure
- `piern/synth/`: synthesis backend routers and services
- `piern/training/`: training backend routers, services, and training core
- `piern/shared/`: shared backend infrastructure such as paths and static hosting
- `piern/simulators/`: Stage 1 simulator pipelines
- `piern/text2comp/`: Stage 2/3 registration, template generation, and filling logic
- `scripts/router/`: Stage 4 router dataset construction and training CLI entrypoints

## Key Project Contracts

- HDF5 naming: `{simulator}_{scenario}.h5`
- Data root: all generated artifacts are currently under `data/`
- Sidecar manifests: `data/.manifests/` is the summary/index layer for Stage 2-4 interactive reads
- Sparse indexes: `data/.indexes/` is the unfiltered pagination layer for large Stage 2-4 JSONL files
- Dashboard summary: `/api/dashboard/summary` is the statistics page aggregation endpoint
- Registry contract: `configs/text2comp/registry.yaml`
- Stage 2/3 default config: `configs/text2comp/default.yaml`

## Documentation Set To Maintain

The maintained documentation set for project overview is:

1. `PROJECT_OVERVIEW.md`
   The source-of-truth high-level system overview.
2. `README.md`
   The user-facing entrypoint for installation, quick start, and common commands.
3. `CLAUDE.md`
   The developer and coding-agent operating context, grounded on the overview above.

`UPGRADE_PLAN.md` remains useful, but it is a roadmap document rather than part of the overview contract.

## Dual Platform Layout
The repository now exposes two clearly separated product surfaces inside one repo and one deployable app:

- Data synthesis platform
  - Frontend namespace: `frontend/src/synth/`
  - Backend namespace: `piern/synth/`
  - Covers Stage 1 simulation, Stage 2 template generation, Stage 3 sample filling, and Stage 4 router dataset construction
- Model training platform
  - Frontend namespace: `frontend/src/training/`
  - Backend namespace: `piern/training/`
  - Frontend route prefix: `/training`
  - Backend API prefix: `/api/training/*`
  - Current scope: single-GPU Token Router training, job management, curve viewing, log viewing

Shared code is intentionally narrowed to `frontend/src/shared/` and `piern/shared/`.
The training platform depends on the synthesis platform only through stable router-data contracts under `data/router/` and `data/.manifests/router.json`, not through Stage 1-3 implementation details.

