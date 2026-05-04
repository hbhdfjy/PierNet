# PiERN

PiERN is a dual-surface system for physics and engineering time-series data.

- `/synth`: Stage 1-4 data synthesis workbench
- `/training`: single-GPU Token Router training workbench

The repository is still one deployable FastAPI + React application. The product surfaces and code ownership are separated by namespace, but they share startup, static hosting, theme, and a small amount of infrastructure.

## Document Map

Keep durable project facts in these root documents:

- `README.md`: installation, startup, quick start, and operational commands
- `PROJECT_OVERVIEW.md`: system boundary, architecture, and data contracts
- `CLAUDE.md`: implementation notes for developers and coding agents

Historical plan documents are not the source of truth.

## Runtime Entrypoints

```text
Landing   http://localhost:8000/
Synth     http://localhost:8000/synth
Training  http://localhost:8000/training
Files     http://localhost:8000/files
API Docs  http://localhost:8000/docs
Vite Dev  http://localhost:5173/
```

Port `8000` is FastAPI. It also serves built frontend assets when `frontend/dist` exists. Port `5173` is the Vite development server.

## Install

Python 3.11 is expected.

```bash
pip install -r requirements.txt
pip install -e .
```

Frontend dependencies:

```bash
cd frontend
npm install
```

`requirements.txt` is the main dependency entry. `setup.py` mirrors the package dependencies and console scripts, but should not be treated as the only source of truth.

### Conda Environment Note

`start_ui.sh` defaults to `/home/fjy/miniconda3/envs/piern-project`. If your conda environment has another name, point the script at it explicitly:

```bash
export PIERN_CONDA_BASE=/home/fjy/miniconda3
export PIERN_CONDA_ENV=/home/fjy/miniconda3/envs/piern
```

Then run the startup command in the same shell.

### MODFLOW Binary

MODFLOW scenarios need an `mf2005` executable. One installation path is:

```bash
python - <<'PY2'
from pathlib import Path
from flopy.utils import get_modflow
get_modflow(str(Path.home() / '.flopy_bin'), subset='mf2005')
PY2
```

Then set:

```bash
export PIERN_MODFLOW_EXE=/path/to/mf2005
```

## Start The App

```bash
./start_ui.sh
./start_ui.sh --dev
```

The script starts:

- FastAPI on `0.0.0.0:8000`
- Vite on `0.0.0.0:5173`

Manual backend start:

```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Manual frontend start:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --strictPort
```

## Product Scope

### `/synth`

The synthesis platform covers:

1. Stage 1 physical simulation or HDF5 upload
2. Stage 2 registry metadata and LLM template generation
3. Stage 3 local sample filling
4. Stage 4 Token Router dataset construction

Main pages:

- data overview
- physical simulation
- HDF5 upload
- scene registration
- template generation
- sample filling
- router data build
- template/sample/router viewers
- file manager
- registry editor
- LLM config

### `/training`

The training platform is intentionally narrow:

- Token Router only
- single GPU only
- no DDP
- no general model-training platform

It supports dataset selection, GPU status, job creation, job list, logs, curves, checkpoints, stop, and delete.

## Supported Stage 1 Simulators

| Simulator | Domain | Math Type | Output Shape | Scenarios |
| --- | --- | --- | --- | --- |
| `modflow` | Groundwater | Parabolic PDE | `(5, 365)` | 7 |
| `simpeg` | Geophysics | Elliptic PDE | `(1, 100)` | 4 |
| `power_flow` | Steady-state power flow | Nonlinear algebraic system | `(43, 365)` | 5 |
| `transient` | Transient stability | DAE | `(5, 1000)` | 3 |
| `gcam` | Energy-climate planning | Dynamic algebraic / LP | `(5, 16)` | 3 |

Scenario configs live under `configs/{simulator}/variants/`.

## Data Contracts

### Stage 1 HDF5

Canonical generated files:

```text
data/{simulator}/{simulator}_{scenario}.h5
```

Uploaded external simulator or big-scene files:

```text
data/{big_scene}/{big_scene}_{scenario}.h5
```

Registration enforces the HDF5 contract:

- `timeseries`: numeric 3D `[N, C, T]`
- `params`: numeric 2D `[N, P]`
- `param_names`: string 1D `[P]`
- root attrs `n_samples`, `n_channels`, `n_timesteps`, `n_params` must match shapes
- `timeseries` and `params` must be finite, with no NaN or Inf

### Stage 2 Templates

```text
data/templates/{scenario}_templates.jsonl
```

### Stage 3 Samples

```text
data/text2comp/{scenario}.jsonl
data/text2comp/all_training_data.jsonl
```

### Stage 4 Router Data

```text
data/router/by_scenario/{scenario}.jsonl
data/router/train.jsonl
```

### Training Artifacts

```text
artifacts/token_router/{simulator}/prepared/{prepared_name}/
artifacts/token_router/{simulator}/runs/{run_name}/
```

## Quick Start CLI

### Stage 1

```bash
python -m piern.simulators.modflow.pipeline \
  --config configs/modflow/variants/unified_aquifer.yaml \
  --n-samples 1000

python -m piern.simulators.simpeg.pipeline \
  --config configs/simpeg/variants/dc_resistivity.yaml \
  --n-samples 1000

python -m piern.simulators.power_flow.pipeline \
  --config configs/power_flow/variants/ieee14_baseload.yaml \
  --n-samples 1000

python -m piern.simulators.transient.pipeline \
  --config configs/transient/variants/ieee14_fault.yaml \
  --n-samples 500

python -m piern.simulators.gcam.pipeline \
  --config configs/gcam/variants/energy_transition.yaml \
  --n-samples 1000
```

### Stage 2

```bash
python -m piern.synth.text2comp.auto_register \
  --config configs/text2comp/default.yaml \
  --output configs/text2comp/registry.yaml

python scripts/text2comp/generate_templates.py \
  --config configs/text2comp/default.yaml \
  --n-templates 1000
```

Template generation calls the configured LLM provider from `configs/text2comp/default.yaml` or the `/synth/llm-config` UI.

### Stage 3

```bash
python scripts/text2comp/fill_samples.py \
  --config configs/text2comp/default.yaml \
  --n-samples 1000 \
  --skip-existing
```

Stage 3 is local deterministic filling; it does not call an LLM.

### Stage 4

```bash
python scripts/router/build_router_data.py --seed 42
python scripts/router/build_router_data.py --seed 42 --chat-template qwen --neg-ratio 2
```

Stage 4 emits Qwen-chat-template contexts for binary Token Router training.

### Token Router Training

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/router/train_token_router.py \
  --simulator modflow \
  --device cuda:0 \
  --epochs 1
```

Current training core:

- model: `FullSeqDilatedConvRouter`
- input representation: Qwen tokenizer + frozen pretrained embedding table
- default embedding backbone: `/data/fjy/Qwen2.5-0.5B-Instruct`
- split: train/test only
- embeddings are looked up during training, not stored offline

## Read Performance Layer

Stage 2-4 source artifacts remain JSONL, but UI reads use sidecar summaries and indexes:

```text
data/.manifests/
data/.indexes/
```

Rebuild manually:

```bash
python scripts/utils/rebuild_manifests.py
python scripts/utils/rebuild_indexes.py
python scripts/utils/rebuild_filter_indexes.py
```

## Repository Layout

```text
api_server.py                  # compatibility entry that exports piern.api.main.app
piern/api/main.py              # unified FastAPI app assembly
piern/core/                    # shared low-level storage, validation, LLM client
piern/shared/                  # static hosting and runtime paths
piern/simulators/              # Stage 1 simulator implementations
piern/synth/                   # synthesis backend and text2comp core
piern/training/                # training API, manager, and Token Router core
frontend/src/platform/         # landing and top-level routing
frontend/src/synth/            # synthesis frontend
frontend/src/training/         # training frontend
frontend/src/files/            # unified file manager surface
scripts/text2comp/             # Stage 2/3 CLI
scripts/router/                # Stage 4 and training CLI
scripts/utils/                 # manifests, indexes, checks, utilities
```

## Verification

Backend syntax check:

```bash
python -m compileall piern scripts api_server.py
```

Frontend build:

```bash
cd frontend
npm run build
```

Targeted tests:

```bash
pytest tests/test_build_router_data_script.py \
  tests/test_hdf5_data_validation.py \
  tests/test_registry_observation_config.py \
  tests/test_router_prepared_inputs.py \
  tests/test_training_manager_fallbacks.py \
  tests/test_check_garbled_text.py
```

Repository consistency check:

```bash
python scripts/ci/check_consistency.py
```

## License

MIT, see `LICENSE`.
