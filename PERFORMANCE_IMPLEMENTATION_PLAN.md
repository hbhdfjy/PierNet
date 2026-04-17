# Performance Implementation Plan

## Goal

Improve frontend responsiveness for large datasets without replacing the core stack or changing Stage 1 HDF5 / Stage 2-4 JSONL source-of-truth formats.

## Guardrails

- Keep FastAPI, React/Vite, HDF5, and JSONL as the current system baseline.
- Do not replace source artifact formats before a sidecar read layer is proven.
- Favor additive changes: manifest, index, cache, query replica.
- Every phase must have:
  - measurable baseline
  - explicit validation
  - rollback path
  - documentation updates when contracts change

## Current Bottlenecks

1. `/api/datasets` rescans Stage 3 files to count rows and infer simulator metadata.
2. `/api/stats` recomputes aggregate stats by scanning all Stage 3 JSONL files.
3. `/api/router/status` scans large router outputs, including `train.jsonl`.
4. `/api/samples`, `/api/files/templates/{scenario}/items`, and `/api/router/samples` paginate by rescanning from file start.
5. `DatasetStats.tsx` currently issues three heavyweight requests and had periodic auto-refresh enabled.

## Execution Order

### Phase 0: Baseline And Measurement

Status: completed

Actions:
- Record P50/P95 latency for:
  - `/api/datasets`
  - `/api/stats`
  - `/api/samples`
  - `/api/files/templates`
  - `/api/files/templates/{scenario}/items`
  - `/api/router/status`
  - `/api/router/samples`
- Log request duration, file touched, and cache-hit state for heavy endpoints.
- Freeze representative dataset sizes for regression checks.

Validation:
- A baseline table exists before any backend read-path refactor.

Rollback:
- Remove instrumentation only.

### Phase 1: Reduce Unnecessary Frontend Reads

Status: completed

Actions:
- Remove periodic auto-refresh from the statistics page.
- Disable focus-triggered revalidation for the statistics page.
- Keep explicit manual refresh as the operator-controlled refresh path.

Validation:
- Statistics page no longer polls every 30 seconds.
- Frontend build passes.

Rollback:
- Restore prior SWR polling options.

### Phase 2: Introduce Sidecar Manifest Layer

Status: completed

Actions:
- Add sidecar manifests for:
  - `data/templates/`
  - `data/text2comp/`
  - `data/router/`
- Manifest fields should minimally include:
  - scenario
  - simulator
  - count
  - file_size_bytes
  - mtime
- Extend manifests with summary breakdowns where applicable:
  - template: language/style
  - sample: language/style/time_mode
  - router: label_counts
- Add a manifest rebuild script for offline regeneration.

Validation:
- Manifest files can be rebuilt from source artifacts.
- Manifest summaries match sampled ground truth.

Rollback:
- Keep manifests unused and continue serving from direct scans.

### Phase 3: Switch Summary Endpoints To Manifest Reads

Status: completed

Actions:
- Move these endpoints to manifest-backed reads:
  - `/api/datasets`
  - `/api/stats`
  - `/api/files/templates`
  - `/api/files/samples`
  - `/api/router/status`
- Keep an emergency fallback path to legacy scans.

Validation:
- Summary endpoints no longer open large JSONL files in the common path.
- Endpoint latency drops relative to Phase 0 baselines.

Rollback:
- Feature flag or code-path switch back to legacy scanners.

### Phase 4: Add Offset Indexes For Pagination

Status: completed

Actions:
- Build sparse byte-offset indexes for large JSONL files.
- Use indexes for:
  - `/api/samples`
  - `/api/files/templates/{scenario}/items`
  - `/api/router/samples`
- Seek to nearest stored offset and scan only local page windows.

Validation:
- Deep-page latency no longer scales linearly with page number in unfiltered mode.

Rollback:
- If index is missing or invalid, fall back to legacy pagination scans.

### Phase 5: Add Lightweight Filter Indexes

Status: completed

Actions:
- Add side indexes for high-frequency filters:
  - language
  - style
  - label
- Start with sparse mapping or row-range metadata instead of full inverted indexes.

Validation:
- Filtered pagination no longer requires full-file scans for common filters.

Rollback:
- Fallback to unindexed scan path for unsupported filters.

### Phase 6: Add Dashboard Summary Endpoint

Status: completed

Actions:
- Add a single dashboard summary endpoint that returns:
  - stats summary
  - dataset summary
  - router summary
- Update `DatasetStats.tsx` to consume one summary request instead of three heavyweight requests.

Validation:
- Statistics page request count drops on initial load.

Rollback:
- Restore the prior three-request frontend pattern.

### Phase 7: Evaluate SQLite Query Replica

Status: planned

Actions:
- Only if manifest + indexes are still insufficient.
- Keep JSONL as source artifacts.
- Add SQLite as a query-side replica for Stage 3 and Stage 4 interactive reads.

Validation:
- Query latency stabilizes under large data growth.

Rollback:
- Route reads back to manifest + index mode.

### Phase 8: Hardening And Deployment

Status: planned

Actions:
- Persist long-running job metadata.
- Improve service startup and recovery.
- Consolidate environment definitions.

Validation:
- Service recovery and long-running operations become predictable.

Rollback:
- Operate with current startup flow.

## Mandatory Documentation Sync

When the implementation changes user-visible behavior, runtime contracts, or architecture, review and update:

- `README.md`
- `PROJECT_OVERVIEW.md`
- `CLAUDE.md`
- `UPGRADE_PLAN.md` when roadmap priorities change

## Immediate Next Steps

1. Reassess whether Phase 7 is still warranted after the current manifest/index/dashboard gains.
2. If Phase 7 stays deferred, scope Phase 8 down to job persistence and service startup hardening only.
3. Keep the benchmark JSON artifacts under `.runlogs/` as the regression baseline for future changes.
