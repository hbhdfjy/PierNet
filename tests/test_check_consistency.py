from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci import check_consistency  # noqa: E402


def test_frontend_api_prefixes_detects_base_and_get_calls() -> None:
    api_text = """
const BASE = '/api'
const api = {
  getDatasets: () => get('/datasets'),
  getStatus: (jobId: string) => get(`/generate/${jobId}/status`),
  saveConfig: () => apiFetch(`${BASE}/llm-config`, { method: 'POST' }),
  removeAsset: (assetId: string) => apiFetch(`${BASE}/files/catalog/assets/${assetId}`, { method: 'DELETE' }),
  upload: () => new URL(`${BASE}/simulation/upload`, window.location.origin),
}
"""

    assert check_consistency.frontend_api_prefixes(api_text) == {
        "datasets",
        "files",
        "generate",
        "llm-config",
        "simulation",
    }


def test_api_alignment_warns_for_missing_base_prefix(monkeypatch, tmp_path: Path) -> None:
    frontend_dir = tmp_path / "frontend" / "src" / "lib"
    synth_router_dir = tmp_path / "PierNet" / "synth" / "api" / "routers"
    training_router_dir = tmp_path / "PierNet" / "training" / "api" / "routers"
    frontend_dir.mkdir(parents=True)
    synth_router_dir.mkdir(parents=True)
    training_router_dir.mkdir(parents=True)

    (frontend_dir / "api.ts").write_text(
        """
const BASE = '/api'
export const api = {
  known: () => get('/known'),
  missing: () => apiFetch(`${BASE}/missing/item`),
}
""",
        encoding="utf-8",
    )
    (synth_router_dir / "known.py").write_text(
        'from fastapi import APIRouter\nrouter = APIRouter()\n@router.get("/known")\ndef known():\n    return {}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_api_alignment()

    assert checker.warnings == ["WARN: frontend calls /api/missing/... but no matching router prefix was found"]
    assert checker.info == []


def test_frontend_api_path_templates_normalize_dynamic_paths() -> None:
    api_text = """
const BASE = '/api'
const api = {
  getStatus: (jobId: string) => get(`/generate/${encodeURIComponent(jobId)}/status`),
  stop: (jobId: string) => apiFetch(`${BASE}/generate/${jobId}`, { method: 'DELETE' }),
  upload: () => new URL(`${BASE}/simulation/upload?overwrite=${overwrite}`, window.location.origin),
}
"""

    assert check_consistency.frontend_api_path_templates(api_text) == {
        "/generate/{param}",
        "/generate/{param}/status",
        "/simulation/upload",
    }


def test_frontend_api_method_path_templates_include_methods_and_url_variables() -> None:
    api_text = """
const BASE = '/api'
const api = {
  getDatasets: () => get('/datasets'),
  stop: (jobId: string) => apiFetch(`${BASE}/generate/${jobId}`, { method: 'DELETE' }),
  stream: (jobId: string) => new EventSource(`${BASE}/generate/${jobId}/stream`),
  upload: (file: File) => {
    const url = new URL(`${BASE}/simulation/upload?overwrite=${overwrite}`, window.location.origin)
    return apiFetch(url.toString(), { method: 'POST', body: file })
  },
}
"""

    assert check_consistency.frontend_api_method_path_templates(api_text) == {
        ("DELETE", "/generate/{param}"),
        ("GET", "/datasets"),
        ("GET", "/generate/{param}/stream"),
        ("POST", "/simulation/upload"),
    }


def test_frontend_openapi_path_check_warns_for_missing_nested_path(monkeypatch, tmp_path: Path) -> None:
    frontend_dir = tmp_path / "frontend" / "src" / "lib"
    generated_dir = frontend_dir / "generated"
    generated_dir.mkdir(parents=True)
    (frontend_dir / "api.ts").write_text(
        """
const BASE = '/api'
export const api = {
  ok: () => get('/datasets'),
  missing: (id: string) => apiFetch(`${BASE}/files/${id}/bad`, { method: 'DELETE' }),
}
""",
        encoding="utf-8",
    )
    (generated_dir / "openapi.json").write_text(
        '{"paths": {"/api/datasets": {"get": {}}, "/api/files/{asset_id}": {"delete": {}}}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_frontend_openapi_paths()

    assert checker.warnings == [
        "WARN: frontend calls DELETE /api/files/{param}/bad but generated OpenAPI has no matching operation"
    ]


def test_frontend_openapi_path_check_warns_for_method_mismatch(monkeypatch, tmp_path: Path) -> None:
    frontend_dir = tmp_path / "frontend" / "src" / "lib"
    generated_dir = frontend_dir / "generated"
    generated_dir.mkdir(parents=True)
    (frontend_dir / "api.ts").write_text(
        """
const BASE = '/api'
export const api = {
  wrong: () => apiFetch(`${BASE}/datasets`, { method: 'POST' }),
}
""",
        encoding="utf-8",
    )
    (generated_dir / "openapi.json").write_text('{"paths": {"/api/datasets": {"get": {}}}}', encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_frontend_openapi_paths()

    assert checker.warnings == ["WARN: frontend calls POST /api/datasets but generated OpenAPI has no matching operation"]


def test_frontend_import_graph_errors_for_orphan_production_file(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "frontend" / "src"
    src.mkdir(parents=True)
    (src / "main.tsx").write_text("import './used'\nimport('./lazy')\n", encoding="utf-8")
    (src / "used.ts").write_text("import './nested'\n", encoding="utf-8")
    (src / "nested.ts").write_text("export const nested = true\n", encoding="utf-8")
    (src / "lazy.tsx").write_text("export default function Lazy() { return null }\n", encoding="utf-8")
    (src / "orphan.ts").write_text("export const orphan = true\n", encoding="utf-8")
    (src / "orphan.test.ts").write_text("export const testOnly = true\n", encoding="utf-8")
    (src / "generated").mkdir()
    (src / "generated" / "openapi.d.ts").write_text("export type Generated = string\n", encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_frontend_import_graph()

    assert checker.errors == [
        "ERROR: frontend production file is not reachable from main.tsx: frontend/src/orphan.ts"
    ]


def test_frontend_import_graph_accepts_secondary_html_entry(monkeypatch, tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    src = frontend / "src"
    secondary = src / "secondary"
    secondary.mkdir(parents=True)
    (frontend / "index.html").write_text(
        '<script type="module" src="/src/main.tsx"></script>\n',
        encoding="utf-8",
    )
    (frontend / "secondary.html").write_text(
        '<script type="module" src="/src/secondary/main.tsx"></script>\n',
        encoding="utf-8",
    )
    (src / "main.tsx").write_text("export const main = true\n", encoding="utf-8")
    (secondary / "main.tsx").write_text("import './workflow'\n", encoding="utf-8")
    (secondary / "workflow.ts").write_text("export const workflow = true\n", encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_frontend_import_graph()

    assert checker.errors == []


def test_stale_setup_py_warns_but_documented_note_does_not(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("README\n", encoding="utf-8")
    (tmp_path / "PROJECT_OVERVIEW.md").write_text("overview\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("Do not reintroduce `setup.py`.\n", encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_stale_files()
    checker.check_docs_structure()

    assert "WARN: stale path still exists: setup.py" in checker.warnings
    assert "WARN: CLAUDE.md still references stale path: setup.py" not in checker.warnings


def test_text_encoding_check_reports_garbled_text(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "frontend" / "src"
    src.mkdir(parents=True)
    suspicious = "?" * 4
    (src / "bad.tsx").write_text(f'const label = "{suspicious}"\n', encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_text_encoding()

    assert "ERROR: text encoding marker: frontend/src/bad.tsx:1: suspicious-question-run" in checker.errors


def test_markdown_local_link_issues_report_missing_targets(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ok.md").write_text("ok\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[ok](docs/ok.md) [missing](docs/missing.md) [external](https://example.com)\n",
        encoding="utf-8",
    )

    issues = check_consistency.markdown_local_link_issues(tmp_path)

    assert issues == ["README.md links to missing local target: docs/missing.md"]


def test_markdown_local_link_issues_skip_runtime_artifact_dirs(tmp_path: Path) -> None:
    result_dir = tmp_path / "frontend" / "test-results" / "failed-test"
    result_dir.mkdir(parents=True)
    (result_dir / "error-context.md").write_text("[missing](missing.png)\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("README\n", encoding="utf-8")

    issues = check_consistency.markdown_local_link_issues(tmp_path)

    assert issues == []


def test_removed_docs_and_simulator_requirement_files_warn(monkeypatch, tmp_path: Path) -> None:
    stale_paths = [
        "docs/INDUSTRIALIZATION_PLAN.md",
        "docs/generated-review.docx",
        "PierNet/simulators/modflow/requirements.txt",
        "scripts/utils/rebuild_filter_indexes.py",
    ]
    for rel in stale_paths:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_stale_files()

    assert "WARN: stale path still exists: docs/INDUSTRIALIZATION_PLAN.md" in checker.warnings
    assert "WARN: stale path still exists: PierNet/simulators/modflow/requirements.txt" in checker.warnings
    assert "WARN: stale path still exists: scripts/utils/rebuild_filter_indexes.py" in checker.warnings
    assert "WARN: stale generated document remains: docs/generated-review.docx" in checker.warnings


def test_docs_structure_errors_for_stale_jsonl_primary_claim(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Stage 2-4 的源产物仍以 JSONL 为主\n", encoding="utf-8")
    (tmp_path / "PROJECT_OVERVIEW.md").write_text("overview\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("notes\n", encoding="utf-8")
    script = tmp_path / "scripts" / "utils" / "rebuild_indexes.py"
    script.parent.mkdir(parents=True)
    script.write_text("JSONL pagination indexes for Stage 2-4 artifacts\n", encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_docs_structure()

    assert "ERROR: README.md has stale storage-format claim: Stage 2-4 的源产物仍以 JSONL 为主" in checker.errors
    assert (
        "ERROR: scripts/utils/rebuild_indexes.py has stale storage-format claim: "
        "JSONL pagination indexes for Stage 2-4 artifacts"
    ) in checker.errors


def test_docs_structure_errors_for_stale_api_server_stage2_claim(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("README\n", encoding="utf-8")
    (tmp_path / "PROJECT_OVERVIEW.md").write_text("overview\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("notes\n", encoding="utf-8")
    (tmp_path / "api_server.py").write_text("PierNet Stage 2 API 入口\n", encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_docs_structure()

    assert "ERROR: api_server.py describes the unified API as a stale Stage 2-only entrypoint" in checker.errors


def test_registry_check_errors_for_invalid_observation_config(monkeypatch, tmp_path: Path) -> None:
    registry_path = tmp_path / "configs" / "text2comp" / "registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        """
demo:
  domain_context: demo
  output_description: demo {ch} {ts}
  param_info: {}
  output_info:
    - name: output
      description: output
      unit: '-'
      slice: [0, null]
  observation_config:
    channel_level: row
    fixed_channels: []
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_registry()

    assert any(item.startswith("ERROR: registry[demo] invalid:") for item in checker.errors)


def test_simulator_default_config_check_errors_for_missing_path(monkeypatch, tmp_path: Path) -> None:
    pipeline = tmp_path / "PierNet" / "simulators" / "modflow" / "pipeline.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text('parser.add_argument("--config", default="configs/missing.yaml")\n', encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_simulator_default_configs()

    assert (
        "ERROR: PierNet/simulators/modflow/pipeline.py references missing default config: configs/missing.yaml"
        in checker.errors
    )


def test_frontend_node_runtime_check_rejects_stale_stage2_package_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "frontend").mkdir()
    (tmp_path / "scripts" / "frontend").mkdir(parents=True)
    (tmp_path / "frontend" / "package.json").write_text(
        '{"name": "PierNet-stage2-ui", "engines": {"node": ">=20.19.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "frontend" / "package-lock.json").write_text(
        '{"name": "PierNet-other-ui", "packages": {"": {"name": "PierNet-other-ui"}}}',
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "frontend" / "run-vite.mjs").write_text(
        "const MIN_NODE_LABEL = '20.19.0'\nprocess.env.PierNet_NODE_BIN\n",
        encoding="utf-8",
    )
    (tmp_path / "start_ui.sh").write_text(
        'MIN_NODE_VERSION="20.19.0"\nPierNet_NODE_BIN:-${PierNet_NODE:-}\nPierNet_NPM:-npm\n',
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("要求 Python 3.11 和 Node.js 20.19.0+。\n", encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_frontend_node_runtime()

    assert "ERROR: frontend/package.json package name still uses stale Stage 2 identity" in checker.errors
    assert "ERROR: frontend/package-lock.json package name does not match frontend/package.json" in checker.errors


def test_frontend_node_runtime_check_requires_script_and_doc_alignment(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "frontend").mkdir()
    (tmp_path / "scripts" / "frontend").mkdir(parents=True)
    (tmp_path / "frontend" / "package.json").write_text(
        '{"engines": {"node": ">=20.19.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "frontend" / "run-vite.mjs").write_text(
        "const MIN_NODE_LABEL = '18.0.0'\n",
        encoding="utf-8",
    )
    (tmp_path / "start_ui.sh").write_text('MIN_NODE_VERSION="20.19.0"\n', encoding="utf-8")
    (tmp_path / "README.md").write_text("要求 Python 3.11 和 Node.js 20.19.0+。\n", encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_frontend_node_runtime()

    assert "ERROR: scripts/frontend/run-vite.mjs does not mirror frontend Node minimum 20.19.0" in checker.errors
    assert "ERROR: scripts/frontend/run-vite.mjs does not mirror PierNet_NODE_BIN support 20.19.0" in checker.errors


def test_frontend_node_runtime_check_does_not_report_ok_when_node_bin_support_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "frontend").mkdir()
    (tmp_path / "scripts" / "frontend").mkdir(parents=True)
    (tmp_path / "frontend" / "package.json").write_text(
        '{"engines": {"node": ">=20.19.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "frontend" / "run-vite.mjs").write_text(
        "const MIN_NODE_LABEL = '20.19.0'\n",
        encoding="utf-8",
    )
    (tmp_path / "start_ui.sh").write_text('MIN_NODE_VERSION="20.19.0"\n', encoding="utf-8")
    (tmp_path / "README.md").write_text("要求 Python 3.11 和 Node.js 20.19.0+。\n", encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_frontend_node_runtime()

    assert "ERROR: scripts/frontend/run-vite.mjs does not mirror PierNet_NODE_BIN support 20.19.0" in checker.errors
    assert "OK: frontend Node runtime minimum: 20.19.0" not in checker.info


def test_frontend_node_runtime_check_rejects_public_node_bin_dir_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "frontend").mkdir()
    (tmp_path / "scripts" / "frontend").mkdir(parents=True)
    (tmp_path / "frontend" / "package.json").write_text(
        '{"engines": {"node": ">=20.19.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "frontend" / "run-vite.mjs").write_text(
        "const MIN_NODE_LABEL = '20.19.0'\nprocess.env.PierNet_NODE_BIN\n",
        encoding="utf-8",
    )
    (tmp_path / "start_ui.sh").write_text(
        'MIN_NODE_VERSION="20.19.0"\nPierNet_NODE_BIN:-${PierNet_NODE:-}\nPierNet_NPM:-npm\n',
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("要求 Python 3.11 和 Node.js 20.19.0+。\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("PierNet_NODE=/old/node\nPierNet_NODE_BIN_DIR=/old/path\n", encoding="utf-8")
    (tmp_path / "compose.yaml").write_text("PierNet_NODE: /old/node\nPierNet_NODE_BIN_DIR: /old/path\n", encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_frontend_node_runtime()

    assert "ERROR: .env.example exposes legacy PierNet_NODE; use PierNet_NODE_BIN instead" in checker.errors
    assert "ERROR: .env.example exposes legacy PierNet_NODE_BIN_DIR; use PierNet_NODE_BIN instead" in checker.errors
    assert "ERROR: compose.yaml exposes legacy PierNet_NODE; use PierNet_NODE_BIN instead" in checker.errors
    assert "ERROR: compose.yaml exposes legacy PierNet_NODE_BIN_DIR; use PierNet_NODE_BIN instead" in checker.errors


def test_python_entrypoint_check_errors_for_missing_targets(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "PierNet" / "ok").mkdir(parents=True)
    (tmp_path / "PierNet" / "ok" / "tool.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    (tmp_path / "PierNet" / "ok" / "bad_func.py").write_text("def other():\n    return 0\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "demo"

[project.scripts]
ok-tool = "PierNet.ok.tool:main"
missing-module = "PierNet.missing.tool:main"
missing-func = "PierNet.ok.bad_func:main"
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("python -m PierNet.docs.missing\n", encoding="utf-8")

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_python_entrypoints()

    assert "ERROR: pyproject script missing-module references missing module: PierNet.missing.tool" in checker.errors
    assert "ERROR: pyproject script missing-func references missing function: PierNet.ok.bad_func:main" in checker.errors
    assert "ERROR: documented python -m module is missing: PierNet.docs.missing" in checker.errors


def test_compose_model_mount_default_matches_repo_model_path(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "models" / "Qwen" / "Qwen2.5-0.5B-Instruct").mkdir(parents=True)
    (tmp_path / "compose.yaml").write_text(
        "${PierNet_QWEN_EMBEDDING_MODEL:-./models/Qwen/Qwen2.5-0.5B-Instruct}:/models/qwen:ro\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_compose_model_mount_defaults()

    assert checker.errors == []
    assert checker.info == ["OK: compose.yaml Qwen model mount default matches local model path"]


def test_compose_model_mount_default_rejects_missing_qwen_directory(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "compose.yaml").write_text(
        "${PierNet_QWEN_EMBEDDING_MODEL:-./models/Qwen2.5-0.5B-Instruct}:/models/qwen:ro\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_compose_model_mount_defaults()

    assert (
        "ERROR: compose.yaml Qwen model mount default must be ./models/Qwen/Qwen2.5-0.5B-Instruct"
        in checker.errors
    )


def test_container_runtime_assets_requires_configs_copy(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text(
        "\n".join(
            [
                "COPY PierNet ./PierNet",
                "COPY scripts ./scripts",
                "COPY api_server.py ./",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_container_runtime_assets()

    assert (
        "ERROR: Dockerfile does not copy runtime asset(s): configs (text2comp and simulator configs)"
        in checker.errors
    )


def test_container_runtime_assets_accepts_required_copies(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text(
        "\n".join(
            [
                "COPY PierNet ./PierNet",
                "COPY scripts ./scripts",
                "COPY configs ./configs",
                "COPY api_server.py ./",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_container_runtime_assets()

    assert checker.errors == []
    assert checker.info == ["OK: Dockerfile copies required runtime assets"]


def test_dependency_metadata_checks_requirements_and_pyproject(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
dependencies = ["numpy>=1", "pillow>=9"]

[project.optional-dependencies]
dev = ["pytest>=7"]
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text(
        """
numpy>=1
pytest>=7
pandas>=1
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(check_consistency, "ROOT", tmp_path)
    checker = check_consistency.Checker()

    checker.check_dependency_metadata()

    assert "ERROR: requirements.txt dependencies missing from pyproject metadata: ['pandas']" in checker.errors
    assert "ERROR: pyproject runtime dependencies missing from requirements.txt: ['pillow']" in checker.errors
