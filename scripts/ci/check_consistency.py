#!/usr/bin/env python3
"""Repository consistency checks for PierNet."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
import json
import tomllib
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PierNet.synth.api.routers.registry import _validate_registry_entry  # noqa: E402
from scripts.utils.check_garbled_text import find_garbled_text, iter_files as iter_text_files  # noqa: E402


def _first_api_segment(path: str) -> str:
    return path.lstrip("/").split("/", 1)[0].split("?", 1)[0]


def dependency_name(requirement: str) -> str | None:
    cleaned = requirement.split("#", 1)[0].strip()
    if not cleaned or cleaned.startswith(("-", "--")):
        return None
    match = re.match(r"([A-Za-z0-9_.-]+)", cleaned)
    return match.group(1).lower().replace("_", "-") if match else None


def dependency_names(requirements: list[str]) -> set[str]:
    return {name for item in requirements if (name := dependency_name(item))}


def python_module_path(module: str, *, root: Path | None = None) -> Path | None:
    root = ROOT if root is None else root
    module_rel = Path(*module.split("."))
    py_file = root / module_rel.with_suffix(".py")
    if py_file.exists():
        return py_file
    package_dir = root / module_rel
    for candidate in [package_dir / "__main__.py", package_dir / "__init__.py"]:
        if candidate.exists():
            return candidate
    return None


def python_entrypoint_function_exists(module_path: Path, function_name: str) -> bool:
    text = module_path.read_text(encoding="utf-8")
    return re.search(rf"(?m)^def\s+{re.escape(function_name)}\s*\(", text) is not None


def documented_PierNet_modules(text: str) -> set[str]:
    return set(re.findall(r"python\s+-m\s+(PierNet(?:\.[A-Za-z_][\w]*)+)", text))


def iter_project_markdown_files(root: Path) -> Iterable[Path]:
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files", "*.md"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
    except Exception:
        tracked = []
    if tracked:
        for rel in tracked:
            path = root / rel
            if path.exists():
                yield path
        return

    skip_dirs = {
        ".cache",
        ".conda",
        ".git",
        ".mypy_cache",
        ".node",
        ".pytest_cache",
        ".ruff_cache",
        ".tmp",
        "artifacts",
        "build",
        "data",
        "dist",
        "logs",
        "migration_backup",
        "migration_exports",
        "models",
        "node_modules",
        "outputs",
        "playwright-report",
        "test-results",
        "wandb",
    }
    stack = [root]
    while stack:
        directory = stack.pop()
        for child in sorted(directory.iterdir()):
            if child.is_dir():
                if child.name not in skip_dirs:
                    stack.append(child)
                continue
            if child.suffix == ".md":
                yield child


def markdown_local_link_issues(root: Path) -> list[str]:
    link_re = re.compile(r"(?<!!\[)\[[^\]]+\]\(([^)]+)\)")
    issues: list[str] = []
    resolved_root = root.resolve()
    for path in iter_project_markdown_files(root):
        rel_path = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for match in link_re.finditer(text):
            target = match.group(1).strip().split("#", 1)[0].strip()
            if not target or target.startswith(("#", "http://", "https://", "mailto:", "app://")):
                continue
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            probe = resolved_root / target.lstrip("/") if target.startswith("/") else (path.parent / target).resolve()
            try:
                probe.relative_to(resolved_root)
            except ValueError:
                continue
            if not probe.exists():
                issues.append(f"{rel_path} links to missing local target: {target}")
    return issues


def frontend_api_prefixes(api_text: str) -> set[str]:
    prefixes: set[str] = set()
    for match in re.finditer(r"\bget(?:<[^>]+>)?\(\s*([\"'`])(/[^\"'`]+)\1", api_text):
        prefixes.add(_first_api_segment(match.group(2)))
    for match in re.finditer(r"\$\{BASE\}(/[^\s\"'`)]+)", api_text):
        prefixes.add(_first_api_segment(match.group(1)))
    for match in re.finditer(r"\bBASE\s*\+\s*([\"'`])(/[^\"'`]+)\1", api_text):
        prefixes.add(_first_api_segment(match.group(2)))
    return {prefix for prefix in prefixes if prefix}


def normalize_api_path_template(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith("${BASE}"):
        normalized = normalized[len("${BASE}"):]
    if normalized.startswith("/api"):
        normalized = normalized[4:]
    normalized = normalized.split("?", 1)[0]
    normalized = re.sub(r"\$\{[^}]+\}", "{param}", normalized)
    normalized = re.sub(r"\{[^}/]+\}", "{param}", normalized)
    normalized = normalized.rstrip("/")
    return normalized or "/"


def frontend_api_path_templates(api_text: str) -> set[str]:
    paths: set[str] = set()
    for match in re.finditer(r"\bget(?:<[^>]+>)?\(\s*([\"'`])([^\"'`]+)\1", api_text):
        raw_path = match.group(2)
        if raw_path.startswith("/"):
            paths.add(normalize_api_path_template(raw_path))
    for match in re.finditer(r"\$\{BASE\}([^`\"']+)", api_text):
        paths.add(normalize_api_path_template("${BASE}" + match.group(1)))
    for match in re.finditer(r"\bBASE\s*\+\s*([\"'`])([^\"'`]+)\1", api_text):
        raw_path = match.group(2)
        if raw_path.startswith("/"):
            paths.add(normalize_api_path_template(raw_path))
    return {path for path in paths if path.startswith("/")}


def _method_from_fetch_init(init_text: str | None) -> str:
    if not init_text:
        return "GET"
    match = re.search(r"\bmethod\s*:\s*([\"'])([A-Za-z]+)\1", init_text)
    return match.group(2).upper() if match else "GET"


def frontend_api_method_path_templates(api_text: str) -> set[tuple[str, str]]:
    requests: set[tuple[str, str]] = set()
    url_vars: dict[str, str] = {}

    for match in re.finditer(r"\bget(?:<[^>]+>)?\(\s*([\"'`])([^\"'`]+)\1", api_text):
        raw_path = match.group(2)
        if raw_path.startswith("/"):
            requests.add(("GET", normalize_api_path_template(raw_path)))

    for match in re.finditer(r"\bnew\s+EventSource\(\s*([\"'`])([^\"'`]+)\1", api_text):
        raw_path = match.group(2)
        if raw_path.startswith("${BASE}") or raw_path.startswith("/api"):
            requests.add(("GET", normalize_api_path_template(raw_path)))

    for match in re.finditer(
        r"\bconst\s+([A-Za-z_]\w*)\s*=\s*new\s+URL\(\s*([\"'`])([^\"'`]+)\2",
        api_text,
    ):
        raw_path = match.group(3)
        if raw_path.startswith("${BASE}") or raw_path.startswith("/api"):
            url_vars[match.group(1)] = normalize_api_path_template(raw_path)

    for match in re.finditer(
        r"\bapiFetch\(\s*([\"'`])([^\"'`]+)\1\s*(?:,\s*(\{.*?\}))?\s*\)",
        api_text,
        re.DOTALL,
    ):
        raw_path = match.group(2)
        if raw_path.startswith("${BASE}") or raw_path.startswith("/api"):
            requests.add((_method_from_fetch_init(match.group(3)), normalize_api_path_template(raw_path)))

    for match in re.finditer(
        r"\bapiFetch\(\s*([A-Za-z_]\w*)\.toString\(\)\s*,\s*(\{.*?\})\s*\)",
        api_text,
        re.DOTALL,
    ):
        path = url_vars.get(match.group(1))
        if path:
            requests.add((_method_from_fetch_init(match.group(2)), path))

    return {(method, path) for method, path in requests if path.startswith("/")}


FRONTEND_IMPORT_RE = re.compile(
    r"(?:import\s+(?:[^'\"()]+?\s+from\s+)?|import\s*\(|export\s+[^'\"()]+?\s+from\s+)(['\"])(.*?)\1"
)


def frontend_production_files(src_root: Path) -> list[Path]:
    return sorted(
        path
        for path in src_root.rglob("*")
        if path.suffix in {".ts", ".tsx"}
        and not path.name.endswith((".test.ts", ".test.tsx", ".d.ts"))
        and "generated" not in path.parts
    )


def resolve_frontend_import(source: Path, specifier: str, production_files: set[Path]) -> Path | None:
    if not specifier.startswith("."):
        return None
    base = (source.parent / specifier).resolve()
    candidates = [
        base,
        base.with_suffix(".ts"),
        base.with_suffix(".tsx"),
        base / "index.ts",
        base / "index.tsx",
    ]
    for candidate in candidates:
        if candidate in production_files:
            return candidate
    return None


def frontend_reachable_files(src_root: Path) -> set[Path]:
    files = frontend_production_files(src_root)
    file_set = {path.resolve() for path in files}
    entry = (src_root / "main.tsx").resolve()
    if entry not in file_set:
        return set()

    imports: dict[Path, set[Path]] = {path.resolve(): set() for path in files}
    for path in files:
        resolved_path = path.resolve()
        text = path.read_text(encoding="utf-8")
        for match in FRONTEND_IMPORT_RE.finditer(text):
            target = resolve_frontend_import(resolved_path, match.group(2), file_set)
            if target is not None:
                imports[resolved_path].add(target)

    reachable = {entry}
    stack = [entry]
    while stack:
        path = stack.pop()
        for target in imports.get(path, set()):
            if target not in reachable:
                reachable.add(target)
                stack.append(target)
    return reachable


def unreachable_frontend_production_files(src_root: Path) -> list[Path]:
    files = [path.resolve() for path in frontend_production_files(src_root)]
    reachable = frontend_reachable_files(src_root)
    return sorted(path for path in files if path not in reachable)


class Checker:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(f"ERROR: {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(f"WARN: {msg}")

    def ok(self, msg: str) -> None:
        self.info.append(f"OK: {msg}")

    def check_stale_files(self) -> None:
        skip_parts = {".git", "node_modules", "data", "artifacts", "dist", ".venv"}
        trash = []
        for pattern in [".DS_Store", "Thumbs.db"]:
            trash.extend(p for p in ROOT.rglob(pattern) if not any(part in skip_parts for part in p.parts))
        for path in trash:
            self.warn(f"trash file remains: {path.relative_to(ROOT)}")
        if not trash:
            self.ok("no OS trash files")

        stale_paths = [
            "PierNet/api/routers",
            "PierNet/api/schemas",
            "PierNet/api/services",
            "PierNet/api/deps.py",
            "PierNet/text2comp",
            "PierNet/simulators/power_system",
            "PierNet/simulators/gcam/requirements.txt",
            "PierNet/simulators/modflow/requirements.txt",
            "PierNet/simulators/power_flow/requirements.txt",
            "PierNet/simulators/simpeg/requirements.txt",
            "PierNet/simulators/transient/requirements.txt",
            "TOKEN_ROUTER_TRAINING_PLAN.md",
            "PERFORMANCE_IMPLEMENTATION_PLAN.md",
            "UPGRADE_PLAN.md",
            "docs/INDUSTRIALIZATION_PLAN.md",
            "setup.py",
            "scripts/utils/rebuild_filter_indexes.py",
        ]
        for rel in stale_paths:
            if (ROOT / rel).exists():
                self.warn(f"stale path still exists: {rel}")

        for path in sorted((ROOT / "docs").glob("*.docx")):
            self.warn(f"stale generated document remains: {path.relative_to(ROOT)}")

    def check_text_encoding(self) -> None:
        base_paths = [
            ROOT / "README.md",
            ROOT / "PROJECT_OVERVIEW.md",
            ROOT / "CLAUDE.md",
            ROOT / "docs",
            ROOT / "frontend",
            ROOT / "PierNet",
            ROOT / "scripts",
            ROOT / "tests",
        ]
        findings: list[str] = []
        for path in iter_text_files(ROOT, [path for path in base_paths if path.exists()]):
            findings.extend(find_garbled_text(path))
        if not findings:
            self.ok("no garbled text markers")
            return

        root_prefix = f"{ROOT}/"
        for finding in findings[:50]:
            rel_finding = finding[len(root_prefix):] if finding.startswith(root_prefix) else finding
            self.error(f"text encoding marker: {rel_finding}")
        if len(findings) > 50:
            self.error(f"text encoding marker scan truncated: {len(findings) - 50} additional finding(s)")

    def check_backend_routes(self) -> None:
        main_py = ROOT / "PierNet/api/main.py"
        if not main_py.exists():
            self.error("missing PierNet/api/main.py")
            return

        content = main_py.read_text(encoding="utf-8")
        registered = set(re.findall(r"\b([A-Za-z_]\w*)\.router", content))
        for router_dir in [ROOT / "PierNet/synth/api/routers", ROOT / "PierNet/training/api/routers"]:
            if not router_dir.exists():
                self.error(f"missing router dir: {router_dir.relative_to(ROOT)}")
                continue
            for router_file in sorted(router_dir.glob("*.py")):
                if router_file.stem == "__init__":
                    continue
                rel = router_file.relative_to(ROOT)
                if router_file.stem in registered:
                    self.ok(f"registered router: {rel}")
                else:
                    self.warn(f"router not included by PierNet/api/main.py: {rel}")

    def check_docs_structure(self) -> None:
        required_files = [
            "README.md",
            "PROJECT_OVERVIEW.md",
            "CLAUDE.md",
            "api_server.py",
            "PierNet/api/main.py",
            "PierNet/synth/text2comp/generator.py",
            "PierNet/synth/text2comp/interview_agent.py",
            "PierNet/synth/text2comp/template_store.py",
            "scripts/text2comp/generate_templates.py",
            "scripts/text2comp/fill_samples.py",
            "configs/text2comp/default.yaml",
            "configs/text2comp/registry.yaml",
        ]
        for rel in required_files:
            if (ROOT / rel).exists():
                self.ok(rel)
            else:
                self.error(f"missing documented file: {rel}")

        stale_refs = [
            "PierNet/text2comp",
            "PierNet.synth.text2comp/",
            "PierNet/api/routers",
            "PierNet/api/schemas",
            "PierNet/api/services",
            "PierNet/simulators/power_system",
            "TOKEN_ROUTER_TRAINING_PLAN.md",
            "PERFORMANCE_IMPLEMENTATION_PLAN.md",
            "UPGRADE_PLAN.md",
        ]
        docs = ["README.md", "PROJECT_OVERVIEW.md", "CLAUDE.md"]
        for doc in docs:
            path = ROOT / doc
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for ref in stale_refs:
                if ref in text:
                    self.warn(f"{doc} still references stale path: {ref}")

        stale_storage_claims = [
            "Stage 2-4 的源产物仍以 JSONL 为主",
            "Stage 2-4 source artifacts are JSONL",
            "training dynamically reopens JSONL",
            "JSONL pagination indexes for Stage 2-4 artifacts",
        ]
        for rel in [*docs, "scripts/utils/rebuild_indexes.py"]:
            path = ROOT / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for claim in stale_storage_claims:
                if claim in text:
                    self.error(f"{rel} has stale storage-format claim: {claim}")

        for issue in markdown_local_link_issues(ROOT):
            self.error(issue)

        api_server = ROOT / "api_server.py"
        if api_server.exists() and "Stage 2 API" in api_server.read_text(encoding="utf-8"):
            self.error("api_server.py describes the unified API as a stale Stage 2-only entrypoint")

    @staticmethod
    def _router_prefixes(router_files: Iterable[Path]) -> set[str]:
        prefixes: set[str] = set()
        for router_file in router_files:
            text = router_file.read_text(encoding="utf-8")
            prefix_match = re.search(r"APIRouter\([^)]*prefix=[\"']/?([^\"']*)", text)
            if prefix_match and prefix_match.group(1):
                prefixes.add(prefix_match.group(1).split("/")[0])
            for match in re.finditer(r"@router\.\w+\([\"']/?([^\"'/?]+)", text):
                prefixes.add(match.group(1).split("/")[0])
        return prefixes

    def check_api_alignment(self) -> None:
        api_ts = ROOT / "frontend/src/lib/api.ts"
        if not api_ts.exists():
            self.warn("frontend/src/lib/api.ts missing")
            return
        api_text = api_ts.read_text(encoding="utf-8")
        frontend_paths = frontend_api_prefixes(api_text)

        router_files = []
        for router_dir in [ROOT / "PierNet/synth/api/routers", ROOT / "PierNet/training/api/routers"]:
            router_files.extend(p for p in router_dir.glob("*.py") if p.stem != "__init__")
        backend_prefixes = self._router_prefixes(router_files)
        missing = frontend_paths - backend_prefixes - {""}
        if missing:
            for prefix in sorted(missing):
                self.warn(f"frontend calls /api/{prefix}/... but no matching router prefix was found")
        else:
            self.ok("frontend API calls align with backend router prefixes")

    def check_frontend_openapi_paths(self) -> None:
        api_ts = ROOT / "frontend/src/lib/api.ts"
        openapi_json = ROOT / "frontend/src/lib/generated/openapi.json"
        if not api_ts.exists():
            self.warn("frontend/src/lib/api.ts missing")
            return
        if not openapi_json.exists():
            self.warn("frontend generated OpenAPI snapshot missing")
            return

        try:
            openapi = json.loads(openapi_json.read_text(encoding="utf-8"))
        except Exception as exc:
            self.error(f"frontend generated OpenAPI snapshot parse failed: {exc}")
            return

        frontend_requests = frontend_api_method_path_templates(api_ts.read_text(encoding="utf-8"))
        openapi_requests = {
            (method.upper(), normalize_api_path_template(path))
            for path, methods in (openapi.get("paths") or {}).items()
            if isinstance(path, str) and isinstance(methods, dict)
            for method in methods
            if isinstance(method, str)
        }
        missing = sorted(frontend_requests - openapi_requests)
        if missing:
            for method, path in missing:
                self.warn(f"frontend calls {method} /api{path} but generated OpenAPI has no matching operation")
        else:
            self.ok("frontend API method/path templates align with generated OpenAPI")

    def check_registry(self) -> None:
        registry_path = ROOT / "configs/text2comp/registry.yaml"
        if not registry_path.exists():
            self.error("missing configs/text2comp/registry.yaml")
            return
        try:
            registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            self.error(f"registry.yaml parse failed: {exc}")
            return
        required = {"domain_context", "output_description", "param_info", "output_info", "observation_config"}
        for simulator, entry in registry.items():
            if not isinstance(entry, dict):
                self.error(f"registry[{simulator}] is not an object")
                continue
            missing = required - set(entry)
            if missing:
                self.warn(f"registry[{simulator}] missing fields: {sorted(missing)}")
            try:
                _validate_registry_entry(str(simulator), copy.deepcopy(entry))
            except Exception as exc:
                detail = getattr(exc, "detail", str(exc))
                self.error(f"registry[{simulator}] invalid: {detail}")
            else:
                self.ok(f"registry[{simulator}]")

    def check_python_entrypoints(self) -> None:
        pyproject_path = ROOT / "pyproject.toml"
        if not pyproject_path.exists():
            self.error("missing pyproject.toml")
            return
        try:
            pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.error(f"pyproject.toml parse failed: {exc}")
            return

        scripts = (pyproject.get("project") or {}).get("scripts") or {}
        for name, target in sorted(scripts.items()):
            if not isinstance(target, str) or ":" not in target:
                self.error(f"pyproject script {name} must be module:function")
                continue
            module, function_name = target.split(":", 1)
            module_path = python_module_path(module)
            if module_path is None:
                self.error(f"pyproject script {name} references missing module: {module}")
                continue
            if not python_entrypoint_function_exists(module_path, function_name):
                self.error(f"pyproject script {name} references missing function: {target}")

        docs = [ROOT / "README.md", ROOT / "PROJECT_OVERVIEW.md", ROOT / "CLAUDE.md", ROOT / "docs/MIGRATION.md"]
        missing_modules: set[str] = set()
        for path in docs:
            if not path.exists():
                continue
            for module in documented_PierNet_modules(path.read_text(encoding="utf-8")):
                if python_module_path(module) is None:
                    missing_modules.add(module)
        for module in sorted(missing_modules):
            self.error(f"documented python -m module is missing: {module}")

        if not any("pyproject script" in error or "documented python -m" in error for error in self.errors):
            self.ok("python entrypoints and documented PierNet modules resolve")

    def check_dependency_metadata(self) -> None:
        pyproject_path = ROOT / "pyproject.toml"
        requirements_path = ROOT / "requirements.txt"
        if not pyproject_path.exists() or not requirements_path.exists():
            self.error("missing pyproject.toml or requirements.txt")
            return
        try:
            pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.error(f"pyproject.toml parse failed: {exc}")
            return

        project = pyproject.get("project") or {}
        runtime_deps = dependency_names(list(project.get("dependencies") or []))
        optional_deps: set[str] = set()
        for values in (project.get("optional-dependencies") or {}).values():
            optional_deps.update(dependency_names(list(values or [])))
        requirements_deps = dependency_names(requirements_path.read_text(encoding="utf-8").splitlines())

        missing_from_pyproject = sorted(requirements_deps - runtime_deps - optional_deps)
        if missing_from_pyproject:
            self.error(f"requirements.txt dependencies missing from pyproject metadata: {missing_from_pyproject}")

        missing_from_requirements = sorted(runtime_deps - requirements_deps)
        if missing_from_requirements:
            self.error(f"pyproject runtime dependencies missing from requirements.txt: {missing_from_requirements}")

        if not missing_from_pyproject and not missing_from_requirements:
            self.ok("dependency metadata aligns with requirements.txt")

    def check_config_paths(self) -> None:
        cfg_path = ROOT / "configs/text2comp/default.yaml"
        if not cfg_path.exists():
            self.error("missing configs/text2comp/default.yaml")
            return
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        for key, value in (cfg.get("data_dirs") or {}).items():
            rel = value if isinstance(value, str) else value.get("path", "")
            if rel and not (ROOT / rel).exists():
                self.warn(f"default.yaml data_dirs[{key}] path missing: {rel}")
        for key in ["generation_config", "registry"]:
            rel = cfg.get(key)
            if rel and not (ROOT / rel).exists():
                self.warn(f"default.yaml {key} path missing: {rel}")

    def check_simulator_default_configs(self) -> None:
        simulator_root = ROOT / "PierNet/simulators"
        if not simulator_root.exists():
            self.error("missing PierNet/simulators")
            return
        for pipeline in sorted(simulator_root.glob("*/pipeline.py")):
            text = pipeline.read_text(encoding="utf-8")
            for match in re.finditer(r"default=[\"'](configs/[^\"']+\.ya?ml)[\"']", text):
                rel = match.group(1)
                if (ROOT / rel).exists():
                    self.ok(f"simulator default config: {pipeline.relative_to(ROOT)} -> {rel}")
                else:
                    self.error(f"{pipeline.relative_to(ROOT)} references missing default config: {rel}")

    def check_frontend_import_graph(self) -> None:
        src_root = ROOT / "frontend/src"
        if not src_root.exists():
            self.warn("frontend/src missing")
            return
        entry = src_root / "main.tsx"
        if not entry.exists():
            self.error("frontend/src/main.tsx missing")
            return

        unreachable = unreachable_frontend_production_files(src_root)
        if unreachable:
            for path in unreachable:
                self.error(f"frontend production file is not reachable from main.tsx: {path.relative_to(ROOT)}")
        else:
            self.ok("frontend production files are reachable from main.tsx")

    def check_container_runtime_assets(self) -> None:
        dockerfile = ROOT / "Dockerfile"
        if not dockerfile.exists():
            self.warn("Dockerfile missing")
            return

        text = dockerfile.read_text(encoding="utf-8")
        required = {
            "PierNet": "Python package",
            "scripts": "runtime scripts",
            "configs": "text2comp and simulator configs",
            "api_server.py": "FastAPI compatibility entrypoint",
        }
        missing: list[str] = []
        for source, label in required.items():
            if not re.search(rf"(?m)^COPY\s+{re.escape(source)}(?:\s|$)", text):
                missing.append(f"{source} ({label})")
        if missing:
            self.error("Dockerfile does not copy runtime asset(s): " + ", ".join(missing))
            return
        self.ok("Dockerfile copies required runtime assets")

    def check_frontend_node_runtime(self) -> None:
        package_json = ROOT / "frontend/package.json"
        run_vite = ROOT / "scripts/frontend/run-vite.mjs"
        start_ui = ROOT / "start_ui.sh"
        readme = ROOT / "README.md"
        required = [package_json, run_vite, start_ui, readme]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
        if missing:
            self.error("missing frontend runtime files: " + ", ".join(missing))
            return

        try:
            package = json.loads(package_json.read_text(encoding="utf-8"))
        except Exception as exc:
            self.error(f"frontend/package.json parse failed: {exc}")
            return

        package_name = str(package.get("name") or "")
        if re.search(r"stage[-_ ]?2", package_name, re.IGNORECASE):
            self.error("frontend/package.json package name still uses stale Stage 2 identity")

        lock_path = ROOT / "frontend/package-lock.json"
        if lock_path.exists():
            try:
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.error(f"frontend/package-lock.json parse failed: {exc}")
                return
            lock_names = [lock.get("name"), ((lock.get("packages") or {}).get("") or {}).get("name")]
            if any(name and name != package_name for name in lock_names):
                self.error("frontend/package-lock.json package name does not match frontend/package.json")

        node_engine = str((package.get("engines") or {}).get("node") or "")
        match = re.search(r">=\s*(\d+\.\d+\.\d+)", node_engine)
        if not match:
            self.error("frontend/package.json engines.node must declare >=x.y.z")
            return
        minimum = match.group(1)

        errors_before = len(self.errors)
        checks = [
            ("scripts/frontend/run-vite.mjs", f"MIN_NODE_LABEL = '{minimum}'", "frontend Node minimum"),
            ("scripts/frontend/run-vite.mjs", "process.env.PierNet_NODE_BIN", "PierNet_NODE_BIN support"),
            ("start_ui.sh", f'MIN_NODE_VERSION="{minimum}"', "frontend Node minimum"),
            ("start_ui.sh", "PierNet_NODE_BIN:-${PierNet_NODE:-", "PierNet_NODE_BIN support"),
            ("start_ui.sh", "PierNet_NPM:-npm", "PierNet_NPM support"),
            ("README.md", f"Node.js {minimum}+", "frontend Node minimum"),
        ]
        for rel, needle, label in checks:
            if needle not in (ROOT / rel).read_text(encoding="utf-8"):
                self.error(f"{rel} does not mirror {label} {minimum}")
        legacy_public_node_patterns = {
            "PierNet_NODE": r"(?m)^\s*PierNet_NODE\s*[:=]",
            "PierNet_NODE_BIN_DIR": r"(?m)^\s*PierNet_NODE_BIN_DIR\s*[:=]",
        }
        for rel in [".env.example", "compose.yaml"]:
            path = ROOT / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for key, pattern in legacy_public_node_patterns.items():
                if re.search(pattern, text):
                    self.error(f"{rel} exposes legacy {key}; use PierNet_NODE_BIN instead")

        if len(self.errors) == errors_before:
            self.ok(f"frontend Node runtime minimum: {minimum}")

    def check_compose_model_mount_defaults(self) -> None:
        compose = ROOT / "compose.yaml"
        if not compose.exists():
            self.warn("compose.yaml missing")
            return

        text = compose.read_text(encoding="utf-8")
        expected = "./models/Qwen/Qwen2.5-0.5B-Instruct"
        if expected not in text:
            self.error(f"compose.yaml Qwen model mount default must be {expected}")
            return
        if not (ROOT / expected).exists():
            self.warn(f"compose.yaml Qwen model mount default is not present locally: {expected}")
            return
        self.ok("compose.yaml Qwen model mount default matches local model path")

    def report(self) -> bool:
        for line in self.info:
            print(line)
        for line in self.warnings:
            print(line)
        for line in self.errors:
            print(line)
        if self.errors:
            print(f"FAILED: {len(self.errors)} errors, {len(self.warnings)} warnings")
            return False
        print(f"PASSED: {len(self.warnings)} warnings")
        return True


def main() -> None:
    checker = Checker()
    for check in [
        checker.check_stale_files,
        checker.check_text_encoding,
        checker.check_backend_routes,
        checker.check_docs_structure,
        checker.check_api_alignment,
        checker.check_frontend_openapi_paths,
        checker.check_frontend_import_graph,
        checker.check_registry,
        checker.check_python_entrypoints,
        checker.check_dependency_metadata,
        checker.check_config_paths,
        checker.check_simulator_default_configs,
        checker.check_container_runtime_assets,
        checker.check_frontend_node_runtime,
        checker.check_compose_model_mount_defaults,
    ]:
        try:
            check()
        except Exception as exc:
            checker.error(f"{check.__name__} failed: {exc}")
    sys.exit(0 if checker.report() else 1)


if __name__ == "__main__":
    main()
