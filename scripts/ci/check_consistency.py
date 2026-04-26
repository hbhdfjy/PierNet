#!/usr/bin/env python3
"""Repository consistency checks for PiERN."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]


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
            "piern/api/routers",
            "piern/api/schemas",
            "piern/api/services",
            "piern/api/deps.py",
            "piern/text2comp",
            "piern/simulators/power_system",
            "TOKEN_ROUTER_TRAINING_PLAN.md",
            "PERFORMANCE_IMPLEMENTATION_PLAN.md",
            "UPGRADE_PLAN.md",
        ]
        for rel in stale_paths:
            if (ROOT / rel).exists():
                self.warn(f"stale path still exists: {rel}")

    def check_backend_routes(self) -> None:
        main_py = ROOT / "piern/api/main.py"
        if not main_py.exists():
            self.error("missing piern/api/main.py")
            return

        content = main_py.read_text(encoding="utf-8")
        registered = set(re.findall(r"\b([A-Za-z_]\w*)\.router", content))
        for router_dir in [ROOT / "piern/synth/api/routers", ROOT / "piern/training/api/routers"]:
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
                    self.warn(f"router not included by piern/api/main.py: {rel}")

    def check_docs_structure(self) -> None:
        required_files = [
            "README.md",
            "PROJECT_OVERVIEW.md",
            "CLAUDE.md",
            "api_server.py",
            "piern/api/main.py",
            "piern/synth/text2comp/generator.py",
            "piern/synth/text2comp/interview_agent.py",
            "piern/synth/text2comp/template_store.py",
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
            "piern/text2comp",
            "piern.synth.text2comp/",
            "piern/api/routers",
            "piern/api/schemas",
            "piern/api/services",
            "piern/simulators/power_system",
            "TOKEN_ROUTER_TRAINING_PLAN.md",
            "PERFORMANCE_IMPLEMENTATION_PLAN.md",
            "UPGRADE_PLAN.md",
        ]
        for doc in ["README.md", "PROJECT_OVERVIEW.md", "CLAUDE.md"]:
            path = ROOT / doc
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for ref in stale_refs:
                if ref in text:
                    self.warn(f"{doc} still references stale path: {ref}")

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
        frontend_paths = {
            match.group(1).split("/")[0]
            for match in re.finditer(r"['\"`]/api/([^'\"` \n?{]+)", api_text)
        }

        router_files = []
        for router_dir in [ROOT / "piern/synth/api/routers", ROOT / "piern/training/api/routers"]:
            router_files.extend(p for p in router_dir.glob("*.py") if p.stem != "__init__")
        backend_prefixes = self._router_prefixes(router_files)
        missing = frontend_paths - backend_prefixes - {""}
        if missing:
            for prefix in sorted(missing):
                self.warn(f"frontend calls /api/{prefix}/... but no matching router prefix was found")
        else:
            self.ok("frontend API calls align with backend router prefixes")

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
            else:
                self.ok(f"registry[{simulator}]")

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
        checker.check_backend_routes,
        checker.check_docs_structure,
        checker.check_api_alignment,
        checker.check_registry,
        checker.check_config_paths,
    ]:
        try:
            check()
        except Exception as exc:
            checker.error(f"{check.__name__} failed: {exc}")
    sys.exit(0 if checker.report() else 1)


if __name__ == "__main__":
    main()
