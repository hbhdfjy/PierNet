#!/usr/bin/env python3
"""
PiERN pre-push 检查脚本。

检查项：
  1. 废弃文件检测（孤立脚本、空目录、系统垃圾文件）
  2. 代码引用完整性（import/路由引用的文件是否存在）
  3. 前端路由与页面文件对齐
  4. 后端 API 路由注册完整性
  5. README / CLAUDE.md 项目结构描述与实际文件对齐
  6. registry 结构合法性
  7. 配置文件引用的路径是否存在

使用方式：
    python scripts/ci/check_consistency.py

返回值：
    0 - 所有检查通过（或只有警告）
    1 - 发现必须修复的错误
"""

import re
import sys
import yaml
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent.parent


class Checker:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def error(self, msg: str):
        self.errors.append(f"❌  {msg}")

    def warn(self, msg: str):
        self.warnings.append(f"⚠️   {msg}")

    def ok(self, msg: str):
        self.info.append(f"✓  {msg}")

    # ── 1. 废弃文件检测 ─────────────────────────────────────────────

    def check_stale_files(self):
        print("\n📁 检查 1: 废弃文件与垃圾文件")
        print("-" * 60)

        # .DS_Store
        ds_stores = list(ROOT.rglob(".DS_Store"))
        ds_stores = [p for p in ds_stores if ".git" not in p.parts]
        if ds_stores:
            for p in ds_stores:
                self.warn(f".DS_Store 未清理: {p.relative_to(ROOT)}")
        else:
            self.ok(".DS_Store 已清理")

        # 空目录（排除 .git、node_modules、__pycache__、data）
        skip_dirs = {".git", "node_modules", "__pycache__", "data", ".venv", "dist"}
        empty_dirs = []
        for d in ROOT.rglob("*"):
            if not d.is_dir():
                continue
            if any(s in d.parts for s in skip_dirs):
                continue
            # 目录为空（或只含 __pycache__）
            children = [c for c in d.iterdir() if c.name != "__pycache__"]
            if not children:
                empty_dirs.append(d.relative_to(ROOT))
        if empty_dirs:
            for d in empty_dirs:
                self.warn(f"空目录: {d}")
        else:
            self.ok("无空目录")

        # 已知废弃文件
        stale_known = [
            "scripts/text2comp/generate_stage2.py",  # 旧版一步流程
            "piern/router/__init__.py",               # Stage 3 空占位
            "tests/__init__.py",                      # 空测试
            "CHANGE_CHECKLIST.md",                    # 临时文档
            "PROJECT_GUIDE.md",                       # 已合并到 CLAUDE.md
        ]
        for rel in stale_known:
            p = ROOT / rel
            if p.exists():
                self.warn(f"废弃文件仍存在: {rel}")

    # ── 2. 前端路由 vs 页面文件 ──────────────────────────────────────

    def check_frontend_routes(self):
        print("\n🖥️  检查 2: 前端路由与页面文件对齐")
        print("-" * 60)

        app_tsx = ROOT / "frontend/src/App.tsx"
        pages_dir = ROOT / "frontend/src/pages"
        if not app_tsx.exists():
            self.warn("App.tsx 不存在，跳过前端路由检查")
            return

        content = app_tsx.read_text(encoding="utf-8")

        # 提取 import 的页面组件
        imports = re.findall(r"import\s+(\w+)\s+from\s+'\.\/pages\/(\w+)'", content)
        for component, filename in imports:
            page_file = pages_dir / f"{filename}.tsx"
            if not page_file.exists():
                self.error(f"App.tsx import {component} 但文件不存在: pages/{filename}.tsx")
            else:
                self.ok(f"pages/{filename}.tsx")

        # 检查 pages/ 下是否有未被 import 的页面
        existing_pages = {p.stem for p in pages_dir.glob("*.tsx")}
        imported_pages = {filename for _, filename in imports}
        orphan = existing_pages - imported_pages
        for p in sorted(orphan):
            self.warn(f"pages/{p}.tsx 未在 App.tsx 中 import")

    # ── 3. 后端路由注册完整性 ────────────────────────────────────────

    def check_backend_routes(self):
        print("\n🔌 检查 3: 后端路由注册完整性")
        print("-" * 60)

        main_py = ROOT / "piern/api/main.py"
        routers_dir = ROOT / "piern/api/routers"
        if not main_py.exists():
            self.warn("piern/api/main.py 不存在，跳过后端路由检查")
            return

        content = main_py.read_text(encoding="utf-8")

        # 提取 include_router 注册的模块名
        registered = set(re.findall(r"from\s+\.routers\s+import\s+(\w+)", content))
        registered |= set(re.findall(r"routers\.(\w+)\.router", content))

        # 检查 routers/ 下每个文件是否都被注册
        router_files = [p.stem for p in routers_dir.glob("*.py") if p.stem != "__init__"]
        for rf in sorted(router_files):
            if rf not in registered:
                # 尝试检查是否通过其他方式引入
                if rf not in content:
                    self.warn(f"routers/{rf}.py 可能未在 main.py 中注册")
                else:
                    self.ok(f"routers/{rf}.py（间接引入）")
            else:
                self.ok(f"routers/{rf}.py 已注册")

    # ── 4. README / CLAUDE.md 结构描述对齐 ──────────────────────────

    def check_docs_structure(self):
        print("\n📝 检查 4: 文档结构描述与实际文件对齐")
        print("-" * 60)

        # 检查 README.md 和 CLAUDE.md 中提到的关键文件是否存在
        key_files_in_docs = {
            "api_server.py":                        ROOT / "api_server.py",
            "piern/api/main.py":                    ROOT / "piern/api/main.py",
            "piern/text2comp/generator.py":         ROOT / "piern/text2comp/generator.py",
            "piern/text2comp/interview_agent.py":   ROOT / "piern/text2comp/interview_agent.py",
            "piern/text2comp/template_store.py":    ROOT / "piern/text2comp/template_store.py",
            "scripts/text2comp/generate_templates.py": ROOT / "scripts/text2comp/generate_templates.py",
            "scripts/text2comp/fill_samples.py":    ROOT / "scripts/text2comp/fill_samples.py",
            "start_ui.sh":                          ROOT / "start_ui.sh",
            "configs/text2comp/default.yaml":       ROOT / "configs/text2comp/default.yaml",
            "configs/text2comp/registry.yaml":      ROOT / "configs/text2comp/registry.yaml",
        }
        for desc, path in key_files_in_docs.items():
            if not path.exists():
                self.error(f"文档中提到的文件不存在: {desc}")
            else:
                self.ok(desc)

        # 检查 README/CLAUDE.md 是否还引用已删除的文件
        deleted_refs = [
            "generate_stage2.py",
            "piern/router/",
            "GenerationMonitor.tsx",
            "SampleRefinement.tsx",
            "TaskLauncher.tsx",
        ]
        for doc in ["README.md", "CLAUDE.md"]:
            p = ROOT / doc
            if not p.exists():
                continue
            content = p.read_text(encoding="utf-8")
            for ref in deleted_refs:
                if ref in content:
                    self.warn(f"{doc} 仍引用已删除的 '{ref}'")

        # 检查 README 中的 simulator 状态标记
        readme = ROOT / "README.md"
        if readme.exists():
            content = readme.read_text(encoding="utf-8")
            sim_dirs = {
                "modflow":     ROOT / "piern/simulators/modflow/pipeline.py",
                "simpeg":      ROOT / "piern/simulators/simpeg/pipeline.py",
                "power_flow":  ROOT / "piern/simulators/power_flow/pipeline.py",
                "transient":   ROOT / "piern/simulators/transient/pipeline.py",
                "gcam":        ROOT / "piern/simulators/gcam/pipeline.py",
            }
            for sim, pipeline in sim_dirs.items():
                if not pipeline.exists():
                    if "✅" in content and sim in content:
                        self.error(f"README 标记 {sim} 为 ✅ 但 pipeline.py 不存在")
                else:
                    if "❌" in content:
                        # 找到 ❌ 标记的行，看是否是这个 simulator
                        for line in content.split("\n"):
                            if sim in line and "❌" in line:
                                self.warn(f"README 标记 {sim} 为 ❌ 但 pipeline.py 存在")

    # ── 5. 前端 API 调用与后端路由对齐 ──────────────────────────────

    def check_api_alignment(self):
        print("\n🔗 检查 5: 前端 API 调用与后端路由对齐")
        print("-" * 60)

        api_ts = ROOT / "frontend/src/lib/api.ts"
        if not api_ts.exists():
            self.warn("api.ts 不存在，跳过 API 对齐检查")
            return

        api_content = api_ts.read_text(encoding="utf-8")

        # 提取前端调用的路径（去掉参数）
        frontend_paths = set()
        for m in re.finditer(r"['\"`]/api/([^'\"` \n?{]+)", api_content):
            path = m.group(1).split("/")[0]  # 取第一段
            frontend_paths.add(path)

        # 收集后端实际注册的路由前缀
        backend_prefixes = set()
        routers_dir = ROOT / "piern/api/routers"
        for router_file in routers_dir.glob("*.py"):
            if router_file.stem == "__init__":
                continue
            content = router_file.read_text(encoding="utf-8")
            for m in re.finditer(r'@router\.\w+\(["\']/?([^"\'/?]+)', content):
                backend_prefixes.add(m.group(1).split("/")[0])

        orphan_calls = frontend_paths - backend_prefixes - {""}
        if orphan_calls:
            for p in sorted(orphan_calls):
                self.warn(f"前端调用 /api/{p}/... 但后端可能无对应路由")
        else:
            self.ok("前端 API 调用与后端路由基本对齐")

    # ── 6. registry.yaml 结构合法性 ─────────────────────────────────

    def check_registry(self):
        print("\n📋 检查 6: registry.yaml 结构合法性")
        print("-" * 60)

        registry_path = ROOT / "configs/text2comp/registry.yaml"
        if not registry_path.exists():
            self.warn("registry.yaml 不存在")
            return

        try:
            with open(registry_path, encoding="utf-8") as f:
                reg = yaml.safe_load(f) or {}
        except Exception as e:
            self.error(f"registry.yaml 解析失败: {e}")
            return

        if not reg:
            self.warn("registry.yaml 为空，尚未注册任何 simulator")
            return

        required_fields = ["domain_context", "output_description", "param_info",
                           "output_info", "observation_config"]

        for sim, entry in reg.items():
            if not isinstance(entry, dict):
                self.error(f"registry[{sim}] 不是 dict")
                continue

            # 检查必要字段
            missing = [f for f in required_fields if f not in entry]
            if missing:
                self.warn(f"registry[{sim}] 缺少字段: {missing}")
            else:
                self.ok(f"{sim}: 完整")

            # 检查 scenarios 子字段
            scenarios = entry.get("scenarios", {})
            if not isinstance(scenarios, dict):
                self.error(f"registry[{sim}].scenarios 不是 dict")
            elif not scenarios:
                self.warn(f"registry[{sim}].scenarios 为空，建议注册场景描述")
            else:
                self.ok(f"{sim}.scenarios: {len(scenarios)} 个场景")

            # 检查 output_description 含 {ts} 占位符（{ch} 可选，单通道时可硬编码）
            od = entry.get("output_description", "")
            if od and "{ts}" not in od:
                self.error(f"registry[{sim}].output_description 缺少 {{ts}} 占位符")

            # 检查 param_info 格式
            pi = entry.get("param_info", {})
            if pi:
                for name, info in list(pi.items())[:3]:  # 只检查前3个
                    if not isinstance(info, (list, tuple)) or len(info) < 2:
                        self.error(f"registry[{sim}].param_info[{name}] 格式应为 [含义, 单位]")
                        break

    # ── 7. 配置文件路径引用 ──────────────────────────────────────────

    def check_config_paths(self):
        print("\n⚙️  检查 7: 配置文件路径引用")
        print("-" * 60)

        default_yaml = ROOT / "configs/text2comp/default.yaml"
        if not default_yaml.exists():
            self.warn("configs/text2comp/default.yaml 不存在")
            return

        with open(default_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # 检查 data_dirs 中的路径是否存在
        data_dirs = cfg.get("data_dirs", {})
        for key, val in data_dirs.items():
            if isinstance(val, str):
                path = ROOT / val
            else:
                path = ROOT / val.get("path", "")
            if not path.exists():
                self.warn(f"default.yaml data_dirs[{key}].path 不存在: {path.relative_to(ROOT)}")
            else:
                self.ok(f"data_dirs[{key}]: {path.relative_to(ROOT)}")

        # 检查 generation_config 引用
        gen_cfg = cfg.get("generation_config", "")
        if gen_cfg:
            gen_path = ROOT / gen_cfg
            if not gen_path.exists():
                self.error(f"default.yaml generation_config 引用不存在: {gen_cfg}")
            else:
                self.ok(f"generation_config: {gen_cfg}")

        # 检查 registry 路径
        reg_path_str = cfg.get("registry", "")
        if reg_path_str:
            reg_path = ROOT / reg_path_str
            if not reg_path.exists():
                self.warn(f"default.yaml registry 路径不存在: {reg_path_str}（可能尚未生成）")
            else:
                self.ok(f"registry: {reg_path_str}")

    # ── 报告 ─────────────────────────────────────────────────────────

    def report(self) -> bool:
        print("\n" + "=" * 60)
        print("📊 检查报告")
        print("=" * 60)

        if self.info:
            for msg in self.info:
                print(f"  {msg}")
            print()

        if self.warnings:
            print(f"⚠️  {len(self.warnings)} 个警告：")
            for msg in self.warnings:
                print(f"  {msg}")
            print()

        if self.errors:
            print(f"❌ {len(self.errors)} 个错误（必须修复）：")
            for msg in self.errors:
                print(f"  {msg}")
            print()
            print("=" * 60)
            print("❌ 检查失败！请修复上述错误后重新 push。")
            print("   跳过检查（不推荐）：git push --no-verify")
            print("=" * 60)
            return False

        print("=" * 60)
        if self.warnings:
            print("⚠️  检查通过，但存在警告，建议修复。")
        else:
            print("✅ 所有检查通过！")
        print("=" * 60)
        return True


def main():
    print("=" * 60)
    print("🔍 PiERN pre-push 检查")
    print("=" * 60)

    import os
    os.chdir(ROOT)

    checker = Checker()
    checks = [
        checker.check_stale_files,
        checker.check_frontend_routes,
        checker.check_backend_routes,
        checker.check_docs_structure,
        checker.check_api_alignment,
        checker.check_registry,
        checker.check_config_paths,
    ]

    for check in checks:
        try:
            check()
        except Exception as e:
            checker.error(f"{check.__name__} 执行异常: {e}")

    passed = checker.report()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
