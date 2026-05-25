"""配置相关路由：/api/config, /api/llm-config, /api/config/text2comp-scenarios。"""

import time
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

from PierNet.shared.runtime.paths import CONFIG_DIR, DATA_ROOT, PROJECT_ROOT
from PierNet.shared.storage import portable
from PierNet.shared.storage.hdf5_files import iter_hdf5_files
from PierNet.synth.api.schemas.config import LLMConfigRequest
from PierNet.synth.services import file_manager

router = APIRouter()


@router.get("/config")
def get_config():
    """读取 default.yaml 的 generation 节配置。"""
    default_yaml = CONFIG_DIR / "default.yaml"
    if not default_yaml.exists():
        return {}
    try:
        with open(default_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        # 只返回 generation 超参数节，不暴露 llm（含 api_key）
        return {k: v for k, v in cfg.items() if k not in ("llm",)}
    except Exception as e:
        raise HTTPException(500, f"读取配置失败: {e}")


@router.get("/llm-config")
def get_llm_config():
    """读取当前 LLM 配置（default.yaml 的 llm 节），api_key 脱敏返回。"""
    default_yaml = CONFIG_DIR / "default.yaml"
    try:
        cfg = {}
        if default_yaml.exists():
            with open(default_yaml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        llm = cfg.get("llm", {})

        raw_key = llm.get("api_key") or ""
        if raw_key and len(raw_key) > 8:
            masked = raw_key[:4] + "****" + raw_key[-4:]
        elif raw_key:
            masked = "****"
        else:
            masked = ""

        return {
            "provider": llm.get("provider", "siliconflow"),
            "model":    llm.get("model", ""),
            "base_url": llm.get("base_url", ""),
            "api_key_masked": masked,
            "has_api_key": bool(raw_key),
            "temperature": llm.get("temperature", 1),
            "max_tokens": llm.get("max_tokens", 1024),
            "thinking": llm.get("thinking", "disabled"),
        }
    except Exception as e:
        raise HTTPException(500, f"读取 LLM 配置失败: {e}")


@router.post("/llm-config")
def save_llm_config(req: LLMConfigRequest):
    """保存 LLM 配置到 default.yaml 的 llm 节。api_key 为空时保持原有值不变。"""
    provider = req.provider.strip() or "siliconflow"
    model = req.model.strip()
    if not model:
        raise HTTPException(status_code=400, detail="模型名称不能为空")

    default_yaml = CONFIG_DIR / "default.yaml"
    try:
        cfg: dict = {}
        if default_yaml.exists():
            with open(default_yaml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        llm = cfg.get("llm", {})
        llm["provider"] = provider
        llm["model"] = model
        api_key = req.api_key.strip()
        if api_key:
            llm["api_key"] = api_key
        llm["base_url"] = req.base_url.strip()
        llm["temperature"] = req.temperature
        llm["max_tokens"] = req.max_tokens
        llm["thinking"] = req.thinking if req.thinking in ("enabled", "disabled") else "disabled"
        cfg["llm"] = llm

        default_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(default_yaml, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, indent=2)

        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"保存 LLM 配置失败: {e}")


@router.post("/llm-config/test")
def test_llm_config(req: LLMConfigRequest):
    """
    用给定配置测试 LLM 连通性（不保存）。
    api_key 为空时从已保存的配置中读取。
    返回 {"ok": True/False, "message": "...", "response_preview": "..."}
    """
    from PierNet.core.llm_client import LLMClient

    default_yaml = CONFIG_DIR / "default.yaml"

    # 若 api_key 未传入，从已保存配置中补全
    api_key = req.api_key
    if not api_key and default_yaml.exists():
        try:
            with open(default_yaml, "r", encoding="utf-8") as f:
                saved = yaml.safe_load(f) or {}
            api_key = saved.get("llm", {}).get("api_key", "")
        except Exception:
            pass

    if not api_key:
        return {"ok": False, "message": "未配置 API Key，请先填写", "response_preview": ""}

    try:
        client = LLMClient(
            provider=req.provider,
            model=req.model or "gpt-3.5-turbo",
            api_key=api_key,
            base_url=req.base_url or None,
            max_retries=1,
            timeout=15,
            thinking=req.thinking,
        )
        reply = client.generate("Say 'ok' in one word.", max_tokens=16, temperature=0)
        preview = reply.strip()[:120]
        return {"ok": True, "message": "连接成功", "response_preview": preview}
    except Exception as e:
        return {"ok": False, "message": str(e), "response_preview": ""}


_t2c_cache: dict = {"result": None, "ts": 0.0}
_T2C_TTL = 10.0  # seconds


def invalidate_text2comp_scenarios_cache() -> None:
    _t2c_cache.update({"result": None, "ts": 0.0})


def _resolve_data_path(value: str | None, *, default: str = "data") -> Path:
    raw = (value or default).strip()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "data":
        return DATA_ROOT.joinpath(*parts[1:])
    return PROJECT_ROOT / path


@router.get("/config/text2comp-scenarios")
def get_text2comp_scenarios():
    """
    返回 Stage 2 可用场景列表（按 simulator 子目录分组）。
    约定：data/{simulator}/{simulator}_{scenario}.h5/.hdf5
    """
    if _t2c_cache["result"] is not None and time.time() - _t2c_cache["ts"] < _T2C_TTL:
        return _t2c_cache["result"]

    default_yaml = CONFIG_DIR / "default.yaml"
    if not default_yaml.exists():
        raise HTTPException(404, "configs/text2comp/default.yaml 不存在")

    with open(default_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_root = _resolve_data_path(cfg.get("data_root", "data"))

    # 加载 registry
    registry_path_str = cfg.get("registry", "configs/text2comp/registry.yaml")
    registry: dict = {}
    registry_path = PROJECT_ROOT / registry_path_str
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = yaml.safe_load(f) or {}
        except Exception:
            pass

    import h5py

    parquet_counts: dict[tuple[str, str], int] = {}
    try:
        for part in portable.discover_partitions("text2comp"):
            parquet_counts[(part.simulator, part.scenario)] = max(
                parquet_counts.get((part.simulator, part.scenario), 0),
                int(part.row_count or 0),
            )
    except Exception:
        parquet_counts = {}

    def _jsonl_status(simulator_name: str, scenario_name: str) -> tuple[int, bool]:
        try:
            jsonl_path = file_manager.resolve_sample_file(scenario_name, simulator=simulator_name)
        except ValueError:
            return 0, True
        if jsonl_path is None or not jsonl_path.exists():
            return 0, False
        try:
            with open(jsonl_path, "rb") as jf:
                content = jf.read()
                count = content.count(b"\n")
                if content and not content.endswith(b"\n"):
                    count += 1
                return count, True
        except Exception:
            return 0, True

    def _sample_status(simulator_name: str, scenario_name: str, jsonl_count: int, has_jsonl: bool) -> dict:
        parquet_count = parquet_counts.get((simulator_name, scenario_name), 0)
        has_parquet = parquet_count > 0
        if has_parquet and has_jsonl:
            storage = "mixed"
            existing_count = max(jsonl_count, parquet_count)
        elif has_parquet:
            storage = "parquet"
            existing_count = parquet_count
        elif has_jsonl:
            storage = "jsonl"
            existing_count = jsonl_count
        else:
            storage = None
            existing_count = 0
        return {
            "existing_jsonl_count": jsonl_count,
            "existing_parquet_count": parquet_count,
            "existing_sample_count": existing_count,
            "existing_storage": storage,
            "has_jsonl": has_jsonl,
            "has_parquet": has_parquet,
            "has_samples": existing_count > 0,
        }

    # 收集所有 simulator 名（来自 data/ 子目录 + registry 顶层 key + Parquet 分区）
    sim_dirs = {d.name for d in data_root.iterdir() if d.is_dir()} if data_root.exists() else set()
    reg_sims = set(registry.keys())
    parquet_sims = {simulator for simulator, _ in parquet_counts}
    # 排除非 simulator 目录（templates、text2comp 等）
    skip = {"templates", "text2comp", "router"}
    all_sims = sorted((sim_dirs | reg_sims | parquet_sims) - skip)

    result: dict = {}

    for simulator in all_sims:
        dir_path = data_root / simulator
        scenarios_map: dict = {}

        # 1. 扫描 HDF5 文件：文件名格式 {simulator}_{scenario}.h5/.hdf5
        if dir_path.exists():
            prefix = simulator + "_"
            for h5_file in iter_hdf5_files(dir_path):
                stem = h5_file.stem
                scenario_name = stem[len(prefix):] if stem.startswith(prefix) else stem

                sample_count = 0
                output_shape = None
                try:
                    with h5py.File(h5_file, "r") as hf:
                        if "timeseries" in hf:
                            ts_shape = hf["timeseries"].shape
                            sample_count = ts_shape[0]
                            if len(ts_shape) >= 3:
                                output_shape = list(ts_shape[1:])
                except Exception:
                    pass

                existing_count, has_jsonl = _jsonl_status(simulator, scenario_name)

                scenarios_map[scenario_name] = {
                    "name": scenario_name,
                    "simulator": simulator,
                    "h5_file": h5_file.name,
                    "sample_count": sample_count,
                    "output_shape": output_shape,
                    **_sample_status(simulator, scenario_name, existing_count, has_jsonl),
                    "has_h5": True,
                    "registered": False,
                }

        # 2. 合并 registry 中已注册但无 HDF5 的场景
        sim_entry = registry.get(simulator, {})
        reg_scenarios = sim_entry.get("scenarios", {}) if isinstance(sim_entry, dict) else {}
        for sc_name in reg_scenarios:
            if sc_name not in scenarios_map:
                existing_count, has_jsonl = _jsonl_status(simulator, sc_name)
                scenarios_map[sc_name] = {
                    "name": sc_name,
                    "simulator": simulator,
                    "h5_file": None,
                    "sample_count": 0,
                    "output_shape": None,
                    **_sample_status(simulator, sc_name, existing_count, has_jsonl),
                    "has_h5": False,
                    "registered": True,
                }

        # 3. 合并只有 Parquet 样本、但无 HDF5/registry 记录的场景
        for (part_simulator, sc_name), _ in parquet_counts.items():
            if part_simulator != simulator or sc_name in scenarios_map:
                continue
            existing_count, has_jsonl = _jsonl_status(simulator, sc_name)
            scenarios_map[sc_name] = {
                "name": sc_name,
                "simulator": simulator,
                "h5_file": None,
                "sample_count": 0,
                "output_shape": None,
                **_sample_status(simulator, sc_name, existing_count, has_jsonl),
                "has_h5": False,
                "registered": sc_name in reg_scenarios,
            }

        # 4. 标记已注册的场景
        for sc_name in scenarios_map:
            scenarios_map[sc_name]["registered"] = sc_name in reg_scenarios

        if scenarios_map:
            result[simulator] = sorted(scenarios_map.values(), key=lambda x: x["name"])

    _t2c_cache.update({"result": result, "ts": time.time()})
    return result
