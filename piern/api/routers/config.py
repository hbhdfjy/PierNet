"""配置相关路由：/api/config, /api/llm-config, /api/config/scenarios。"""

import time
import yaml
from fastapi import APIRouter, HTTPException

from piern.api.deps import CONFIG_DIR, CONFIGS_ROOT, PROJECT_ROOT
from piern.api.schemas.config import LLMConfigRequest, DataDirEntry, DataDirsRequest

router = APIRouter()


@router.get("/config")
def get_config():
    """读取 generation.yaml 配置。"""
    gen_yaml = CONFIG_DIR / "generation.yaml"
    if not gen_yaml.exists():
        return {}
    try:
        with open(gen_yaml, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise HTTPException(500, f"读取配置失败: {e}")


@router.get("/llm-config")
def get_llm_config():
    """读取当前 LLM 配置（generation.yaml 的 llm 节），api_key 脱敏返回。"""
    gen_yaml = CONFIG_DIR / "generation.yaml"
    try:
        cfg = {}
        if gen_yaml.exists():
            with open(gen_yaml, "r", encoding="utf-8") as f:
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
        }
    except Exception as e:
        raise HTTPException(500, f"读取 LLM 配置失败: {e}")


@router.post("/llm-config")
def save_llm_config(req: LLMConfigRequest):
    """保存 LLM 配置到 generation.yaml 的 llm 节。api_key 为空时保持原有值不变。"""
    gen_yaml = CONFIG_DIR / "generation.yaml"
    try:
        cfg: dict = {}
        if gen_yaml.exists():
            with open(gen_yaml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        llm = cfg.get("llm", {})
        llm["provider"] = req.provider
        if req.model:
            llm["model"] = req.model
        if req.api_key:
            llm["api_key"] = req.api_key
        if req.base_url is not None:
            # 空字符串表示清空 base_url，存为空字符串（不存 None，避免 YAML null 解析歧义）
            llm["base_url"] = req.base_url
        llm["temperature"] = req.temperature
        llm["max_tokens"] = req.max_tokens
        cfg["llm"] = llm

        gen_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(gen_yaml, "w", encoding="utf-8") as f:
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
    from piern.core.llm_client import LLMClient

    gen_yaml = CONFIG_DIR / "generation.yaml"

    # 若 api_key 未传入，从已保存配置中补全
    api_key = req.api_key
    if not api_key and gen_yaml.exists():
        try:
            with open(gen_yaml, "r", encoding="utf-8") as f:
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
        )
        reply = client.generate("Say 'ok' in one word.", max_tokens=16, temperature=0)
        preview = reply.strip()[:120]
        return {"ok": True, "message": "连接成功", "response_preview": preview}
    except Exception as e:
        return {"ok": False, "message": str(e), "response_preview": ""}


@router.get("/config/scenarios")
def get_scenarios():
    """扫描 configs/*/variants/*.yaml，返回各 simulator 的场景列表。"""
    result = {}
    for sim_dir in CONFIGS_ROOT.iterdir():
        if not sim_dir.is_dir() or sim_dir.name == "text2comp":
            continue
        variants_dir = sim_dir / "variants"
        if not variants_dir.exists():
            continue
        scenarios = []
        for yaml_file in sorted(variants_dir.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                scenarios.append({
                    "name": yaml_file.stem,
                    "scenario": cfg.get("scenario", yaml_file.stem),
                    "output_file": cfg.get("output_file", ""),
                    "n_samples": cfg.get("n_samples", 0),
                })
            except Exception:
                scenarios.append({"name": yaml_file.stem})
        if scenarios:
            result[sim_dir.name] = scenarios
    return result


_t2c_cache: dict = {"result": None, "ts": 0.0}
_T2C_TTL = 10.0  # seconds


@router.get("/config/text2comp-scenarios")
def get_text2comp_scenarios():
    """
    返回 Stage 2 可用场景列表（按 data_dirs key 分组）。

    每个场景包含：
      - has_h5: 是否有 HDF5 数据文件（可直接生成）
      - registered: 是否在 registry.yaml 中有注册信息
      - sample_count: HDF5 中的样本数（无 HDF5 时为 0）
    """
    if _t2c_cache["result"] is not None and time.time() - _t2c_cache["ts"] < _T2C_TTL:
        return _t2c_cache["result"]

    default_yaml = CONFIG_DIR / "default.yaml"
    if not default_yaml.exists():
        raise HTTPException(404, "configs/text2comp/default.yaml 不存在")

    with open(default_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_dirs = cfg.get("data_dirs", {})

    # 加载 registry
    registry_path_str = cfg.get("registry", "configs/text2comp/registry.yaml")
    registry_path = PROJECT_ROOT / registry_path_str
    registry: dict = {}
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = yaml.safe_load(f) or {}
        except Exception:
            pass

    from piern.api.deps import DATA_DIR as _DATA_DIR
    import h5py

    result: dict = {}

    for dir_key, dir_cfg in data_dirs.items():
        if isinstance(dir_cfg, str):
            dir_path = PROJECT_ROOT / dir_cfg
            simulator = dir_key
            file_suffix = None
            transient_simulator = None
            transient_keywords: list = []
            include_keywords: list = []
        else:
            dir_path = PROJECT_ROOT / dir_cfg["path"]
            simulator = dir_cfg.get("simulator", dir_key)
            file_suffix = dir_cfg.get("file_suffix", None)
            transient_simulator = dir_cfg.get("transient_simulator", None)
            transient_keywords = dir_cfg.get("transient_keywords", [])
            include_keywords: list = dir_cfg.get("include_keywords", [])

        scenarios_map: dict = {}  # scenario_name -> entry

        # 1. 扫描 HDF5 文件
        if dir_path.exists():
            for h5_file in sorted(dir_path.glob("*.h5")):
                stem = h5_file.stem

                # 白名单过滤
                if include_keywords and not any(kw.lower() in stem.lower() for kw in include_keywords):
                    continue

                scenario_name = stem
                if file_suffix and stem.endswith(file_suffix):
                    scenario_name = stem[: -len(file_suffix)]

                sim_type = simulator
                if transient_simulator and transient_keywords:
                    if any(kw in stem.lower() for kw in transient_keywords):
                        sim_type = transient_simulator

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

                jsonl_path = _DATA_DIR / f"{scenario_name}.jsonl"
                existing_count = 0
                if jsonl_path.exists():
                    try:
                        with open(jsonl_path, "rb") as jf:
                            existing_count = jf.read().count(b"\n")
                    except Exception:
                        pass

                scenarios_map[scenario_name] = {
                    "name": scenario_name,
                    "simulator": sim_type,
                    "h5_file": h5_file.name,
                    "sample_count": sample_count,
                    "output_shape": output_shape,
                    "existing_jsonl_count": existing_count,
                    "has_jsonl": jsonl_path.exists(),
                    "has_h5": True,
                    "registered": False,  # 下面补充
                }

        # 2. 合并 registry 中已注册的场景（可能没有 HDF5）
        sim_entry = registry.get(simulator, {})
        reg_scenarios = sim_entry.get("scenarios", {}) if isinstance(sim_entry, dict) else {}
        for sc_name in reg_scenarios:
            # 白名单过滤：registry 中的场景也要通过白名单
            if include_keywords and not any(kw.lower() in sc_name.lower() for kw in include_keywords):
                continue
            if sc_name not in scenarios_map:
                # 有注册但无 HDF5
                scenarios_map[sc_name] = {
                    "name": sc_name,
                    "simulator": simulator,
                    "h5_file": None,
                    "sample_count": 0,
                    "existing_jsonl_count": 0,
                    "has_jsonl": False,
                    "has_h5": False,
                    "registered": True,
                }

        # 3. 标记已注册的场景
        for sc_name in scenarios_map:
            scenarios_map[sc_name]["registered"] = sc_name in reg_scenarios

        if scenarios_map:
            result[dir_key] = sorted(scenarios_map.values(), key=lambda x: x["name"])

    _t2c_cache["result"] = result
    _t2c_cache["ts"] = time.time()
    return result


@router.get("/config/data-dirs", response_model=list[DataDirEntry])
def get_data_dirs():
    """读取 default.yaml 的 data_dirs 配置，返回结构化列表。"""
    default_yaml = CONFIG_DIR / "default.yaml"
    if not default_yaml.exists():
        return []
    try:
        with open(default_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        data_dirs = cfg.get("data_dirs", {})
        entries = []
        for key, val in data_dirs.items():
            if isinstance(val, str):
                entries.append(DataDirEntry(key=key, path=val, simulator=key))
            else:
                entries.append(DataDirEntry(
                    key=key,
                    path=val.get("path", ""),
                    simulator=val.get("simulator", key),
                    file_suffix=val.get("file_suffix", "") or "",
                    transient_simulator=val.get("transient_simulator", "") or "",
                    transient_keywords=val.get("transient_keywords", []) or [],
                ))
        return entries
    except Exception as e:
        raise HTTPException(500, f"读取 data_dirs 失败: {e}")


@router.post("/config/data-dirs")
def save_data_dirs(req: DataDirsRequest):
    """保存 data_dirs 配置到 default.yaml，其余字段保持不变。"""
    default_yaml = CONFIG_DIR / "default.yaml"
    try:
        cfg: dict = {}
        if default_yaml.exists():
            with open(default_yaml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        data_dirs: dict = {}
        for entry in req.entries:
            has_transient = bool(entry.transient_simulator)
            has_suffix = bool(entry.file_suffix)
            has_keywords = bool(entry.transient_keywords)

            if not has_transient and not has_suffix and not has_keywords:
                # 简单形式：只有 path 和 simulator 相同时可用字符串，但保持 dict 更统一
                data_dirs[entry.key] = {
                    "path": entry.path,
                    "simulator": entry.simulator,
                }
            else:
                d: dict = {
                    "path": entry.path,
                    "simulator": entry.simulator,
                }
                if has_suffix:
                    d["file_suffix"] = entry.file_suffix
                if has_transient:
                    d["transient_simulator"] = entry.transient_simulator
                if has_keywords:
                    d["transient_keywords"] = entry.transient_keywords
                data_dirs[entry.key] = d

        cfg["data_dirs"] = data_dirs
        default_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(default_yaml, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, indent=2)
        # data_dirs 变更后场景列表会变，清空缓存
        _t2c_cache["result"] = None
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"保存 data_dirs 失败: {e}")
