"""Registry CRUD 路由：/api/registry。"""

import yaml
from fastapi import APIRouter, HTTPException

from piern.shared.runtime.paths import REGISTRY_PATH
from piern.synth.api.routers.config import invalidate_text2comp_scenarios_cache
from piern.synth.services.hdf5_data import validate_name

router = APIRouter()


def _load_registry_raw() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_registry_raw(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)


def _registry_key_parts(key: str) -> tuple[str, str | None]:
    raw = str(key or "").strip()
    parts = raw.split("/") if raw else []
    if len(parts) not in {1, 2}:
        raise HTTPException(400, "registry key 必须是 simulator 或 simulator/scenario")
    try:
        simulator = validate_name("simulator", parts[0])
        scenario = validate_name("scenario", parts[1]) if len(parts) == 2 else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return simulator, scenario


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_output_info(output_info: object) -> set[str]:
    if output_info is None:
        return set()
    if not isinstance(output_info, list) or len(output_info) == 0:
        raise HTTPException(400, "至少需要保留 1 个 output_info 输出定义")

    output_names = set()
    for index, item in enumerate(output_info):
        if not isinstance(item, dict):
            raise HTTPException(400, f"output_info[{index}] 必须是 JSON 对象")
        name = str(item.get("name", "")).strip()
        if not name:
            raise HTTPException(400, f"output_info[{index}].name 不能为空")
        if name in output_names:
            raise HTTPException(400, f"output_info.name 重复: {name}")
        output_names.add(name)
        slice_value = item.get("slice")
        if not isinstance(slice_value, list) or len(slice_value) != 2:
            raise HTTPException(400, f"output_info[{index}].slice 必须是 [start, end_or_null]")
        start, end = slice_value
        if not _is_plain_int(start) or start < 0:
            raise HTTPException(400, f"output_info[{index}].slice start 必须是非负整数")
        if end is not None and (not _is_plain_int(end) or end < start):
            raise HTTPException(400, f"output_info[{index}].slice end 必须是 null 或不小于 start 的整数")
    return output_names


def _validate_optional_positive_int(name: str, value: object) -> None:
    if value is not None and (not _is_plain_int(value) or value < 1):
        raise HTTPException(400, f"{name} 必须是正整数或 null")


def _validate_registry_entry(key: str, body: dict) -> None:
    if not isinstance(body, dict):
        raise HTTPException(400, "registry 条目必须是 JSON 对象")

    output_info = body.get("output_info")
    output_names = _validate_output_info(output_info)

    obs = body.get("observation_config")
    if obs is None:
        return
    if not isinstance(obs, dict):
        raise HTTPException(400, "observation_config 必须是 JSON 对象")

    channel_level = str(obs.get("channel_level", "row") or "row").lower()
    if channel_level not in {"row", "output", "output_info"}:
        raise HTTPException(400, "channel_level 必须是 row、output 或 output_info")
    is_output_level = channel_level in {"output", "output_info"}
    obs["channel_level"] = "output_info" if is_output_level else "row"

    channel_min = obs.get("channel_min")
    channel_max = obs.get("channel_max")
    _validate_optional_positive_int("channel_min", channel_min)
    _validate_optional_positive_int("channel_max", channel_max)
    if channel_min is not None and channel_max is not None and channel_max < channel_min:
        raise HTTPException(400, "channel_max 必须大于或等于 channel_min")

    fixed_channels = obs.get("fixed_channels", None)
    if fixed_channels is None:
        return
    if not isinstance(fixed_channels, list):
        raise HTTPException(400, "fixed_channels 必须是 null 或通道索引列表")
    if len(fixed_channels) == 0:
        raise HTTPException(
            400,
            "至少选择 1 个输出通道；如需全选，请将 fixed_channels 设为 null",
        )
    if is_output_level and not isinstance(output_info, list):
        raise HTTPException(400, "按输出维度采样需要先定义 output_info")

    invalid = []
    invalid_output = []
    for value in fixed_channels:
        if isinstance(value, bool):
            invalid.append(value)
        elif isinstance(value, int):
            if value < 0:
                invalid.append(value)
            elif is_output_level and isinstance(output_info, list) and value >= len(output_info):
                invalid_output.append(value)
        elif isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                invalid.append(value)
            elif is_output_level:
                if cleaned in output_names:
                    continue
                try:
                    idx = int(cleaned)
                except ValueError:
                    invalid_output.append(value)
                else:
                    if idx < 0 or idx >= len(output_info):
                        invalid_output.append(value)
        else:
            invalid.append(value)
    if invalid:
        raise HTTPException(
            400,
            f"fixed_channels 包含无效通道值: {invalid}",
        )
    if invalid_output:
        raise HTTPException(
            400,
            f"output_info 通道选择超出范围或不存在: {invalid_output}",
        )


@router.get("/registry")
def get_registry():
    """读取 registry.yaml，返回完整内容。"""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise HTTPException(500, f"读取 registry 失败: {e}")


@router.put("/registry/{key:path}")
def update_registry_entry(key: str, body: dict):
    """更新（或新建）registry 中的记录。

    key = "simulator"              → 更新 simulator 级字段（body 为完整条目）
    key = "simulator/scenario"     → 更新 simulator.scenarios.scenario（body 为字符串或 {"scenario_description": "..."}）
    """
    try:
        data = _load_registry_raw()
        simulator, scenario = _registry_key_parts(key)
        normalized_key = f"{simulator}/{scenario}" if scenario else simulator
        if scenario:
            sim_entry = data.setdefault(simulator, {})
            scenarios = sim_entry.setdefault("scenarios", {})
            # body 可以是 {"scenario_description": "..."} 或直接字符串
            desc = body.get("scenario_description", "") if isinstance(body, dict) else str(body)
            scenarios[scenario] = desc
        else:
            _validate_registry_entry(simulator, body)
            # simulator 级：合并 scenarios 子字段（不丢失已有场景描述）
            existing_scenarios = data.get(simulator, {}).get("scenarios", {}) if isinstance(data.get(simulator), dict) else {}
            data[simulator] = body
            if existing_scenarios and isinstance(body, dict):
                # body 中的 scenarios 优先，但保留 body 中没有的场景描述
                merged = {**existing_scenarios, **(body.get("scenarios") or {})}
                data[simulator]["scenarios"] = merged
        _save_registry_raw(data)
        invalidate_text2comp_scenarios_cache()
        return {"ok": True, "key": normalized_key}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"保存 registry 失败: {e}")


@router.delete("/registry/{key:path}")
def delete_registry_entry(key: str):
    """删除 registry 中的记录。

    key = "simulator"          → 删除整个 simulator 条目
    key = "simulator/scenario" → 只删除 simulator.scenarios.scenario
    """
    try:
        data = _load_registry_raw()
        simulator, scenario = _registry_key_parts(key)
        normalized_key = f"{simulator}/{scenario}" if scenario else simulator
        if scenario:
            sim_entry = data.get(simulator, {})
            scenarios = sim_entry.get("scenarios", {})
            if scenario not in scenarios:
                raise HTTPException(404, f"场景 '{normalized_key}' 不存在")
            del scenarios[scenario]
            if not scenarios:
                sim_entry.pop("scenarios", None)
        else:
            if simulator not in data:
                raise HTTPException(404, f"key '{simulator}' 不存在")
            del data[simulator]
        _save_registry_raw(data)
        invalidate_text2comp_scenarios_cache()
        return {"ok": True, "key": normalized_key}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"删除 registry 失败: {e}")
