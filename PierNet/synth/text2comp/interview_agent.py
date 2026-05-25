"""
多智能体任务注册框架（Interactive Registration）。

通过 6 步固定顺序对话引导用户逐步提供信息，
再由 4 个 Agent 流水线转换为 registry.yaml 条目。

流程：
  用户输入
    ↓ InterviewerAgent (temperature=0.7) — 生成问题
    ↓ ExtractorAgent   (temperature=0.1) — 解析回答 → JSON
    ↓ ValidatorAgent   (纯 Python)       — 校验格式
    ↓ 失败(<3次) → Interviewer 追问
    ↓ 成功 → 前端展示 + 等待用户确认
    ↓ 用户确认
    WriterAgent — 合并字段 → registry.yaml

6 步内容：
  1. 仿真器名称、物理域、控制方程  → domain_context, output_description
  2. 参数含义逐一确认（批量预推断）  → param_info
  3. 输出通道结构（Case A/B）       → output_info
  4. 通道采样策略                   → obs.fixed_channels (int indices), channel_name_template
  5. 时间采样模式                   → obs.fixed_time_mode, obs.time_modes
  6. 预览（纯 Python 渲染）+ 确认写入
"""

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from PierNet.core.llm_client import LLMClient

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# 合法的时间索引模式（every_N 通用，此处列出常用值供提示）
VALID_TIME_INDICES = {"monthly", "weekly", "full"}  # every_N 也合法，N 为任意正整数

STEP_LABELS = {
    0: "GitHub 预填充",
    1: "仿真器基本信息",
    2: "参数含义确认",
    3: "输出通道结构",
    4: "通道降采样策略",
    5: "时间采样模式",
    6: "预览与确认",
    7: "场景描述",   # scenario 模式专用
}

# GitHub 预填充可以覆盖的步骤（step number → 对应的 session 字段列表）
GITHUB_FILLABLE_STEPS = {
    1: ["domain_context", "output_description"],
    2: ["param_info"],
    3: ["output_info"],
    4: ["observation_config"],   # channel 部分
    5: ["observation_config"],   # time 部分
}

# ─────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────

@dataclass
class InterviewSession:
    session_id: str
    simulator: str
    scenario: str
    mode: str = "simulator"    # "simulator"（6步，写 simulator 级 key）| "scenario"（1步，写 scenario 级 key）
    step: int = 0              # 0=GitHub询问阶段, 1-6=正式步骤, 7=场景描述（scenario 模式）
    status: str = "interviewing"   # "interviewing"|"confirming"|"done"|"error"

    # HDF5 预加载（可选）
    param_names: list = field(default_factory=list)
    timeseries_shape: Optional[tuple] = None   # (n_samples, channels, timesteps)
    params_sample: Optional[object] = None     # np.ndarray，前3行，用于预推断

    # GitHub 预填充
    github_url: Optional[str] = None
    github_context: str = ""   # 抓取到的原始文本（README + 代码片段）
    prefilled_steps: set = field(default_factory=set)  # 已被 GitHub 预填的步骤编号

    # 累积字段
    domain_context: str = ""
    output_description: str = ""
    param_info: dict = field(default_factory=dict)
    output_info: list = field(default_factory=list)
    observation_config: dict = field(default_factory=dict)
    scenario_description: str = ""   # scenario 模式专用：该场景的一句话描述

    # 对话历史（全程累积，每条消息含 step 标记）
    history: list = field(default_factory=list)
    # [{"role": "assistant"|"user", "content": str, "step": int, "ts": float}]

    # 待确认的提取结果
    pending_extraction: dict = field(default_factory=dict)
    extraction_retries: int = 0
    MAX_RETRIES: int = 3

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    step: int
    step_label: str
    total_steps: int = 6
    question: Optional[str] = None
    extracted: Optional[dict] = None
    needs_confirmation: bool = False
    extraction_uncertain: bool = False
    done: bool = False
    saved: bool = False
    registry_key: Optional[str] = None
    error: Optional[str] = None
    hdf5_loaded: bool = False
    github_prefilled: Optional[dict] = None   # {"steps": [1,2,3], "summary": "从 README 预填了..."}

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "step_label": self.step_label,
            "total_steps": self.total_steps,
            "question": self.question,
            "extracted": self.extracted,
            "needs_confirmation": self.needs_confirmation,
            "extraction_uncertain": self.extraction_uncertain,
            "done": self.done,
            "saved": self.saved,
            "registry_key": self.registry_key,
            "error": self.error,
            "hdf5_loaded": self.hdf5_loaded,
            "github_prefilled": self.github_prefilled,
        }


# ─────────────────────────────────────────────────────────────────
# InterviewerAgent
# ─────────────────────────────────────────────────────────────────

class InterviewerAgent:
    """生成面向用户的自然语言问题（temperature=0.7）。"""

    SYSTEM_PROMPT = (
        "You are a helpful assistant guiding a user through registering a new physics simulation dataset "
        "into a metadata registry. Ask clear, concise questions in the user's language (Chinese preferred). "
        "Be friendly and direct. When asking about technical details, provide examples to help the user understand."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _call(self, prompt: str) -> str:
        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=600,
        ).strip()

    def generate_opening_question(self, session: InterviewSession) -> str:
        """为当前步骤生成开场问题。"""
        step = session.step
        sim = session.simulator
        if step == 1:
            hdf5_hint = ""
            if session.timeseries_shape:
                _, ch, ts = session.timeseries_shape
                hdf5_hint = f"\n\n（已检测到数据集：输出形状 {ch} 通道 × {ts} 时间步）"
            prompt = f"""我们正在为 simulator="{sim}" 注册 Simulator 级元数据（物理域背景、参数含义、输出结构等）。{hdf5_hint}
这些信息将被该 simulator 下的所有场景共享复用。

请用一条消息向用户询问以下三件事：
1. 仿真器/框架的名称（如 FEniCS、OpenFOAM、自研代码等）
2. 物理域和控制方程（如 PDE 类型、数值方法）
3. 输出张量代表什么（物理量、单位、空间分辨率）

如果 HDF5 未提供，还需询问：输出有多少通道（channels）和多少时间步（timesteps）。
{"（HDF5 已提供输出形状，无需再问通道数和时间步数）" if session.timeseries_shape else "（未提供 HDF5，请同时询问输出形状）"}

用中文，语气友好，把三个问题合并成一段话。"""
            return self._call(prompt)

        elif step == 2:
            n_params = len(session.param_names)
            if n_params > 0:
                # 已有参数名，展示预推断结果让用户纠错
                param_table = "\n".join(
                    f"  [{i+1}] {name}: {info[0] if isinstance(info, list) else info} ({info[1] if isinstance(info, list) else '-'})"
                    for i, (name, info) in enumerate(session.param_info.items())
                ) if session.param_info else "\n".join(f"  [{i+1}] {name}" for i, name in enumerate(session.param_names))

                prompt = f"""我们已从 HDF5 自动推断了 {n_params} 个参数的含义，请向用户展示并请其确认或纠正：

{param_table}

告诉用户：
- 如果全部正确，回复"全部正确"或"OK"
- 如果需要修改，指出序号和正确的含义/单位，如"#3 应该是储水系数，单位 1/m"

用中文，简洁展示，不要逐条解释。"""
            else:
                prompt = """我们需要了解这个仿真器的输入参数含义。

请向用户询问：请列出所有参数名称及其物理含义和单位。
例如格式：E - 杨氏模量 (GPa)，nu - 泊松比 (无量纲)

用中文，给出格式示例。"""
            return self._call(prompt)

        elif step == 3:
            ch = session.timeseries_shape[1] if session.timeseries_shape else "unknown"
            prompt = f"""现在需要确定输出张量的通道结构。输出有 {ch} 个通道。

请向用户询问：这 {ch} 个通道是：
(A) 同一物理量在不同空间位置的测量值？（如 5 口观测井的水头，10 台发电机的转子角）
(B) 不同物理量垂直拼接？（如 [0:14] 母线电压 + [14:28] 相角 + [28:43] 线路功率）

请用户选择 A 或 B，并描述每个通道（或每组通道）代表什么物理量及单位。
同时询问是否有中文名称。

用中文，给出具体例子帮助用户理解。"""
            return self._call(prompt)

        elif step == 4:
            ch = session.timeseries_shape[1] if session.timeseries_shape else "unknown"
            n_oi = len(session.output_info)
            names = [o.get("name", f"output_{i}") for i, o in enumerate(session.output_info)]
            names_hint = f"（通道组：{', '.join(names)}）" if n_oi > 1 else ""

            prompt = f"""输出共有 {ch} 个通道{names_hint}，需要确定通道降采样策略。

请向用户询问：
1. 默认选取全部 {ch} 个通道，还是随机选取子集？
2. 如果是子集，最少选几个？最多选几个？（用 0-based 整数索引）
3. 每个通道有没有名字模板？（如"第{{i}}号观测井"或"node {{i}}"，可省略）

说明：通道选取始终使用 0-based 整数索引（如 [0,1,2,3,4]），null 表示全选。

用中文，给出示例格式。"""
            return self._call(prompt)

        elif step == 5:
            ts = session.timeseries_shape[2] if session.timeseries_shape else "unknown"
            prompt = f"""输出有 {ts} 个时间步。

请向用户询问：适合哪些时间降采样模式？

合法的模式名称（indices）：monthly, weekly, quarterly, bimonthly, every_100, every_10, every_2, full

根据时间步数的参考：
- 365 步（日分辨率一年）→ 建议 monthly/weekly/quarterly
- 1000 步（100Hz 暂态）→ 建议 every_100/every_10/full
- ≤20 步 → 建议 full

请用户：
1. 选择 2-3 种模式及其权重（权重之和为 1.0）
2. 指定默认模式（权重最高的那个）

用中文，给出示例。"""
            return self._call(prompt)

        else:
            return "请确认以上所有信息无误后，我将写入 registry.yaml。"

    def generate_followup(
        self,
        session: InterviewSession,
        validation_errors: list[str],
    ) -> str:
        """校验失败时生成追问。"""
        error_text = "\n".join(f"- {e}" for e in validation_errors)
        # 获取当前步骤最后几条对话
        step_history = [h for h in session.history if h.get("step") == session.step]
        last_user = next((h["content"] for h in reversed(step_history) if h["role"] == "user"), "")

        prompt = f"""用户的回答存在以下问题，需要追问：

{error_text}

用户上一条回答："{last_user[:200]}"

请用中文生成一条简短的追问，针对具体错误给出纠正指引，并给出正确格式的示例。
不要重复列出所有要求，只针对出错的地方。"""
        return self._call(prompt)

    def generate_confirmation_summary(
        self, session: InterviewSession, extracted: dict
    ) -> str:
        """提取成功后，生成摘要让用户确认。"""
        try:
            extracted_str = json.dumps(extracted, ensure_ascii=False, indent=2)
        except Exception:
            extracted_str = str(extracted)

        prompt = f"""提取结果如下，请向用户展示并请其确认：

```json
{extracted_str[:800]}
```

用一句话告诉用户：请确认以上信息是否正确，如无问题请回复"确认"，如需修改请直接编辑 JSON。
用中文，简洁。"""
        return self._call(prompt)

    def render_final_preview(self, session: InterviewSession) -> str:
        """Step 6：纯 Python 渲染最终 registry 条目（不调 LLM）。

        新两层结构预览：
          simulator:           ← 顶层 key
            domain_context: ...
            param_info: ...
            ...
            scenarios:
              scenario_name: "（已有场景描述，不变）"
        """
        entry = _build_entry_from_session(session)
        # 在预览中加入 scenarios 占位（实际写入时会保留已有场景）
        preview_entry = {session.simulator: entry}
        try:
            yaml_str = yaml.dump(preview_entry, allow_unicode=True, sort_keys=True, indent=2)
        except Exception:
            yaml_str = json.dumps({session.simulator: entry}, ensure_ascii=False, indent=2)

        return (
            f"以下是将写入 registry.yaml 的完整条目（key: `{session.simulator}`）：\n\n"
            f"```yaml\n{yaml_str}\n```\n\n"
            "请确认无误后点击「确认写入」，或直接编辑 JSON 后确认。\n"
            "（已有的场景描述 `scenarios` 子字段将自动保留）"
        )


# ─────────────────────────────────────────────────────────────────
# ExtractorAgent
# ─────────────────────────────────────────────────────────────────

class ExtractorAgent:
    """将用户回答解析为结构化 JSON（temperature=0.1）。"""

    SYSTEM_PROMPT = (
        "You are a data extraction assistant. Given a conversation about a physics simulation dataset, "
        "extract the requested structured metadata fields as valid JSON. "
        "Return ONLY valid JSON, no markdown fences, no explanation."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _call(self, prompt: str, max_tokens: int = 2000) -> str:
        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=max_tokens,
        ).strip()

    def _parse_json(self, text: str) -> dict:
        """解析 LLM 返回的 JSON，去掉 markdown fence。"""
        text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        return json.loads(text.strip())

    def pre_infer_params(self, session: InterviewSession) -> dict:
        """
        Step 2 批量预推断：复用 auto_register 的 _build_domain_prompt 逻辑，
        返回 {"param_info": {name: [meaning, unit], ...}}。
        仅在有 HDF5 数据时调用。
        """
        if not session.param_names or session.timeseries_shape is None:
            return {}

        from PierNet.synth.text2comp.auto_register import _build_domain_prompt

        # 构造 dummy params_sample（若有则用真实数据）
        if session.params_sample is not None:
            params_sample = session.params_sample
        else:
            params_sample = np.zeros((3, len(session.param_names)))

        prompt = _build_domain_prompt(
            simulator=session.simulator,
            scenario_name=session.scenario,
            param_names=session.param_names,
            timeseries_shape=session.timeseries_shape,
            params_sample=params_sample,
        )

        try:
            text = self._call(prompt, max_tokens=2000)
            result = self._parse_json(text)
            return {"param_info": result.get("param_info", {})}
        except Exception as e:
            logger.warning(f"参数预推断失败: {e}")
            # 返回空推断（含参数名，含义待用户填写）
            return {"param_info": {name: ["pending", "-"] for name in session.param_names}}

    def extract(self, step: int, session: InterviewSession, user_message: str) -> dict:
        """
        解析用户回答，返回该步骤的结构化字段。
        抛出 ValueError 或 json.JSONDecodeError 表示解析失败。
        """
        # 当前步骤的对话历史（不含最新用户消息）
        step_history = [
            h for h in session.history
            if h.get("step") == step and h["role"] == "assistant"
        ]
        last_question = step_history[-1]["content"] if step_history else ""

        prompt = self._build_extraction_prompt(step, session, user_message, last_question)
        text = self._call(prompt)
        result = self._parse_json(text)
        return result

    def _build_extraction_prompt(
        self,
        step: int,
        session: InterviewSession,
        user_message: str,
        last_question: str,
    ) -> str:
        sim = session.simulator
        sc = session.scenario
        ch = session.timeseries_shape[1] if session.timeseries_shape else "unknown"
        ts = session.timeseries_shape[2] if session.timeseries_shape else "unknown"

        context = f'Simulator: "{sim}", Scenario: "{sc}", Output shape: ({ch} channels, {ts} timesteps)'

        if step == 1:
            no_hdf5_extra = ""
            if not session.timeseries_shape:
                no_hdf5_extra = '\n  "n_channels": <integer>,\n  "n_timesteps": <integer>,'
            return f"""Extract metadata from the user's answer about their physics simulation dataset.

Context: {context}
Question asked: {last_question}
User's answer: {user_message}

Return JSON with exactly these keys:
{{
  "domain_context": "One paragraph: physical domain, governing equations, numerical method, what output represents.",
  "output_description": "Short phrase. MUST contain literal {{ch}} and {{ts}} placeholders, e.g. '{{ch}} observation wells × {{ts}} days of hydraulic head (m)'"{no_hdf5_extra}
}}

Rules:
- domain_context must be at least 50 characters.
- output_description MUST contain the literal strings {{ch}} and {{ts}}.
- Return JSON only."""

        elif step == 2:
            base_param_info = json.dumps(session.param_info, ensure_ascii=False, indent=2) if session.param_info else "{}"
            param_names_str = json.dumps(session.param_names)
            return f"""The user is confirming/correcting parameter meanings for a physics simulation dataset.

Context: {context}
Parameter names from dataset: {param_names_str}
Pre-inferred param_info (base): {base_param_info[:1000]}
User's corrections/confirmation: {user_message}

Apply the user's corrections to the base param_info. If the user said "all correct" or similar, keep the base unchanged.
For each correction like "#3 should be X, unit Y", update that parameter.

Return JSON:
{{
  "param_info": {{
    "<param_name>": ["<physical meaning>", "<unit>"],
    ...
  }}
}}

Include ALL parameter names from the dataset. Return JSON only."""

        elif step == 3:
            return f"""Extract the output channel structure from the user's description.

Context: {context}
User's answer: {user_message}

Decide: Case A (all channels = same physical quantity at different locations → ONE output_info entry with slice [0, null])
        Case B (different physical quantities stacked → multiple entries with correct slices)

Return JSON:
{{
  "output_info": [
    {{
      "name": "<snake_case_name>",
      "name_zh": "<Chinese name>",
      "description": "<physical meaning>",
      "unit": "<unit>",
      "slice": [<start>, <end_or_null>]
    }}
  ]
}}

Return JSON only."""

        elif step == 4:
            n_ch = session.timeseries_shape[1] if session.timeseries_shape else "unknown"
            return f"""Extract the channel sampling configuration from the user's answer.

Context: {context}
Total channels in dataset: {n_ch}
User's answer: {user_message}

Return JSON:
{{
  "fixed_channels": <list_of_int_or_null>,
  "channel_name_template": "<optional, template with {{i}}, e.g. 'well {{i}}', 'generator {{i}}'>",
  "channel_name_template_zh": "<optional Chinese template with {{i}}>"
}}

Rules:
- fixed_channels: list of 0-based integer indices (e.g. [0,1,2]), or null to select all channels.
- channel_name_template / channel_name_template_zh: optional. Include only when channels have meaningful names (e.g. observation wells, generators). Must contain {{i}}.
- Return JSON only."""

        elif step == 5:
            return f"""Extract the time sampling configuration from the user's answer.

Context: {context}
User's answer: {user_message}

Valid indices: "monthly" (12 pts), "weekly" (52 pts), "full" (all pts), "every_N" (every N steps, N is any positive integer)

Return JSON:
{{
  "fixed_time_mode": "<mode_name, e.g. monthly / weekly / full / every_10>",
  "time_modes": [
    {{
      "name": "<same as fixed_time_mode>",
      "indices": "<same as fixed_time_mode>",
      "desc_en": "<English description, e.g. 'monthly, 12 time points'>",
      "desc_zh": "<Chinese description, e.g. '月度，共12个时间点'>"
    }}
  ]
}}

Rules:
- Only one time mode (the fixed one). No weights needed.
- fixed_time_mode must match the name in time_modes.
- Return JSON only."""

        else:
            return f"User message: {user_message}\nReturn empty JSON: {{}}"


# ─────────────────────────────────────────────────────────────────
# ValidatorAgent
# ─────────────────────────────────────────────────────────────────

class ValidatorAgent:
    """纯 Python 校验，不调 LLM。复用 auto_register 的校验逻辑。"""

    def validate(
        self, step: int, extracted: dict, session: InterviewSession
    ) -> tuple[bool, list[str]]:
        """返回 (is_valid, error_list)。"""
        errors: list[str] = []

        try:
            if step == 1:
                errors.extend(self._validate_step1(extracted, session))
            elif step == 2:
                errors.extend(self._validate_step2(extracted, session))
            elif step == 3:
                errors.extend(self._validate_step3(extracted))
            elif step == 4:
                errors.extend(self._validate_step4(extracted, session))
            elif step == 5:
                errors.extend(self._validate_step5(extracted))
        except Exception as e:
            errors.append(f"校验异常: {e}")

        return len(errors) == 0, errors

    def _validate_step1(self, d: dict, session: InterviewSession) -> list[str]:
        errors = []
        dc = d.get("domain_context", "")
        if not dc or len(dc) < 50:
            errors.append("domain_context 太短（需至少50字符），请更详细地描述物理域和控制方程")
        od = d.get("output_description", "")
        if "{ch}" not in od or "{ts}" not in od:
            errors.append("output_description 必须包含 {ch} 和 {ts} 占位符，例如：'{ch} 观测井 × {ts} 天的水力水头'")
        if not session.timeseries_shape:
            if "n_channels" not in d:
                errors.append("未提供 HDF5，需要指定输出通道数（n_channels）")
            if "n_timesteps" not in d:
                errors.append("未提供 HDF5，需要指定时间步数（n_timesteps）")
        return errors

    def _validate_step2(self, d: dict, session: InterviewSession) -> list[str]:
        errors = []
        pi = d.get("param_info", {})
        if not isinstance(pi, dict) or len(pi) == 0:
            errors.append("param_info 不能为空")
            return errors
        if session.param_names:
            missing = [n for n in session.param_names if n not in pi]
            if missing:
                errors.append(f"以下参数缺失：{missing[:5]}{'...' if len(missing) > 5 else ''}")
        for name, info in pi.items():
            if not isinstance(info, (list, tuple)) or len(info) < 2:
                errors.append(f"参数 {name} 的格式应为 [含义, 单位]，如 ['水力传导系数', 'm/day']")
                break
        return errors

    def _validate_step3(self, d: dict) -> list[str]:
        errors = []
        oi = d.get("output_info", [])
        if not isinstance(oi, list) or len(oi) == 0:
            errors.append("output_info 不能为空，至少需要一个通道描述")
            return errors
        for i, entry in enumerate(oi):
            for k in ("name", "description", "unit", "slice"):
                if k not in entry:
                    errors.append(f"output_info[{i}] 缺少字段 '{k}'")
            sl = entry.get("slice", [])
            if not isinstance(sl, list) or len(sl) != 2:
                errors.append(f"output_info[{i}].slice 格式应为 [start, end_or_null]")
        return errors

    def _validate_step4(self, d: dict, session: InterviewSession) -> list[str]:
        errors = []
        fc = d.get("fixed_channels")
        if fc is not None:
            if not isinstance(fc, list):
                errors.append("fixed_channels 必须是整数列表（0-based）或 null（全选）")
            elif fc and not all(isinstance(i, int) for i in fc):
                errors.append("fixed_channels 列表中的元素必须全是整数")
        tmpl = d.get("channel_name_template", "")
        if tmpl and "{i}" not in tmpl:
            errors.append(f"channel_name_template='{tmpl}' 必须包含 {{i}} 占位符")
        return errors

    def _validate_step5(self, d: dict) -> list[str]:
        errors = []
        modes = d.get("time_modes", [])
        fixed = d.get("fixed_time_mode", "")

        if not modes:
            errors.append("time_modes 不能为空")
            return errors

        for i, m in enumerate(modes):
            idx = m.get("indices", "")
            # 合法：monthly/weekly/full 或 every_N
            if idx not in VALID_TIME_INDICES and not (idx.startswith("every_") and idx[6:].isdigit()):
                errors.append(
                    f"time_modes[{i}].indices='{idx}' 不合法，"
                    f"支持：monthly / weekly / full / every_N（N为正整数）"
                )

        mode_names = {m.get("name") for m in modes}
        if fixed not in mode_names:
            errors.append(f"fixed_time_mode='{fixed}' 不在 time_modes 名称列表中：{mode_names}")

        return errors


# ─────────────────────────────────────────────────────────────────
# WriterAgent
# ─────────────────────────────────────────────────────────────────

class WriterAgent:
    """将 session 中的字段合并为 registry 条目并写入 YAML。"""

    def write(self, session: InterviewSession, registry_path: Path) -> str:
        """写入 registry.yaml（新两层结构），返回写入的 key。

        simulator 模式：写入 registry[simulator]，包含完整 5 字段
        scenario 模式：写入 registry[simulator]["scenarios"][scenario]，只写描述字符串
        """
        # 加载现有 registry
        existing: dict = {}
        if registry_path.exists():
            with open(registry_path, "r", encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}

        if session.mode == "scenario":
            # 确保 simulator 条目存在，在其 scenarios 子字段写入场景描述
            sim_entry = existing.setdefault(session.simulator, {})
            scenarios = sim_entry.setdefault("scenarios", {})
            scenarios[session.scenario] = session.scenario_description
            key = f"{session.simulator}/{session.scenario}"
        else:
            # simulator 模式：写入顶层 simulator key，保留已有 scenarios 子字段
            entry = _build_entry_from_session(session)
            existing_scenarios = existing.get(session.simulator, {}).get("scenarios", {})
            if existing_scenarios:
                entry["scenarios"] = existing_scenarios
            existing[session.simulator] = entry
            key = session.simulator

        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_path, "w", encoding="utf-8") as f:
            yaml.dump(existing, f, allow_unicode=True, sort_keys=True, indent=2)

        logger.info(f"已写入 registry: {key} → {registry_path}")
        return key


def _build_entry_from_session(session: InterviewSession) -> dict:
    """从 session 字段组装完整的 registry 条目。"""
    obs_cfg = dict(session.observation_config)

    entry = {
        "domain_context": session.domain_context,
        "output_description": session.output_description,
        "param_info": session.param_info,
        "output_info": session.output_info,
        "observation_config": obs_cfg,
    }
    # 去掉空字段
    return {k: v for k, v in entry.items() if v}


# ─────────────────────────────────────────────────────────────────
# GitHub 预填充
# ─────────────────────────────────────────────────────────────────

def _parse_github_url(url: str) -> Optional[tuple[str, str]]:
    """
    从 GitHub URL 解析 owner/repo。
    支持：
      https://github.com/owner/repo
      https://github.com/owner/repo/tree/branch/...
      https://github.com/owner/repo/blob/...
    返回 (owner, repo) 或 None。
    """
    m = re.match(r'https?://github\.com/([^/]+)/([^/?\s#]+)', url.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


def _fetch_github_content(url: str, timeout: int = 15) -> tuple[str, str]:
    """
    抓取 GitHub 仓库的关键内容：README + 顶层 Python 文件片段。

    Returns:
        (content_text, status_message)
        content_text: 拼接的文本，供 LLM 分析
        status_message: 给用户看的状态描述
    """
    if not _REQUESTS_AVAILABLE:
        return "", "requests 库未安装，无法抓取 GitHub 内容"

    parsed = _parse_github_url(url)
    if not parsed:
        return "", f"无法解析 GitHub URL：{url}"

    owner, repo = parsed
    headers = {"Accept": "application/vnd.github.v3+json"}
    # 如果有 GITHUB_TOKEN 环境变量，加入认证头（避免 rate limit）
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    parts: list[str] = []
    status_parts: list[str] = []

    # 1. 抓 README
    for readme_name in ["README.md", "readme.md", "README.rst", "README.txt", "README"]:
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{readme_name}"
        try:
            resp = _requests.get(raw_url, timeout=timeout, headers=headers)
            if resp.status_code == 200:
                text = resp.text[:8000]  # 最多 8000 字符
                parts.append(f"=== {readme_name} ===\n{text}")
                status_parts.append(f"README ({len(text)} 字符)")
                break
        except Exception:
            pass

    # 如果 main 分支没有，试 master
    if not parts:
        for readme_name in ["README.md", "readme.md"]:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{readme_name}"
            try:
                resp = _requests.get(raw_url, timeout=timeout, headers=headers)
                if resp.status_code == 200:
                    text = resp.text[:8000]
                    parts.append(f"=== {readme_name} ===\n{text}")
                    status_parts.append(f"README ({len(text)} 字符)")
                    break
            except Exception:
                pass

    # 2. 抓仓库顶层文件列表，找 Python 文件
    try:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        resp = _requests.get(api_url, timeout=timeout, headers=headers)
        if resp.status_code == 200:
            files = resp.json()
            py_files = [f for f in files if isinstance(f, dict) and f.get("name", "").endswith(".py")][:3]
            for f_info in py_files:
                try:
                    raw_url = f_info.get("download_url", "")
                    if raw_url:
                        r2 = _requests.get(raw_url, timeout=timeout)
                        if r2.status_code == 200:
                            snippet = r2.text[:3000]
                            parts.append(f"=== {f_info['name']} (前3000字符) ===\n{snippet}")
                            status_parts.append(f_info["name"])
                except Exception:
                    pass
    except Exception:
        pass

    if not parts:
        return "", f"无法获取仓库 {owner}/{repo} 的内容（可能是私有仓库或网络问题）"

    content = "\n\n".join(parts)
    status = f"已获取：{', '.join(status_parts)}"
    return content, status


_GITHUB_PREFILL_SYSTEM = (
    "You are a scientific computing expert. Given the GitHub repository content of a physics simulation tool, "
    "extract metadata for a simulation dataset registry. "
    "Return ONLY valid JSON, no markdown fences, no explanation."
)

_GITHUB_PREFILL_PROMPT = """
Based on the following GitHub repository content, extract as much metadata as possible for a physics simulation dataset registry.

Repository: {simulator}/{scenario}

GitHub Content:
{content}

Return a JSON object with any of these fields you can confidently infer (omit fields you're unsure about):

{{
  "domain_context": "One paragraph: physical domain, governing equations, numerical method, what output represents. Must be >50 chars.",
  "output_description": "Short phrase with {{ch}} and {{ts}} placeholders, e.g. '{{ch}} nodes × {{ts}} timesteps of displacement (mm)'",
  "param_info": {{
    "<param_name>": ["<physical meaning>", "<unit>"],
    ...
  }},
  "output_info": [
    {{
      "name": "<snake_case>",
      "name_zh": "<Chinese name>",
      "description": "<physical meaning>",
      "unit": "<unit>",
      "slice": [0, null]
    }}
  ],
  "observation_config": {{
    "fixed_channels": null,
    "channel_name_template": "channel {{i}}",
    "channel_name_template_zh": "第{{i}}个通道",
    "fixed_time_mode": "full",
    "time_modes": [
      {{"name": "full", "indices": "full", "desc_en": "full time series", "desc_zh": "全量时间序列"}}
    ]
  }},
  "confidence": {{
    "domain_context": 0.0-1.0,
    "output_description": 0.0-1.0,
    "param_info": 0.0-1.0,
    "output_info": 0.0-1.0,
    "observation_config": 0.0-1.0
  }}
}}

Rules:
- Only include fields you can confidently infer from the repository content.
- output_description MUST contain literal {{ch}} and {{ts}} if included.
- confidence: 0.9+ means very confident, 0.6-0.9 means reasonably confident, <0.6 means uncertain (skip the field).
- If param_info cannot be inferred, omit it entirely.
- Return JSON only.
"""

# 置信度阈值：低于此值的字段视为"不确定"，仍需用户确认
_CONFIDENCE_THRESHOLD = 0.6


def _prefill_from_github(
    session: InterviewSession,
    content: str,
    llm: LLMClient,
) -> dict:
    """
    用 LLM 从 GitHub 内容预填充 session 字段。

    Returns:
        prefill_summary: {"steps": [1,2,3], "summary": "...", "fields": {...}}
    """
    prompt = _GITHUB_PREFILL_PROMPT.format(
        simulator=session.simulator,
        scenario=session.scenario,
        content=content[:12000],  # 限制 context 长度
    )

    try:
        text = llm.generate(
            prompt=prompt,
            system_prompt=_GITHUB_PREFILL_SYSTEM,
            temperature=0.1,
            max_tokens=3000,
        ).strip()
        text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        result = json.loads(text.strip())
    except Exception as e:
        logger.warning(f"GitHub 预填充 LLM 解析失败: {e}")
        return {}

    confidence = result.pop("confidence", {})
    filled_steps: list[int] = []
    filled_fields: dict = {}

    # Step 1: domain_context + output_description
    # Bug #9 fix: 两个字段都必须成功才算 step 1 预填充，避免一个为空
    dc_ok = False
    od_ok = False
    if "domain_context" in result and confidence.get("domain_context", 0) >= _CONFIDENCE_THRESHOLD:
        dc = result["domain_context"]
        if isinstance(dc, str) and len(dc) >= 50:
            session.domain_context = dc
            filled_fields["domain_context"] = dc
            dc_ok = True
    if "output_description" in result and confidence.get("output_description", 0) >= _CONFIDENCE_THRESHOLD:
        od = result["output_description"]
        if isinstance(od, str) and "{ch}" in od and "{ts}" in od:
            session.output_description = od
            filled_fields["output_description"] = od
            od_ok = True
    if dc_ok and od_ok:
        filled_steps.append(1)
    else:
        # 部分填充：清除已写入的字段，保持 step 1 为空，走正常流程
        if dc_ok and not od_ok:
            session.domain_context = ""
            filled_fields.pop("domain_context", None)
        if od_ok and not dc_ok:
            session.output_description = ""
            filled_fields.pop("output_description", None)

    # Step 2: param_info
    # Bug #3 fix: 若 HDF5 已预推断 param_info，GitHub 预填充不覆盖（HDF5 更可靠）
    if "param_info" in result and confidence.get("param_info", 0) >= _CONFIDENCE_THRESHOLD:
        pi = result["param_info"]
        if isinstance(pi, dict) and pi:
            if not session.param_info:  # 只在没有 HDF5 预推断结果时才用 GitHub 的
                session.param_info = pi
                filled_fields["param_info"] = pi
                filled_steps.append(2)
            else:
                logger.info("param_info 已由 HDF5 预推断，跳过 GitHub 覆盖")

    # Step 3: output_info
    if "output_info" in result and confidence.get("output_info", 0) >= _CONFIDENCE_THRESHOLD:
        oi = result["output_info"]
        if isinstance(oi, list) and oi:
            session.output_info = oi
            filled_fields["output_info"] = oi
            filled_steps.append(3)

    # Steps 4+5: observation_config
    if "observation_config" in result and confidence.get("observation_config", 0) >= _CONFIDENCE_THRESHOLD:
        obs = result["observation_config"]
        if isinstance(obs, dict) and obs:
            session.observation_config = obs
            filled_fields["observation_config"] = obs
            filled_steps.extend([4, 5])

    session.prefilled_steps = set(filled_steps)

    # 生成摘要
    field_names = {
        "domain_context": "物理域描述",
        "output_description": "输出描述",
        "param_info": f"参数含义（{len(filled_fields.get('param_info', {}))} 个）",
        "output_info": f"输出通道结构（{len(filled_fields.get('output_info', []))} 个）",
        "observation_config": "观测配置",
    }
    filled_names = [field_names[k] for k in filled_fields if k in field_names]

    summary = (
        f"从 GitHub 仓库预填充了 {len(filled_names)} 个字段：{', '.join(filled_names)}。"
        if filled_names
        else "GitHub 仓库内容不足以预填充任何字段，将进行正常注册流程。"
    )

    return {
        "steps": filled_steps,
        "summary": summary,
        "fields": filled_fields,
    }


# ─────────────────────────────────────────────────────────────────
# 模块级会话管理
# ─────────────────────────────────────────────────────────────────

_interview_sessions: dict[str, InterviewSession] = {}


def _make_llm(llm_cfg: Optional[dict]) -> LLMClient:
    """从配置字典创建 LLMClient。"""
    cfg = llm_cfg or {}
    provider = cfg.get("provider", "siliconflow")
    model = cfg.get("model", "Qwen/Qwen2.5-72B-Instruct")

    # api_key：优先用配置，其次从环境变量读取
    api_key = cfg.get("api_key")
    if not api_key:
        env_map = {
            "siliconflow": ["SILICONFLOW_API_KEY"],
            "openai":      ["OPENAI_API_KEY"],
            "deepseek":    ["DEEPSEEK_API_KEY"],
            "anthropic":   ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
        }
        for env_var in env_map.get(provider, []):
            val = os.getenv(env_var)
            if val:
                api_key = val
                break

    return LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=cfg.get("base_url"),
        thinking=cfg.get("thinking"),
        max_retries=cfg.get("max_retries", 3),
        timeout=cfg.get("timeout", 60),
    )


def create_session(
    simulator: str,
    scenario: str,
    hdf5_path: Optional[str] = None,
    llm_cfg: Optional[dict] = None,
    registry_path: Optional[Path] = None,
    mode: str = "simulator",
) -> tuple[str, AgentResponse]:
    """
    创建新的面试会话。

    mode="simulator"：完整 6 步流程，注册 simulator 级元数据（key=simulator）
    mode="scenario"：极简 1 步流程，只注册场景描述（key=simulator/scenario）

    若提供 hdf5_path，预加载 param_names 和 timeseries_shape，
    并用 ExtractorAgent 批量预推断 param_info（Step 2 起点）。

    Returns:
        (session_id, AgentResponse with step=1/7 question)
    """
    session_id = "iv-" + str(uuid.uuid4())[:8]
    session = InterviewSession(
        session_id=session_id,
        simulator=simulator,
        scenario=scenario,
        mode=mode,
    )

    hdf5_loaded = False

    # 尝试加载 HDF5
    if hdf5_path:
        try:
            from PierNet.core.storage import load_dataset
            ts_arr, params_arr, param_names = load_dataset(str(hdf5_path))
            session.param_names = list(param_names)
            session.timeseries_shape = ts_arr.shape  # (n_samples, ch, ts)
            session.params_sample = params_arr[:3] if len(params_arr) >= 3 else params_arr
            hdf5_loaded = True
            logger.info(f"HDF5 已加载: shape={ts_arr.shape}, params={len(param_names)}")
        except Exception as e:
            logger.warning(f"HDF5 加载失败（继续无 HDF5 模式）: {e}")

    # 存储会话（含 llm_cfg 和 registry_path 供后续使用）
    session._llm_cfg = llm_cfg  # type: ignore[attr-defined]
    session._registry_path = registry_path  # type: ignore[attr-defined]
    _interview_sessions[session_id] = session

    # 创建 Agent 实例
    llm = _make_llm(llm_cfg)
    extractor = ExtractorAgent(llm)

    # Step 2 预推断（有 HDF5 时）
    if hdf5_loaded and session.param_names:
        try:
            pre_result = extractor.pre_infer_params(session)
            session.param_info = pre_result.get("param_info", {})
            logger.info(f"参数预推断完成: {len(session.param_info)} 个参数")
        except Exception as e:
            logger.warning(f"参数预推断失败: {e}")

    # ── scenario 模式：直接跳到 step 7，只问场景描述 ──────────────
    if mode == "scenario":
        session.step = 7
        # 尝试从 registry 读取已有的 simulator 级 domain context 作为参考
        sim_context_hint = ""
        if registry_path and registry_path.exists():
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    reg = yaml.safe_load(f) or {}
                sim_entry = reg.get(simulator, {})
                if sim_entry.get("domain_context"):
                    sim_context_hint = f"\n\n（参考：{simulator} 的物理域描述已注册，无需重复填写）"
            except Exception:
                pass

        q = (
            f"正在为场景 **{simulator}/{scenario}** 注册描述。{sim_context_hint}\n\n"
            f"请用一句话描述该场景的具体物理设置，例如：\n"
            f"- 「单层均匀含水层，均匀水力传导系数」\n"
            f"- 「IEEE 14节点系统，基础负荷工况」\n"
            f"- 「海岸含水层，受海水入侵边界条件影响」\n\n"
            f"这句话将用于 LLM 生成模板时描述该场景的物理背景。"
        )
        session.history.append({"role": "assistant", "content": q, "step": 7, "ts": time.time()})
        return session_id, AgentResponse(
            step=7,
            step_label=STEP_LABELS[7],
            total_steps=7,
            question=q,
            hdf5_loaded=hdf5_loaded,
        )

    # ── simulator 模式：Step 0 询问 GitHub 链接 ───────────────────
    step0_q = (
        f"我们正在为 **{simulator}** 注册 Simulator 级元数据。\n\n"
        "如果该仿真器有 GitHub 仓库，请粘贴链接，我可以自动从 README 和代码中预填大部分信息。\n"
        "没有的话直接回复「跳过」即可，进入正常注册流程。\n\n"
        "例如：`https://github.com/owner/repo`"
    )
    session.history.append({
        "role": "assistant", "content": step0_q,
        "step": 0, "ts": time.time(),
    })

    resp = AgentResponse(
        step=0,
        step_label=STEP_LABELS[0],
        question=step0_q,
        hdf5_loaded=hdf5_loaded,
    )
    return session_id, resp


def _handle_github_step(
    session: InterviewSession,
    user_message: str,
    llm: LLMClient,
) -> AgentResponse:
    """处理 step=0 的用户输入（GitHub 链接或跳过）。"""
    interviewer = InterviewerAgent(llm)

    # 检测是否跳过
    skip_keywords = {"跳过", "skip", "no", "没有", "无", "n", "none", "pass", "略过", "不需要"}
    msg_lower = user_message.strip().lower()
    is_skip = msg_lower in skip_keywords or len(msg_lower) < 5

    # 检测是否包含 GitHub URL
    github_url_match = re.search(r'https?://github\.com/[^\s]+', user_message)
    github_url = github_url_match.group(0) if github_url_match else None

    if is_skip and not github_url:
        # 直接跳到 step 1
        session.step = 1
        question = interviewer.generate_opening_question(session)
        session.history.append({"role": "assistant", "content": question, "step": 1, "ts": time.time()})
        return AgentResponse(
            step=1,
            step_label=STEP_LABELS[1],
            question=question,
        )

    if not github_url:
        # 用户输入了什么但不像 URL，追问
        followup = (
            "请粘贴完整的 GitHub 仓库链接（以 `https://github.com/` 开头），"
            "或回复「跳过」进入正常注册流程。"
        )
        session.history.append({"role": "assistant", "content": followup, "step": 0, "ts": time.time()})
        return AgentResponse(step=0, step_label=STEP_LABELS[0], question=followup)

    # 有 URL：抓取内容
    session.github_url = github_url
    fetch_msg = f"正在抓取 {github_url} 的内容，请稍候…"
    session.history.append({"role": "assistant", "content": fetch_msg, "step": 0, "ts": time.time()})

    content, fetch_status = _fetch_github_content(github_url)
    if not content:
        # 抓取失败，直接进入 step 1
        fail_msg = f"⚠️ {fetch_status}\n\n继续正常注册流程。"
        session.history.append({"role": "assistant", "content": fail_msg, "step": 0, "ts": time.time()})
        session.step = 1
        question = interviewer.generate_opening_question(session)
        session.history.append({"role": "assistant", "content": question, "step": 1, "ts": time.time()})
        return AgentResponse(step=1, step_label=STEP_LABELS[1], question=question)

    session.github_context = content

    # LLM 预填充
    prefill_result = _prefill_from_github(session, content, llm)
    filled_steps: list[int] = prefill_result.get("steps", [])
    summary: str = prefill_result.get("summary", "")

    if not filled_steps:
        # 预填充失败，直接进入 step 1
        fail_msg = f"已获取仓库内容（{fetch_status}），但无法从中推断足够信息。\n\n{summary}\n\n继续正常注册流程。"
        session.history.append({"role": "assistant", "content": fail_msg, "step": 0, "ts": time.time()})
        session.step = 1
        question = interviewer.generate_opening_question(session)
        session.history.append({"role": "assistant", "content": question, "step": 1, "ts": time.time()})
        return AgentResponse(step=1, step_label=STEP_LABELS[1], question=question)

    # 预填充成功，进入第一个需要处理的步骤
    # Bug #2 fix: 若预填充跳过了早期步骤（如只填了 3/4/5），
    # 必须从 step 1 开始，缺失的步骤走正常 Interviewer 流程
    min_prefilled = min(filled_steps)
    first_step = min(1, min_prefilled)  # 始终从 step 1 开始
    if min_prefilled > 1:
        # 有未预填的早期步骤，从 step 1 开始正常注册
        first_step = 1
    else:
        first_step = min_prefilled
    session.step = first_step

    # 构建确认消息
    prefilled_fields = prefill_result.get("fields", {})

    if first_step in session.prefilled_steps:
        # 第一步是预填充的，直接进入确认
        confirm_msg = _build_prefill_confirm_message(
            summary, first_step, prefilled_fields, session
        )
        session.history.append({"role": "assistant", "content": confirm_msg, "step": first_step, "ts": time.time()})
        pending = _extract_step_data_from_prefill(first_step, prefilled_fields)
        session.pending_extraction = pending
        session.status = "confirming"
        return AgentResponse(
            step=first_step,
            step_label=STEP_LABELS[first_step],
            question=confirm_msg,
            extracted=pending,
            needs_confirmation=True,
            github_prefilled={"steps": sorted(filled_steps), "summary": summary},
        )
    else:
        # 第一步不是预填充的（早期步骤被跳过），走正常 Interviewer 流程
        notice = f"{summary}\n\nStep 1~{min_prefilled-1} 未能从 GitHub 推断，请手动填写。"
        question = interviewer.generate_opening_question(session)
        full_q = f"{notice}\n\n{question}"
        session.history.append({"role": "assistant", "content": full_q, "step": first_step, "ts": time.time()})
        return AgentResponse(
            step=first_step,
            step_label=STEP_LABELS[first_step],
            question=full_q,
            github_prefilled={"steps": sorted(filled_steps), "summary": summary},
        )


def _build_prefill_confirm_message(
    summary: str,
    step: int,
    prefilled_fields: dict,
    session: InterviewSession,
) -> str:
    """
    为预填充结果构建确认消息。

    Step 3/4/5（输出通道、降采样策略）会生成详细的可读描述，
    明确告知用户原始数据将如何被降采样，需要仔细确认。
    """
    lines = [f"✅ {summary}", "", f"**Step {step} — {STEP_LABELS[step]}** 的预填结果如下，请仔细确认："]

    if step == 1:
        dc = prefilled_fields.get("domain_context", "")
        od = prefilled_fields.get("output_description", "")
        if dc:
            lines.append(f"\n**domain_context（物理域描述）**:\n{dc[:400]}{'…' if len(dc) > 400 else ''}")
        if od:
            lines.append(f"\n**output_description（输出格式）**: {od}")

    elif step == 2:
        pi = prefilled_fields.get("param_info", {})
        if isinstance(pi, dict) and pi:
            lines.append(f"\n**param_info（{len(pi)} 个输入参数）**:")
            for k, v in list(pi.items())[:10]:
                meaning = v[0] if isinstance(v, list) else str(v)
                unit = v[1] if isinstance(v, list) and len(v) > 1 else "-"
                lines.append(f"  - {k}: {meaning}（{unit}）")
            if len(pi) > 10:
                lines.append(f"  … 共 {len(pi)} 个参数")

    elif step == 3:
        # ⚠️ 输出通道结构直接决定训练数据的 target 格式，需要仔细核对
        oi = prefilled_fields.get("output_info", [])
        if isinstance(oi, list) and oi:
            lines.append("\n⚠️ **输出通道结构（直接影响 target 数据格式）**")
            lines.append(f"共 {len(oi)} 个通道组：")
            for o in oi:
                s = o.get("slice", [0, None])
                s_end = s[1] if s[1] is not None else "…（到末尾）"
                # 单条 output_info 且 slice=[0,null] 时显示"全部通道"
                if len(oi) == 1 and s[0] == 0 and s[1] is None:
                    s_str = "[全部通道]"
                else:
                    s_str = f"[{s[0]}:{s_end}]"
                lines.append(
                    f"  {s_str} **{o.get('name', 'output')}**（{o.get('name_zh', '')}）"
                    f" — {o.get('description', 'unknown')}，单位：{o.get('unit', '-')}"
                )
            lines.append("")
            lines.append("请确认：通道数量、每组的物理含义和单位是否正确？")
            lines.append("（错误的 output_info 会导致时序数据被错误切分）")

    elif step == 4:
        # ⚠️ 通道采样策略
        obs = prefilled_fields.get("observation_config", {})
        if isinstance(obs, dict):
            fc = obs.get("fixed_channels")
            tmpl = obs.get("channel_name_template", "")
            tmpl_zh = obs.get("channel_name_template_zh", "")

            lines.append("\n⚠️ **通道采样（决定每条样本观测哪些通道）**")
            if fc is None:
                lines.append("  - fixed_channels：**null**（全选所有通道）")
            else:
                lines.append(f"  - fixed_channels：**{fc}**（0-based 整数索引）")
            if tmpl:
                lines.append(f"  - 通道名模板：{tmpl}（中文：{tmpl_zh}）")
            lines.append("")
            lines.append("请确认：是否全选所有通道？还是只选其中部分（请给出索引列表，如 [0,1,2]）？")

    elif step == 5:
        # ⚠️ 时间采样策略
        obs = prefilled_fields.get("observation_config", {})
        if isinstance(obs, dict):
            fixed = obs.get("fixed_time_mode", "unknown")
            modes = obs.get("time_modes", [])
            mode_desc = next((m.get("desc_en", "") for m in modes if m.get("name") == fixed), "")

            lines.append("\n⚠️ **时间采样（决定每条样本观测哪些时间点）**")
            lines.append(f"  - 固定时间模式：**{fixed}**")
            if mode_desc:
                lines.append(f"    （{mode_desc}）")
            lines.append("")
            lines.append("请确认：时间采样模式是否合理？")
            lines.append("支持：monthly（12点）/ weekly（52点）/ full（全量）/ every_N（每N步取1点）")

    lines.append("\n请确认以上内容，或在 JSON 编辑框中修改后确认。")
    return "\n".join(lines)


def _extract_step_data_from_prefill(step: int, prefilled_fields: dict) -> dict:
    """从预填充字段中提取指定步骤的数据。"""
    if step == 1:
        result = {}
        if "domain_context" in prefilled_fields:
            result["domain_context"] = prefilled_fields["domain_context"]
        if "output_description" in prefilled_fields:
            result["output_description"] = prefilled_fields["output_description"]
        return result
    elif step == 2:
        return {"param_info": prefilled_fields.get("param_info", {})}
    elif step == 3:
        return {"output_info": prefilled_fields.get("output_info", [])}
    elif step in (4, 5):
        obs = prefilled_fields.get("observation_config", {})
        if step == 4:
            return {k: obs.get(k) for k in ["fixed_channels", "channel_name_template", "channel_name_template_zh"] if k in obs}
        else:
            return {k: obs.get(k) for k in ["fixed_time_mode", "time_modes"] if k in obs}
    return {}


def process_user_message(
    session_id: str,
    user_message: str,
) -> AgentResponse:
    """
    处理用户消息（status=="interviewing" 时调用）。

    状态转移：
      interviewing → 提取 → 校验
        成功: status="confirming", needs_confirmation=True
        失败 & retries<3: 追问
        失败 & retries>=3: 部分结果 + extraction_uncertain=True
    """
    session = _interview_sessions.get(session_id)
    if not session:
        raise KeyError(f"会话 {session_id} 不存在")
    if session.status == "confirming":
        raise ValueError("当前步骤等待确认，请调用 process_confirm()")
    if session.status == "done":
        raise ValueError("会话已完成")

    # 记录用户消息
    session.history.append({
        "role": "user", "content": user_message,
        "step": session.step, "ts": time.time(),
    })
    session.updated_at = time.time()

    llm = _make_llm(session._llm_cfg)  # type: ignore[attr-defined]
    interviewer = InterviewerAgent(llm)
    extractor = ExtractorAgent(llm)
    validator = ValidatorAgent()

    step = session.step

    # ── Step 0：处理 GitHub 链接 ──────────────────────────────────
    if step == 0:
        return _handle_github_step(session, user_message, llm)

    # ── Step 7：scenario 模式，直接接收场景描述 ───────────────────
    if step == 7:
        desc = user_message.strip()
        if len(desc) < 5:
            followup = "请用一句话描述该场景的具体物理设置（至少 5 个字符）。"
            session.history.append({"role": "assistant", "content": followup, "step": 7, "ts": time.time()})
            return AgentResponse(step=7, step_label=STEP_LABELS[7], total_steps=7, question=followup)

        session.scenario_description = desc
        preview = (
            f"将写入 registry.yaml 的条目（key: `{session.simulator}/{session.scenario}`）：\n\n"
            f"```yaml\n{session.simulator}/{session.scenario}:\n"
            f"  scenario_description: \"{desc}\"\n```\n\n"
            "确认写入？"
        )
        session.pending_extraction = {"scenario_description": desc}
        session.status = "confirming"
        session.history.append({"role": "assistant", "content": preview, "step": 7, "ts": time.time()})
        return AgentResponse(
            step=7,
            step_label=STEP_LABELS[7],
            total_steps=7,
            question=preview,
            extracted={"scenario_description": desc},
            needs_confirmation=True,
        )

    # Step 6 特殊处理：直接进入确认
    if step == 6:
        preview = interviewer.render_final_preview(session)
        entry = _build_entry_from_session(session)
        session.pending_extraction = entry
        session.status = "confirming"
        session.history.append({
            "role": "assistant", "content": preview,
            "step": 6, "ts": time.time(),
        })
        return AgentResponse(
            step=6,
            step_label=STEP_LABELS[6],
            question=preview,
            extracted=entry,
            needs_confirmation=True,
        )

    # 提取
    extracted: dict = {}
    parse_error: Optional[str] = None
    try:
        extracted = extractor.extract(step, session, user_message)
    except (json.JSONDecodeError, ValueError) as e:
        parse_error = str(e)
        logger.warning(f"提取失败 (step={step}, retry={session.extraction_retries}): {e}")

    # 校验
    valid = False
    validation_errors: list[str] = []
    if not parse_error:
        valid, validation_errors = validator.validate(step, extracted, session)

    if valid:
        # 成功：进入确认模式
        session.pending_extraction = extracted
        session.status = "confirming"
        session.extraction_retries = 0

        summary = interviewer.generate_confirmation_summary(session, extracted)
        session.history.append({
            "role": "assistant", "content": summary,
            "step": step, "ts": time.time(),
        })
        return AgentResponse(
            step=step,
            step_label=STEP_LABELS[step],
            question=summary,
            extracted=extracted,
            needs_confirmation=True,
        )

    # 失败
    session.extraction_retries += 1
    all_errors = ([parse_error] if parse_error else []) + validation_errors

    if session.extraction_retries >= session.MAX_RETRIES:
        # 3次失败：用部分结果
        logger.warning(f"Step {step} 提取失败达 {session.MAX_RETRIES} 次，使用部分结果")
        partial = extracted if extracted else _make_partial_default(step, session)
        session.pending_extraction = partial
        session.status = "confirming"
        session.extraction_retries = 0

        uncertain_msg = (
            "自动提取不够确定，以下是当前最佳解析结果，请检查并编辑后确认：\n"
            f"（问题：{'; '.join(all_errors[:2])}）"
        )
        session.history.append({
            "role": "assistant", "content": uncertain_msg,
            "step": step, "ts": time.time(),
        })
        return AgentResponse(
            step=step,
            step_label=STEP_LABELS[step],
            question=uncertain_msg,
            extracted=partial,
            needs_confirmation=True,
            extraction_uncertain=True,
        )

    # 还有重试机会：追问
    followup = interviewer.generate_followup(session, all_errors)
    session.history.append({
        "role": "assistant", "content": followup,
        "step": step, "ts": time.time(),
    })
    return AgentResponse(
        step=step,
        step_label=STEP_LABELS[step],
        question=followup,
    )


def process_confirm(
    session_id: str,
    confirmed: bool,
    edited_data: Optional[dict] = None,
) -> AgentResponse:
    """
    处理用户确认/拒绝（status=="confirming" 时调用）。

    confirmed=True:
      - 将 pending_extraction（或 edited_data）合并到 session 字段
      - step<6: 推进到下一步，生成 opening question
      - step==6: WriterAgent 写入 registry.yaml
    confirmed=False:
      - 回到 interviewing 状态，生成追问
    """
    session = _interview_sessions.get(session_id)
    if not session:
        raise KeyError(f"会话 {session_id} 不存在")
    if session.status != "confirming":
        raise ValueError("当前不在确认状态，请先发送消息")

    step = session.step
    data = edited_data if edited_data is not None else session.pending_extraction

    if not confirmed:
        # 拒绝：回到 interviewing，生成追问
        session.status = "interviewing"
        session.extraction_retries = 0
        llm = _make_llm(session._llm_cfg)  # type: ignore[attr-defined]
        interviewer = InterviewerAgent(llm)
        followup = interviewer.generate_followup(session, ["请重新描述或修改上面的信息"])
        session.history.append({
            "role": "assistant", "content": followup,
            "step": step, "ts": time.time(),
        })
        return AgentResponse(
            step=step,
            step_label=STEP_LABELS[step],
            question=followup,
        )

    # 确认：将数据合并到 session
    _merge_step_data(session, step, data)
    # Bug #1 fix: 已确认的步骤从 prefilled_steps 中移除，避免重复展示预填数据
    session.prefilled_steps.discard(step)
    session.status = "interviewing"
    session.pending_extraction = {}
    session.updated_at = time.time()

    # ── Step 7（scenario 模式）：直接写入 registry ─────────────────
    if step == 7:
        registry_path: Optional[Path] = getattr(session, "_registry_path", None)
        if registry_path is None:
            registry_path = Path(__file__).parent.parent.parent / "configs" / "text2comp" / "registry.yaml"
        writer = WriterAgent()
        try:
            key = writer.write(session, registry_path)
            session.status = "done"
            return AgentResponse(
                step=7,
                step_label=STEP_LABELS[7],
                total_steps=7,
                done=True,
                saved=True,
                registry_key=key,
            )
        except Exception as e:
            session.status = "error"
            return AgentResponse(
                step=7,
                step_label=STEP_LABELS[7],
                total_steps=7,
                error=f"写入 registry 失败: {e}",
            )

    if step < 6:
        session.extraction_retries = 0
        llm = _make_llm(session._llm_cfg)  # type: ignore[attr-defined]
        interviewer = InterviewerAgent(llm)

        # 找下一个需要处理的步骤（跳过已预填但未到的步骤，直接进入确认）
        next_step = step + 1

        # 如果下一步已被 GitHub 预填，直接进入确认
        prefilled_fields = _get_prefilled_fields_for_session(session)
        if next_step in session.prefilled_steps and next_step < 6:
            session.step = next_step
            pending = _extract_step_data_from_prefill(next_step, prefilled_fields)
            confirm_msg = _build_prefill_confirm_message(
                f"Step {next_step} 已从 GitHub 预填充", next_step, prefilled_fields, session
            )
            session.pending_extraction = pending
            session.status = "confirming"
            session.history.append({"role": "assistant", "content": confirm_msg, "step": next_step, "ts": time.time()})
            return AgentResponse(
                step=next_step,
                step_label=STEP_LABELS[next_step],
                question=confirm_msg,
                extracted=pending,
                needs_confirmation=True,
            )

        session.step = next_step

        if next_step == 6:
            # 直接渲染预览（不等用户消息）
            preview = interviewer.render_final_preview(session)
            entry = _build_entry_from_session(session)
            session.pending_extraction = entry
            session.status = "confirming"
            session.history.append({
                "role": "assistant", "content": preview,
                "step": 6, "ts": time.time(),
            })
            return AgentResponse(
                step=6,
                step_label=STEP_LABELS[6],
                question=preview,
                extracted=entry,
                needs_confirmation=True,
            )
        else:
            question = interviewer.generate_opening_question(session)
            session.history.append({
                "role": "assistant", "content": question,
                "step": next_step, "ts": time.time(),
            })
            return AgentResponse(
                step=next_step,
                step_label=STEP_LABELS[next_step],
                question=question,
            )

    else:
        # Step 6 确认：写入 registry
        registry_path: Optional[Path] = getattr(session, "_registry_path", None)
        if registry_path is None:
            # 默认路径
            registry_path = Path(__file__).parent.parent.parent / "configs" / "text2comp" / "registry.yaml"

        writer = WriterAgent()
        try:
            key = writer.write(session, registry_path)
            session.status = "done"
            return AgentResponse(
                step=6,
                step_label=STEP_LABELS[6],
                done=True,
                saved=True,
                registry_key=key,
            )
        except Exception as e:
            session.status = "error"
            return AgentResponse(
                step=6,
                step_label=STEP_LABELS[6],
                error=f"写入 registry 失败: {e}",
            )


def get_session(session_id: str) -> Optional[InterviewSession]:
    return _interview_sessions.get(session_id)


def delete_session(session_id: str) -> bool:
    if session_id in _interview_sessions:
        del _interview_sessions[session_id]
        return True
    return False


# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────

def _merge_step_data(session: InterviewSession, step: int, data: dict) -> None:
    """将提取结果合并到 session 的对应字段。"""
    if step == 1:
        session.domain_context = data.get("domain_context", session.domain_context)
        session.output_description = data.get("output_description", session.output_description)
        # 无 HDF5 时从 step1 获取形状
        if not session.timeseries_shape:
            n_ch = data.get("n_channels")
            n_ts = data.get("n_timesteps")
            if n_ch and n_ts:
                session.timeseries_shape = (0, int(n_ch), int(n_ts))
    elif step == 2:
        session.param_info = data.get("param_info", session.param_info)
    elif step == 3:
        session.output_info = data.get("output_info", session.output_info)
    elif step == 4:
        obs = dict(session.observation_config)
        # 清除旧通道字段（防止残留）
        for _k in ["channel_level", "fixed_channels", "channel_min", "channel_max",  # 兼容旧 session
                   "channel_name_template", "channel_name_template_zh"]:
            obs.pop(_k, None)
        # 写入新通道字段（统一整数索引）
        obs["fixed_channels"] = data.get("fixed_channels")  # None=全选，list[int]=指定子集
        if data.get("channel_name_template"):
            obs["channel_name_template"] = data["channel_name_template"]
        if data.get("channel_name_template_zh"):
            obs["channel_name_template_zh"] = data["channel_name_template_zh"]
        session.observation_config = obs
    elif step == 5:
        obs = dict(session.observation_config)
        for _k in ["time_modes", "time_mode_weights", "fixed_time_mode"]:  # time_mode_weights 兼容旧 session
            obs.pop(_k, None)
        obs["time_modes"] = data.get("time_modes", [])
        obs["fixed_time_mode"] = data.get("fixed_time_mode", "")
        session.observation_config = obs
    elif step == 6:
        # Step 6 数据就是完整 entry，直接更新各字段
        session.domain_context = data.get("domain_context", session.domain_context)
        session.output_description = data.get("output_description", session.output_description)
        session.param_info = data.get("param_info", session.param_info)
        session.output_info = data.get("output_info", session.output_info)
        session.observation_config = data.get("observation_config", session.observation_config)
    elif step == 7:
        # Step 7（scenario 模式）：只更新 scenario_description
        session.scenario_description = data.get("scenario_description", session.scenario_description)


def _get_prefilled_fields_for_session(session: InterviewSession) -> dict:
    """从 session 当前字段重建 prefilled_fields dict（供后续步骤确认使用）。"""
    return {
        "domain_context":   session.domain_context,
        "output_description": session.output_description,
        "param_info":       session.param_info,
        "output_info":      session.output_info,
        "observation_config": session.observation_config,
    }


def _make_partial_default(step: int, session: InterviewSession) -> dict:
    """3次失败后生成最小合法默认值。"""
    if step == 1:
        return {
            "domain_context": f"{session.simulator} physics simulation.",
            "output_description": "{ch} channels × {ts} timesteps of simulation output",
        }
    elif step == 2:
        return {"param_info": {n: ["pending", "-"] for n in session.param_names}}
    elif step == 3:
        return {"output_info": [{"name": "output", "name_zh": "输出", "description": "simulation output", "unit": "-", "slice": [0, None]}]}
    elif step == 4:
        return {"fixed_channels": None, "channel_name_template": "channel {i}", "channel_name_template_zh": "第{i}个通道"}
    elif step == 5:
        return {"fixed_time_mode": "full", "time_modes": [{"name": "full", "indices": "full", "desc_en": "full time series", "desc_zh": "全量时间序列"}]}
    return {}
