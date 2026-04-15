"""
Stage 4：Token Router 训练数据生成脚本。

从 Stage 3 生成的 data/text2comp/*.jsonl 中构建二分类路由数据：
  label=1：context = chat_template(input + trigger_prefix)  → 下一步调用科学计算专家
  label=0：context = chat_template(input 的随机截断前缀)    → 继续 LLM 生成

正负比例 1:1（可配置）。每条 Stage 3 样本 → 1 条正样本 + neg_ratio 条负样本。

支持多种 chat template，通过 --chat-template 指定：
  qwen        Qwen2/2.5 系列（<|im_start|> 格式）
  deepseek    DeepSeek-V2/V3 系列（<｜User｜> 格式）
  llama3      LLaMA-3 系列（<|start_header_id|> 格式）
  mistral     Mistral/Mixtral 系列（[INST] 格式）
  chatml      通用 ChatML（与 Qwen 相同）
  plain       不包裹（仅 input 文本，向后兼容）
  custom      自定义（需同时指定 --user-prefix / --user-suffix / --assistant-prefix）

输出结构：
  data/router/
    ├── by_scenario/
    │   ├── unified_aquifer.jsonl
    │   └── ...
    └── train.jsonl

用法：
    python scripts/router/build_router_data.py
    python scripts/router/build_router_data.py --chat-template qwen
    python scripts/router/build_router_data.py --chat-template custom \\
        --user-prefix "<|im_start|>user\\n" --user-suffix "<|im_end|>\\n" \\
        --assistant-prefix "<|im_start|>assistant\\n"
"""

import argparse
import json
import random
from pathlib import Path


# ── 内置 Chat Template 定义 ──────────────────────────────────────
#
# 每个模板定义三段：
#   user_prefix      — 放在用户消息之前
#   user_suffix      — 放在用户消息之后
#   assistant_prefix — 放在 assistant 回复开头（路由器看到的最后一段）
#
# 最终 context = user_prefix + content + user_suffix + assistant_prefix
# 其中 content = input_text（负样本）或 input_text + trigger_prefix（正样本）

CHAT_TEMPLATES: dict[str, dict[str, str]] = {
    # Qwen2 / Qwen2.5 / ChatML 通用格式
    "qwen": {
        "user_prefix":      "<|im_start|>user\n",
        "user_suffix":      "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
    },
    "chatml": {
        "user_prefix":      "<|im_start|>user\n",
        "user_suffix":      "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
    },
    # DeepSeek-V2 / V3 格式
    "deepseek": {
        "user_prefix":      "<｜User｜>",
        "user_suffix":      "",
        "assistant_prefix": "<｜Assistant｜>",
    },
    # LLaMA-3 格式
    "llama3": {
        "user_prefix":      "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_suffix":      "<|eot_id|>",
        "assistant_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    # Mistral / Mixtral 格式
    "mistral": {
        "user_prefix":      "[INST] ",
        "user_suffix":      " [/INST]",
        "assistant_prefix": "",
    },
}


def _apply_chat_template_pos(input_text: str, trigger_prefix: str, tmpl: dict[str, str]) -> str:
    """
    正样本：input 完整，assistant 开始说引导语。
    结构：user_prefix + input + user_suffix + assistant_prefix + trigger_prefix
    """
    return (
        tmpl["user_prefix"]
        + input_text
        + tmpl["user_suffix"]
        + tmpl["assistant_prefix"]
        + trigger_prefix
    )


def _apply_chat_template_neg(truncated_input: str, tmpl: dict[str, str]) -> str:
    """
    负样本：input 在 user 轮次内部被截断，对话尚未结束。
    结构：user_prefix + 截断的input
    不加 user_suffix / assistant_prefix，因为 user 还没说完。
    """
    return tmpl["user_prefix"] + truncated_input


# ── 工具函数 ──────────────────────────────────────────────────────

def _extract_trigger_prefix(target_template: str) -> str:
    """从 target_template 中提取 {output_0} 之前的引导语。"""
    idx = target_template.find("{output_0}")
    if idx == -1:
        return ""
    return target_template[:idx]


def _random_truncation(text: str, rng: random.Random) -> str:
    """在 text 中随机选一个截断位置，返回截断后的前缀。"""
    n = len(text)
    if n == 0:
        return ""
    if n < 10:
        return text[: max(1, n // 2)]

    lo = max(1, int(n * 0.2))
    hi = max(lo + 1, int(n * 0.9))

    candidates = []
    boundary_chars = set("。！？，；,.!? \t")
    for i in range(lo, hi + 1):
        if i < n and text[i] in boundary_chars:
            candidates.append(i + 1)

    if candidates:
        pos = rng.choice(candidates)
    else:
        pos = rng.randint(lo, hi - 1)

    return text[:pos]


def _build_samples_from_file(
    jsonl_path: Path,
    rng: random.Random,
    chat_tmpl: dict[str, str],
    neg_ratio: int = 1,
    progress_callback=None,
    progress_interval: int = 500,
) -> list[dict]:
    """从单个场景 JSONL 文件生成正负样本对。

    每条原始样本 → 1 条正样本 + neg_ratio 条负样本（不同截断位置）。
    context 用 chat_tmpl 包裹。

    Args:
        progress_callback: 可选回调 (done_samples: int)，每 progress_interval 条源样本调用一次。
    """
    samples = []
    source_processed = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            input_text = record.get("input", "")
            meta = record.get("metadata", {})
            target_template = meta.get("target_template", "")

            if not input_text:
                continue

            trigger_prefix = _extract_trigger_prefix(target_template)
            if not trigger_prefix:
                continue  # 旧格式模板，跳过

            base_meta = {
                "simulator": meta.get("simulator", "unknown"),
                "scenario":  meta.get("scenario",  "unknown"),
                "language":  meta.get("language",  "unknown"),
                "chat_template": chat_tmpl.get("_name", "plain"),
            }

            # 正样本（1条）：input 完整，assistant 开始说引导语
            samples.append({
                "context": _apply_chat_template_pos(input_text, trigger_prefix, chat_tmpl),
                "label": 1,
                "metadata": {**base_meta, "trigger_prefix": trigger_prefix},
            })

            # 负样本（neg_ratio 条）：input 在 user 轮次内截断，不闭合对话轮次
            for _ in range(neg_ratio):
                neg_user = _random_truncation(input_text, rng)
                samples.append({
                    "context": _apply_chat_template_neg(neg_user, chat_tmpl),
                    "label": 0,
                    "metadata": {**base_meta, "trigger_prefix": ""},
                })

            source_processed += 1
            if progress_callback and source_processed % progress_interval == 0:
                progress_callback(len(samples))

    return samples


def _write_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _write_all(samples: list[dict], output_dir: Path, seed: int) -> int:
    """打乱后写入单个 train.jsonl。"""
    rng = random.Random(seed)
    rng.shuffle(samples)
    out_path = output_dir / "train.jsonl"
    _write_jsonl(samples, out_path)
    print(f"  [train] {len(samples)} 条 → {out_path}")
    return len(samples)


# ── 主函数 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="构建 Token Router 训练数据")
    parser.add_argument("--data-dir",   type=str,   default="data/text2comp")
    parser.add_argument("--output-dir", type=str,   default="data/router")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--scenarios",  nargs="*",  default=None,
                        help="只处理指定场景（空=全部），例：--scenarios unified_aquifer ieee14_baseload")
    parser.add_argument("--neg-ratio",  type=int,   default=1,
                        help="每条正样本对应的负样本数量（默认1，即1:1）")
    parser.add_argument("--chat-template", type=str, default="custom",
                        choices=list(CHAT_TEMPLATES.keys()) + ["custom"],
                        help="Chat template 类型（默认 plain，即不包裹）")
    parser.add_argument("--user-prefix",      type=str, default="",
                        help="自定义 chat template：用户消息前缀（--chat-template custom 时生效）")
    parser.add_argument("--user-suffix",      type=str, default="",
                        help="自定义 chat template：用户消息后缀")
    parser.add_argument("--assistant-prefix", type=str, default="",
                        help="自定义 chat template：assistant 回复前缀")
    args = parser.parse_args()

    # 解析 chat template
    tmpl_name = args.chat_template
    if tmpl_name == "custom":
        chat_tmpl = {
            "_name":            "custom",
            "user_prefix":      args.user_prefix,
            "user_suffix":      args.user_suffix,
            "assistant_prefix": args.assistant_prefix,
        }
    else:
        chat_tmpl = dict(CHAT_TEMPLATES.get(tmpl_name, {"user_prefix": "", "user_suffix": "", "assistant_prefix": ""}))
        chat_tmpl["_name"] = tmpl_name

    print(f"[Router 数据生成] Chat template: {tmpl_name}", flush=True)

    project_root = Path(__file__).resolve().parents[2]
    data_dir   = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    scenario_dir = output_dir / "by_scenario"

    if not data_dir.exists():
        print(f"[错误] 数据目录不存在：{data_dir}")
        return

    jsonl_files = sorted(
        f for f in data_dir.glob("*.jsonl")
        if f.name != "all_training_data.jsonl"
    )
    if not jsonl_files:
        print(f"[错误] {data_dir} 中没有找到 JSONL 文件")
        return

    # 按场景过滤
    if args.scenarios:
        scenario_set = set(args.scenarios)
        jsonl_files = [f for f in jsonl_files if f.stem in scenario_set]
        if not jsonl_files:
            print(f"[错误] 指定的场景均未找到：{args.scenarios}")
            return

    print(f"[Router 数据生成] 扫描到 {len(jsonl_files)} 个场景文件")

    # 预统计各场景源样本数，用于进度上报
    scenario_source_counts: dict[str, int] = {}
    for f in jsonl_files:
        try:
            count = sum(1 for line in open(f, "rb") if line.strip())
        except Exception:
            count = 0
        scenario_source_counts[f.stem] = count

    # 发送 init 事件：告知前端各场景的预期生成条数
    # 格式：PROGRESS_INIT:场景名:预期总条数（= source * (1 + neg_ratio)）
    for f in jsonl_files:
        expected = scenario_source_counts[f.stem] * (1 + args.neg_ratio)
        print(f"PROGRESS_INIT:{f.stem}:{expected}", flush=True)

    rng = random.Random(args.seed)
    new_samples: list[dict] = []
    processed_scenarios: set[str] = set()

    # 处理本次指定的场景
    for jsonl_path in jsonl_files:
        scenario_name = jsonl_path.stem
        expected = scenario_source_counts[scenario_name] * (1 + args.neg_ratio)
        print(f"  处理：{jsonl_path.name} ...", flush=True)

        def _progress(done_samples: int, sc=scenario_name, exp=expected):
            print(f"PROGRESS_UPDATE:{sc}:{done_samples}:{exp}", flush=True)

        samples = _build_samples_from_file(
            jsonl_path, rng, chat_tmpl=chat_tmpl, neg_ratio=args.neg_ratio,
            progress_callback=_progress,
        )
        n_pos = sum(1 for s in samples if s["label"] == 1)
        n_neg = len(samples) - n_pos
        print(f"  完成：{len(samples)} 条（正={n_pos}, 负={n_neg}）", flush=True)

        # 发送进度完成事件
        print(f"PROGRESS_DONE:{scenario_name}:{len(samples)}:{expected}", flush=True)

        # 写入场景独立文件（覆盖）
        _write_jsonl(samples, scenario_dir / f"{scenario_name}.jsonl")
        new_samples.extend(samples)
        processed_scenarios.add(scenario_name)

    if not new_samples:
        print("[警告] 没有生成任何样本。Stage 3 数据需使用新版 target_template 格式（{output_0} 前有引导语）。")
        return

    # 部分生成时：读取其他已有场景的独立文件，合并后重新划分
    all_samples: list[dict] = list(new_samples)
    if scenario_dir.exists():
        for existing_file in sorted(scenario_dir.glob("*.jsonl")):
            sc = existing_file.stem
            if sc in processed_scenarios:
                continue  # 本次已处理，跳过（已覆盖写入）
            try:
                with open(existing_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_samples.append(json.loads(line))
                print(f"  复用已有：{existing_file.name}")
            except Exception as e:
                print(f"  [警告] 读取 {existing_file.name} 失败: {e}")

    n_pos_total = sum(1 for s in all_samples if s["label"] == 1)
    print(f"\n  总计：{len(all_samples)} 条（正={n_pos_total}, 负={len(all_samples)-n_pos_total}）")

    total = _write_all(all_samples, output_dir, args.seed)
    print(f"\n[完成] 共 {total} 条 → {output_dir}/train.jsonl")


if __name__ == "__main__":
    main()
