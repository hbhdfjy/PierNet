"""
Stage 4：Token Router 训练数据生成脚本。

从 Stage 3 生成的 data/text2comp/*.jsonl 中构建二分类路由数据：
  label=1：user_prefix+input+user_suffix+assistant_prefix+trigger_prefix（完整引导语）
  label=0：user_prefix+input+user_suffix+assistant_prefix+trigger_prefix[:pos]（引导语内随机截断）

正负比例 1:1（可配置）。每条 Stage 3 样本 → 1 条正样本 + neg_ratio 条负样本。
Router 推理时永远看到完整 input，截断只发生在 assistant 侧引导语内部。

支持多种 chat template，通过 --chat-template 指定：
  qwen        Qwen2/2.5 系列（<|im_start|> 格式）
  deepseek    DeepSeek-V2/V3 系列（<｜User｜> 格式）
  llama3      LLaMA-3 系列（<|start_header_id|> 格式）
  mistral     Mistral/Mixtral 系列（[INST] 格式）
  chatml      通用 ChatML（与 Qwen 相同）
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
import sys
from pathlib import Path


# ── 内置 Chat Template 定义 ──────────────────────────────────────
#
# 每个模板定义三段：
#   user_prefix      — 放在用户消息之前
#   user_suffix      — 放在用户消息之后
#   assistant_prefix — 放在 assistant 回复开头（路由器看到的最后一段）
#
# 正样本：user_prefix + input + user_suffix + assistant_prefix + trigger_prefix（完整）
# 负样本：user_prefix + input + user_suffix + assistant_prefix + trigger_prefix[:pos]（截断）

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


def _apply_chat_template_neg(
    input_text: str,
    trigger_prefix: str,
    tmpl: dict[str, str],
    rng: random.Random,
) -> str:
    """
    负样本：input 完整，assistant 在引导语内部均匀随机截断。

    推理时 Router 永远看到完整的 input，区别只在于 assistant 侧生成了多少。
    完整串：user_prefix + input + user_suffix + assistant_prefix + trigger_prefix
    截断范围：[tp_start, tp_end - 1)，即只在 trigger_prefix 内部截断。
    """
    up = tmpl["user_prefix"]
    us = tmpl["user_suffix"]
    ap = tmpl["assistant_prefix"]

    tp_len = len(trigger_prefix)
    if tp_len <= 1:
        # 引导语极短，截到空（只保留 assistant_prefix）
        return up + input_text + us + ap

    # 在 trigger_prefix 内均匀随机截断，保留 [0, pos) 部分
    pos = rng.randint(0, tp_len - 1)
    return up + input_text + us + ap + trigger_prefix[:pos]


# ── 工具函数 ──────────────────────────────────────────────────────

def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _build_source_signature(jsonl_path: Path) -> dict | None:
    try:
        stat = jsonl_path.stat()
    except FileNotFoundError:
        return None
    return {
        "name": jsonl_path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "row_count": _count_nonempty_lines(jsonl_path),
    }


def _scenario_meta_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".meta.json")


def _write_scenario_meta(
    output_path: Path,
    *,
    source_path: Path,
    chat_template_name: str,
    neg_ratio: int,
    source_signature: dict | None,
    output_count: int,
) -> None:
    meta = {
        "scenario": output_path.stem,
        "source_file": source_path.name,
        "chat_template": chat_template_name,
        "neg_ratio": neg_ratio,
        "source_signature": source_signature,
        "output_count": output_count,
    }
    _scenario_meta_path(output_path).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_scenario_meta(output_path: Path) -> dict | None:
    meta_path = _scenario_meta_path(output_path)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _can_reuse_existing_scenario(
    output_path: Path,
    *,
    source_path: Path,
    chat_template_name: str,
    neg_ratio: int,
    source_signature: dict | None,
) -> tuple[bool, str]:
    if source_signature is None:
        return False, f"missing current Stage 3 source file: {source_path.name}"

    meta = _load_scenario_meta(output_path)
    if not meta:
        return False, "missing verified build metadata; rebuild the scenario"
    if meta.get("chat_template") != chat_template_name:
        return False, "chat template mismatch"
    if int(meta.get("neg_ratio", -1)) != neg_ratio:
        return False, "neg_ratio mismatch"
    if meta.get("source_signature") != source_signature:
        return False, "Stage 3 source file changed"
    return True, ""


def _extract_trigger_prefix(target_template: str) -> str:
    """从 target_template 中提取 {output_0} 之前的引导语。"""
    idx = target_template.find("{output_0}")
    if idx == -1:
        return ""
    return target_template[:idx]



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

            # 负样本（neg_ratio 条）：input 完整，引导语内部均匀随机截断
            for _ in range(neg_ratio):
                context = _apply_chat_template_neg(
                    input_text, trigger_prefix, chat_tmpl, rng
                )
                samples.append({
                    "context": context,
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
        print(f"[错误] 未找到数据目录：{data_dir}")
        raise SystemExit(1)

    all_jsonl_files = sorted(
        f for f in data_dir.glob("*.jsonl")
        if f.name != "all_training_data.jsonl"
    )
    if not all_jsonl_files:
        print(f"[错误] {data_dir} 下没有可用的 JSONL 文件")
        raise SystemExit(1)

    scenario_source_signatures: dict[str, dict | None] = {
        f.stem: _build_source_signature(f) for f in all_jsonl_files
    }
    scenario_source_counts: dict[str, int] = {
        scenario: (sig or {}).get("row_count", 0)
        for scenario, sig in scenario_source_signatures.items()
    }

    jsonl_files = list(all_jsonl_files)
    if args.scenarios:
        scenario_set = set(args.scenarios)
        jsonl_files = [f for f in jsonl_files if f.stem in scenario_set]
        if not jsonl_files:
            print(f"[错误] 未匹配到指定场景：{args.scenarios}")
            raise SystemExit(1)

    print(f"[Router 数据生成] 共处理 {len(jsonl_files)} 个场景")

    # 先发送 init，让前端知道每个场景的预计总数。
    # 格式：PROGRESS_INIT:场景名:预计总条数 = source_count * (1 + neg_ratio)
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
        # 供场景级别复用和删除接口使用
        scenario_output_path = scenario_dir / f"{scenario_name}.jsonl"
        _write_jsonl(samples, scenario_output_path)
        _write_scenario_meta(
            scenario_output_path,
            source_path=jsonl_path,
            chat_template_name=chat_tmpl["_name"],
            neg_ratio=args.neg_ratio,
            source_signature=scenario_source_signatures.get(scenario_name),
            output_count=len(samples),
        )
        new_samples.extend(samples)
        processed_scenarios.add(scenario_name)

    if not new_samples:
        print("[错误] 没有生成任何 router 样本；请先确认 Stage 3 输出存在，且 target_template 能解析出 {output_0} 时间序列")
        raise SystemExit(1)

    all_samples: list[dict] = list(new_samples)
    if scenario_dir.exists():
        for existing_file in sorted(scenario_dir.glob("*.jsonl")):
            sc = existing_file.stem
            if sc in processed_scenarios:
                continue  # 当前场景已在本轮重建，无需复用旧文件

            source_path = data_dir / f"{sc}.jsonl"
            reusable, reason = _can_reuse_existing_scenario(
                existing_file,
                source_path=source_path,
                chat_template_name=chat_tmpl["_name"],
                neg_ratio=args.neg_ratio,
                source_signature=scenario_source_signatures.get(sc),
            )
            if not reusable:
                print(f"  [跳过] 无法复用 {existing_file.name}: {reason}")
                continue

            try:
                with open(existing_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_samples.append(json.loads(line))
                print(f"  复用旧文件：{existing_file.name}")
            except Exception as e:
                print(f"  [警告] 读取 {existing_file.name} 失败: {e}")

    n_pos_total = sum(1 for s in all_samples if s["label"] == 1)
    print(f"\n  总计：{len(all_samples)} 条（正={n_pos_total}, 负={len(all_samples)-n_pos_total}）")

    total = _write_all(all_samples, output_dir, args.seed)
    print(f"\n[完成] 共 {total} 条 → {output_dir}/train.jsonl")


if __name__ == "__main__":
    main()
