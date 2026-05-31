"""
一键训练入口：从HDF5到训练模型的完整自动化流程

整合调用原有数据生成模块（auto_register、generate_templates、fill_samples）
并添加格式转换和训练步骤。

用户只需提供HDF5文件，系统自动完成：
1. LLM推断物理领域元数据（auto_register）
2. LLM生成语言模板（generate_templates）
3. 数值填充（fill_samples）
4. 格式转换（data_converter）
5. 模型训练（train）

用法：
    # CLI方式
    python -m PierNet.training.text2comp.run_pipeline \
        --h5-path data/my_simulator/data.h5 \
        --simulator my_simulator \
        --n-samples 1000 \
        --epochs 50 \
        --gpu 0

    # Python API方式
    from PierNet.training.text2comp import run_full_pipeline
    result = run_full_pipeline(h5_path="data/my_sim.h5", simulator="my_sim")
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 项目根目录 (PierNet)
# run_pipeline.py 位于: PierNet/training/text2comp/run_pipeline.py
# parents[3] = PierNet
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 默认配置
DEFAULT_CONFIG = {
    "n_samples": 1000,
    "n_templates": 1000,
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "base_model": "Qwen2.5-0.5B-Instruct",
}


def setup_h5_directory(h5_path: str | Path, simulator: str) -> Path:
    """
    将HDF5文件移动到约定目录结构

    约定结构: data/{simulator}/{simulator}_{scenario}.h5

    Args:
        h5_path: 用户提供的HDF5原始路径
        simulator: 模拟器名称

    Returns:
        移动后的HDF5路径
    """
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5文件不存在: {h5_path}")

    # 目标目录
    data_root = PROJECT_ROOT / "data"
    target_dir = data_root / simulator
    target_dir.mkdir(parents=True, exist_ok=True)

    # 从原文件名提取scenario
    original_name = h5_path.stem

    # 如果文件名已经符合约定（{simulator}_{scenario}），保持不变
    if original_name.startswith(f"{simulator}_"):
        scenario = original_name[len(simulator) + 1:]
        target_path = target_dir / h5_path.name
    else:
        # 否则重命名为约定格式
        scenario = original_name
        target_path = target_dir / f"{simulator}_{scenario}.h5"

    # 如果目标路径已存在且不同，备份原文件
    if target_path.exists() and target_path != h5_path:
        logger.warning(f"目标路径已存在: {target_path}")

    # 复制文件到目标位置（不移动原文件）
    if target_path != h5_path:
        shutil.copy2(h5_path, target_path)
        logger.info(f"复制HDF5文件: {h5_path} → {target_path}")

    return target_path


def step_auto_register(
    config_path: str,
    scenarios: list[str] | None = None,
    fields: list[str] | None = None,
) -> bool:
    """
    Step 1: 调用auto_register自动推断元数据

    Args:
        config_path: 配置文件路径
        scenarios: 指定场景列表
        fields: 指定字段组（domain, output_info, observation）

    Returns:
        是否成功
    """
    logger.info("=" * 60)
    logger.info("Step 1: 自动注册元数据 (auto_register)")
    logger.info("=" * 60)

    cmd = [
        sys.executable,
        "-m",
        "PierNet.text2comp.auto_register",
        "--config", config_path,
    ]

    if scenarios:
        cmd.extend(["--scenarios", *scenarios])

    if fields:
        cmd.extend(["--fields", *fields])

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"auto_register失败: {result.stderr}")
            return False

        logger.info(f"auto_register完成: {result.stdout[-500:]}")
        return True

    except subprocess.SubprocessError as e:
        logger.error(f"执行auto_register异常: {e}")
        return False


def step_generate_templates(
    config_path: str,
    scenarios: list[str] | None = None,
    n_templates: int = 1000,
    skip_existing: bool = False,
) -> bool:
    """
    Step 2: 调用generate_templates生成语言模板

    Args:
        config_path: 配置文件路径
        scenarios: 指定场景列表
        n_templates: 每场景模板数
        skip_existing: 跳过已存在的模板

    Returns:
        是否成功
    """
    logger.info("=" * 60)
    logger.info("Step 2: 生成语言模板 (generate_templates)")
    logger.info("=" * 60)

    # 原始脚本路径
    script_path = PROJECT_ROOT / "scripts" / "text2comp" / "generate_templates.py"

    if not script_path.exists():
        logger.error(f"脚本不存在: {script_path}")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        "--config", config_path,
        "--n-templates", str(n_templates),
    ]

    if scenarios:
        cmd.extend(["--scenarios", *scenarios])

    if skip_existing:
        cmd.append("--skip-existing")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"generate_templates失败: {result.stderr}")
            return False

        logger.info(f"generate_templates完成: {result.stdout[-500:]}")
        return True

    except subprocess.SubprocessError as e:
        logger.error(f"执行generate_templates异常: {e}")
        return False


def step_fill_samples(
    config_path: str,
    scenarios: list[str] | None = None,
    n_samples: int = 1000,
    skip_existing: bool = False,
) -> bool:
    """
    Step 3: 调用fill_samples填充数值

    Args:
        config_path: 配置文件路径
        scenarios: 指定场景列表
        n_samples: 每场景样本数
        skip_existing: 跳过已存在的输出

    Returns:
        是否成功
    """
    logger.info("=" * 60)
    logger.info("Step 3: 数值填充 (fill_samples)")
    logger.info("=" * 60)

    # 原始脚本路径
    script_path = PROJECT_ROOT / "scripts" / "text2comp" / "fill_samples.py"

    if not script_path.exists():
        logger.error(f"脚本不存在: {script_path}")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        "--config", config_path,
        "--n-samples", str(n_samples),
    ]

    if scenarios:
        cmd.extend(["--scenarios", *scenarios])

    if skip_existing:
        cmd.append("--skip-existing")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"fill_samples失败: {result.stderr}")
            return False

        logger.info(f"fill_samples完成: {result.stdout[-500:]}")
        return True

    except subprocess.SubprocessError as e:
        logger.error(f"执行fill_samples异常: {e}")
        return False


def step_convert_data(
    scenarios: list[str] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Step 4: 格式转换（5字段 → 2字段）

    Args:
        scenarios: 指定场景列表
        output_dir: 输出目录

    Returns:
        转换统计信息
    """
    logger.info("=" * 60)
    logger.info("Step 4: 格式转换 (data_converter)")
    logger.info("=" * 60)

    from .data_converter import convert_jsonl

    # 输入输出目录
    input_dir = PROJECT_ROOT / "data" / "text2comp"
    output_dir_path = Path(output_dir) if output_dir else input_dir

    stats = {}
    converted_scenarios = []

    # 查找所有5字段JSONL文件
    for samples_file in input_dir.glob("*.jsonl"):
        # 跳过合并文件和已转换文件
        if samples_file.name.startswith("all_") or samples_file.name.endswith("_train.jsonl"):
            continue

        scenario = samples_file.stem

        # 检查是否在指定场景范围内
        if scenarios and scenario not in scenarios:
            continue

        # 输出路径
        train_file = output_dir_path / f"{scenario}_train.jsonl"

        logger.info(f"转换: {samples_file} → {train_file}")

        try:
            result = convert_jsonl(samples_file, train_file, skip_invalid=True)
            stats[scenario] = result
            converted_scenarios.append(scenario)
        except Exception as e:
            logger.error(f"转换失败: {scenario}, {e}")
            stats[scenario] = {"error": str(e)}

    logger.info(f"格式转换完成: {len(converted_scenarios)}个场景")
    return stats


def step_train(
    simulator: str,
    scenario: str,
    train_data_path: str,
    output_dim: int,
    config: dict[str, Any],
    gpu_id: int = 0,
) -> dict[str, Any]:
    """
    Step 5: 模型训练

    Args:
        simulator: 模拟器名称
        scenario: 场景名称
        train_data_path: 训练数据路径
        output_dim: 输出维度
        config: 训练配置
        gpu_id: GPU编号

    Returns:
        任务信息
    """
    logger.info("=" * 60)
    logger.info("Step 5: 模型训练 (train)")
    logger.info("=" * 60)

    from .text2comp_manager import create_job, EXPERT_MODEL_LIBRARY

    # 更新expert_info中的output_dim
    expert_info = EXPERT_MODEL_LIBRARY.get(simulator, {})
    expert_info["output_dim"] = output_dim
    EXPERT_MODEL_LIBRARY[simulator] = expert_info

    # 构建任务payload
    payload = {
        "simulator": simulator,
        "train_data_path": train_data_path,
        "gpu_id": gpu_id,
        "name": f"{simulator}_{scenario}",
        "output_dim": output_dim,
        "epochs": config.get("epochs", DEFAULT_CONFIG["epochs"]),
        "batch_size": config.get("batch_size", DEFAULT_CONFIG["batch_size"]),
        "learning_rate": config.get("learning_rate", DEFAULT_CONFIG["learning_rate"]),
        "base_model_path": config.get("base_model", DEFAULT_CONFIG["base_model"]),
    }

    try:
        job = create_job(payload)
        logger.info(f"训练任务创建成功: job_id={job['job_id']}")
        return job

    except Exception as e:
        logger.error(f"创建训练任务失败: {e}")
        raise


def run_full_pipeline(
    h5_path: str | Path,
    simulator: str = None,
    scenario: str = None,
    config_path: str = "configs/text2comp/default.yaml",
    training_config: dict[str, Any] = None,
    gpu_id: int = 0,
    skip_steps: list[str] = None,
) -> dict[str, Any]:
    """
    一键运行完整训练流程

    Args:
        h5_path: HDF5数据文件路径
        simulator: 模拟器名称（可选，从路径推断）
        scenario: 场景名称（可选，从路径推断）
        config_path: 配置文件路径
        training_config: 训练配置（可选）
        gpu_id: GPU编号
        skip_steps: 跳过的步骤 ['register', 'templates', 'fill', 'convert', 'train']

    Returns:
        {
            "simulator": str,
            "scenario": str,
            "h5_path": str,
            "registry_path": str,
            "templates_path": str,
            "samples_path": str,
            "train_data_path": str,
            "output_dim": int,
            "job_id": str,
            "model_path": str,
        }
    """
    h5_path = Path(h5_path)
    skip_steps = skip_steps or []
    training_config = training_config or {}

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 70)
    logger.info("开始一键训练流程")
    logger.info(f"HDF5: {h5_path}")
    logger.info("=" * 70)

    # 1. 推断simulator和scenario
    if simulator is None:
        simulator = h5_path.parent.name or h5_path.stem.split("_")[0]
    if scenario is None:
        # 从文件名提取scenario
        stem = h5_path.stem
        if stem.startswith(f"{simulator}_"):
            scenario = stem[len(simulator) + 1:]
        else:
            scenario = stem

    logger.info(f"simulator: {simulator}, scenario: {scenario}")

    # 2. 设置HDF5目录结构
    final_h5_path = setup_h5_directory(h5_path, simulator)

    results = {
        "simulator": simulator,
        "scenario": scenario,
        "h5_path": str(final_h5_path),
    }

    # 3. Step 1: auto_register
    if "register" not in skip_steps:
        success = step_auto_register(config_path, scenarios=[scenario])
        if not success:
            logger.warning("auto_register未成功，使用默认配置继续")
        results["registry_path"] = str(PROJECT_ROOT / "configs" / "text2comp" / "registry.yaml")

    # 4. Step 2: generate_templates
    if "templates" not in skip_steps:
        n_templates = training_config.get("n_templates", DEFAULT_CONFIG["n_templates"])
        success = step_generate_templates(
            config_path,
            scenarios=[scenario],
            n_templates=n_templates,
        )
        if not success:
            raise RuntimeError("generate_templates失败")
        results["templates_path"] = str(PROJECT_ROOT / "data" / "templates" / f"{scenario}_templates.jsonl")

    # 5. Step 3: fill_samples
    if "fill" not in skip_steps:
        n_samples = training_config.get("n_samples", DEFAULT_CONFIG["n_samples"])
        success = step_fill_samples(
            config_path,
            scenarios=[scenario],
            n_samples=n_samples,
        )
        if not success:
            raise RuntimeError("fill_samples失败")
        results["samples_path"] = str(PROJECT_ROOT / "data" / "text2comp" / f"{scenario}.jsonl")

    # 6. Step 4: format conversion
    if "convert" not in skip_steps:
        convert_stats = step_convert_data(scenarios=[scenario])
        if scenario not in convert_stats or "error" in convert_stats.get(scenario, {}):
            raise RuntimeError("格式转换失败")

        output_dim = convert_stats[scenario].get("output_dim", 0)
        results["train_data_path"] = str(PROJECT_ROOT / "data" / "text2comp" / f"{scenario}_train.jsonl")
        results["output_dim"] = output_dim
    else:
        # 从配置获取output_dim
        output_dim = training_config.get("output_dim", 0)
        results["train_data_path"] = training_config.get("train_data_path", "")
        results["output_dim"] = output_dim

    # 7. Step 5: train
    if "train" not in skip_steps:
        if results["output_dim"] == 0:
            raise ValueError("output_dim未指定，请先完成数据转换或手动指定")

        job = step_train(
            simulator=simulator,
            scenario=scenario,
            train_data_path=results["train_data_path"],
            output_dim=results["output_dim"],
            config=training_config,
            gpu_id=gpu_id,
        )
        results["job_id"] = job["job_id"]
        results["model_path"] = job.get("run_dir", "")
    else:
        results["job_id"] = None
        results["model_path"] = ""

    logger.info("=" * 70)
    logger.info("一键训练流程完成")
    logger.info(f"结果: {json.dumps(results, ensure_ascii=False, indent=2)}")
    logger.info("=" * 70)

    return results


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(
        description="一键训练：从HDF5到模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必需参数
    parser.add_argument(
        "--h5-path",
        required=True,
        help="HDF5数据文件路径",
    )

    # 可选参数
    parser.add_argument(
        "--simulator",
        default=None,
        help="模拟器名称（默认从路径推断）",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="场景名称（默认从路径推断）",
    )
    parser.add_argument(
        "--config",
        default="configs/text2comp/default.yaml",
        help="配置文件路径",
    )

    # 数据生成配置
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="每场景样本数",
    )
    parser.add_argument(
        "--n-templates",
        type=int,
        default=1000,
        help="每场景模板数",
    )

    # 训练配置
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批大小",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="学习率",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU编号",
    )

    # 跳过步骤
    parser.add_argument(
        "--skip",
        nargs="+",
        default=None,
        choices=["register", "templates", "fill", "convert", "train"],
        help="跳过的步骤",
    )

    args = parser.parse_args()

    # 构建训练配置
    training_config = {
        "n_samples": args.n_samples,
        "n_templates": args.n_templates,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    # 运行
    result = run_full_pipeline(
        h5_path=args.h5_path,
        simulator=args.simulator,
        scenario=args.scenario,
        config_path=args.config,
        training_config=training_config,
        gpu_id=args.gpu,
        skip_steps=args.skip,
    )

    print("\n" + "=" * 60)
    print("训练任务已提交")
    print("=" * 60)
    print(f"simulator:    {result['simulator']}")
    print(f"scenario:     {result['scenario']}")
    print(f"output_dim:   {result['output_dim']}")
    print(f"train_data:   {result['train_data_path']}")
    print(f"job_id:       {result['job_id']}")
    print(f"model_path:   {result['model_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()