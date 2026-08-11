"""
文生计算模块训练流程

完整训练流程：
1. 数据准备（可选：从HDF5生成）
2. 加载预训练LLM embedding
3. 构建模型（冻结base + 训练回归头）
4. 训练循环
5. 评估和checkpoint保存
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from .config import Text2CompTrainingConfig, create_custom_expert_info, get_expert_info
from .data import PromptNumbersDataset, generate_text2comp_data
from .model import Text2CompModel, create_text2comp_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_loss_fn(name: str) -> nn.Module:
    """获取损失函数"""
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "huber":
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")


def _split_indices_by_group(
    dataset: PromptNumbersDataset,
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split semantic samples as groups so prompt variants cannot leak into validation."""

    grouped: dict[str, list[int]] = {}
    group_ids = getattr(dataset, "group_ids", [])
    for index in range(len(dataset)):
        group_id = group_ids[index] if index < len(group_ids) else f"row:{index}"
        grouped.setdefault(str(group_id), []).append(index)
    if len(grouped) < 2:
        raise ValueError("Text2Comp training requires at least two distinct sample groups")

    group_names = sorted(grouped)
    random.Random(seed).shuffle(group_names)
    test_group_count = max(1, int(round(len(group_names) * test_ratio)))
    test_group_count = min(test_group_count, len(group_names) - 1)
    test_groups = set(group_names[:test_group_count])
    train_indices: list[int] = []
    test_indices: list[int] = []
    for group_name, indices in grouped.items():
        (test_indices if group_name in test_groups else train_indices).extend(indices)
    return sorted(train_indices), sorted(test_indices)


def _label_statistics(
    dataset: PromptNumbersDataset,
    train_indices: list[int],
    *,
    enabled: bool,
) -> dict[str, Any]:
    labels = torch.stack([dataset.samples[index][1] for index in train_indices])
    mean = labels.mean(dim=0)
    scale = labels.std(dim=0, correction=0).clamp(min=1e-6)
    return {"enabled": enabled, "mean": mean, "scale": scale}


def _semantic_group_count(dataset: PromptNumbersDataset) -> int:
    group_ids = getattr(dataset, "group_ids", [])
    return len(set(group_ids)) if group_ids else len(dataset)


def prepare_data(
    config: Text2CompTrainingConfig,
) -> tuple[DataLoader, DataLoader, int, dict[str, Any]]:
    """
    准备训练数据

    Returns:
        train_loader, test_loader, output_dim, label_statistics
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path,
        trust_remote_code=True,
        use_fast=False, local_files_only=True,
    )

    # 如果训练数据路径不存在，尝试从HDF5生成
    train_path = Path(config.train_data_path)
    if not train_path.exists():
        if config.expert_info and config.expert_info.data_path:
            logger.info(f"Generating training data from {config.expert_info.data_path}")
            stats = generate_text2comp_data(
                h5_path=config.expert_info.data_path,
                output_path=str(train_path),
                expert_info=config.expert_info,
            )
            config.output_dim = stats["output_dim"]
        else:
            raise FileNotFoundError(
                f"Training data not found: {train_path}. "
                "Provide train_data_path or expert_info.data_path"
            )

    # 加载数据集
    dataset = PromptNumbersDataset(
        file_path=str(train_path),
        tokenizer=tokenizer,
        max_length=config.max_length,
        expected_len=config.output_dim if config.output_dim > 0 else None,
        skip_invalid=config.skip_invalid,
    )

    # 自动推断output_dim
    if config.output_dim == 0:
        config.output_dim = dataset.label_dim

    n_total = len(dataset)
    semantic_samples = _semantic_group_count(dataset)
    if semantic_samples < max(2, int(config.min_samples)):
        raise ValueError(
            "Text2Comp dataset has "
            f"{semantic_samples} distinct samples ({n_total} prompt variants); "
            f"at least {max(2, int(config.min_samples))} distinct samples are required"
        )
    train_indices, test_indices = _split_indices_by_group(
        dataset,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    label_statistics = _label_statistics(dataset, train_indices, enabled=config.normalize_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    logger.info(
        "Train rows: %s, Test rows: %s, Distinct samples: %s, Output dim: %s, Label normalization: %s",
        len(train_indices),
        len(test_indices),
        semantic_samples,
        config.output_dim,
        config.normalize_labels,
    )
    return train_loader, test_loader, config.output_dim, label_statistics


def evaluate(
    model: Text2CompModel,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    label_mean: torch.Tensor,
    label_scale: torch.Tensor,
    normalize_labels: bool,
) -> dict[str, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model_outputs = model(input_ids, attention_mask)
            normalized_labels = (labels - label_mean) / label_scale if normalize_labels else labels
            loss = loss_fn(model_outputs, normalized_labels)
            preds = model_outputs * label_scale + label_mean if normalize_labels else model_outputs

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / total_samples

    # 计算额外指标
    preds_all = torch.cat(all_preds, dim=0)
    labels_all = torch.cat(all_labels, dim=0)

    mse = nn.MSELoss()(preds_all, labels_all).item()
    mae = nn.L1Loss()(preds_all, labels_all).item()
    scale_cpu = label_scale.detach().cpu().clamp(min=1e-6)
    normalized_rmse = torch.sqrt((((preds_all - labels_all) / scale_cpu) ** 2).mean()).item()
    centered = labels_all - labels_all.mean(dim=0, keepdim=True)
    residual_sum = ((preds_all - labels_all) ** 2).sum()
    total_sum = (centered**2).sum().clamp(min=1e-12)
    r2 = (1.0 - residual_sum / total_sum).item()

    # 相对误差
    relative_error = (preds_all - labels_all).abs() / (labels_all.abs().clamp(min=1e-6))
    mean_relative_error = relative_error.mean().item()

    return {
        "loss": avg_loss,
        "mse": mse,
        "mae": mae,
        "mean_relative_error": mean_relative_error,
        "normalized_rmse": normalized_rmse,
        "r2": r2,
    }


def run_training(config: Text2CompTrainingConfig) -> Path:
    """
    执行训练

    Returns:
        run_dir: 训练输出目录
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # 设置设备
    device = torch.device(config.device)
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = torch.device("cpu")

    # 准备数据
    train_loader, test_loader, output_dim, label_statistics = prepare_data(config)

    # 创建模型
    model = create_text2comp_model(config)
    model = model.to(device)
    label_mean = label_statistics["mean"].to(device)
    label_scale = label_statistics["scale"].to(device)

    # 打印参数统计
    params_info = model.get_num_parameters()
    logger.info(f"Model parameters: {params_info}")

    # 损失函数和优化器
    loss_fn = get_loss_fn(config.loss_fn)
    base_parameters = [parameter for parameter in model.base_model.parameters() if parameter.requires_grad]
    parameter_groups: list[dict[str, Any]] = []
    if base_parameters:
        parameter_groups.append({"params": base_parameters, "lr": config.learning_rate})
    parameter_groups.append({"params": model.head.parameters(), "lr": config.head_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)

    # 创建输出目录
    run_dir = Path(config.output_dir) / config.simulator / "runs" / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_path = run_dir / "config.json"
    config_dict = {
        "task_name": config.task_name,
        "simulator": config.simulator,
        "base_model_path": config.base_model_path,
        "output_dim": config.output_dim,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "head_learning_rate": config.head_learning_rate,
        "weight_decay": config.weight_decay,
        "loss_fn": config.loss_fn,
        "head_layers": config.head_layers,
        "head_dropout": config.head_dropout,
        "freeze_base": config.freeze_base_model,
        "trainable_base_layers": config.trainable_base_layers,
        "normalize_labels": config.normalize_labels,
        "min_samples": config.min_samples,
        "min_epochs": config.min_epochs,
        "early_stop_patience": config.early_stop_patience,
        "target_normalized_rmse": config.target_normalized_rmse,
        "max_normalized_rmse": config.max_normalized_rmse,
        "require_quality": config.require_quality,
    }
    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    # 训练日志
    log_path = run_dir / "train_log.jsonl"

    best_score = float("inf")
    best_metrics: dict[str, float] | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    completed_epochs = 0
    stop_reason = "max_epochs"
    global_step = 0
    start_time = time.time()

    logger.info(f"Starting training, output to {run_dir}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            training_labels = (labels - label_mean) / label_scale if config.normalize_labels else labels
            loss = loss_fn(preds, training_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_norm=1.0,
            )
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # 日志
            if step % config.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                eta = (config.epochs - epoch) * len(train_loader) / steps_per_sec

                log_entry = {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "loss": loss.item(),
                    "avg_loss": epoch_loss / epoch_steps,
                    "steps_per_sec": steps_per_sec,
                    "eta_seconds": eta,
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

                logger.info(
                    f"[epoch {epoch} step {step}] loss={loss.item():.6f}, "
                    f"avg={epoch_loss/epoch_steps:.6f}, eta={eta:.1f}s"
                )

        completed_epochs = epoch

        should_evaluate = epoch % config.eval_interval == 0 or epoch == config.epochs
        if should_evaluate:
            metrics = evaluate(
                model,
                test_loader,
                loss_fn,
                device,
                label_mean,
                label_scale,
                config.normalize_labels,
            )
            metrics["epoch"] = epoch
            logger.info(f"[eval epoch {epoch}] {metrics}")

            # 保存评估结果
            eval_path = run_dir / f"eval_epoch_{epoch:04d}.json"
            eval_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            score = float(metrics["normalized_rmse"])
            if score + 1e-6 < best_score:
                best_score = score
                best_metrics = dict(metrics)
                best_epoch = epoch
                epochs_without_improvement = 0
                checkpoint_path = run_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": config_dict,
                    "metrics": metrics,
                    "label_normalization": {
                        "enabled": config.normalize_labels,
                        "mean": label_statistics["mean"].tolist(),
                        "scale": label_statistics["scale"].tolist(),
                    },
                }, checkpoint_path)
                logger.info(f"New best model saved: {checkpoint_path}")
            else:
                epochs_without_improvement += 1

            if epoch >= max(1, config.min_epochs):
                if best_score <= config.target_normalized_rmse:
                    stop_reason = "quality_target_reached"
                    logger.info(
                        "Quality target reached at epoch %s: normalized_rmse=%.6f <= %.6f",
                        epoch,
                        best_score,
                        config.target_normalized_rmse,
                    )
                    break
                if config.early_stop_patience > 0 and epochs_without_improvement >= config.early_stop_patience:
                    stop_reason = "validation_plateau"
                    logger.info(
                        "Early stopping at epoch %s after %s evaluations without improvement",
                        epoch,
                        epochs_without_improvement,
                    )
                    break

        # 仅覆盖一个恢复点，避免每轮复制完整LLM导致磁盘持续增长。
        checkpoint_path = run_dir / "checkpoint_latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config_dict,
            "label_normalization": {
                "enabled": config.normalize_labels,
                "mean": label_statistics["mean"].tolist(),
                "scale": label_statistics["scale"].tolist(),
            },
        }, checkpoint_path)

    if best_metrics is None or best_epoch is None:
        raise RuntimeError("Text2Comp training completed without validation metrics")

    quality_passed = (not config.require_quality) or best_score <= config.max_normalized_rmse
    summary = {
        "status": "passed" if quality_passed else "quality_failed",
        "quality_passed": quality_passed,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "stop_reason": stop_reason,
        "best_metrics": best_metrics,
        "quality_gate": {
            "metric": "normalized_rmse",
            "required_max": config.max_normalized_rmse,
            "target": config.target_normalized_rmse,
        },
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_path = run_dir / "final_model.pt"
    best_path = run_dir / "best_model.pt"
    if final_path.exists() or final_path.is_symlink():
        final_path.unlink()
    if quality_passed:
        latest_checkpoint = run_dir / "checkpoint_latest.pt"
        if latest_checkpoint.exists():
            latest_checkpoint.unlink()
        final_path.symlink_to(best_path.name)
        logger.info(
            "Training complete, quality gate passed: normalized_rmse=%.6f, final model=%s",
            best_score,
            final_path,
        )
    else:
        raise RuntimeError(
            "Text2Comp quality gate failed: "
            f"best normalized_rmse={best_score:.6f}, required <= {config.max_normalized_rmse:.6f}"
        )

    return run_dir


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(description="Train Text-to-Computation model")

    # 任务配置
    parser.add_argument("--simulator", default="modflow", help="Simulator type")
    parser.add_argument("--task-name", default="", help="Task name")
    parser.add_argument("--run-name", default="", help="Run name")

    # 模型配置
    parser.add_argument("--base-model", required=True, help="Base LLM path")
    parser.add_argument("--output-dim", type=int, default=0, help="Output dimension")
    parser.add_argument("--freeze-base", dest="freeze_base", action="store_true", default=False)
    parser.add_argument("--unfreeze-base", dest="freeze_base", action="store_false")
    parser.add_argument("--trainable-base-layers", type=int, default=0)
    parser.add_argument("--head-layers", nargs="+", type=int, default=[128, 256, 512, 1024])

    # 训练配置
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-fn", default="mse", choices=["mse", "mae", "huber"])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--normalize-labels", action="store_true")
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--target-normalized-rmse", type=float, default=0.15)
    parser.add_argument("--max-normalized-rmse", type=float, default=0.25)
    parser.add_argument("--require-quality", action="store_true")

    # 数据配置
    parser.add_argument("--train-data", required=True, help="Training data path (JSONL)")
    parser.add_argument("--test-data", default="", help="Test data path")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--h5-path", default="", help="HDF5 path (for data generation)")

    # 运行配置
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/text2comp")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=20)

    args = parser.parse_args()

    # 构建配置
    expert_info = None
    if args.simulator:
        try:
            expert_info = get_expert_info(args.simulator)
        except ValueError:
            if args.output_dim <= 0:
                raise
            expert_info = create_custom_expert_info(
                name=args.simulator,
                expert_input_dim=args.output_dim,
                domain=f"Uploaded Expert adapter for {args.simulator}",
                expert_type="UploadedExpert",
            )
    if args.h5_path:
        expert_info.data_path = args.h5_path

    config = Text2CompTrainingConfig(
        task_name=args.task_name or f"{args.simulator}_train",
        simulator=args.simulator,
        expert_info=expert_info,
        base_model_path=args.base_model,
        freeze_base_model=args.freeze_base,
        trainable_base_layers=max(0, args.trainable_base_layers),
        output_dim=args.output_dim,
        head_layers=args.head_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        loss_fn=args.loss_fn,
        max_length=args.max_length,
        normalize_labels=args.normalize_labels,
        min_samples=max(2, args.min_samples),
        min_epochs=max(1, args.min_epochs),
        early_stop_patience=max(0, args.early_stop_patience),
        target_normalized_rmse=max(0.0, args.target_normalized_rmse),
        max_normalized_rmse=max(0.0, args.max_normalized_rmse),
        require_quality=args.require_quality,
        train_data_path=args.train_data,
        test_data_path=args.test_data,
        test_ratio=args.test_ratio,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name or args.task_name,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
    )

    run_dir = run_training(config)
    print(f"[done] run_dir={run_dir}")


if __name__ == "__main__":
    main()
