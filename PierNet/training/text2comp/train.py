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
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from .config import Text2CompTrainingConfig, get_expert_info
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


def prepare_data(config: Text2CompTrainingConfig) -> tuple[DataLoader, DataLoader, int]:
    """
    准备训练数据

    Returns:
        train_loader, test_loader, output_dim
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

    # 切分训练/测试集
    n_total = len(dataset)
    n_test = int(n_total * config.test_ratio)
    n_train = n_total - n_test

    train_dataset, test_dataset = random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    logger.info(f"Train: {n_train}, Test: {n_test}, Output dim: {config.output_dim}")
    return train_loader, test_loader, config.output_dim


def evaluate(
    model: Text2CompModel,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
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

            preds = model(input_ids, attention_mask)
            loss = loss_fn(preds, labels)

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

    # 相对误差
    relative_error = (preds_all - labels_all).abs() / (labels_all.abs().clamp(min=1e-6))
    mean_relative_error = relative_error.mean().item()

    return {
        "loss": avg_loss,
        "mse": mse,
        "mae": mae,
        "mean_relative_error": mean_relative_error,
    }


def run_training(config: Text2CompTrainingConfig) -> Path:
    """
    执行训练

    Returns:
        run_dir: 训练输出目录
    """
    # 设置设备
    device = torch.device(config.device)
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = torch.device("cpu")

    # 准备数据
    train_loader, test_loader, output_dim = prepare_data(config)

    # 创建模型
    model = create_text2comp_model(config)
    model = model.to(device)

    # 打印参数统计
    params_info = model.get_num_parameters()
    logger.info(f"Model parameters: {params_info}")

    # 损失函数和优化器
    loss_fn = get_loss_fn(config.loss_fn)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

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
        "weight_decay": config.weight_decay,
        "loss_fn": config.loss_fn,
        "head_layers": config.head_layers,
        "head_dropout": config.head_dropout,
        "freeze_base": config.freeze_base_model,
    }
    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    # 训练日志
    log_path = run_dir / "train_log.jsonl"

    best_loss = float("inf")
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
            loss = loss_fn(preds, labels)
            loss.backward()
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

        # 每epoch评估
        if epoch % config.eval_interval == 0:
            metrics = evaluate(model, test_loader, loss_fn, device)
            logger.info(f"[eval epoch {epoch}] {metrics}")

            # 保存评估结果
            eval_path = run_dir / f"eval_epoch_{epoch:04d}.json"
            eval_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            # 保存最佳模型
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                checkpoint_path = run_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config_dict,
                    "metrics": metrics,
                }, checkpoint_path)
                logger.info(f"New best model saved: {checkpoint_path}")

        # 定期保存checkpoint
        checkpoint_path = run_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config_dict,
        }, checkpoint_path)

    # 保存最终模型
    final_path = run_dir / "final_model.pt"
    torch.save({
        "epoch": config.epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config_dict,
    }, final_path)
    logger.info(f"Training complete, final model: {final_path}")

    # 更新最佳模型路径
    best_path = run_dir / "best_model.pt"
    best_path.symlink_to(final_path.name) if best_path.exists() else None

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
    parser.add_argument("--head-layers", nargs="+", type=int, default=[128, 256, 512, 1024])

    # 训练配置
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-fn", default="mse", choices=["mse", "mae", "huber"])
    parser.add_argument("--max-length", type=int, default=2048)

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
    expert_info = get_expert_info(args.simulator) if args.simulator else None
    if args.h5_path:
        expert_info.data_path = args.h5_path

    config = Text2CompTrainingConfig(
        task_name=args.task_name or f"{args.simulator}_train",
        simulator=args.simulator,
        expert_info=expert_info,
        base_model_path=args.base_model,
        freeze_base_model=args.freeze_base,
        output_dim=args.output_dim,
        head_layers=args.head_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_fn=args.loss_fn,
        max_length=args.max_length,
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
