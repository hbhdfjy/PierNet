from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    CHAR_TOKENS,
    LengthBucketBatchSampler,
    PackedSequenceDataset,
    collate_batch,
    prepare_router_dataset,
)
from .metrics import binary_classification_metrics
from .model import FullSeqDilatedConvRouter
from .pretrained_embeddings import EmbeddingBackboneSpec, PretrainedEmbeddingEncoder
from .tokenizer import CharTokenizer


@dataclass(slots=True)
class RouterTrainingConfig:
    simulator: str = "modflow"
    scenarios: tuple[str, ...] | None = None
    router_dir: str = "data/router"
    artifact_root: str = "artifacts/token_router/modflow"
    prepared_name: str | None = None
    run_name: str | None = None
    test_ratio: float = 0.10
    batch_size: int = 256
    test_batch_size: int = 256
    epochs: int = 1
    eval_interval: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    embedding_dim: int = 192
    model_dim: int = 256
    scene_dim: int = 16
    dropout: float = 0.10
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    num_workers: int = 8
    device: str = "cuda:0"
    seed: int = 42
    force_prepare: bool = False
    resume_from: str | None = None
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    input_representation: str = "auto"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _forward_logits(model: FullSeqDilatedConvRouter, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    kwargs = {
        "attention_mask": batch["attention_mask"],
        "scenario_ids": batch["scenario_ids"],
    }
    if "input_embeds" in batch:
        kwargs["input_embeds"] = batch["input_embeds"]
    else:
        kwargs["input_ids"] = batch["input_ids"]
    return model(**kwargs)


def _evaluate(
    model: FullSeqDilatedConvRouter,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    model.eval()
    logits_buffer: list[np.ndarray] = []
    labels_buffer: list[np.ndarray] = []
    scenario_buffer: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            with _autocast_context(device):
                logits = _forward_logits(model, batch)
            logits_buffer.append(logits.float().cpu().numpy())
            labels_buffer.append(batch["labels"].cpu().numpy())
            scenario_buffer.append(batch["scenario_ids"].cpu().numpy())

    logits_np = np.concatenate(logits_buffer) if logits_buffer else np.empty((0,), dtype=np.float32)
    labels_np = np.concatenate(labels_buffer) if labels_buffer else np.empty((0,), dtype=np.float32)
    scenarios_np = np.concatenate(scenario_buffer) if scenario_buffer else np.empty((0,), dtype=np.int64)

    overall = binary_classification_metrics(labels_np, logits_np)
    return overall, {"_scenario_ids": scenarios_np.tolist(), "_logits": logits_np.tolist(), "_labels": labels_np.tolist()}


def _save_checkpoint(
    path: Path,
    *,
    model: FullSeqDilatedConvRouter,
    optimizer: torch.optim.Optimizer,
    config: RouterTrainingConfig,
    summary,
    epoch: int,
    global_step: int,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "prepared_summary": summary.to_dict(),
            "vocab_size": summary.vocab_size,
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def _run_test(
    *,
    model: FullSeqDilatedConvRouter,
    test_loader: DataLoader,
    summary,
    run_dir: Path,
    epoch: int,
    train_sample_count: int,
    test_sample_count: int,
    device: torch.device,
) -> dict[str, object]:
    overall_metrics, raw_outputs = _evaluate(model, test_loader, device=device)
    scenario_names = summary.scenarios
    scenario_ids = np.asarray(raw_outputs.pop("_scenario_ids"), dtype=np.int64)
    logits = np.asarray(raw_outputs.pop("_logits"), dtype=np.float32)
    labels = np.asarray(raw_outputs.pop("_labels"), dtype=np.float32)
    per_scenario: dict[str, dict[str, float | int]] = {}
    for idx, name in enumerate(scenario_names):
        mask = scenario_ids == idx
        if mask.any():
            per_scenario[name] = binary_classification_metrics(labels[mask], logits[mask])

    metrics = {
        "epoch": epoch,
        "overall": overall_metrics,
        "per_scenario": per_scenario,
        "train_samples": train_sample_count,
        "test_samples": test_sample_count,
    }
    epoch_metrics_path = run_dir / f"test_metrics_epoch_{epoch:04d}.json"
    latest_metrics_path = run_dir / "test_metrics_latest.json"
    epoch_metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    latest_metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[test] epoch={epoch} accuracy={overall_metrics['accuracy']:.4f} "
        f"precision={overall_metrics['precision']:.4f} recall={overall_metrics['recall']:.4f} "
        f"f1={overall_metrics['f1']:.4f} pr_auc={overall_metrics['pr_auc']:.4f}"
    )
    print(f"[test] metrics written to {epoch_metrics_path}")
    return metrics


def run_training(config: RouterTrainingConfig) -> Path:
    _set_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is not available: {config.device}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    artifact_root = Path(config.artifact_root)
    prepared_dir = artifact_root / "prepared"
    if config.prepared_name:
        prepared_dir = prepared_dir / config.prepared_name
    summary = prepare_router_dataset(
        simulator=config.simulator,
        scenarios=list(config.scenarios) if config.scenarios else None,
        router_dir=Path(config.router_dir),
        output_dir=prepared_dir,
        test_ratio=config.test_ratio,
        force=config.force_prepare,
        input_representation=config.input_representation,
    )
    tokenizer = CharTokenizer.load(prepared_dir / "vocab.json") if summary.input_representation == CHAR_TOKENS else None
    pad_id = tokenizer.pad_id if tokenizer is not None else 0
    pretrained_embedding_weights = None
    if summary.input_representation != CHAR_TOKENS:
        encoder = PretrainedEmbeddingEncoder(
            EmbeddingBackboneSpec(
                model_name=summary.embedding_model,
                tokenizer_name=summary.embedding_tokenizer or summary.embedding_model,
                provider=summary.embedding_provider,
                chat_template=summary.chat_template,
                source=summary.embedding_source,
            )
        )
        pretrained_embedding_weights = encoder.build_model_embedding_tensor()
        if summary.vocab_size != encoder.model_vocab_size:
            raise ValueError(
                f"Prepared vocab_size mismatch: expected {summary.vocab_size}, got {encoder.model_vocab_size}"
            )
        if summary.input_hidden_size != encoder.hidden_size:
            raise ValueError(
                f"Prepared input_hidden_size mismatch: expected {summary.input_hidden_size}, got {encoder.hidden_size}"
            )

    train_dataset = PackedSequenceDataset(
        prepared_dir=prepared_dir,
        split="train",
        summary=summary,
        pad_id=pad_id,
        max_samples=config.max_train_samples,
    )
    test_dataset = PackedSequenceDataset(
        prepared_dir=prepared_dir,
        split="test",
        summary=summary,
        pad_id=pad_id,
        max_samples=config.max_test_samples,
    )

    train_sampler = LengthBucketBatchSampler(
        train_dataset.lengths,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    test_sampler = LengthBucketBatchSampler(
        test_dataset.lengths,
        batch_size=config.test_batch_size,
        shuffle=False,
        seed=config.seed,
    )
    collate = partial(collate_batch, pad_id=pad_id)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    test_num_workers = max(config.num_workers // 2, 0)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate,
        num_workers=test_num_workers,
        pin_memory=pin_memory,
        persistent_workers=test_num_workers > 0,
    )
    model = FullSeqDilatedConvRouter(
        vocab_size=summary.vocab_size,
        num_scenarios=len(summary.scenarios),
        max_sequence_length=summary.max_sequence_length,
        input_representation=summary.input_representation,
        input_embedding_dim=(
            config.embedding_dim
            if summary.input_representation == CHAR_TOKENS
            else summary.input_hidden_size
        ),
        model_dim=config.model_dim,
        scene_dim=config.scene_dim,
        kernel_size=config.kernel_size,
        dilations=config.dilations,
        dropout=config.dropout,
        pad_id=pad_id,
        pretrained_embedding_weights=pretrained_embedding_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_epoch = 0
    global_step = 0
    if config.resume_from:
        checkpoint = torch.load(config.resume_from, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        checkpoint_config = checkpoint.get("config", {})
        start_epoch = int(checkpoint.get("epoch", checkpoint_config.get("epochs", 0)))
        global_step = int(checkpoint.get("global_step", 0))

    run_dir = artifact_root / "runs" / (config.run_name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] run_dir={run_dir.resolve()}")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "training": asdict(config),
                "prepared_summary": summary.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log_path = run_dir / "train_log.jsonl"
    total_steps = len(train_loader)
    print(
        f"[train] simulator={config.simulator} train_samples={len(train_dataset)} "
        f"test_samples={len(test_dataset)} steps_per_epoch={total_steps} device={config.device} "
        f"input_representation={summary.input_representation}"
    )
    if summary.input_representation != CHAR_TOKENS:
        print(
            f"[train] embedding_model={summary.embedding_model} "
            f"tokenizer={summary.embedding_tokenizer} hidden_size={summary.input_hidden_size}"
        )
    if config.resume_from:
        print(
            f"[train] resume_from={config.resume_from} start_epoch={start_epoch} "
            f"eval_interval={config.eval_interval}"
        )

    epochs_to_run = config.epochs
    current_epoch = start_epoch
    try:
        while epochs_to_run <= 0 or current_epoch - start_epoch < epochs_to_run:
            train_sampler.set_epoch(current_epoch)
            model.train()
            running_loss = 0.0
            start_time = time.time()
            current_epoch += 1
            for step, batch in enumerate(train_loader, start=1):
                batch = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device):
                    logits = _forward_logits(model, batch)
                    loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
                loss.backward()
                optimizer.step()
                global_step += 1

                running_loss += float(loss.item())
                if step % 100 == 0 or step == total_steps:
                    avg_loss = running_loss / step
                    elapsed = time.time() - start_time
                    steps_per_sec = step / max(elapsed, 1e-6)
                    eta = (total_steps - step) / max(steps_per_sec, 1e-6)
                    message = {
                        "epoch": current_epoch,
                        "step": step,
                        "global_step": global_step,
                        "steps_per_epoch": total_steps,
                        "avg_loss": avg_loss,
                        "steps_per_sec": steps_per_sec,
                        "eta_seconds": eta,
                    }
                    print(
                        f"[train] epoch={current_epoch} step={step}/{total_steps} "
                        f"loss={avg_loss:.4f} steps_per_sec={steps_per_sec:.2f} eta={eta/60:.1f}m"
                    )
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(message, ensure_ascii=False) + "\n")

            _save_checkpoint(
                run_dir / "router_latest.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                summary=summary,
                epoch=current_epoch,
                global_step=global_step,
            )

            should_eval = current_epoch % max(config.eval_interval, 1) == 0
            is_last_finite_epoch = epochs_to_run > 0 and (current_epoch - start_epoch) >= epochs_to_run
            if should_eval or is_last_finite_epoch:
                _save_checkpoint(
                    run_dir / f"router_epoch_{current_epoch:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    summary=summary,
                    epoch=current_epoch,
                    global_step=global_step,
                )
                _run_test(
                    model=model,
                    test_loader=test_loader,
                    summary=summary,
                    run_dir=run_dir,
                    epoch=current_epoch,
                    train_sample_count=len(train_dataset),
                    test_sample_count=len(test_dataset),
                    device=device,
                )
    except KeyboardInterrupt:
        print(f"[train] interrupted at epoch={current_epoch}, saving checkpoint")
        _save_checkpoint(
            run_dir / "router_interrupted.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            summary=summary,
            epoch=current_epoch,
            global_step=global_step,
        )
        return run_dir

    _save_checkpoint(
        run_dir / "router_final.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        summary=summary,
        epoch=current_epoch,
        global_step=global_step,
    )
    return run_dir

