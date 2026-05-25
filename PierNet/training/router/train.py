from __future__ import annotations

import json
import signal
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

from PierNet.shared.runtime.paths import ARTIFACT_ROOT, DATA_ROOT

from .data import (
    LengthBucketBatchSampler,
    PackedSequenceDataset,
    collate_batch,
    prepare_router_dataset,
)
from .metrics import binary_classification_metrics
from .model import FullSeqDilatedConvRouter
from .pretrained_embeddings import EmbeddingBackboneSpec, PretrainedEmbeddingEncoder


@dataclass(slots=True)
class RouterTrainingConfig:
    simulator: str = "modflow"
    scenarios: tuple[str, ...] | None = None
    router_dir: str = str(DATA_ROOT / "router")
    artifact_root: str | None = None
    prepared_name: str | None = None
    run_name: str | None = None
    test_ratio: float = 0.10
    batch_size: int = 256
    test_batch_size: int = 256
    epochs: int = 1
    eval_interval: int = 1
    keep_last_epochs: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    embedding_dim: int = 192
    model_dim: int = 256
    scene_dim: int = 16
    dropout: float = 0.10
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    num_workers: int = 8
    prepare_workers: int | None = None
    device: str = "cuda:0"
    seed: int = 42
    force_prepare: bool = False
    resume_from: str | None = None
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    input_representation: str = "embedding"
    stop_file: str | None = None

    def __post_init__(self) -> None:
        if self.artifact_root is None:
            self.artifact_root = str(ARTIFACT_ROOT / "token_router" / self.simulator)


class PlatformStopController:
    def __init__(self, stop_file: str | None) -> None:
        self.stop_file = Path(stop_file) if stop_file else None
        self._requested = False

    @property
    def enabled(self) -> bool:
        return self.stop_file is not None

    def install(self) -> None:
        if not self.enabled:
            return
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        print(f"[control] platform stop enabled stop_file={self.stop_file}")

    def _stop_request_exists(self) -> bool:
        if not self.enabled or self.stop_file is None or not self.stop_file.exists():
            return False
        try:
            payload = json.loads(self.stop_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("reason") == "platform_stop"

    def _handle_signal(self, signum: int, _frame) -> None:
        signal_name = signal.Signals(signum).name
        if self._stop_request_exists():
            self._requested = True
            print(f"[stop] platform stop signal received signal={signal_name}")
            return
        print(f"[signal] ignored signal={signal_name}; platform stop file is missing")

    def requested(self) -> bool:
        if self._requested:
            return True
        if self._stop_request_exists():
            self._requested = True
        return self._requested


def _write_stop_state(
    run_dir: Path,
    *,
    reason: str,
    epoch: int | None,
    step: int | None,
    global_step: int,
) -> None:
    payload = {
        "reason": reason,
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "stopped_at": time.time(),
    }
    (run_dir / "stop_state.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _log_startup(phase: str, message: str) -> None:
    print(f"[startup] phase={phase} {message}")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_context(device: torch.device, dtype: torch.dtype | None = None):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype or _autocast_dtype(device) or torch.float16)
    return nullcontext()


def _make_grad_scaler(device: torch.device, amp_dtype: torch.dtype | None):
    enabled = device.type == "cuda" and amp_dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover - compatibility for older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


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
    amp_dtype: torch.dtype | None = None,
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    model.eval()
    logits_buffer: list[np.ndarray] = []
    labels_buffer: list[np.ndarray] = []
    scenario_buffer: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            with _autocast_context(device, amp_dtype):
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


def _epoch_from_checkpoint_name(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("router_epoch_"):
        return None
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return None


def _prune_epoch_checkpoints(run_dir: Path, keep_last_epochs: int) -> None:
    keep_count = max(0, int(keep_last_epochs))
    checkpoints = [
        (epoch, path)
        for path in run_dir.glob("router_epoch_*.pt")
        if (epoch := _epoch_from_checkpoint_name(path)) is not None
    ]
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    for _, path in checkpoints[keep_count:]:
        try:
            path.unlink()
            print(f"[checkpoint] pruned_epoch_checkpoint={path}")
        except OSError as exc:
            print(f"[checkpoint] prune_failed path={path} error={exc}")


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
    amp_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    overall_metrics, raw_outputs = _evaluate(model, test_loader, device=device, amp_dtype=amp_dtype)
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
    startup_started_at = time.perf_counter()
    _log_startup("bootstrap", f"seed={config.seed} device={config.device}")
    stop_controller = PlatformStopController(config.stop_file)
    stop_controller.install()
    _set_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is not available: {config.device}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    artifact_root = Path(config.artifact_root)
    run_dir = artifact_root / "runs" / (config.run_name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    if stop_controller.requested():
        print("[stop] platform stop accepted before dataset preparation")
        _write_stop_state(run_dir, reason="platform_stop_before_prepare", epoch=None, step=None, global_step=0)
        return run_dir

    prepared_dir = artifact_root / "prepared"
    if config.prepared_name:
        prepared_dir = prepared_dir / config.prepared_name
    prepare_started_at = time.perf_counter()
    _log_startup(
        "prepare",
        f"router_dir={config.router_dir} output_dir={prepared_dir} force_prepare={config.force_prepare} "
        f"prepare_workers={config.prepare_workers if config.prepare_workers is not None else config.num_workers}",
    )
    summary = prepare_router_dataset(
        simulator=config.simulator,
        scenarios=list(config.scenarios) if config.scenarios else None,
        router_dir=Path(config.router_dir),
        output_dir=prepared_dir,
        test_ratio=config.test_ratio,
        force=config.force_prepare,
        input_representation=config.input_representation,
        prepare_workers=config.prepare_workers if config.prepare_workers is not None else config.num_workers,
    )
    _log_startup(
        "prepare",
        "done "
        f"elapsed={time.perf_counter() - prepare_started_at:.1f}s "
        f"train_samples={summary.train_samples} test_samples={summary.test_samples} "
        f"max_sequence_length={summary.max_sequence_length}",
    )
    if stop_controller.requested():
        print("[stop] platform stop accepted after dataset preparation")
        _write_stop_state(run_dir, reason="platform_stop_after_prepare", epoch=None, step=None, global_step=0)
        return run_dir

    pad_id = 0
    _log_startup(
        "encoder",
        f"initializing tokenizer/model metadata for {summary.embedding_model}",
    )
    encoder = PretrainedEmbeddingEncoder(
        EmbeddingBackboneSpec(
            model_name=summary.embedding_model,
            tokenizer_name=summary.embedding_tokenizer or summary.embedding_model,
            provider=summary.embedding_provider,
            chat_template=summary.chat_template,
            source=summary.embedding_source,
        )
    )
    embedding_started_at = time.perf_counter()
    _log_startup(
        "encoder",
        f"loading pretrained embedding weights tokenizer={summary.embedding_tokenizer or summary.embedding_model}",
    )
    pretrained_embedding_weights = encoder.build_model_embedding_tensor()
    _log_startup(
        "encoder",
        "weights ready "
        f"elapsed={time.perf_counter() - embedding_started_at:.1f}s "
        f"vocab_size={encoder.model_vocab_size} hidden_size={encoder.hidden_size}",
    )
    if stop_controller.requested():
        print("[stop] platform stop accepted after embedding weights load")
        _write_stop_state(run_dir, reason="platform_stop_after_encoder", epoch=None, step=None, global_step=0)
        return run_dir

    if summary.vocab_size != encoder.model_vocab_size:
        raise ValueError(
            f"Prepared vocab_size mismatch: expected {summary.vocab_size}, got {encoder.model_vocab_size}"
        )
    if summary.input_hidden_size != encoder.hidden_size:
        raise ValueError(
            f"Prepared input_hidden_size mismatch: expected {summary.input_hidden_size}, got {encoder.hidden_size}"
        )

    _log_startup("dataset", "constructing train/test datasets from dynamic router records")
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
    _log_startup(
        "dataset",
        f"datasets ready train={len(train_dataset)} test={len(test_dataset)}",
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
    _log_startup(
        "dataloader",
        f"building DataLoaders train_workers={config.num_workers} pin_memory={pin_memory}",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )
    test_num_workers = max(config.num_workers // 2, 0)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate,
        num_workers=test_num_workers,
        pin_memory=pin_memory,
        persistent_workers=test_num_workers > 0,
        prefetch_factor=4 if test_num_workers > 0 else None,
    )
    _log_startup(
        "dataloader",
        f"dataloaders ready train_steps={len(train_loader)} test_steps={len(test_loader)}",
    )
    if stop_controller.requested():
        print("[stop] platform stop accepted after dataloader setup")
        _write_stop_state(run_dir, reason="platform_stop_after_dataloader", epoch=None, step=None, global_step=0)
        return run_dir

    _log_startup(
        "model",
        f"initializing router model model_dim={config.model_dim} scene_dim={config.scene_dim}",
    )
    model = FullSeqDilatedConvRouter(
        vocab_size=summary.vocab_size,
        num_scenarios=len(summary.scenarios),
        max_sequence_length=summary.max_sequence_length,
        input_representation=summary.input_representation,
        input_embedding_dim=summary.input_hidden_size,
        model_dim=config.model_dim,
        scene_dim=config.scene_dim,
        kernel_size=config.kernel_size,
        dilations=config.dilations,
        dropout=config.dropout,
        pad_id=pad_id,
        pretrained_embedding_weights=pretrained_embedding_weights,
    ).to(device)

    _log_startup(
        "optimizer",
        f"initializing AdamW learning_rate={config.learning_rate} weight_decay={config.weight_decay}",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_dtype = _autocast_dtype(device)
    scaler = _make_grad_scaler(device, amp_dtype)
    _log_startup(
        "amp",
        f"autocast_dtype={amp_dtype} grad_scaler={scaler.is_enabled()}",
    )
    start_epoch = 0
    global_step = 0
    if config.resume_from:
        _log_startup("resume", f"loading checkpoint {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        checkpoint_config = checkpoint.get("config", {})
        start_epoch = int(checkpoint.get("epoch", checkpoint_config.get("epochs", 0)))
        global_step = int(checkpoint.get("global_step", 0))
        _log_startup(
            "resume",
            f"checkpoint ready start_epoch={start_epoch} global_step={global_step}",
        )

    print(f"[run] run_dir={run_dir.resolve()}")
    _log_startup("run_dir", f"writing run config to {run_dir / 'config.json'}")
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
    _log_startup(
        "loop",
        f"entering training loop after {time.perf_counter() - startup_started_at:.1f}s",
    )

    log_path = run_dir / "train_log.jsonl"
    total_steps = len(train_loader)
    print(
        f"[train] simulator={config.simulator} train_samples={len(train_dataset)} "
        f"test_samples={len(test_dataset)} steps_per_epoch={total_steps} device={config.device} "
        f"input_representation={summary.input_representation}"
    )
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
    keep_last_epochs = max(0, int(config.keep_last_epochs))
    current_epoch = start_epoch
    try:
        while epochs_to_run <= 0 or current_epoch - start_epoch < epochs_to_run:
            if stop_controller.requested():
                print(f"[stop] platform stop accepted before epoch={current_epoch + 1}")
                _write_stop_state(
                    run_dir,
                    reason="platform_stop_before_epoch",
                    epoch=current_epoch,
                    step=None,
                    global_step=global_step,
                )
                return run_dir

            train_sampler.set_epoch(current_epoch)
            model.train()
            running_loss = 0.0
            start_time = time.time()
            current_epoch += 1
            for step, batch in enumerate(train_loader, start=1):
                if stop_controller.requested():
                    print(f"[stop] platform stop accepted at epoch={current_epoch} step={step}; saving checkpoint")
                    _save_checkpoint(
                        run_dir / "router_interrupted.pt",
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        summary=summary,
                        epoch=current_epoch,
                        global_step=global_step,
                    )
                    _write_stop_state(
                        run_dir,
                        reason="platform_stop_during_epoch",
                        epoch=current_epoch,
                        step=step,
                        global_step=global_step,
                    )
                    print(f"[stop] interrupted_checkpoint={run_dir / 'router_interrupted.pt'}")
                    return run_dir

                batch = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, amp_dtype):
                    logits = _forward_logits(model, batch)
                    loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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

                if stop_controller.requested():
                    print(f"[stop] platform stop accepted at epoch={current_epoch} step={step}; saving checkpoint")
                    _save_checkpoint(
                        run_dir / "router_interrupted.pt",
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        summary=summary,
                        epoch=current_epoch,
                        global_step=global_step,
                    )
                    _write_stop_state(
                        run_dir,
                        reason="platform_stop_after_step",
                        epoch=current_epoch,
                        step=step,
                        global_step=global_step,
                    )
                    print(f"[stop] interrupted_checkpoint={run_dir / 'router_interrupted.pt'}")
                    return run_dir

            _save_checkpoint(
                run_dir / "router_latest.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                summary=summary,
                epoch=current_epoch,
                global_step=global_step,
            )

            if keep_last_epochs > 0:
                _save_checkpoint(
                    run_dir / f"router_epoch_{current_epoch:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    summary=summary,
                    epoch=current_epoch,
                    global_step=global_step,
                )
                _prune_epoch_checkpoints(run_dir, keep_last_epochs)

            should_eval = current_epoch % max(config.eval_interval, 1) == 0
            is_last_finite_epoch = epochs_to_run > 0 and (current_epoch - start_epoch) >= epochs_to_run
            if should_eval or is_last_finite_epoch:
                _run_test(
                    model=model,
                    test_loader=test_loader,
                    summary=summary,
                    run_dir=run_dir,
                    epoch=current_epoch,
                    train_sample_count=len(train_dataset),
                    test_sample_count=len(test_dataset),
                    device=device,
                    amp_dtype=amp_dtype,
                )
            if stop_controller.requested():
                print(f"[stop] platform stop accepted after epoch={current_epoch}; saving checkpoint")
                _save_checkpoint(
                    run_dir / "router_interrupted.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    summary=summary,
                    epoch=current_epoch,
                    global_step=global_step,
                )
                _write_stop_state(
                    run_dir,
                    reason="platform_stop_after_epoch",
                    epoch=current_epoch,
                    step=total_steps,
                    global_step=global_step,
                )
                print(f"[stop] interrupted_checkpoint={run_dir / 'router_interrupted.pt'}")
                return run_dir
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
        _write_stop_state(
            run_dir,
            reason="keyboard_interrupt",
            epoch=current_epoch,
            step=None,
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
    print(f"[done] training finished final_checkpoint={run_dir / 'router_final.pt'}")
    return run_dir
