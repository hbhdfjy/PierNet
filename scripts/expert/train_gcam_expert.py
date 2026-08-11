#!/usr/bin/env python3
"""Train and package a unified GCAM surrogate expert for PiERN."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SCENARIO_FILES = {
    "carbon_pricing": "gcam_carbon_pricing.h5",
    "climate_feedback": "gcam_climate_feedback.h5",
    "energy_transition": "gcam_energy_transition.h5",
}
EXPECTED_INPUT_DIM = 18
EXPECTED_OUTPUT_SHAPE = (5, 16)
GCAM_DEFAULT_PROMPT = (
    "请使用 GCAM 能源—气候专家完成 carbon_pricing 碳定价情景预测。"
    "参数：solar_cost_init=53.953953; cost_learning_rate=0.153154; "
    "carbon_tax=155.896484; gdp_growth=3.156048; population_growth=0.433008; "
    "energy_intensity=0.922160; fossil_reserve=1.136147; "
    "climate_sensitivity=2.735805; discount_rate=0.041285; n_regions=3; "
    "nuclear_cost=178.193756; ccs_cost=84.528244; ev_penetration=0.895704; "
    "industrial_eff=0.313091; land_use_change=1.911192; scenario_type=1; "
    "output_type=1; complexity=2。请预测 2025—2100 年每 5 年的煤电占比、"
    "可再生能源占比、CO₂ 排放、能源价格和温度变化，并用中文概括主要趋势。"
)


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    split_seed: int
    hidden_dim: int
    residual_blocks: int
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    train_fraction: float
    val_fraction: float


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.norm(values)
        residual = self.activation(self.fc1(residual))
        residual = self.dropout(residual)
        residual = self.fc2(residual)
        return values + residual


class GCAMExpert(nn.Module):
    def __init__(
        self,
        input_dim: int = EXPECTED_INPUT_DIM,
        output_dim: int = math.prod(EXPECTED_OUTPUT_SHAPE),
        hidden_dim: int = 256,
        residual_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden_dim, dropout) for _ in range(residual_blocks)
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.input(values)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/gcam"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/expert_models/gcam_unified_mlp_v1"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--model-name", default="gcam_unified_mlp_v1")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=1800)
    parser.add_argument("--patience", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    ]


def load_data(data_dir: Path) -> dict[str, Any]:
    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_scenarios: list[np.ndarray] = []
    param_names: list[str] | None = None
    output_names: list[str] | None = None

    for scenario_index, (scenario, file_name) in enumerate(SCENARIO_FILES.items()):
        path = data_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing GCAM dataset: {path}")
        with h5py.File(path, "r") as handle:
            inputs = np.asarray(handle["params"], dtype=np.float32)
            targets = np.asarray(handle["timeseries"], dtype=np.float32)
            current_param_names = decode_strings(handle["param_names"][:])
            current_output_names = decode_strings(
                handle["metadata/output_variables"][:]
            )

        if inputs.ndim != 2 or inputs.shape[1] != EXPECTED_INPUT_DIM:
            raise ValueError(f"{path}: expected params [N, 18], got {inputs.shape}")
        if targets.shape[1:] != EXPECTED_OUTPUT_SHAPE:
            raise ValueError(
                f"{path}: expected timeseries [N, 5, 16], got {targets.shape}"
            )
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(f"{path}: params/timeseries sample counts differ")
        if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
            raise ValueError(f"{path}: non-finite training values")
        if param_names is not None and current_param_names != param_names:
            raise ValueError(f"{path}: inconsistent parameter ordering")
        if output_names is not None and current_output_names != output_names:
            raise ValueError(f"{path}: inconsistent output ordering")

        param_names = current_param_names
        output_names = current_output_names
        all_inputs.append(inputs)
        all_targets.append(targets.reshape(targets.shape[0], -1))
        all_scenarios.append(
            np.full(inputs.shape[0], scenario_index, dtype=np.int64)
        )

    return {
        "inputs": np.concatenate(all_inputs, axis=0),
        "targets": np.concatenate(all_targets, axis=0),
        "scenarios": np.concatenate(all_scenarios, axis=0),
        "param_names": param_names,
        "output_names": output_names,
        "scenario_names": list(SCENARIO_FILES),
    }


def stratified_split(
    scenarios: np.ndarray,
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 < val_fraction < 1 - train_fraction:
        raise ValueError("val_fraction leaves no room for the test split")

    rng = np.random.default_rng(seed)
    splits: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for scenario_index in sorted(np.unique(scenarios).tolist()):
        indices = np.flatnonzero(scenarios == scenario_index)
        rng.shuffle(indices)
        train_end = int(round(len(indices) * train_fraction))
        val_end = train_end + int(round(len(indices) * val_fraction))
        splits["train"].append(indices[:train_end])
        splits["val"].append(indices[train_end:val_end])
        splits["test"].append(indices[val_end:])

    result: dict[str, np.ndarray] = {}
    for name, parts in splits.items():
        merged = np.concatenate(parts)
        rng.shuffle(merged)
        result[name] = merged
    return result


def compute_normalization(
    inputs: np.ndarray, targets: np.ndarray
) -> dict[str, np.ndarray]:
    input_mean = inputs.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_std = inputs.std(axis=0, dtype=np.float64).astype(np.float32)
    target_mean = targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_std = targets.std(axis=0, dtype=np.float64).astype(np.float32)
    input_std = np.where(input_std < 1e-6, 1.0, input_std).astype(np.float32)
    target_std = np.where(target_std < 1e-6, 1.0, target_std).astype(np.float32)
    return {
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
            total_loss += float(batch_loss.item()) * inputs.shape[0]
            total_samples += inputs.shape[0]
    return total_loss / max(total_samples, 1)


def predict_normalized(
    model: nn.Module,
    inputs: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(inputs)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            output = model(batch.to(device, non_blocking=True))
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def apply_physical_bounds(values: np.ndarray) -> np.ndarray:
    shaped = values.reshape(-1, *EXPECTED_OUTPUT_SHAPE).copy()
    shaped[:, 0:2, :] = np.clip(shaped[:, 0:2, :], 0.0, 1.0)
    shaped[:, 2:4, :] = np.maximum(shaped[:, 2:4, :], 0.0)
    shaped[:, 4, :] = np.clip(shaped[:, 4, :], -1.0, 8.0)
    return shaped.reshape(values.shape)


def regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    output_names: list[str],
) -> dict[str, Any]:
    target_cube = targets.reshape(-1, *EXPECTED_OUTPUT_SHAPE)
    prediction_cube = predictions.reshape(-1, *EXPECTED_OUTPUT_SHAPE)
    error = prediction_cube - target_cube
    variable_metrics: dict[str, Any] = {}

    for index, name in enumerate(output_names):
        truth = target_cube[:, index, :].reshape(-1)
        estimate = prediction_cube[:, index, :].reshape(-1)
        residual = estimate - truth
        mse = float(np.mean(np.square(residual)))
        mae = float(np.mean(np.abs(residual)))
        variance = float(np.sum(np.square(truth - truth.mean())))
        r2 = 1.0 - float(np.sum(np.square(residual))) / max(variance, 1e-12)
        variable_metrics[name] = {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": mae,
            "r2": r2,
            "target_mean": float(truth.mean()),
            "target_std": float(truth.std()),
        }

    return {
        "mse": float(np.mean(np.square(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "variables": variable_metrics,
    }


def physical_validity(predictions: np.ndarray) -> dict[str, Any]:
    values = predictions.reshape(-1, *EXPECTED_OUTPUT_SHAPE)
    sample_valid = (
        np.isfinite(values).all(axis=(1, 2))
        & (values[:, 0:2, :] >= 0).all(axis=(1, 2))
        & (values[:, 0:2, :] <= 1).all(axis=(1, 2))
        & (values[:, 2:4, :] >= 0).all(axis=(1, 2))
        & (values[:, 4, :] >= -1).all(axis=1)
        & (values[:, 4, :] <= 8).all(axis=1)
    )
    return {
        "valid_samples": int(sample_valid.sum()),
        "total_samples": int(sample_valid.size),
        "valid_rate": float(sample_valid.mean()),
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


ADAPTER_SOURCE = '''from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

INPUT_DIM = 18
OUTPUT_DIM = 80
OUTPUT_SHAPE = (5, 16)
WEIGHTS = Path(__file__).with_name("gcam_expert.pt")
_MODEL = None
_PAYLOAD = None


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        residual = self.norm(values)
        residual = self.activation(self.fc1(residual))
        residual = self.dropout(residual)
        residual = self.fc2(residual)
        return values + residual


class GCAMExpert(nn.Module):
    def __init__(self, hidden_dim, residual_blocks, dropout):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(INPUT_DIM, hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden_dim, dropout) for _ in range(residual_blocks)
        )
        self.output = nn.Linear(hidden_dim, OUTPUT_DIM)

    def forward(self, values):
        hidden = self.input(values)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def _load():
    global _MODEL, _PAYLOAD
    if _MODEL is None:
        try:
            payload = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(WEIGHTS, map_location="cpu")
        model = GCAMExpert(
            hidden_dim=int(payload["hidden_dim"]),
            residual_blocks=int(payload["residual_blocks"]),
            dropout=float(payload["dropout"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        torch.set_num_threads(1)
        _MODEL = model
        _PAYLOAD = payload
    return _MODEL, _PAYLOAD


def _apply_physical_bounds(values):
    values = values.reshape(OUTPUT_SHAPE)
    values[0:2, :] = np.clip(values[0:2, :], 0.0, 1.0)
    values[2:4, :] = np.maximum(values[2:4, :], 0.0)
    values[4, :] = np.clip(values[4, :], -1.0, 8.0)
    return values


def predict(inputs):
    if len(inputs) != INPUT_DIM:
        raise ValueError(f"expected {INPUT_DIM} GCAM parameters, got {len(inputs)}")
    array = np.asarray(inputs, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("GCAM parameters must be finite")
    model, payload = _load()
    input_mean = payload["input_mean"].numpy()
    input_std = payload["input_std"].numpy()
    target_mean = payload["target_mean"].numpy()
    target_std = payload["target_std"].numpy()
    normalized = np.clip((array - input_mean) / input_std, -4.0, 4.0)
    tensor = torch.from_numpy(normalized).reshape(1, -1)
    with torch.no_grad():
        output_normalized = model(tensor)[0].numpy()
    output = output_normalized * target_std + target_mean
    output = _apply_physical_bounds(output)
    return [float(value) for value in output.reshape(-1)]
'''


def build_package(
    output_dir: Path,
    *,
    model_name: str,
    checkpoint: dict[str, Any],
    example_input: list[float],
    metrics: dict[str, Any],
    param_names: list[str],
    output_names: list[str],
) -> Path:
    package_dir = output_dir / "package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    weights_path = package_dir / "gcam_expert.pt"
    torch.save(checkpoint, weights_path)
    (package_dir / "adapter.py").write_text(ADAPTER_SOURCE, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "runtime": "python",
        "entrypoint": "adapter.py",
        "callable": "predict",
        "example_input": example_input,
        "input_dim": EXPECTED_INPUT_DIM,
        "output_dim": math.prod(EXPECTED_OUTPUT_SHAPE),
        "name": model_name,
        "domain": "energy_climate",
        "simulator": "gcam",
        "demo_prompt": GCAM_DEFAULT_PROMPT,
        "demo_prompt_label": "GCAM 碳定价样例",
        "assembly_enabled": True,
        "data_generation_enabled": True,
    }
    save_json(package_dir / "piernet_expert_model.json", manifest)
    save_json(
        package_dir / "model_card.json",
        {
            "description": (
                "Neural surrogate for the PiERN simplified GCAM/PyPSA simulator."
            ),
            "input": {
                "shape": [EXPECTED_INPUT_DIM],
                "names": param_names,
            },
            "output": {
                "shape": list(EXPECTED_OUTPUT_SHAPE),
                "flatten_order": "C",
                "variables": output_names,
                "years": list(range(2025, 2105, 5)),
            },
            "supported_scenarios": list(SCENARIO_FILES),
            "test_metrics": metrics,
        },
    )

    archive_path = output_dir / f"{model_name}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.iterdir()):
            archive.write(path, arcname=path.name)
    return archive_path


def smoke_test_package(
    package_dir: Path, example_input: list[float]
) -> dict[str, Any]:
    adapter_path = package_dir / "adapter.py"
    spec = importlib.util.spec_from_file_location("gcam_expert_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load packaged adapter")
    module = importlib.util.module_from_spec(spec)
    previous_path = list(sys.path)
    previous_cwd = Path.cwd()
    try:
        sys.path.insert(0, str(package_dir))
        import os

        os.chdir(package_dir)
        spec.loader.exec_module(module)
        result = module.predict(example_input)
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
    values = np.asarray(result, dtype=np.float64)
    if values.shape != (math.prod(EXPECTED_OUTPUT_SHAPE),):
        raise ValueError(f"Packaged adapter returned unexpected shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Packaged adapter returned non-finite values")
    return {
        "output_dim": int(values.size),
        "output_min": float(values.min()),
        "output_max": float(values.max()),
        "physical_validity": physical_validity(values.reshape(1, -1)),
    }


def main() -> int:
    args = parse_args()
    config = TrainConfig(
        seed=args.seed,
        split_seed=args.split_seed,
        hidden_dim=args.hidden_dim,
        residual_blocks=args.residual_blocks,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    set_determinism(config.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "training_config.json", asdict(config))

    data = load_data(args.data_dir)
    inputs = data["inputs"]
    targets = data["targets"]
    scenarios = data["scenarios"]
    splits = stratified_split(
        scenarios,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        seed=config.split_seed,
    )
    np.savez_compressed(output_dir / "split_indices.npz", **splits)

    normalization = compute_normalization(
        inputs[splits["train"]], targets[splits["train"]]
    )
    normalized_inputs = (
        inputs - normalization["input_mean"]
    ) / normalization["input_std"]
    normalized_targets = (
        targets - normalization["target_mean"]
    ) / normalization["target_std"]

    train_loader = make_loader(
        normalized_inputs[splits["train"]],
        normalized_targets[splits["train"]],
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = make_loader(
        normalized_inputs[splits["val"]],
        normalized_targets[splits["val"]],
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    model = GCAMExpert(
        hidden_dim=config.hidden_dim,
        residual_blocks=config.residual_blocks,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=50,
        min_lr=1e-6,
    )
    criterion = nn.MSELoss()
    best_path = output_dir / "best_checkpoint.pt"
    metrics_path = output_dir / "training_metrics.jsonl"
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started_at = time.time()

    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            total_loss = 0.0
            total_samples = 0
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += float(loss.item()) * batch_inputs.shape[0]
                total_samples += batch_inputs.shape[0]

            train_loss = total_loss / max(total_samples, 1)
            val_loss = evaluate_loss(model, val_loader, device, criterion)
            scheduler.step(val_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])
            improved = val_loss < best_val - 1e-7
            if improved:
                best_val = val_loss
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "hidden_dim": config.hidden_dim,
                        "residual_blocks": config.residual_blocks,
                        "dropout": config.dropout,
                        **{
                            key: torch.from_numpy(value.copy())
                            for key, value in normalization.items()
                        },
                        "epoch": epoch,
                        "validation_loss": val_loss,
                    },
                    best_path,
                )
            else:
                stale_epochs += 1

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "learning_rate": current_lr,
                "best_validation_loss": best_val,
                "best_epoch": best_epoch,
            }
            metrics_handle.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_handle.flush()
            if epoch == 1 or epoch % 25 == 0 or improved and epoch % 5 == 0:
                print(json.dumps(record, sort_keys=True), flush=True)
            if stale_epochs >= config.patience:
                print(
                    f"Early stopping at epoch {epoch}; best epoch {best_epoch}",
                    flush=True,
                )
                break

    try:
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    split_metrics: dict[str, Any] = {}
    predictions_by_split: dict[str, np.ndarray] = {}
    for split_name, indices in splits.items():
        predicted_normalized = predict_normalized(
            model,
            normalized_inputs[indices],
            batch_size=config.batch_size,
            device=device,
        )
        predictions = (
            predicted_normalized * normalization["target_std"]
            + normalization["target_mean"]
        )
        predictions = apply_physical_bounds(predictions)
        predictions_by_split[split_name] = predictions
        split_metrics[split_name] = regression_metrics(
            targets[indices],
            predictions,
            output_names=data["output_names"],
        )
        split_metrics[split_name]["physical_validity"] = physical_validity(
            predictions
        )

    scenario_metrics: dict[str, Any] = {}
    test_indices = splits["test"]
    test_predictions = predictions_by_split["test"]
    for scenario_index, scenario_name in enumerate(data["scenario_names"]):
        mask = scenarios[test_indices] == scenario_index
        scenario_metrics[scenario_name] = regression_metrics(
            targets[test_indices][mask],
            test_predictions[mask],
            output_names=data["output_names"],
        )

    baseline = np.broadcast_to(
        targets[splits["train"]].mean(axis=0, keepdims=True),
        targets[splits["test"]].shape,
    )
    baseline_metrics = regression_metrics(
        targets[splits["test"]],
        baseline,
        output_names=data["output_names"],
    )
    final_metrics = {
        "dataset_samples": int(inputs.shape[0]),
        "split_sizes": {name: int(len(value)) for name, value in splits.items()},
        "device": str(device),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "best_epoch": int(best_epoch),
        "best_validation_loss_normalized": float(best_val),
        "elapsed_seconds": float(time.time() - started_at),
        "splits": split_metrics,
        "test_by_scenario": scenario_metrics,
        "test_mean_baseline": baseline_metrics,
    }
    save_json(output_dir / "evaluation.json", final_metrics)

    samples: list[dict[str, Any]] = []
    for scenario_index, scenario_name in enumerate(data["scenario_names"]):
        positions = np.flatnonzero(scenarios[test_indices] == scenario_index)
        position = int(positions[0])
        samples.append(
            {
                "scenario": scenario_name,
                "input": inputs[test_indices[position]].tolist(),
                "target": targets[test_indices[position]]
                .reshape(EXPECTED_OUTPUT_SHAPE)
                .tolist(),
                "prediction": test_predictions[position]
                .reshape(EXPECTED_OUTPUT_SHAPE)
                .tolist(),
            }
        )
    save_json(output_dir / "sample_predictions.json", samples)

    archive_path = build_package(
        output_dir,
        model_name=args.model_name,
        checkpoint=checkpoint,
        example_input=samples[0]["input"],
        metrics=final_metrics["splits"]["test"],
        param_names=data["param_names"],
        output_names=data["output_names"],
    )
    package_smoke = smoke_test_package(
        output_dir / "package", samples[0]["input"]
    )
    summary = {
        "status": "completed",
        "archive": str(archive_path),
        "checkpoint": str(best_path),
        "evaluation": str(output_dir / "evaluation.json"),
        "package_smoke": package_smoke,
        "test_metrics": final_metrics["splits"]["test"],
        "best_epoch": best_epoch,
        "elapsed_seconds": final_metrics["elapsed_seconds"],
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
