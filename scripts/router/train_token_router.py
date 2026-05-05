from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from piern.training.router import RouterTrainingConfig, run_training  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a full-sequence Token Router.")
    parser.add_argument("--simulator", default="modflow")
    parser.add_argument("--scenarios", nargs="*")
    parser.add_argument("--router-dir", default="data/router")
    parser.add_argument("--artifact-root", default="artifacts/token_router/modflow")
    parser.add_argument("--prepared-name")
    parser.add_argument("--run-name")
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--keep-last-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--scene-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--resume-from")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--input-representation", choices=("embedding",), default="embedding")
    parser.add_argument("--stop-file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RouterTrainingConfig(
        simulator=args.simulator,
        scenarios=tuple(args.scenarios) if args.scenarios else None,
        router_dir=args.router_dir,
        artifact_root=args.artifact_root,
        prepared_name=args.prepared_name,
        run_name=args.run_name,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        keep_last_epochs=args.keep_last_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        model_dim=args.model_dim,
        scene_dim=args.scene_dim,
        dropout=args.dropout,
        kernel_size=args.kernel_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        force_prepare=args.force_prepare,
        resume_from=args.resume_from,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        input_representation=args.input_representation,
        stop_file=args.stop_file,
    )
    run_dir = run_training(config)
    print(f"[done] run_dir={Path(run_dir).resolve()}")


if __name__ == "__main__":
    main()
