"""
Build Stage 4 Token Router training data from Stage 3 `data/text2comp/*.jsonl`.

Each Stage 3 sample becomes:
- 1 positive sample:
  `user_prefix + input + user_suffix + assistant_prefix + trigger_prefix`
- `neg_ratio` negative samples:
  `user_prefix + input + user_suffix + assistant_prefix + trigger_prefix[:pos]`

The input portion is always complete. Negatives only truncate inside the assistant
trigger prefix so router inference sees the same user context shape as training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CHAT_TEMPLATES: dict[str, dict[str, str]] = {
    "qwen": {
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
    },
    "chatml": {
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
    },
    "deepseek": {
        "user_prefix": "<｜User｜>",
        "user_suffix": "",
        "assistant_prefix": "<｜Assistant｜>",
    },
    "llama3": {
        "user_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_suffix": "<|eot_id|>",
        "assistant_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "mistral": {
        "user_prefix": "[INST] ",
        "user_suffix": " [/INST]",
        "assistant_prefix": "",
    },
}


def _apply_chat_template_pos(input_text: str, trigger_prefix: str, tmpl: dict[str, str]) -> str:
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
    user_prefix = tmpl["user_prefix"]
    user_suffix = tmpl["user_suffix"]
    assistant_prefix = tmpl["assistant_prefix"]
    trigger_length = len(trigger_prefix)
    if trigger_length <= 1:
        return user_prefix + input_text + user_suffix + assistant_prefix
    pos = rng.randint(0, trigger_length - 1)
    return user_prefix + input_text + user_suffix + assistant_prefix + trigger_prefix[:pos]


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _build_source_signature(jsonl_path: Path) -> dict[str, object] | None:
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


def _build_embedding_metadata(
    *,
    provider: str,
    model: str,
    tokenizer: str,
    source: str,
) -> dict[str, str]:
    model_name = model.strip()
    tokenizer_name = tokenizer.strip() or model_name
    return {
        "embedding_provider": provider.strip(),
        "embedding_model": model_name,
        "embedding_tokenizer": tokenizer_name,
        "embedding_source": source.strip(),
    }


def _write_scenario_meta(
    output_path: Path,
    *,
    source_path: Path,
    chat_template_name: str,
    neg_ratio: int,
    source_signature: dict[str, object] | None,
    output_count: int,
    embedding_metadata: dict[str, str],
) -> None:
    payload = {
        "scenario": output_path.stem,
        "source_file": source_path.name,
        "chat_template": chat_template_name,
        "neg_ratio": neg_ratio,
        "source_signature": source_signature,
        "output_count": output_count,
        **embedding_metadata,
    }
    _scenario_meta_path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_scenario_meta(output_path: Path) -> dict[str, object] | None:
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
    source_signature: dict[str, object] | None,
    embedding_metadata: dict[str, str],
) -> tuple[bool, str]:
    if source_signature is None:
        return False, f"missing current Stage 3 source file: {source_path.name}"
    meta = _load_scenario_meta(output_path)
    if not meta:
        return False, "missing build metadata"
    if meta.get("chat_template") != chat_template_name:
        return False, "chat_template mismatch"
    if int(meta.get("neg_ratio", -1)) != neg_ratio:
        return False, "neg_ratio mismatch"
    if meta.get("source_signature") != source_signature:
        return False, "Stage 3 source file changed"
    for key, value in embedding_metadata.items():
        if (meta.get(key) or "") != value:
            return False, f"{key} mismatch"
    return True, ""


def _extract_trigger_prefix(target_template: str) -> str:
    marker = "{output_0}"
    pos = target_template.find(marker)
    if pos == -1:
        return ""
    return target_template[:pos]


def _build_samples_from_file(
    jsonl_path: Path,
    rng: random.Random,
    chat_tmpl: dict[str, str],
    embedding_metadata: dict[str, str],
    neg_ratio: int = 1,
    progress_callback=None,
    progress_interval: int = 500,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    source_processed = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue

            input_text = record.get("input", "")
            metadata = record.get("metadata", {})
            target_template = metadata.get("target_template", "")
            if not input_text:
                continue

            trigger_prefix = _extract_trigger_prefix(target_template)
            if not trigger_prefix:
                continue

            base_meta = {
                "simulator": metadata.get("simulator", "unknown"),
                "scenario": metadata.get("scenario", "unknown"),
                "language": metadata.get("language", "unknown"),
                "chat_template": chat_tmpl.get("_name", "plain"),
                **embedding_metadata,
            }

            samples.append(
                {
                    "context": _apply_chat_template_pos(input_text, trigger_prefix, chat_tmpl),
                    "label": 1,
                    "metadata": {**base_meta, "trigger_prefix": trigger_prefix},
                }
            )

            for _ in range(neg_ratio):
                samples.append(
                    {
                        "context": _apply_chat_template_neg(input_text, trigger_prefix, chat_tmpl, rng),
                        "label": 0,
                        "metadata": {**base_meta, "trigger_prefix": ""},
                    }
                )

            source_processed += 1
            if progress_callback and source_processed % progress_interval == 0:
                progress_callback(len(samples))
    return samples


def _write_jsonl(samples: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _write_all(samples: list[dict[str, object]], output_dir: Path, seed: int) -> int:
    rng = random.Random(seed)
    rng.shuffle(samples)
    out_path = output_dir / "train.jsonl"
    _write_jsonl(samples, out_path)
    print(f"  [train] {len(samples)} rows -> {out_path}", flush=True)
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Token Router training data.")
    parser.add_argument("--data-dir", type=str, default="data/text2comp")
    parser.add_argument("--output-dir", type=str, default="data/router")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--chat-template", type=str, default="custom", choices=list(CHAT_TEMPLATES.keys()) + ["custom"])
    parser.add_argument("--user-prefix", type=str, default="")
    parser.add_argument("--user-suffix", type=str, default="")
    parser.add_argument("--assistant-prefix", type=str, default="")
    parser.add_argument("--embedding-provider", type=str, default="")
    parser.add_argument("--embedding-model", type=str, default="")
    parser.add_argument("--embedding-tokenizer", type=str, default="")
    parser.add_argument("--embedding-source", type=str, default="router_build")
    args = parser.parse_args()

    template_name = args.chat_template
    if template_name == "custom":
        chat_tmpl = {
            "_name": "custom",
            "user_prefix": args.user_prefix,
            "user_suffix": args.user_suffix,
            "assistant_prefix": args.assistant_prefix,
        }
    else:
        chat_tmpl = dict(CHAT_TEMPLATES.get(template_name, {"user_prefix": "", "user_suffix": "", "assistant_prefix": ""}))
        chat_tmpl["_name"] = template_name

    embedding_metadata = _build_embedding_metadata(
        provider=args.embedding_provider,
        model=args.embedding_model,
        tokenizer=args.embedding_tokenizer,
        source=args.embedding_source,
    )

    print(f"[router-build] chat_template={template_name}", flush=True)
    if embedding_metadata["embedding_model"]:
        print(
            "[router-build] embedding "
            f"provider={embedding_metadata['embedding_provider'] or 'unknown'} "
            f"model={embedding_metadata['embedding_model']} "
            f"tokenizer={embedding_metadata['embedding_tokenizer']}",
            flush=True,
        )
    else:
        print(
            "[router-build] embedding metadata missing; "
            "router training will remain on char tokens unless embedding mode is forced",
            flush=True,
        )

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    scenario_dir = output_dir / "by_scenario"
    if not data_dir.exists():
        print(f"[error] missing data directory: {data_dir}", flush=True)
        raise SystemExit(1)

    all_jsonl_files = sorted(
        path for path in data_dir.glob("*.jsonl")
        if path.name != "all_training_data.jsonl"
    )
    if not all_jsonl_files:
        print(f"[error] no Stage 3 jsonl files found under {data_dir}", flush=True)
        raise SystemExit(1)

    scenario_source_signatures = {
        path.stem: _build_source_signature(path) for path in all_jsonl_files
    }
    scenario_source_counts = {
        scenario: int((signature or {}).get("row_count", 0))
        for scenario, signature in scenario_source_signatures.items()
    }

    jsonl_files = list(all_jsonl_files)
    if args.scenarios:
        wanted = set(args.scenarios)
        jsonl_files = [path for path in jsonl_files if path.stem in wanted]
        if not jsonl_files:
            print(f"[error] scenarios not found: {sorted(wanted)}", flush=True)
            raise SystemExit(1)

    print(f"[router-build] scenarios={len(jsonl_files)}", flush=True)
    for path in jsonl_files:
        expected = scenario_source_counts[path.stem] * (1 + args.neg_ratio)
        print(f"PROGRESS_INIT:{path.stem}:{expected}", flush=True)

    rng = random.Random(args.seed)
    new_samples: list[dict[str, object]] = []
    processed_scenarios: set[str] = set()

    for jsonl_path in jsonl_files:
        scenario_name = jsonl_path.stem
        expected = scenario_source_counts[scenario_name] * (1 + args.neg_ratio)
        print(f"  processing {jsonl_path.name} ...", flush=True)

        def _progress(done_samples: int, sc: str = scenario_name, exp: int = expected) -> None:
            print(f"PROGRESS_UPDATE:{sc}:{done_samples}:{exp}", flush=True)

        samples = _build_samples_from_file(
            jsonl_path,
            rng,
            chat_tmpl=chat_tmpl,
            embedding_metadata=embedding_metadata,
            neg_ratio=args.neg_ratio,
            progress_callback=_progress,
        )
        n_pos = sum(1 for sample in samples if sample["label"] == 1)
        print(
            f"  done: {len(samples)} rows (pos={n_pos}, neg={len(samples) - n_pos})",
            flush=True,
        )
        print(f"PROGRESS_DONE:{scenario_name}:{len(samples)}:{expected}", flush=True)

        scenario_output_path = scenario_dir / f"{scenario_name}.jsonl"
        _write_jsonl(samples, scenario_output_path)
        _write_scenario_meta(
            scenario_output_path,
            source_path=jsonl_path,
            chat_template_name=chat_tmpl["_name"],
            neg_ratio=args.neg_ratio,
            source_signature=scenario_source_signatures.get(scenario_name),
            output_count=len(samples),
            embedding_metadata=embedding_metadata,
        )
        new_samples.extend(samples)
        processed_scenarios.add(scenario_name)

    if not new_samples:
        print(
            "[error] no router samples were generated; "
            "make sure Stage 3 data exists and target_template contains {output_0}",
            flush=True,
        )
        raise SystemExit(1)

    all_samples: list[dict[str, object]] = list(new_samples)
    if scenario_dir.exists():
        for existing_file in sorted(scenario_dir.glob("*.jsonl")):
            scenario_name = existing_file.stem
            if scenario_name in processed_scenarios:
                continue
            reusable, reason = _can_reuse_existing_scenario(
                existing_file,
                source_path=data_dir / f"{scenario_name}.jsonl",
                chat_template_name=chat_tmpl["_name"],
                neg_ratio=args.neg_ratio,
                source_signature=scenario_source_signatures.get(scenario_name),
                embedding_metadata=embedding_metadata,
            )
            if not reusable:
                print(f"  [skip] cannot reuse {existing_file.name}: {reason}", flush=True)
                continue
            try:
                with existing_file.open("r", encoding="utf-8") as handle:
                    for raw in handle:
                        raw = raw.strip()
                        if raw:
                            all_samples.append(json.loads(raw))
                print(f"  reused {existing_file.name}", flush=True)
            except Exception as exc:
                print(f"  [warn] failed to read {existing_file.name}: {exc}", flush=True)

    n_pos_total = sum(1 for sample in all_samples if sample["label"] == 1)
    print(
        f"\n  total={len(all_samples)} rows "
        f"(pos={n_pos_total}, neg={len(all_samples) - n_pos_total})",
        flush=True,
    )
    total = _write_all(all_samples, output_dir, args.seed)
    print(f"\n[done] total={total} -> {output_dir / 'train.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
