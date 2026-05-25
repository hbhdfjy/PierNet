from __future__ import annotations

from pathlib import Path

from piern.synth.services import router_executor


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_router_build_command_uses_runtime_data_root(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "runtime-data"
    text2comp_dir = data_root / "text2comp_parquet"
    router_dir = data_root / "router_parquet"
    monkeypatch.setattr(router_executor, "DATA_ROOT", data_root)
    monkeypatch.setattr(router_executor.portable, "TEXT2COMP_PARQUET_DIR", text2comp_dir)
    monkeypatch.setattr(router_executor.portable, "ROUTER_PARQUET_DIR", router_dir)
    monkeypatch.setattr(router_executor.portable, "has_partitions", lambda kind: kind == "text2comp")

    command = router_executor._router_build_command(7, 2, 3, ["gcam/shared"])

    assert _option_value(command, "--data-dir") == str(text2comp_dir)
    assert _option_value(command, "--output-dir") == str(router_dir)
    assert command[command.index("--scenarios") + 1 :] == ["gcam/shared"]


def test_router_build_command_falls_back_to_legacy_jsonl_when_no_parquet(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "runtime-data"
    router_dir = data_root / "router_parquet"
    monkeypatch.setattr(router_executor, "DATA_ROOT", data_root)
    monkeypatch.setattr(router_executor.portable, "ROUTER_PARQUET_DIR", router_dir)
    monkeypatch.setattr(router_executor.portable, "has_partitions", lambda kind: False)

    command = router_executor._router_build_command(7, 2, 3, ["gcam/shared"])

    assert _option_value(command, "--data-dir") == str(data_root / "text2comp")
    assert _option_value(command, "--input-format") == "jsonl"
    assert _option_value(command, "--output-dir") == str(router_dir)


def test_router_build_command_uses_auto_input_for_mixed_storage(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "runtime-data"
    jsonl_dir = data_root / "text2comp"
    jsonl_dir.mkdir(parents=True)
    (jsonl_dir / "simpeg_shared.jsonl").write_text("{}\n", encoding="utf-8")
    router_dir = data_root / "router_parquet"
    monkeypatch.setattr(router_executor, "DATA_ROOT", data_root)
    monkeypatch.setattr(router_executor.portable, "ROUTER_PARQUET_DIR", router_dir)
    monkeypatch.setattr(router_executor.portable, "has_partitions", lambda kind: kind == "text2comp")

    command = router_executor._router_build_command(7, 2, 3, ["gcam/shared", "simpeg/shared"])

    assert _option_value(command, "--data-dir") == str(data_root)
    assert _option_value(command, "--input-format") == "auto"
    assert command[command.index("--scenarios") + 1 :] == ["gcam/shared", "simpeg/shared"]
