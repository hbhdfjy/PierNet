from __future__ import annotations

import json

from PierNet.training.text2comp.data_converter import convert_jsonl, convert_sample


def test_convert_sample_uses_transformed_expert_input_not_physical_target():
    result = convert_sample(
        {
            "input": "参数 a=3，参数 b=4",
            "number": [1.0, 2.0],
            "params_transformed": [3.0, 4.0],
            "target": "物理结果 [[100.0, 200.0, 300.0]]",
        }
    )

    assert result is not None
    assert result["label"] == [3.0, 4.0]
    assert result["expert_input"] == [3.0, 4.0]
    assert result["metadata"]["label_semantics"] == "expert_input"
    assert result["metadata"]["label_source"] == "params_transformed"


def test_convert_sample_prefers_explicit_expert_input():
    result = convert_sample(
        {
            "input": "计算这个样本",
            "expert_input": [[7.0, 8.0]],
            "params_transformed": [3.0, 4.0],
            "target": "最终物理输出 [999.0]",
        }
    )

    assert result is not None
    assert result["label"] == [7.0, 8.0]
    assert result["metadata"]["label_source"] == "expert_input"


def test_convert_sample_rejects_target_only_record():
    assert convert_sample({"input": "计算", "target": "物理输出 [1.0, 2.0]"}) is None


def test_convert_jsonl_validates_expert_input_dimension(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "train.jsonl"
    rows = [
        {
            "input": "样本一",
            "params_transformed": [1.0, 2.0],
            "target": "物理输出 [9.0, 9.0, 9.0]",
        },
        {
            "input": "样本二",
            "params_transformed": [3.0, 4.0, 5.0],
            "target": "物理输出 [8.0]",
        },
    ]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    stats = convert_jsonl(source, output, expected_dim=2)

    assert stats["converted"] == 1
    assert stats["skipped"] == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["label"] == [1.0, 2.0]
    assert saved["metadata"]["label_semantics"] == "expert_input"
