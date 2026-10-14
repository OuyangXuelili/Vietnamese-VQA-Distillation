import hashlib
import json
from pathlib import Path

import pytest
import yaml

from vqa_explanation.inference import study_validation as study


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"question_id": str(index), "split": "validation"} for index in range(1407)]
    annotations = data / "validation.jsonl"
    annotations.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(study, "VALIDATION_SHA256", study.sha256_file(annotations))
    monkeypatch.setattr(study, "validate_selected_records", lambda *args: None)
    environment = tmp_path / "environment.yaml"
    environment.write_text(yaml.safe_dump({"annotations": str(data / "all.jsonl"),
                                         "image_root": str(data),
                                         "run_root": str(tmp_path / "runs")}), encoding="utf-8")
    m1 = tmp_path / "m1"
    names = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"m1"}
    m1.mkdir()
    for name, value in names.items():
        (m1 / name).write_bytes(value)
    parent = {"files": {name: digest(value) for name, value in names.items()}}
    group = tmp_path / "training"
    for arm in ("m2", "m3"):
        stage = group / f"{arm}_train"
        adapter = stage / "training/checkpoint-24"
        adapter.mkdir(parents=True)
        for name, value in names.items():
            (adapter / name).write_bytes(value)
        plan = {"pairs": 186, "m1_adapter": parent, "preferences_manifest_sha256": "p",
                "model": {"id": study.MODEL_ID, "revision": study.MODEL_REVISION}}
        plan["sha256"] = digest(json.dumps(plan, sort_keys=True).encode())
        report = {"status": "passed", "global_step": 24, "epoch": 1,
                  "policy_updated": True, "saved_policy_verified": True,
                  "reference_unchanged": arm == "m3", "plan_sha256": plan["sha256"],
                  "output_adapter": parent}
        (stage / "data_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (stage / "training_result.json").write_text(json.dumps(report), encoding="utf-8")
    return environment, group, m1, tmp_path / "evaluation"


def test_inspection_loads_no_model_and_writes_no_predictions(inputs):
    result = study.run_validation(*inputs, inspect_only=True)
    assert result["status"] == "inputs_verified"
    assert result["questions"] == 1407
    assert [command[command.index("--experiment-id") + 1] for command in result["commands"]] == [
        "M2_SFT_CHOSEN", "M3_DPO", "M0_PRETRAINED", "M1_SFT"]
    assert not inputs[-1].exists()
    assert all(command[command.index("--limit") + 1] == "0" for command in result["commands"])


def test_export_is_resumable_but_new_runs_do_not_use_resume(inputs):
    result = study.run_validation(*inputs, inspect_only=True)
    assert all("--resume" not in command for command in result["commands"])
    inputs[-1].mkdir()
    (inputs[-1] / "m2.jsonl.manifest.json").write_text("{}", encoding="utf-8")
    result = study.run_validation(*inputs, inspect_only=True)
    assert "--resume" in result["commands"][0]


@pytest.mark.parametrize("target", ["m1", "m2", "m3"])
def test_changed_adapter_blocks_every_export(inputs, target):
    adapter = inputs[2] if target == "m1" else inputs[1] / f"{target}_train/training/checkpoint-24"
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Adapter changed"):
        study.run_validation(*inputs, inspect_only=True)


def test_unfinished_training_blocks_export(inputs):
    path = inputs[1] / "m3_train/training_result.json"
    report = json.loads(path.read_text())
    report["global_step"] = 5
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not complete"):
        study.run_validation(*inputs, inspect_only=True)


def test_changed_plan_blocks_export(inputs):
    path = inputs[1] / "m2_train/data_plan.json"
    plan = json.loads(path.read_text())
    plan["pairs"] = 187
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan changed"):
        study.run_validation(*inputs, inspect_only=True)


def test_changed_validation_blocks_export(inputs):
    (inputs[0].parent / "data/validation.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="annotations changed"):
        study.run_validation(*inputs, inspect_only=True)


def test_full_pipeline_calls_exports_then_four_paired_comparisons(inputs):
    calls = []
    result = study.run_validation(*inputs, runner=lambda cmd, **kw: calls.append(cmd))
    assert result["status"] == "completed" and len(calls) == 8
    assert all(Path(command[2]).name == "run_m0.py" for command in calls[:4])
    assert all("--bootstrap-resamples" in command for command in calls[4:])
