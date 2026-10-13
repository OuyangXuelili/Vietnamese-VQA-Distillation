"""Launch records cannot turn a partial review into silent GPU training."""

import json
from pathlib import Path

import pytest

from vqa_explanation.training import preference_launch as launch


def record_file(tmp_path, **updates):
    record = {"schema": "preference_training_launch_v1", "scope": "main_preference_experiment",
              "data_locked": False, **updates}
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["inspect", "gate", "train"])
def test_partial_dataset_stops_before_stage_or_model(tmp_path, monkeypatch, mode):
    path = record_file(tmp_path)
    monkeypatch.setattr(launch, "prepare_stage", lambda args: pytest.fail("Must not load stage"))
    report, args = launch.prepare_launch(path, tmp_path, "m2", mode)
    assert report["status"] == "blocked" and args is None
    assert report["main_training_started"] is False


@pytest.mark.parametrize("flag", [None, "true", 1, False])
def test_only_explicit_boolean_lock_is_accepted(tmp_path, flag):
    record, paths = launch.load_launch(record_file(tmp_path, data_locked=flag), tmp_path)
    assert paths is None


def test_locked_record_resolves_cloud_data_separately_from_source(tmp_path):
    path = record_file(tmp_path, data_locked=True, expected_pairs=100,
                       preferences="preference_local/final/frozen", m1_adapter="runs/m1/adapter",
                       m1_selection="reports/selection.json", environment="configs/cloud.yaml",
                       study_config="configs/study.yaml")
    _, paths = launch.load_launch(path, tmp_path)
    assert paths["preferences"] == tmp_path / "preference_local/final/frozen"
    assert paths["environment"] == launch.project_root() / "configs/cloud.yaml"


def test_abandoned_pilot_record_is_rejected(tmp_path):
    path = record_file(tmp_path, scope="technical_gpu_gate_not_main_experiment")
    with pytest.raises(ValueError, match="main preference"):
        launch.load_launch(path, tmp_path)


@pytest.mark.parametrize("arm,mode", [("m0", "gate"), ("m2", "automatic")])
def test_unknown_arm_or_mode_is_rejected(tmp_path, arm, mode):
    with pytest.raises(ValueError, match="Unknown"):
        launch.prepare_launch(record_file(tmp_path), tmp_path, arm, mode)


def test_main_launch_uses_same_gate_folder_after_reexecution(tmp_path):
    command = launch.launch_command(tmp_path, tmp_path / "launch.json", tmp_path,
                                    "m3", "train", tmp_path / "run")
    assert Path(command[1]) == tmp_path / "scripts/launch_preference_training.py"
    assert command[command.index("--gpu-gate") + 1] == str(
        tmp_path / "run/m3_gate/training_result.json")
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "run/m3_train")


@pytest.mark.parametrize("mode,returncode", [("inspect", 0), ("gate", 2), ("train", 2)])
def test_launch_cli_blocks_an_unlocked_record_before_training(tmp_path, mode, returncode):
    import subprocess
    import sys

    result = subprocess.run([
        sys.executable, str(launch.project_root() / "scripts/launch_preference_training.py"),
        "--launch-record", str(record_file(tmp_path)), "--project-dir", str(tmp_path),
        "--arm", "m2", "--mode", mode,
    ], capture_output=True, text=True, check=False)
    assert result.returncode == returncode, result.stderr
    assert json.loads(result.stdout)["main_training_started"] is False


def test_inspect_command_cannot_request_resume(tmp_path):
    with pytest.raises(ValueError, match="Resume"):
        launch.launch_command(tmp_path, tmp_path / "launch.json", tmp_path,
                              "m2", "inspect", tmp_path / "run", resume=Path("checkpoint-4"))


def test_resume_keeps_gate_but_uses_new_output(tmp_path):
    command = launch.launch_command(tmp_path, tmp_path / "launch.json", tmp_path,
                                    "m2", "train", tmp_path / "run",
                                    resume=tmp_path / "checkpoint-25", attempt="resume25")
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "run/m2_train_resume25")
    assert command[command.index("--gpu-gate") + 1] == str(
        tmp_path / "run/m2_gate/training_result.json")
    assert command[command.index("--resume-from-checkpoint") + 1] == str(tmp_path / "checkpoint-25")


@pytest.mark.parametrize("attempt", ["../old", "/old", "", "x" * 65])
def test_output_label_cannot_escape_run_folder(tmp_path, attempt):
    with pytest.raises(ValueError, match="attempt label"):
        launch.stage_output(tmp_path, "m3", "train", attempt)


def test_resume_cannot_reuse_initial_output(tmp_path):
    with pytest.raises(ValueError, match="new attempt"):
        launch.launch_command(tmp_path, tmp_path / "launch.json", tmp_path,
                              "m2", "train", tmp_path / "run", resume=Path("checkpoint-25"))


def test_gate_label_cannot_change_the_expected_gate_path(tmp_path):
    with pytest.raises(ValueError, match="main training"):
        launch.stage_output(tmp_path, "m3", "gate", "another")
