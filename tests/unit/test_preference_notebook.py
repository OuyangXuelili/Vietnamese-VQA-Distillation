"""The active notebook is a small launcher, with training disabled by default."""

import ast
import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_notebook_has_one_setup_and_two_short_arm_cells():
    notebook = json.loads((ROOT / "notebooks/04_train_preferences.ipynb").read_text("utf-8"))
    assert [cell["id"] for cell in notebook["cells"]] == [
        "preference-intro", "preference-storage", "preference-source", "preference-preflight",
        "preference-m2", "preference-m3", "preference-results",
    ]
    code = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"}
    for source in code.values():
        ast.parse(source)
    assert sum(source.count("drive.mount(") for source in code.values()) == 1
    assert "MODE = 'inspect'" in code["preference-preflight"]
    assert "launch['run_group']" in code["preference-preflight"]
    assert "datetime" not in code["preference-preflight"]
    assert "ATTEMPT = {'m2': 'initial', 'm3': 'initial'}" in code["preference-preflight"]
    assert "stage_output(RUN_FOLDER, arm, MODE, ATTEMPT[arm])" in code["preference-results"]
    assert len(code["preference-m2"].splitlines()) == 3
    assert len(code["preference-m3"].splitlines()) == 3
    assert "train_sft_full" not in "\n".join(code.values())
    assert "generate_preference_candidates" not in "\n".join(code.values())
    digest = hashlib.sha256((ROOT / "scripts/bootstrap_cloud_source.py").read_bytes()).hexdigest()
    assert digest in code["preference-source"]
    assert all(not cell["outputs"] and cell["execution_count"] is None
               for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_archived_notebook_preserves_its_exact_original_bytes():
    path = ROOT / "notebooks/archive/03_m1_and_preference_before_reorganization_20261013.ipynb"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "83972dbdae24e32b7edfda204b156b1aa988c2f52387e53f4d5adacc8124863d")


def test_storage_setup_allows_time_for_auth_and_checks_launch_record():
    notebook = json.loads((ROOT / "notebooks/04_train_preferences.ipynb").read_text("utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "mount"]
    assert len(calls) == 1
    assert {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords} == {
        "timeout_ms": 600000,
    }
    assert "if not LAUNCH_PATH.is_file():" in source
    assert "print('Project storage ready:', ROOT)" in source


@pytest.mark.parametrize("locked", [False, True])
def test_data_setup_is_conditional_and_precedes_inspection(tmp_path, locked):
    notebook = json.loads((ROOT / "notebooks/04_train_preferences.ipynb").read_text("utf-8"))
    source = "".join(notebook["cells"][3]["source"])
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_text("{}", encoding="utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(yaml.safe_dump({
        "annotations": str(tmp_path / "dataset/metadata/all.jsonl"),
        "image_root": str(tmp_path / "dataset"), "run_root": str(tmp_path / "runs"),
    }), encoding="utf-8")
    calls = []

    class Commands:
        @staticmethod
        def run(command, **kwargs):
            calls.append(command)

    namespace = {
        "ROOT": tmp_path, "REPO": tmp_path, "LAUNCH_PATH": tmp_path / "launch.json",
        "hashlib": hashlib, "subprocess": Commands, "sys": __import__("sys"),
        "launch": {"data_locked": locked, "environment": str(environment),
                   "run_group": "runs/main", "dataset_bundle": "bundle",
                   "dataset_bundle_sha256": hashlib.sha256(b"{}").hexdigest()},
    }
    exec(compile(source, "preference-preflight", "exec"), namespace)
    assert len(calls) == (3 if locked else 2)
    if locked:
        assert calls[0][1].endswith("verify_coco_bundle.py") and "--extract-to" in calls[0]
    assert all("inspect" in command for command in calls[-2:])


def test_wrong_bundle_hash_blocks_extraction_and_inspection(tmp_path):
    notebook = json.loads((ROOT / "notebooks/04_train_preferences.ipynb").read_text("utf-8"))
    source = "".join(notebook["cells"][3]["source"])
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_text("{}", encoding="utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(yaml.safe_dump({
        "annotations": str(tmp_path / "dataset/metadata/all.jsonl"),
        "image_root": str(tmp_path / "dataset"), "run_root": str(tmp_path / "runs"),
    }), encoding="utf-8")
    namespace = {
        "ROOT": tmp_path, "REPO": tmp_path, "hashlib": hashlib,
        "launch": {"data_locked": True, "environment": str(environment),
                   "run_group": "runs/main", "dataset_bundle": "bundle",
                   "dataset_bundle_sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="Dataset bundle changed"):
        exec(compile(source, "preference-preflight", "exec"), namespace)
