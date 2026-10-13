import copy
import json
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from vqa_explanation.data.io import write_json, write_jsonl
from vqa_explanation.inference.prompting import build_user_prompt
from vqa_explanation.preference.freeze import freeze_pairs
from vqa_explanation.training.preference_data import (
    adapter_fingerprint,
    freeze_selection,
    prepare_stage,
)
from vqa_explanation.utils.run_provenance import sha256_file


@pytest.fixture
def stage_inputs(tmp_path):
    root = tmp_path
    train = dict(
        question_id="train1",
        image_id="train1",
        image_path="images/train.jpg",
        question="Animal?",
        answer="cat",
        explanations=["Fur."],
        split="train",
    )
    path = root / train["image_path"]
    path.parent.mkdir()
    Image.new("RGB", (8, 8)).save(path)
    validation = [
        dict(
            question_id=f"val{i}",
            image_id=f"val{i}",
            image_path=f"images/val{i}.jpg",
            question="Animal?",
            answer="cat",
            explanations=["Fur."],
            split="validation",
        )
        for i in range(1407)
    ]
    canonical = root / "all.jsonl"
    write_jsonl(canonical, [train, *validation])
    annotations_hash = sha256_file(canonical)
    reviewed = root / "reviewed.jsonl"
    pair = dict(
        pair_id="p1",
        source_question_id="train1",
        image_path=train["image_path"],
        prompt=build_user_prompt(train["question"]),
        chosen="Đáp án: cat\nGiải thích: Fur.",
        rejected="Đáp án: cat\nGiải thích: Wings.",
        error_type="attribute",
        generation_method="unit_fixture",
        review_status="accepted",
        annotations_sha256=annotations_hash,
        image_sha256=sha256_file(path),
        reviewer="unit fixture",
        reviewed_at="2026-10-06T10:00:00+07:00",
        evidence="Fur is visible",
        rejected_span="Wings",
        provenance={
            side: {"kind": "manual", "source": "unit fixture"} for side in ("chosen", "rejected")
        },
    )
    write_jsonl(reviewed, [pair])
    bundle = root / "frozen"
    adapter = root / "adapter"
    adapter.mkdir()
    write_json(adapter / "adapter_config.json", {"base_model_name_or_path": "model"})
    (adapter / "adapter_model.safetensors").write_bytes(b"unit fixture, not real weights")
    parent = adapter_fingerprint(adapter)
    pair["provenance"]["original_candidate_origin"] = {
        "model_id": "model", "checkpoint_sha256": parent["sha256"]
    }
    write_jsonl(reviewed, [pair])
    freeze_pairs(reviewed, canonical, root, bundle)
    model = root / "model.yaml"
    model.write_text(yaml.safe_dump({"model": {"id": "model", "revision": "rev"}}))
    config = root / "training.yaml"
    cfg = yaml.safe_load(open("configs/training/dpo.yaml", encoding="utf-8"))
    cfg["expected_pairs"] = 1
    config.write_text(yaml.safe_dump(cfg))
    env = root / "env.yaml"
    env.write_text(
        yaml.safe_dump(
            {"annotations": str(canonical), "image_root": str(root), "run_root": str(root / "runs")}
        )
    )
    val_path = root / "validation.jsonl"
    write_jsonl(val_path, validation)
    inputs = dict(
        model_id="model",
        model_revision="rev",
        experiment_id="M1_SFT",
        adapter="checkpoint-selected",
        adapter_files=parent["files"],
        annotations_sha256=sha256_file(val_path),
        split="validation",
        seed=42,
        image_hashes={row["question_id"]: "a" * 64 for row in validation},
    )
    protocol = {"generation": "greedy"}
    predictions = [
        {
            **{
                key: row[key]
                for key in ("question_id", "question", "image_id", "image_path", "split")
            },
            **{
                key: inputs[key]
                for key in ("model_id", "model_revision", "experiment_id", "adapter", "seed")
            },
            **protocol,
            "image_sha256": "a" * 64,
            "reference_answer": row["answer"],
            "reference_explanations": row["explanations"],
            "raw_output": "Đáp án: cat\nGiải thích: Fur.",
            "prediction_answer": "cat",
            "prediction_explanation": "Fur.",
            "format_valid": True,
            "latency_seconds": 0.1,
        }
        for row in validation
    ]
    write_jsonl(root / "predictions.jsonl", predictions)
    write_json(root / "predictions.manifest.json", {"inputs": inputs, "protocol": protocol})
    write_json(root / "epoch_result.json", {"status": "passed", "global_step": 3683})
    write_json(root / "trainer_state.json", {"global_step": 3683, "epoch": 1.0})
    write_json(
        root / "loss_selection.json",
        {
            "status": "passed",
            "selected_adapter_sha256": parent["files"]["adapter_model.safetensors"],
        },
    )
    write_jsonl(
        root / "human.jsonl",
        [
            dict(
                question_id=row["question_id"],
                review_status="reviewed",
                reviewer="unit fixture",
                answer_assessment="correct",
                explanation_support=2,
            )
            for row in validation[:100]
        ],
    )
    evidence = {
        name: {"path": filename, "sha256": sha256_file(root / filename)}
        for name, filename in {
            "epoch_result": "epoch_result.json",
            "trainer_state": "trainer_state.json",
            "loss_selection": "loss_selection.json",
            "validation_predictions": "predictions.jsonl",
            "validation_manifest": "predictions.manifest.json",
            "validation_annotations": "validation.jsonl",
            "human_review": "human.jsonl",
        }.items()
    }
    selection = root / "selection.json"
    write_json(
        selection,
        dict(
            status="selected",
            selection_split="validation",
            adapter=parent,
            model_id="model",
            model_revision="rev",
            annotations_sha256=annotations_hash,
            evidence=evidence,
        ),
    )
    return SimpleNamespace(
        environment=env,
        config=config,
        model_config=model,
        preferences=bundle,
        m1_adapter=adapter,
        m1_selection=selection,
    )


def test_metadata_gate_accepts_matched_frozen_inputs_without_loading_model(stage_inputs):
    _, _, plan, pairs, chosen = prepare_stage(stage_inputs)
    assert plan["pairs"] == 1 and plan["expected_updates"] == 1
    assert pairs[0]["source_question_id"] == chosen[0]["question_id"]


def test_data_plan_binds_both_declared_arm_configs(stage_inputs):
    from pathlib import Path

    root = stage_inputs.preferences.parent
    sft_cfg = yaml.safe_load(
        Path("configs/training/sft_chosen.yaml").read_text(encoding="utf-8")
    )
    sft_cfg["expected_pairs"] = 1
    sft_path = root / "sft.yaml"
    sft_path.write_text(yaml.safe_dump(sft_cfg), encoding="utf-8")
    study = yaml.safe_load(
        Path("configs/experiments/study_budget_v2.yaml").read_text(encoding="utf-8")
    )
    study["training"].update(m2_config=str(sft_path), m3_config=str(stage_inputs.config))
    stage_inputs.study_config = root / "study.yaml"
    stage_inputs.study_config.write_text(yaml.safe_dump(study), encoding="utf-8")
    plan = prepare_stage(stage_inputs)[2]
    assert plan["comparison_protocol"]["arm"] == "m3"
    assert plan["comparison_protocol"]["matched_fields"]["expected_pairs"] == 1
    assert plan["comparison_protocol"]["config_sha256"]["m2"] == sha256_file(sft_path)

    sft_cfg["logging_steps"] = 11
    sft_path.write_text(yaml.safe_dump(sft_cfg), encoding="utf-8")
    changed = prepare_stage(stage_inputs)[2]
    assert changed["sha256"] != plan["sha256"]
    assert changed["config"] == plan["config"]
    assert changed["comparison_protocol"]["config_sha256"]["m2"] != (
        plan["comparison_protocol"]["config_sha256"]["m2"]
    )


@pytest.mark.parametrize("fault", ["missing", "candidate_hash", "model_id", "raw_model_hash"])
def test_stage_rejects_wrong_candidate_lineage_even_with_valid_bundle_hashes(stage_inputs, fault):
    root = stage_inputs.preferences.parent
    pair = json.loads((stage_inputs.preferences / "accepted.jsonl").read_text("utf-8"))
    origin = pair["provenance"]["original_candidate_origin"]
    if fault == "missing":
        del pair["provenance"]["original_candidate_origin"]
    elif fault == "candidate_hash":
        origin["checkpoint_sha256"] = "b" * 64
    elif fault == "model_id":
        origin["model_id"] = "other-backbone"
    else:
        pair["provenance"]["rejected"] = {
            "kind": "model", "source": "old-checkpoint", "model_id": "model",
            "checkpoint_sha256": "b" * 64, "recipe_sha256": "c" * 64, "seed": 42,
        }
    reviewed = root / "wrong_lineage.jsonl"
    write_jsonl(reviewed, [pair])
    stage_inputs.preferences = root / "wrong_lineage_bundle"
    freeze_pairs(reviewed, root / "all.jsonl", root, stage_inputs.preferences)
    with pytest.raises(ValueError, match="differs from selected M1"):
        prepare_stage(stage_inputs)


def test_selection_does_not_require_diagnostic_review(stage_inputs):
    path = stage_inputs.m1_selection
    selection = json.loads(path.read_text())
    del selection["evidence"]["human_review"]
    write_json(path, selection)
    _, _, plan, _, _ = prepare_stage(stage_inputs)
    assert plan["pairs"] == 1


def update_review(stage_inputs, rows):
    selection = json.loads(stage_inputs.m1_selection.read_text())
    path = stage_inputs.m1_selection.parent / "human.jsonl"
    write_jsonl(path, rows)
    selection["evidence"]["human_review"]["sha256"] = sha256_file(path)
    write_json(stage_inputs.m1_selection, selection)


def test_thirty_reviewed_images_can_be_kept_as_optional_diagnostics(stage_inputs):
    path = stage_inputs.m1_selection.parent / "human.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()][:30]
    update_review(stage_inputs, rows)
    assert prepare_stage(stage_inputs)[2]["pairs"] == 1


@pytest.mark.parametrize("fault", ["pending", "empty", "duplicate", "train_id", "score"])
def test_optional_diagnostics_cannot_claim_incomplete_or_invalid_labels(stage_inputs, fault):
    path = stage_inputs.m1_selection.parent / "human.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()][:30]
    if fault == "empty":
        rows = []
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "pending":
        rows[0]["review_status"] = "pending"
    elif fault == "train_id":
        rows[0]["question_id"] = "train1"
    else:
        rows[0]["explanation_support"] = True
    update_review(stage_inputs, rows)
    with pytest.raises(ValueError, match="invalid or pending"):
        prepare_stage(stage_inputs)


def test_full_validation_still_required_without_diagnostics(stage_inputs):
    selection = json.loads(stage_inputs.m1_selection.read_text())
    del selection["evidence"]["human_review"]
    path = stage_inputs.m1_selection.parent / "predictions.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()[:-1]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    selection["evidence"]["validation_predictions"]["sha256"] = sha256_file(path)
    write_json(stage_inputs.m1_selection, selection)
    with pytest.raises(ValueError, match="all 1407"):
        prepare_stage(stage_inputs)


def freeze_args(stage_inputs):
    root = stage_inputs.m1_selection.parent
    return SimpleNamespace(
        adapter=stage_inputs.m1_adapter, model_config=stage_inputs.model_config,
        annotations=root / "all.jsonl", expected_sha256=sha256_file(root / "all.jsonl"),
        epoch_result=root / "epoch_result.json", trainer_state=root / "trainer_state.json",
        loss_selection=root / "loss_selection.json",
        validation_predictions=root / "predictions.jsonl",
        validation_manifest=root / "predictions.manifest.json",
        validation_annotations=root / "validation.jsonl", human_review=None,
        output=root / "frozen_selection/selection.json",
    )


def test_freeze_selection_is_valid_portable_and_idempotent(stage_inputs):
    args = freeze_args(stage_inputs)
    selection = freeze_selection(args)
    before = args.output.read_bytes()
    assert selection["diagnostic_review_required"] is False
    assert "human_review" not in selection["evidence"]
    assert selection["evidence"]["epoch_result"]["path"] == "../epoch_result.json"
    assert freeze_selection(args) == selection
    assert args.output.read_bytes() == before
    stage_inputs.m1_selection = args.output
    assert prepare_stage(stage_inputs)[2]["pairs"] == 1


@pytest.mark.parametrize("fault", ["annotations", "loss", "epoch"])
def test_freeze_does_not_publish_invalid_selection(stage_inputs, fault):
    args = freeze_args(stage_inputs)
    if fault == "annotations":
        args.expected_sha256 = "b" * 64
    elif fault == "loss":
        write_json(args.loss_selection, {"status": "passed", "selected_adapter_sha256": "b" * 64})
    else:
        write_json(args.epoch_result, {"status": "running", "global_step": 3683})
    with pytest.raises(ValueError):
        freeze_selection(args)
    assert not args.output.exists()


def test_freeze_refuses_to_replace_an_existing_decision(stage_inputs):
    args = freeze_args(stage_inputs)
    freeze_selection(args)
    args.human_review = args.output.parent.parent / "human.jsonl"
    before = args.output.read_bytes()
    with pytest.raises(ValueError, match="Existing M1 selection differs"):
        freeze_selection(args)
    assert args.output.read_bytes() == before


def test_chosen_file_tampering_is_rejected(stage_inputs):
    path = stage_inputs.preferences / "chosen_only.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="Frozen preference file changed"):
        prepare_stage(stage_inputs)


def test_silent_subset_is_rejected(stage_inputs):
    cfg = yaml.safe_load(stage_inputs.config.read_text())
    cfg["expected_pairs"] = 1000
    stage_inputs.config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Pair count"):
        prepare_stage(stage_inputs)


@pytest.mark.parametrize("name", ["epoch_result", "validation_manifest", "human_review"])
def test_selection_evidence_is_immutable(stage_inputs, name):
    selection = json.loads(stage_inputs.m1_selection.read_text())
    path = stage_inputs.m1_selection.parent / selection["evidence"][name]["path"]
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="evidence changed"):
        prepare_stage(stage_inputs)


def test_unaccepted_data_cannot_pass_by_updating_its_hash(stage_inputs):
    path = stage_inputs.preferences / "accepted.jsonl"
    pair = json.loads(path.read_text(encoding="utf-8"))
    pair["review_status"] = "pending"
    path.write_text(json.dumps(pair) + "\n")
    manifest_path = stage_inputs.preferences / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["accepted.jsonl"] = sha256_file(path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="No accepted"):
        prepare_stage(stage_inputs)


def test_m2_and_m3_share_dataset_and_parent_but_have_distinct_stage_hashes(stage_inputs):
    _, _, dpo, _, _ = prepare_stage(stage_inputs)
    cfg = yaml.safe_load(open("configs/training/sft_chosen.yaml", encoding="utf-8"))
    cfg["expected_pairs"] = 1
    other = copy.copy(stage_inputs)
    other.config = stage_inputs.config.parent / "m2.yaml"
    other.config.write_text(yaml.safe_dump(cfg))
    _, _, sft, _, _ = prepare_stage(other)
    assert dpo["preferences_files"] == sft["preferences_files"]
    assert dpo["m1_adapter"] == sft["m1_adapter"]
    assert dpo["sha256"] != sft["sha256"]


def test_cpu_preflight_does_not_claim_training_or_write_a_run(stage_inputs, capsys):
    from vqa_explanation.training.preference import run_stage

    stage_inputs.preflight_only = True
    run_stage(stage_inputs)
    report = json.loads(capsys.readouterr().out)
    assert report["gpu_training_verified"] is False
    assert report["status"] == "cpu_metadata_passed"
    assert not (stage_inputs.environment.parent / "runs").exists()


def test_actual_train_on_cpu_fails_before_output_or_model_load(stage_inputs):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU guard only")
    from vqa_explanation.training.preference import run_stage

    stage_inputs.preflight_only = False
    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        run_stage(stage_inputs)
    assert not (stage_inputs.environment.parent / "runs").exists()
