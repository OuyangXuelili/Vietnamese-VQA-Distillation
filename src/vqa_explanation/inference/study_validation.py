"""Export all four study arms in one inference environment."""

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from vqa_explanation.inference.export import read_jsonl, validate_selected_records
from vqa_explanation.inference.export_relocation import (
    prepare_relocation,
    snapshot_export,
    verify_relocation_group,
    write_relocation,
)
from vqa_explanation.utils.environment import load_environment, project_root
from vqa_explanation.utils.process import run_command
from vqa_explanation.utils.run_provenance import sha256_file

VALIDATION_SHA256 = "7f0b2b7d374d386030ddea50bc57d3e8e6c76e257e9fc4d1203bfb331b5f5b6e"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
ARMS = {"m0": "M0_PRETRAINED", "m1": "M1_SFT",
        "m2": "M2_SFT_CHOSEN", "m3": "M3_DPO"}


def verify_adapter(path, expected):
    for name, digest in expected.items():
        if name not in {"adapter_config.json", "adapter_model.safetensors"}:
            raise ValueError("Unsupported adapter evidence file")
        if sha256_file(path / name) != digest:
            raise ValueError(f"Adapter changed: {path / name}")
    if set(expected) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("Incomplete adapter evidence")


def prepare_validation(environment_path, run_group, m1_adapter, output_dir):
    environment = load_environment(Path(environment_path))
    annotations = environment["annotations"].parent / "validation.jsonl"
    if sha256_file(annotations) != VALIDATION_SHA256:
        raise ValueError("Frozen validation annotations changed")
    records = read_jsonl(annotations)
    ids = {row["question_id"] for row in records}
    if len(records) != 1407 or len(ids) != 1407 or any(
        row.get("split") != "validation" for row in records
    ):
        raise ValueError("Require all 1407 unique validation questions")
    validate_selected_records(records, environment["image_root"])
    run_group, output_dir = Path(run_group).resolve(), Path(output_dir).resolve()
    protected = (run_group, environment["image_root"], project_root(), Path(m1_adapter).resolve())
    if any(path.is_relative_to(output_dir) or output_dir.is_relative_to(path)
           for path in protected):
        raise ValueError("Evaluation output cannot contain source, data or training runs")
    adapters, evidence, plans = {"m0": None}, {}, {}
    for arm in ("m2", "m3"):
        stage = run_group / f"{arm}_train"
        report = json.loads((stage / "training_result.json").read_text("utf-8"))
        plan = json.loads((stage / "data_plan.json").read_text("utf-8"))
        content = {key: value for key, value in plan.items() if key != "sha256"}
        plan_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        if plan_hash != plan["sha256"]:
            raise ValueError(f"{arm}: training plan changed")
        if (report.get("status") != "passed" or report.get("global_step") != 24
                or report.get("epoch") != 1 or report.get("policy_updated") is not True
                or report.get("saved_policy_verified") is not True
                or (arm == "m3" and report.get("reference_unchanged") is not True)):
            raise ValueError(f"{arm}: training is not complete")
        if report.get("plan_sha256") != plan.get("sha256") or plan.get("pairs") != 186:
            raise ValueError(f"{arm}: training plan does not match report")
        if (plan["model"]["id"] != MODEL_ID
                or plan["model"]["revision"] != MODEL_REVISION):
            raise ValueError(f"{arm}: unexpected backbone")
        adapter = stage / "training/checkpoint-24"
        verify_adapter(adapter, report["output_adapter"]["files"])
        adapters[arm], plans[arm] = adapter, plan
        evidence[arm] = {"report_sha256": sha256_file(stage / "training_result.json"),
                         "adapter_files": report["output_adapter"]["files"]}
    if (plans["m2"]["m1_adapter"] != plans["m3"]["m1_adapter"]
            or plans["m2"]["preferences_manifest_sha256"]
            != plans["m3"]["preferences_manifest_sha256"]):
        raise ValueError("The study arms have different M1 or preference inputs")
    adapters["m1"] = Path(m1_adapter).resolve()
    verify_adapter(adapters["m1"], plans["m2"]["m1_adapter"]["files"])
    return environment, annotations, adapters, evidence


def export_command(root, environment, annotations, adapter, arm, output):
    command = [sys.executable, "-u", str(root / "scripts/run_m0.py"),
               "--annotations", str(annotations), "--image-root", str(environment["image_root"]),
               "--output", str(output), "--split", "validation", "--limit", "0", "--seed", "42",
               "--model-id", MODEL_ID, "--revision", MODEL_REVISION,
               "--min-pixels", "200704", "--max-pixels", "200704",
               "--max-new-tokens", "128", "--load-in-4bit", "--experiment-id", ARMS[arm]]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    if output.exists() or output.with_suffix(".jsonl.manifest.json").exists():
        command.append("--resume")
    return command


def run_validation(environment_path, run_group, m1_adapter, output_dir, *, inspect_only=False,
                   runner=run_command, relocate_from=None):
    environment, annotations, adapters, evidence = prepare_validation(
        environment_path, run_group, m1_adapter, output_dir
    )
    root, output_dir = project_root(), Path(output_dir).resolve()
    relocations = []
    if relocate_from is not None and not Path(relocate_from).is_dir():
        raise ValueError("Original export directory is missing")
    if relocate_from is not None and output_dir.exists():
        verify_relocation_group(relocate_from, output_dir)
    elif relocate_from is not None:
        for arm in ("m2", "m3", "m0", "m1"):
            args = SimpleNamespace(
                annotations=annotations, image_root=environment["image_root"],
                output=output_dir / f"{arm}.jsonl", adapter=adapters[arm],
                split="validation", limit=0, seed=42, model_id=MODEL_ID,
                revision=MODEL_REVISION, experiment_id=ARMS[arm],
                min_pixels=200704, max_pixels=200704, max_new_tokens=128, load_in_4bit=True,
            )
            plan = prepare_relocation(Path(relocate_from) / f"{arm}.jsonl", args)
            if plan is not None:
                relocations.append(plan)
        if not relocations:
            raise ValueError("Original directory has no verifiable exports")
        if not inspect_only:
            receipts = []
            for plan in relocations:
                receipt = write_relocation(plan)
                receipts.append(receipt)
                print(json.dumps(receipt, ensure_ascii=False), flush=True)
            with (output_dir / "relocation_group.json").open("x", encoding="utf-8") as handle:
                json.dump({"schema": "validation_export_relocation_v1",
                           "source_dir": str(Path(relocate_from).resolve()),
                           "receipts": receipts}, handle, ensure_ascii=False, indent=2)
    commands = [export_command(root, environment, annotations, adapters[arm], arm,
                               output_dir / f"{arm}.jsonl") for arm in ("m2", "m3", "m0", "m1")]
    if inspect_only:
        return {"status": "inputs_verified", "questions": 1407, "commands": commands,
                "relocations": [plan["receipt"] for plan in relocations]}
    records = read_jsonl(annotations)
    for arm, command in zip(("m2", "m3", "m0", "m1"), commands, strict=True):
        print(f"Export {arm}: all 1407 validation questions", flush=True)

        def save_progress(line, output=output_dir / f"{arm}.jsonl"):
            match = re.fullmatch(r"\[(\d+)/1407\] \S+ format=(True|False)\n", line)
            if match and (int(match[1]) == 25 or int(match[1]) % 100 == 0):
                snapshot = snapshot_export(output, records)
                if snapshot:
                    print(f"Recovery snapshot: {snapshot['predictions']} predictions", flush=True)

        runner(command, cwd=root, on_output=save_progress)
    for left, right in (("m2", "m3"), ("m1", "m2"), ("m1", "m3"), ("m0", "m1")):
        destination = output_dir / f"{left}_{right}_comparison.json"
        if destination.exists():
            raise ValueError(f"Comparison already exists; preserve it: {destination}")
        runner([sys.executable, "-u", str(root / "scripts/compare_runs.py"),
                str(output_dir / f"{left}.jsonl"), str(output_dir / f"{right}.jsonl"),
                "--output", str(destination), "--left-experiment", ARMS[left],
                "--right-experiment", ARMS[right], "--expected-validation-annotations",
                str(annotations), "--expected-validation-sha256", VALIDATION_SHA256,
                "--bootstrap-resamples", "2000"], cwd=root)
    return {"status": "completed", "questions": 1407, "training_evidence": evidence,
            "output_dir": str(output_dir)}
