import json
import subprocess
import sys
from pathlib import Path


def test_build_stage1_smoke_variants_writes_spec_and_report(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    k0 = ckpt_dir / "k0.safetensors"
    lora = ckpt_dir / "lora.safetensors"
    fallback = ckpt_dir / "fallback.safetensors"
    full = ckpt_dir / "full.safetensors"
    for path in (k0, lora, fallback, full):
        path.write_text("stub")

    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "phase2/build_stage1_smoke_variants.py",
            "--out-dir",
            str(out_dir),
            "--projinit-k0-ckpt",
            str(k0),
            "--projinit-lora-ckpt",
            str(lora),
            "--projinit-fallback-ckpt",
            str(fallback),
            "--projinit-full-ckpt",
            str(full),
            "--option-b-best-candidate-json",
            str(tmp_path / "missing_best_candidate.json"),
        ],
        check=True,
    )

    spec_path = out_dir / "stage1_hostonly_smoke_variants.json"
    report_path = out_dir / "stage1_hostonly_smoke_variants_report.json"
    metadata_path = out_dir / "stage1_hostonly_smoke_variants_metadata.json"
    variants = json.loads(spec_path.read_text())
    report = json.loads(report_path.read_text())
    metadata = json.loads(metadata_path.read_text())

    assert [row["variant_id"] for row in variants] == ["option_a_base", "legacy_projinit_control"]
    assert variants[0]["initializer_label"] == "none"
    assert variants[1]["initializer_label"] == "probe_nullspace_projinit_control"
    assert variants[1]["k0_ckpt"] == str(k0)
    assert variants[1]["attacked_ckpt"] == str(fallback)
    assert variants[1]["attacked_ckpt_by_recipe"]["lora_r8_lr1e5_l5l9"] == str(lora)
    assert report["variant_spec_json"] == str(spec_path)
    missing_ids = [row["variant_id"] for row in report["missing_planned_variants"]]
    assert "option_b_classification_ce" in missing_ids
    assert metadata["phase"] == "build_stage1_smoke_variants"
    assert metadata["variant_spec_json"] == str(spec_path)


def test_build_stage1_smoke_variants_requires_existing_checkpoints(tmp_path):
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "phase2/build_stage1_smoke_variants.py",
            "--out-dir",
            str(out_dir),
            "--projinit-k0-ckpt",
            str(tmp_path / "missing.safetensors"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "missing.safetensors" in (proc.stderr or proc.stdout)


def test_build_stage1_smoke_variants_adds_option_b_when_checkpoint_exists(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    k0 = ckpt_dir / "k0.safetensors"
    lora = ckpt_dir / "lora.safetensors"
    fallback = ckpt_dir / "fallback.safetensors"
    full = ckpt_dir / "full.safetensors"
    option_b = ckpt_dir / "option_b.safetensors"
    for path in (k0, lora, fallback, full, option_b):
        path.write_text("stub")

    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "phase2/build_stage1_smoke_variants.py",
            "--out-dir",
            str(out_dir),
            "--projinit-k0-ckpt",
            str(k0),
            "--projinit-lora-ckpt",
            str(lora),
            "--projinit-fallback-ckpt",
            str(fallback),
            "--projinit-full-ckpt",
            str(full),
            "--option-b-k0-ckpt",
            str(option_b),
        ],
        check=True,
    )

    variants = json.loads((out_dir / "stage1_hostonly_smoke_variants.json").read_text())
    option_b_rows = [row for row in variants if row["variant_id"] == "option_b_classification_ce"]
    assert len(option_b_rows) == 1
    assert option_b_rows[0]["recipe_ids"] == ["k0_no_attack"]


def test_build_stage1_smoke_variants_falls_back_to_best_candidate_json(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    k0 = ckpt_dir / "k0.safetensors"
    lora = ckpt_dir / "lora.safetensors"
    fallback = ckpt_dir / "fallback.safetensors"
    full = ckpt_dir / "full.safetensors"
    best_option_b = ckpt_dir / "best_option_b.safetensors"
    for path in (k0, lora, fallback, full, best_option_b):
        path.write_text("stub")

    best_candidate_json = tmp_path / "best_candidate.json"
    best_candidate_json.write_text(
        json.dumps(
            {
                "best_candidate": {
                    "config_id": "retain_heavy_40x40",
                    "weights_path": str(best_option_b),
                }
            }
        )
    )

    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "phase2/build_stage1_smoke_variants.py",
            "--out-dir",
            str(out_dir),
            "--projinit-k0-ckpt",
            str(k0),
            "--projinit-lora-ckpt",
            str(lora),
            "--projinit-fallback-ckpt",
            str(fallback),
            "--projinit-full-ckpt",
            str(full),
            "--option-b-k0-ckpt",
            str(tmp_path / "missing_option_b.safetensors"),
            "--option-b-best-candidate-json",
            str(best_candidate_json),
        ],
        check=True,
    )

    variants = json.loads((out_dir / "stage1_hostonly_smoke_variants.json").read_text())
    report = json.loads((out_dir / "stage1_hostonly_smoke_variants_report.json").read_text())
    option_b_rows = [row for row in variants if row["variant_id"] == "option_b_classification_ce"]
    assert len(option_b_rows) == 1
    assert option_b_rows[0]["k0_ckpt"] == str(best_option_b)
    available_option_b = [row for row in report["available_variants"] if row["variant_id"] == "option_b_classification_ce"]
    assert available_option_b[0]["best_candidate_config_id"] == "retain_heavy_40x40"
