from __future__ import annotations

import argparse
import sys
import types

phase1_module = types.ModuleType("phase1")
phase1_utils_module = types.ModuleType("phase1.utils")
phase1_utils_module.load_local_checkpoint = lambda *args, **kwargs: None
phase1_utils_module.read_manifest = lambda *args, **kwargs: []
phase1_module.utils = phase1_utils_module
sys.modules.setdefault("phase1", phase1_module)
sys.modules.setdefault("phase1.utils", phase1_utils_module)

evo_module = types.ModuleType("evo")
evo_tokenizer_module = types.ModuleType("evo.tokenizer")
evo_tokenizer_module.CharLevelTokenizer = object
evo_module.tokenizer = evo_tokenizer_module
sys.modules.setdefault("evo", evo_module)
sys.modules.setdefault("evo.tokenizer", evo_tokenizer_module)

phase2_probe_utils = types.ModuleType("phase2.probe_utils")
phase2_probe_utils.apply_checkpoint = lambda *args, **kwargs: None
phase2_probe_utils.load_probe = lambda *args, **kwargs: None
phase2_probe_utils.load_target_specs = lambda *args, **kwargs: []
phase2_probe_utils.normalized_standard_probe_direction = lambda *args, **kwargs: None
phase2_probe_utils.normalized_raw_probe_direction = lambda *args, **kwargs: None
phase2_probe_utils.orthonormal_basis = lambda *args, **kwargs: None
sys.modules.setdefault("phase2.probe_utils", phase2_probe_utils)

phase2_utils = types.ModuleType("phase2.utils")
phase2_utils.PROBE_LAYERS = list(range(11))
phase2_utils.count_trainable = lambda *args, **kwargs: 0
phase2_utils.freeze_all = lambda *args, **kwargs: None
phase2_utils.get_localized_layers = lambda *args, **kwargs: [5, 6]
phase2_utils.iterate_batches = lambda *args, **kwargs: []
phase2_utils.set_block_grad = lambda *args, **kwargs: None
phase2_utils.tokenize_batch = lambda *args, **kwargs: None
phase2_utils.get_primary_target_layer = lambda *args, **kwargs: 8
phase2_utils.select_random_layers = lambda *args, **kwargs: [8, 9]
sys.modules.setdefault("phase2.utils", phase2_utils)

from phase2.unlearn_gd import build_gd_metadata
from phase2.unlearn_rmu import build_rmu_metadata


def test_build_gd_metadata_merges_run_metadata(monkeypatch) -> None:
    captured = {}

    def fake_build_run_metadata(**kwargs):
        captured.update(kwargs)
        return {"ok": True, **kwargs["extra"]}

    monkeypatch.setattr("phase2.unlearn_gd.build_run_metadata", fake_build_run_metadata)
    args = argparse.Namespace(
        condition="localized",
        internal_target_config="targets.json",
        steps=100,
        lr=1e-4,
        alpha_forget=1.0,
        alpha_retain=2.0,
        retain_cosine_weight=0.5,
        batch_size=4,
        max_length=512,
        forget_csv="forget.csv",
        retain_csv="retain.csv",
        seed=42,
        localized_layers_path="localized_layers.json",
        model_dir="./evo-1-8k-base",
    )

    payload = build_gd_metadata(
        args=args,
        layers=[5, 6],
        loss_layers=[5, 6],
        target_specs=[{"name": "host_tropism"}],
        init_source="base_model",
        init_ckpt="",
        save_steps=[25, 50],
        elapsed_sec=12.5,
    )

    assert payload["method"] == "gradient_difference"
    assert payload["elapsed_sec"] == 12.5
    assert payload["target_names"] == ["host_tropism"]
    assert captured["loss_layers"] == [5, 6]
    assert captured["data_paths"] == ["targets.json", "forget.csv", "retain.csv", "localized_layers.json"]


def test_build_rmu_metadata_merges_run_metadata(monkeypatch) -> None:
    captured = {}

    def fake_build_run_metadata(**kwargs):
        captured.update(kwargs)
        return {"ok": True, **kwargs["extra"]}

    monkeypatch.setattr("phase2.unlearn_rmu.build_run_metadata", fake_build_run_metadata)
    args = argparse.Namespace(
        condition="localized",
        scale_calibrated=True,
        steer_coef=150.0,
        target_direction="joint_probe",
        direction_seqs="",
        internal_target_config="targets.json",
        probe_direction_sign=-1.0,
        retain_cosine_weight=0.25,
        steps=200,
        lr=5e-5,
        alpha_forget=3.0,
        batch_size=2,
        max_length=256,
        forget_csv="forget.csv",
        retain_csv="retain.csv",
        seed=7,
        localized_layers_path="localized_layers.json",
        model_dir="./evo-1-8k-base",
    )

    payload = build_rmu_metadata(
        args=args,
        layers=[8, 9],
        loss_layers=[8, 9],
        requested_target_layer=8,
        normalize_hidden=True,
        direction_metadata={"probe": "joint"},
        effective_alpha_retain=1.5,
        requested_alpha_retain=2.0,
        save_steps=[50, 100],
        checkpoint_step=50,
        parent_run="rmu_run",
    )

    assert payload["method"] == "rmu"
    assert payload["checkpoint_step"] == 50
    assert payload["parent_run"] == "rmu_run"
    assert payload["direction_metadata"] == {"probe": "joint"}
    assert captured["loss_layers"] == [8, 9]
    assert captured["data_paths"] == ["targets.json", "forget.csv", "retain.csv", "localized_layers.json"]
