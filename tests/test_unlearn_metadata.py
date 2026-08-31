from __future__ import annotations

import argparse
import pathlib
import sys
import types

from tests._stub_support import register_stub

phase1_module = types.ModuleType("phase1")
phase1_utils_module = types.ModuleType("phase1.utils")
phase1_utils_module.load_local_checkpoint = lambda *args, **kwargs: None
phase1_utils_module.read_manifest = lambda *args, **kwargs: []
phase1_module.utils = phase1_utils_module
# If this stub is actually installed (bare environment, no torch/stripedhyena),
# keep the real package search path on it so sibling modules such as
# phase1.build_refseq_family_target_dataset still import from disk instead of
# failing with "'phase1' is not a package".
phase1_module.__path__ = [str(pathlib.Path(__file__).resolve().parents[1] / "phase1")]
register_stub("phase1", phase1_module)
register_stub("phase1.utils", phase1_utils_module)

evo_module = types.ModuleType("evo")
evo_tokenizer_module = types.ModuleType("evo.tokenizer")
evo_tokenizer_module.CharLevelTokenizer = object
evo_module.tokenizer = evo_tokenizer_module
register_stub("evo", evo_module)
register_stub("evo.tokenizer", evo_tokenizer_module)

phase2_probe_utils = types.ModuleType("phase2.probe_utils")
phase2_probe_utils.apply_checkpoint = lambda *args, **kwargs: None
phase2_probe_utils.load_probe = lambda *args, **kwargs: None
phase2_probe_utils.load_target_specs = lambda *args, **kwargs: []
phase2_probe_utils.normalized_standard_probe_direction = lambda *args, **kwargs: None
phase2_probe_utils.normalized_raw_probe_direction = lambda *args, **kwargs: None
phase2_probe_utils.orthonormal_basis = lambda *args, **kwargs: None
register_stub("phase2.probe_utils", phase2_probe_utils)

phase2_utils = types.ModuleType("phase2.utils")
phase2_utils.PROBE_LAYERS = list(range(11))
phase2_utils.count_trainable = lambda *args, **kwargs: 0
phase2_utils.freeze_all = lambda *args, **kwargs: None
phase2_utils.get_localized_layers = lambda *args, **kwargs: [5, 6]
phase2_utils.iterate_batches = lambda *args, **kwargs: []
phase2_utils.set_block_grad = lambda *args, **kwargs: None
phase2_utils.tokenize_batch = lambda *args, **kwargs: None
phase2_utils.language_model_loss = lambda *args, **kwargs: None
phase2_utils.get_primary_target_layer = lambda *args, **kwargs: 8
phase2_utils.select_random_layers = lambda *args, **kwargs: [8, 9]
register_stub("phase2.utils", phase2_utils)

import torch

from phase2.unlearn_gd import build_gd_metadata, gd_loss_terms
from phase2.unlearn_probe_repr import build_probe_repr_metadata
from phase2.unlearn_rmu import build_rmu_metadata


def test_build_gd_metadata_records_cross_entropy_objective(monkeypatch) -> None:
    captured = {}

    def fake_build_run_metadata(**kwargs):
        captured.update(kwargs)
        return {"ok": True, **kwargs["extra"]}

    monkeypatch.setattr("phase2.unlearn_gd.build_run_metadata", fake_build_run_metadata)
    args = argparse.Namespace(
        condition="localized",
        steps=100,
        lr=1e-4,
        alpha_forget=1.0,
        alpha_retain=2.0,
        forget_loss_cap=0.0,
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
        trainable_param_count=1234,
        init_source="base_model",
        init_ckpt="",
        save_steps=[25, 50],
        elapsed_sec=12.5,
    )

    assert payload["method"] == "gradient_difference"
    # The label must distinguish the restored CE objective from the probe-guided
    # representation objective that briefly shipped under this filename.
    assert payload["loss_type"] == "cross_entropy_gradient_difference"
    assert payload["objective"].startswith("-alpha_forget * CE(forget)")
    assert payload["elapsed_sec"] == 12.5
    assert captured["trainable_param_count"] == 1234
    assert captured["data_paths"] == ["forget.csv", "retain.csv", "localized_layers.json"]
    # No probe target config is involved in classic gradient difference.
    assert "internal_target_config" not in payload
    assert "target_names" not in payload


def test_build_probe_repr_metadata_merges_run_metadata(monkeypatch) -> None:
    captured = {}

    def fake_build_run_metadata(**kwargs):
        captured.update(kwargs)
        return {"ok": True, **kwargs["extra"]}

    monkeypatch.setattr("phase2.unlearn_probe_repr.build_run_metadata", fake_build_run_metadata)
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

    payload = build_probe_repr_metadata(
        args=args,
        layers=[5, 6],
        loss_layers=[5, 6],
        target_specs=[{"name": "host_tropism"}],
        init_source="base_model",
        init_ckpt="",
        save_steps=[25, 50],
        elapsed_sec=12.5,
    )

    # This method must NOT claim to be gradient difference.
    assert payload["method"] == "probe_guided_representation"
    assert payload["loss_type"] == "probe_guided_representation"
    assert payload["elapsed_sec"] == 12.5
    assert payload["target_names"] == ["host_tropism"]
    assert captured["loss_layers"] == [5, 6]
    assert captured["data_paths"] == ["targets.json", "forget.csv", "retain.csv", "localized_layers.json"]


def test_gd_loss_terms_ascends_forget_and_descends_retain() -> None:
    l_forget = torch.tensor(2.0)
    l_retain = torch.tensor(3.0)

    effective, weighted_forget, weighted_retain = gd_loss_terms(
        l_forget, l_retain, alpha_forget=1.0, alpha_retain=5.0
    )

    # Gradient difference maximizes the forget loss, so its term is negated.
    assert effective.item() == 2.0
    assert weighted_forget.item() == -2.0
    assert weighted_retain.item() == 15.0
    assert (weighted_forget + weighted_retain).item() == 13.0


def test_gd_loss_terms_cap_bounds_the_ascent_term() -> None:
    l_forget = torch.tensor(9.0)
    l_retain = torch.tensor(1.0)

    uncapped = gd_loss_terms(l_forget, l_retain, alpha_forget=2.0, alpha_retain=1.0)
    capped = gd_loss_terms(
        l_forget, l_retain, alpha_forget=2.0, alpha_retain=1.0, forget_loss_cap=4.0
    )

    assert uncapped[0].item() == 9.0
    assert uncapped[1].item() == -18.0
    # The cap clamps the cross-entropy before weighting, bounding the objective.
    assert capped[0].item() == 4.0
    assert capped[1].item() == -8.0
    # A cap above the observed loss must be a no-op.
    assert gd_loss_terms(
        l_forget, l_retain, alpha_forget=2.0, alpha_retain=1.0, forget_loss_cap=100.0
    )[1].item() == -18.0


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
