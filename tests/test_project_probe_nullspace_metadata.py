from __future__ import annotations

import argparse
import sys
import types

phase2_probe_utils = types.ModuleType("phase2.probe_utils")
phase2_probe_utils.apply_checkpoint = lambda *args, **kwargs: None
phase2_probe_utils.load_probe = lambda *args, **kwargs: {"path": "probe.npz"}
phase2_probe_utils.load_target_specs = lambda *args, **kwargs: []
phase2_probe_utils.normalized_raw_probe_direction = lambda *args, **kwargs: None
phase2_probe_utils.normalized_standard_probe_direction = lambda *args, **kwargs: None
phase2_probe_utils.orthonormal_basis = lambda *args, **kwargs: None
phase2_probe_utils.parse_layers = lambda spec: [int(spec)] if spec else []
phase2_probe_utils.projection_matrix = lambda *args, **kwargs: None
sys.modules.setdefault("phase2.probe_utils", phase2_probe_utils)

from phase2.project_probe_nullspace import build_probe_nullspace_metadata


def test_build_probe_nullspace_metadata_merges_run_metadata(monkeypatch) -> None:
    captured = {}

    def fake_build_run_metadata(**kwargs):
        captured.update(kwargs)
        return {"ok": True, **kwargs["extra"]}

    monkeypatch.setattr("phase2.project_probe_nullspace.build_run_metadata", fake_build_run_metadata)
    args = argparse.Namespace(
        model_dir="./evo-1-8k-base",
        internal_target_config="targets.json",
        forget_csv="forget.csv",
        retain_csv="retain.csv",
        basis_dir="basis",
        config_path="configs/evo-1-8k-base_inference.yml",
        layers="5-9",
        target_layers="host_tropism=5-9",
        projection_strength=1.0,
        module_scope="all",
    )

    payload = build_probe_nullspace_metadata(
        args=args,
        layers=[5, 9],
        target_names=["host_tropism"],
        target_strengths={"host_tropism": 1.0},
        suffixes=("mlp.l3",),
        projection_ranks={"5": 2},
        projection_modules={"5": ["blocks.5.mlp.l3"]},
        layer_basis_meta={"5": [{"target": "host_tropism", "strength": 1.0}]},
        layer_probe_paths={"5": {"host_tropism": ["probe.npz"]}},
        elapsed_sec=3.2,
    )

    assert payload["method"] == "probe_nullspace"
    assert payload["phase"] == "project_probe_nullspace"
    assert payload["elapsed_sec"] == 3.2
    assert payload["module_suffixes"] == ["mlp.l3"]
    assert captured["loss_layers"] == [5, 9]
    assert captured["data_paths"] == [
        "targets.json",
        "forget.csv",
        "retain.csv",
        "basis",
        "configs/evo-1-8k-base_inference.yml",
    ]
