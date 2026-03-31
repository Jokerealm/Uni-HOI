from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
CONFIG_SECTION = "dual_branch_fm"
DEFAULT_PYTHON_BIN = "/data3/guanz/miniforge3/envs/cari4d/bin/python"


def _as_config(value: Any) -> DictConfig:
    if value is None:
        return OmegaConf.create({})
    if isinstance(value, DictConfig):
        return value
    if isinstance(value, dict):
        return OmegaConf.create(value)
    raise TypeError(f"Expected mapping-like config, got {type(value).__name__}.")


def _merge_configs(*values: Any) -> DictConfig:
    merged = OmegaConf.create({})
    for value in values:
        merged = OmegaConf.merge(merged, _as_config(value))
    return merged


def _resolve_path(value: Any, *, allow_empty: bool = True) -> str:
    if value is None:
        if allow_empty:
            return ""
        raise ValueError("Expected a non-empty path value, but got None.")
    text = str(value).strip()
    if not text:
        if allow_empty:
            return ""
        raise ValueError("Expected a non-empty path value, but got an empty string.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def _auto_prepare_workers(cpu_count: int | None = None) -> int:
    cpu_total = int(cpu_count or os.cpu_count() or 8)
    if cpu_total <= 2:
        return 1
    if cpu_total <= 8:
        return max(cpu_total // 2, 1)
    return 8


def _normalize_dataset_name(name: str) -> str:
    return str(name).strip().replace("-", "_")


def _resolve_num_processes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text == "auto":
            return 0
        return int(text)
    return int(value)


def _resolve_dist_config(dist_cfg: DictConfig) -> Dict[str, Any]:
    result = OmegaConf.to_container(_as_config(dist_cfg), resolve=True)
    assert isinstance(result, dict)
    return {
        "num_processes": _resolve_num_processes(result.get("num_processes", 0)),
        "master_addr": str(result.get("master_addr", "127.0.0.1")),
        "main_process_port": int(result.get("main_process_port", 29500)),
        "nnodes": int(result.get("nnodes", 1)),
        "node_rank": int(result.get("node_rank", 0)),
    }


def load_dual_branch_fm_config(
    config_path: str | Path | None = None,
    dataset: str | None = None,
) -> Dict[str, Any]:
    resolved_config_path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    root_cfg = OmegaConf.load(resolved_config_path)
    root_cfg = _as_config(OmegaConf.select(root_cfg, CONFIG_SECTION))
    if not root_cfg:
        raise KeyError(
            f"Config file `{resolved_config_path}` is missing the `{CONFIG_SECTION}` section."
        )

    default_dataset = OmegaConf.select(root_cfg, "default_dataset")
    dataset_name = _normalize_dataset_name(dataset or default_dataset or "")
    if not dataset_name:
        raise KeyError("Dual-branch FM config is missing `default_dataset`, and no `dataset` was provided.")

    datasets_cfg = _as_config(OmegaConf.select(root_cfg, "datasets"))
    if dataset_name not in datasets_cfg:
        raise KeyError(f"Unknown dataset preset `{dataset_name}`. Available presets: {sorted(datasets_cfg.keys())}")
    dataset_cfg = _as_config(datasets_cfg[dataset_name])

    runtime_cfg = _merge_configs(
        {"python_bin": OmegaConf.select(root_cfg, "python_bin", default=DEFAULT_PYTHON_BIN)},
        OmegaConf.select(root_cfg, "runtime"),
        OmegaConf.select(dataset_cfg, "runtime"),
    )
    prepare_cfg = _merge_configs(
        OmegaConf.select(root_cfg, "prepare"),
        OmegaConf.select(dataset_cfg, "prepare"),
    )
    train_cfg = _merge_configs(
        OmegaConf.select(root_cfg, "train"),
        OmegaConf.select(dataset_cfg, "train"),
    )
    dist_cfg = _merge_configs(
        OmegaConf.select(root_cfg, "dist"),
        OmegaConf.select(dataset_cfg, "dist"),
    )

    runtime = OmegaConf.to_container(runtime_cfg, resolve=True)
    assert isinstance(runtime, dict)
    dataset_raw = OmegaConf.to_container(dataset_cfg, resolve=True)
    assert isinstance(dataset_raw, dict)

    python_bin = _resolve_path(runtime.get("python_bin"), allow_empty=False)
    output_root = _resolve_path(runtime.get("output_root", "outputs"), allow_empty=False)
    run_name = str(runtime.get("run_name", "procigen_dual_branch_fm"))
    project_name = str(runtime.get("project_name", "dual-branch-fm"))
    log_with = str(runtime.get("log_with", "none"))
    mixed_precision = str(runtime.get("mixed_precision", "bf16"))
    seed = int(runtime.get("seed", 42))

    dataset_section = {
        "name": dataset_name,
        "raw_root": _resolve_path(dataset_raw.get("raw_root", "")),
        "prepared_root": _resolve_path(dataset_raw.get("prepared_root", ""), allow_empty=False),
        "split_file": _resolve_path(dataset_raw.get("split_file", "")),
        "split_key": str(dataset_raw.get("split_key", "train")),
        "camera_id": str(dataset_raw.get("camera_id", "k1")),
        "processed_subdir": str(dataset_raw.get("processed_subdir", "processed")),
        "gs_subdir": str(dataset_raw.get("gs_subdir", "gs_init")),
    }

    train = OmegaConf.to_container(train_cfg, resolve=True)
    assert isinstance(train, dict)
    train_output_dir = str(train.get("output_dir", "")).strip()
    if train_output_dir:
        output_dir = _resolve_path(train_output_dir, allow_empty=False)
    else:
        output_dir = str(Path(output_root) / run_name)

    train_section = dict(train)
    train_section["output_dir"] = output_dir
    train_section["project_name"] = str(train.get("project_name") or project_name)
    train_section["seed"] = int(train.get("seed", seed))
    train_section["log_with"] = str(train.get("log_with", log_with))
    train_section["mixed_precision"] = str(train.get("mixed_precision", mixed_precision))
    train_section["resume_checkpoint"] = _resolve_path(train.get("resume_checkpoint", ""))
    train_section["data_root"] = _resolve_path(train.get("data_root") or dataset_section["prepared_root"], allow_empty=False)
    train_section["processed_subdir"] = str(train.get("processed_subdir") or dataset_section["processed_subdir"])
    train_section["gs_subdir"] = str(train.get("gs_subdir") or dataset_section["gs_subdir"])
    train_section["split_file"] = _resolve_path(train.get("split_file") or dataset_section["split_file"])
    train_section["split_key"] = str(train.get("split_key") or dataset_section["split_key"])

    prepare = OmegaConf.to_container(prepare_cfg, resolve=True)
    assert isinstance(prepare, dict)
    prepare_num_workers = prepare.get("num_workers", "auto")
    if isinstance(prepare_num_workers, str) and prepare_num_workers.strip().lower() == "auto":
        prepare_num_workers = _auto_prepare_workers()
    prepare_section = {
        "skip": bool(prepare.get("skip", False)) or not bool(dataset_section["raw_root"]),
        "only": bool(prepare.get("only", False)),
        "overwrite": bool(prepare.get("overwrite", False)),
        "allow_in_place": bool(prepare.get("allow_in_place", False)),
        "max_sequences": int(prepare.get("max_sequences", 0)),
        "max_frames": int(prepare.get("max_frames", 0)),
        "num_workers": int(prepare_num_workers),
        "heartbeat_interval": int(prepare.get("heartbeat_interval", 30)),
        "detach": bool(prepare.get("detach", False)),
        "status_dir": _resolve_path(prepare.get("status_dir") or (Path(dataset_section["prepared_root"]) / "_preprocess_logs")),
    }

    dist_section = _resolve_dist_config(dist_cfg)

    return {
        "config_path": str(resolved_config_path),
        "repo_root": str(REPO_ROOT),
        "python_bin": python_bin,
        "runtime": {
            "run_name": run_name,
            "output_root": output_root,
            "project_name": project_name,
            "log_with": log_with,
            "mixed_precision": mixed_precision,
            "seed": seed,
        },
        "dataset": dataset_section,
        "prepare": prepare_section,
        "dist": dist_section,
        "train": train_section,
    }
