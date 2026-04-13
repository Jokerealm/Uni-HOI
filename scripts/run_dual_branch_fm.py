#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dual_branch_fm_config import DEFAULT_CONFIG_PATH, load_dual_branch_fm_config

TRUTHY = {"1", "true", "yes", "on"}
FALSEY = {"0", "false", "no", "off"}


def _env_text(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _env_bool(name: str) -> bool | None:
    text = _env_text(name)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in TRUTHY:
        return True
    if lowered in FALSEY:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {text!r}")


def _parse_visible_devices(value: str | None) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text == "-1":
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _resolve_repo_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def _resolve_python_bin(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("`python_bin` must be a non-empty string.")
    if "/" not in text and not text.startswith(".") and not text.startswith("~"):
        return text
    return str(Path(text).expanduser().resolve())


def _append_flag(cmd: list[str], flag: str, value: Any) -> None:
    cmd.extend([flag, str(value)])


def _append_bool_flag(cmd: list[str], value: bool, positive_flag: str, negative_flag: str) -> None:
    cmd.append(positive_flag if bool(value) else negative_flag)


def _auto_train_dataloader_workers(num_processes: int, cpu_count: int | None = None) -> int:
    cpu_total = int(cpu_count or os.cpu_count() or 8)
    process_count = max(int(num_processes), 1)
    if cpu_total <= 2:
        return 1
    per_rank_budget = max(cpu_total // max(process_count * 2, 1), 1)
    return max(1, min(4, per_rank_budget))


def _resolve_train_dataloader_num_workers(value: Any, *, num_processes: int) -> int:
    if value is None:
        return _auto_train_dataloader_workers(num_processes)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text == "auto":
            return _auto_train_dataloader_workers(num_processes)
        return int(text)
    return int(value)


def _finalize_resolved_config(resolved_cfg: dict[str, Any]) -> None:
    runtime = resolved_cfg["runtime"]
    dataset = resolved_cfg["dataset"]
    prepare = resolved_cfg["prepare"]
    dist = resolved_cfg["dist"]
    train = resolved_cfg["train"]

    train["project_name"] = train.get("project_name") or runtime["project_name"]
    train["seed"] = int(train.get("seed", runtime["seed"]))
    train["log_with"] = train.get("log_with") or runtime["log_with"]
    train["mixed_precision"] = train.get("mixed_precision") or runtime["mixed_precision"]
    train["data_root"] = dataset["prepared_root"]
    train["processed_subdir"] = dataset["processed_subdir"]
    train["gs_subdir"] = dataset["gs_subdir"]
    train["split_file"] = dataset["split_file"]
    train["split_key"] = dataset["split_key"]
    if not dataset["raw_root"]:
        prepare["skip"] = True

    visible_devices = _parse_visible_devices(os.getenv("CUDA_VISIBLE_DEVICES"))
    if int(dist["num_processes"]) <= 0:
        dist["num_processes"] = len(visible_devices) if len(visible_devices) > 1 else 1
    if visible_devices and int(dist["num_processes"]) > len(visible_devices):
        raise ValueError(
            "num_processes exceeds the number of visible GPUs from "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')!r}: "
            f"{dist['num_processes']} > {len(visible_devices)}."
        )

    train["dataloader_num_workers"] = _resolve_train_dataloader_num_workers(
        train.get("dataloader_num_workers"),
        num_processes=int(dist["num_processes"]),
    )
    train["dataloader_pin_memory"] = bool(train.get("dataloader_pin_memory", True))
    train["dataloader_persistent_workers"] = bool(train.get("dataloader_persistent_workers", True))
    train["dataloader_prefetch_factor"] = int(train.get("dataloader_prefetch_factor", 2))
    train["batch_sampler_mode"] = str(train.get("batch_sampler_mode", "global_clip_shuffle"))
    train["sequence_batch_interleave_window"] = int(train.get("sequence_batch_interleave_window", 0))
    train["sequence_batch_burst"] = int(train.get("sequence_batch_burst", 1))
    train["wan_pad_to_compatible_frames"] = bool(train.get("wan_pad_to_compatible_frames", True))


def _apply_env_overrides(resolved_cfg: dict[str, Any]) -> None:
    runtime = resolved_cfg["runtime"]
    dataset = resolved_cfg["dataset"]
    prepare = resolved_cfg["prepare"]
    dist = resolved_cfg["dist"]
    train = resolved_cfg["train"]

    explicit_output_override = _env_text("OUTPUT_DIR") is not None
    explicit_status_dir_override = _env_text("STATUS_DIR") is not None

    python_bin = _env_text("PYTHON_BIN")
    if python_bin is not None:
        resolved_cfg["python_bin"] = _resolve_python_bin(python_bin)

    run_name = _env_text("RUN_NAME")
    if run_name is not None:
        runtime["run_name"] = run_name
        if not explicit_output_override:
            train["output_dir"] = str(Path(runtime["output_root"]) / run_name)

    project_name = _env_text("PROJECT_NAME")
    if project_name is not None:
        runtime["project_name"] = project_name
        train["project_name"] = project_name

    log_with = _env_text("LOG_WITH")
    if log_with is not None:
        runtime["log_with"] = log_with
        train["log_with"] = log_with

    mixed_precision = _env_text("MIXED_PRECISION")
    if mixed_precision is not None:
        runtime["mixed_precision"] = mixed_precision
        train["mixed_precision"] = mixed_precision

    seed = _env_text("SEED")
    if seed is not None:
        runtime["seed"] = int(seed)
        train["seed"] = int(seed)

    output_dir = _env_text("OUTPUT_DIR")
    if output_dir is not None:
        train["output_dir"] = _resolve_repo_path(output_dir) if not output_dir.startswith("/") else output_dir

    resume_checkpoint = _env_text("RESUME_CHECKPOINT")
    if resume_checkpoint is not None:
        train["resume_checkpoint"] = _resolve_repo_path(resume_checkpoint) if not resume_checkpoint.startswith("/") else resume_checkpoint
    resume_model_only = _env_bool("RESUME_MODEL_ONLY")
    if resume_model_only is not None:
        train["resume_model_only"] = resume_model_only
    init_checkpoint = _env_text("INIT_CHECKPOINT")
    if init_checkpoint is not None:
        train["init_checkpoint"] = _resolve_repo_path(init_checkpoint) if not init_checkpoint.startswith("/") else init_checkpoint
    init_checkpoint_strict = _env_bool("INIT_CHECKPOINT_STRICT")
    if init_checkpoint_strict is not None:
        train["init_checkpoint_strict"] = init_checkpoint_strict
    training_stage = _env_text("TRAINING_STAGE")
    if training_stage is not None:
        train["training_stage"] = training_stage
    loss_preset = _env_text("LOSS_PRESET")
    if loss_preset is not None:
        train["loss_preset"] = loss_preset
    freeze_video_backbone = _env_bool("FREEZE_VIDEO_BACKBONE")
    if freeze_video_backbone is not None:
        train["freeze_video_backbone"] = freeze_video_backbone

    for env_name, key, caster in (
        ("BATCH_SIZE", "batch_size", int),
        ("MAX_STEPS", "max_steps", int),
        ("SAVE_EVERY", "save_every", int),
        ("LOG_EVERY", "log_every", int),
        ("PRINT_EVERY", "print_every", int),
        ("LR", "lr", float),
    ):
        value = _env_text(env_name)
        if value is not None:
            train[key] = caster(value)

    for env_name, key in (
        ("RAW_ROOT", "raw_root"),
        ("PREPARED_ROOT", "prepared_root"),
        ("SPLIT_FILE", "split_file"),
        ("SPLIT_KEY", "split_key"),
        ("CAMERA_ID", "camera_id"),
        ("PROCESSED_SUBDIR", "processed_subdir"),
        ("GS_SUBDIR", "gs_subdir"),
    ):
        value = _env_text(env_name)
        if value is not None:
            dataset[key] = value if value.startswith("/") or key in {"split_key", "camera_id", "processed_subdir", "gs_subdir"} else _resolve_repo_path(value)

    prepare_skip = _env_bool("SKIP_PREPARE")
    if prepare_skip is not None:
        prepare["skip"] = prepare_skip
    prepare_only = _env_bool("PREPARE_ONLY")
    if prepare_only is not None:
        prepare["only"] = prepare_only
    prepare_overwrite = _env_bool("OVERWRITE_PREPARE")
    if prepare_overwrite is not None:
        prepare["overwrite"] = prepare_overwrite
    allow_in_place = _env_bool("ALLOW_PREPARE_IN_PLACE")
    if allow_in_place is not None:
        prepare["allow_in_place"] = allow_in_place
    prepare_detach = _env_bool("DETACH_PREPARE")
    if prepare_detach is not None:
        prepare["detach"] = prepare_detach

    for env_name, key, caster in (
        ("PREPARE_MAX_SEQUENCES", "max_sequences", int),
        ("PREPARE_MAX_FRAMES", "max_frames", int),
        ("PREPARE_NUM_WORKERS", "num_workers", int),
        ("PREPARE_HEARTBEAT_INTERVAL", "heartbeat_interval", int),
    ):
        value = _env_text(env_name)
        if value is not None:
            prepare[key] = caster(value)

    status_dir = _env_text("STATUS_DIR")
    if status_dir is not None:
        prepare["status_dir"] = _resolve_repo_path(status_dir) if not status_dir.startswith("/") else status_dir

    for env_name, key, caster in (
        ("NUM_PROCESSES", "num_processes", int),
        ("DIST_NUM_PROCESSES", "num_processes", int),
        ("MASTER_PORT", "main_process_port", int),
        ("DIST_MAIN_PROCESS_PORT", "main_process_port", int),
        ("MASTER_ADDR", "master_addr", str),
        ("DIST_MASTER_ADDR", "master_addr", str),
        ("NNODES", "nnodes", int),
        ("DIST_NNODES", "nnodes", int),
        ("NODE_RANK", "node_rank", int),
        ("DIST_NODE_RANK", "node_rank", int),
    ):
        value = _env_text(env_name)
        if value is not None:
            dist[key] = caster(value)

    if not explicit_output_override:
        train["output_dir"] = str(Path(runtime["output_root"]) / runtime["run_name"])
    if not explicit_status_dir_override:
        prepare["status_dir"] = str(Path(dataset["prepared_root"]) / "_preprocess_logs")

    _finalize_resolved_config(resolved_cfg)


def _apply_cli_overrides(resolved_cfg: dict[str, Any], args: argparse.Namespace) -> None:
    runtime = resolved_cfg["runtime"]
    dataset = resolved_cfg["dataset"]
    prepare = resolved_cfg["prepare"]
    dist = resolved_cfg["dist"]
    train = resolved_cfg["train"]

    if args.python_bin is not None:
        resolved_cfg["python_bin"] = _resolve_python_bin(args.python_bin)

    if args.run_name is not None:
        runtime["run_name"] = str(args.run_name)
        if args.output_dir is None:
            train["output_dir"] = str(Path(runtime["output_root"]) / runtime["run_name"])

    if args.output_dir is not None:
        train["output_dir"] = _resolve_repo_path(args.output_dir)

    if args.project_name is not None:
        runtime["project_name"] = str(args.project_name)
        train["project_name"] = str(args.project_name)

    if args.mixed_precision is not None:
        runtime["mixed_precision"] = str(args.mixed_precision)
        train["mixed_precision"] = str(args.mixed_precision)

    if args.split_key is not None:
        dataset["split_key"] = str(args.split_key)

    if args.master_addr is not None:
        dist["master_addr"] = str(args.master_addr)

    if args.master_port is not None:
        dist["main_process_port"] = int(args.master_port)

    if args.nnodes is not None:
        dist["nnodes"] = int(args.nnodes)

    if args.node_rank is not None:
        dist["node_rank"] = int(args.node_rank)

    if args.wandb is not None:
        log_with = "wandb" if bool(args.wandb) else "none"
        runtime["log_with"] = log_with
        train["log_with"] = log_with

    if args.resume_checkpoint is not None:
        train["resume_checkpoint"] = _resolve_repo_path(args.resume_checkpoint)

    if args.resume_model_only is not None:
        train["resume_model_only"] = bool(args.resume_model_only)

    for attr in (
        "batch_size",
        "max_steps",
        "lr",
        "dataloader_num_workers",
        "dataloader_prefetch_factor",
        "sequence_batch_interleave_window",
        "sequence_batch_burst",
    ):
        value = getattr(args, attr)
        if value is not None:
            train[attr] = value

    for attr in ("dataloader_pin_memory", "dataloader_persistent_workers"):
        value = getattr(args, attr)
        if value is not None:
            train[attr] = bool(value)

    if args.batch_sampler_mode is not None:
        train["batch_sampler_mode"] = str(args.batch_sampler_mode)
    if args.wan_pad_to_compatible_frames is not None:
        train["wan_pad_to_compatible_frames"] = bool(args.wan_pad_to_compatible_frames)

    for attr in ("honest_val_every", "honest_val_num_ode_steps"):
        value = getattr(args, attr)
        if value is not None:
            train[attr] = int(value)

    if args.honest_val_sequence is not None:
        train["honest_val_sequence"] = str(args.honest_val_sequence)
    if args.training_stage is not None:
        train["training_stage"] = str(args.training_stage)
    if args.loss_preset is not None:
        train["loss_preset"] = str(args.loss_preset)
    if args.init_checkpoint is not None:
        train["init_checkpoint"] = _resolve_repo_path(args.init_checkpoint)
    if args.init_checkpoint_strict is not None:
        train["init_checkpoint_strict"] = bool(args.init_checkpoint_strict)
    if args.freeze_video_backbone is not None:
        train["freeze_video_backbone"] = bool(args.freeze_video_backbone)

    if args.raw_root is not None:
        dataset["raw_root"] = _resolve_repo_path(args.raw_root)

    if args.prepared_root is not None:
        dataset["prepared_root"] = _resolve_repo_path(args.prepared_root)
        if _env_text("STATUS_DIR") is None:
            prepare["status_dir"] = str(Path(dataset["prepared_root"]) / "_preprocess_logs")

    if args.split_file is not None:
        dataset["split_file"] = _resolve_repo_path(args.split_file)

    if args.prepare is not None:
        prepare["skip"] = not bool(args.prepare)

    if args.num_processes is not None:
        dist["num_processes"] = int(args.num_processes)

    _finalize_resolved_config(resolved_cfg)


def _prepare_command(resolved_cfg: dict[str, Any]) -> list[str]:
    dataset = resolved_cfg["dataset"]
    prepare = resolved_cfg["prepare"]
    return [
        resolved_cfg["python_bin"],
        str(REPO_ROOT / "scripts" / "preprocess_procigen_gt.py"),
        "--raw_root", dataset["raw_root"],
        "--output_root", dataset["prepared_root"],
        "--status_dir", prepare["status_dir"],
        "--split_file", dataset["split_file"],
        "--split_key", dataset["split_key"],
        "--camera_id", dataset["camera_id"],
        "--processed_subdir", dataset["processed_subdir"],
        "--gs_subdir", dataset["gs_subdir"],
        "--num_workers", str(prepare["num_workers"]),
        "--heartbeat_interval", str(prepare["heartbeat_interval"]),
        *( ["--max_sequences", str(prepare["max_sequences"])] if int(prepare["max_sequences"]) > 0 else [] ),
        *( ["--max_frames", str(prepare["max_frames"])] if int(prepare["max_frames"]) > 0 else [] ),
        *( ["--overwrite"] if bool(prepare["overwrite"]) else [] ),
    ]


def _build_h5_cache_command(
    resolved_cfg: dict[str, Any],
    *,
    max_sequences: int | None,
    chunk_frames: int | None,
    overwrite: bool,
    continue_on_error: bool,
) -> list[str]:
    train = resolved_cfg["train"]
    resolved_max_sequences = int(train["max_sequences"]) if max_sequences is None else int(max_sequences)
    resolved_chunk_frames = int(chunk_frames) if chunk_frames is not None else 16
    cmd = [
        resolved_cfg["python_bin"],
        str(REPO_ROOT / "scripts" / "build_dual_branch_h5_cache.py"),
        "--data_root", train["data_root"],
        "--processed_subdir", train["processed_subdir"],
        "--gs_subdir", train["gs_subdir"],
        "--human_gaussian_source", train["human_gaussian_source"],
        "--split_file", train["split_file"],
        "--split_key", train["split_key"],
        "--num_human_gaussians", str(train["num_human_gaussians"]),
        "--num_object_gaussians", str(train["num_object_gaussians"]),
        "--num_joints", str(train["num_joints"]),
        "--contact_dim", str(train["contact_dim"]),
        "--chunk_frames", str(resolved_chunk_frames),
    ]
    if resolved_max_sequences > 0:
        cmd.extend(["--max_sequences", str(resolved_max_sequences)])
    if overwrite:
        cmd.append("--overwrite")
    if continue_on_error:
        cmd.append("--continue_on_error")
    return cmd


def _train_entry(resolved_cfg: dict[str, Any]) -> list[str]:
    train = resolved_cfg["train"]
    cmd = [
        str(REPO_ROOT / "train_dual_branch_fm.py"),
        "--data_root", train["data_root"],
        "--processed_subdir", train["processed_subdir"],
        "--gs_subdir", train["gs_subdir"],
        "--human_gaussian_source", train["human_gaussian_source"],
        "--output_dir", train["output_dir"],
        "--split_file", train["split_file"],
        "--split_key", train["split_key"],
        "--project_name", train["project_name"],
        "--seed", str(train["seed"]),
        "--log_with", train["log_with"],
        "--mixed_precision", train["mixed_precision"],
        "--training_stage", str(train.get("training_stage", "stage1")),
        "--batch_size", str(train["batch_size"]),
        "--max_steps", str(train["max_steps"]),
        "--save_every", str(train["save_every"]),
        "--log_every", str(train["log_every"]),
        "--print_every", str(train["print_every"]),
        "--gradient_accumulation_steps", str(train["gradient_accumulation_steps"]),
        "--max_grad_norm", str(train["max_grad_norm"]),
        "--lr", str(train["lr"]),
        "--weight_decay", str(train["weight_decay"]),
        "--warmup_steps", str(train["warmup_steps"]),
        "--clip_length", str(train["clip_length"]),
        "--clip_stride", str(train["clip_stride"]),
        "--max_sequences", str(train["max_sequences"]),
        "--dataset_cache_sequences", str(train["dataset_cache_sequences"]),
        "--rgb_cache_max_frames", str(train["rgb_cache_max_frames"]),
        "--index_progress_every", str(train["index_progress_every"]),
        "--dataloader_num_workers", str(train.get("dataloader_num_workers", 0)),
        "--dataloader_prefetch_factor", str(train.get("dataloader_prefetch_factor", 2)),
        "--batch_sampler_mode", str(train.get("batch_sampler_mode", "global_clip_shuffle")),
        "--sequence_batch_interleave_window", str(train.get("sequence_batch_interleave_window", 0)),
        "--sequence_batch_burst", str(train.get("sequence_batch_burst", 1)),
        "--background_value", str(train["background_value"]),
        "--image_height", str(train["image_height"]),
        "--image_width", str(train["image_width"]),
        "--patch_size", str(train["patch_size"]),
        "--condition_patch_size", str(train["condition_patch_size"]),
        "--hidden_dim", str(train["hidden_dim"]),
        "--depth", str(train["depth"]),
        "--num_heads", str(train["num_heads"]),
        "--mlp_ratio", str(train["mlp_ratio"]),
        "--dropout", str(train["dropout"]),
        "--video_backend", str(train["video_backend"]),
        "--video_channels", str(train["video_channels"]),
        "--wan_model_id", str(train["wan_model_id"]),
        "--wan_dtype", str(train["wan_dtype"]),
        "--wan_hidden_dim", str(train["wan_hidden_dim"]),
        "--wan_prompt_max_sequence_length", str(train["wan_prompt_max_sequence_length"]),
        "--wan_prompt_override", str(train["wan_prompt_override"]),
        "--num_human_gaussians", str(train["num_human_gaussians"]),
        "--num_object_gaussians", str(train["num_object_gaussians"]),
        "--num_joints", str(train["num_joints"]),
        "--contact_dim", str(train["contact_dim"]),
        "--human_shape_dim", str(train.get("human_shape_dim", 10)),
        "--human_pose_dim", str(train.get("human_pose_dim", 72)),
        "--loss_preset", train["loss_preset"],
        "--lambda_fm", str(train.get("lambda_fm", 1.0)),
        "--lambda_geo_joint_heat", str(train.get("lambda_geo_joint_heat", 1.0)),
        "--lambda_geo_object_silhouette", str(train.get("lambda_geo_object_silhouette", 0.5)),
        "--lambda_geo_depth", str(train.get("lambda_geo_depth", 1.0)),
        "--lambda_reg_pose", str(train.get("lambda_reg_pose", 1.0)),
        "--lambda_reg_shape", str(train.get("lambda_reg_shape", 0.25)),
        "--lambda_reg_translation", str(train.get("lambda_reg_translation", 1.0)),
        "--lambda_reg_object_pose", str(train.get("lambda_reg_object_pose", 1.0)),
        "--lambda_reg_contact", str(train.get("lambda_reg_contact", 0.25)),
        "--lambda_depth", str(train.get("lambda_depth", 1.0)),
        "--lambda_temp_object", str(train.get("lambda_temp_object", 0.1)),
        "--lambda_temp_contact", str(train.get("lambda_temp_contact", 0.1)),
        "--lambda_phys_contact", str(train.get("lambda_phys_contact", 0.05)),
        "--lambda_phys_penetration", str(train.get("lambda_phys_penetration", 0.05)),
        "--lambda_video_fm", str(train["lambda_video_fm"]),
        "--lambda_state_fm", str(train["lambda_state_fm"]),
        "--lambda_video_latent", str(train["lambda_video_latent"]),
        "--lambda_state_latent", str(train["lambda_state_latent"]),
        "--lambda_human_visible", str(train["lambda_human_visible"]),
        "--lambda_human_temporal", str(train["lambda_human_temporal"]),
        "--lambda_object_video", str(train["lambda_object_video"]),
        "--lambda_object_render", str(train["lambda_object_render"]),
        "--lambda_branch_coupling", str(train["lambda_branch_coupling"]),
        "--lambda_human_gaussian", str(train["lambda_human_gaussian"]),
        "--lambda_object_gaussian", str(train["lambda_object_gaussian"]),
        "--lambda_joints", str(train["lambda_joints"]),
        "--lambda_object_motion", str(train["lambda_object_motion"]),
        "--lambda_contact", str(train["lambda_contact"]),
        "--lambda_joint_heat", str(train["lambda_joint_heat"]),
        "--lambda_object_silhouette", str(train["lambda_object_silhouette"]),
        "--lambda_object_depth", str(train["lambda_object_depth"]),
        "--lambda_geometry_distill", str(train["lambda_geometry_distill"]),
        "--human_visible_region_weight", str(train["human_visible_region_weight"]),
        "--human_completion_region_weight", str(train["human_completion_region_weight"]),
        "--num_human_video_points", str(train["num_human_video_points"]),
        "--human_proxy_gaussian_scale", str(train["human_proxy_gaussian_scale"]),
        "--object_visible_region_weight", str(train["object_visible_region_weight"]),
        "--object_primary_region_weight", str(train["object_primary_region_weight"]),
        "--object_secondary_region_weight", str(train["object_secondary_region_weight"]),
        "--reconstruction_warmup_ratio", str(train["reconstruction_warmup_ratio"]),
        "--curriculum_fusion_start_ratio", str(train["curriculum_fusion_start_ratio"]),
        "--curriculum_full_start_ratio", str(train["curriculum_full_start_ratio"]),
        "--video_unfreeze_start_ratio", str(train["video_unfreeze_start_ratio"]),
        "--video_stage2_num_top_blocks", str(train["video_stage2_num_top_blocks"]),
        "--honest_val_every", str(train["honest_val_every"]),
        "--honest_val_num_ode_steps", str(train["honest_val_num_ode_steps"]),
        "--honest_val_prior_noise_std", str(train["honest_val_prior_noise_std"]),
        "--honest_val_sequence", str(train["honest_val_sequence"]),
    ]
    _append_bool_flag(cmd, bool(train["cache_rgb"]), "--cache_rgb", "--no-cache_rgb")
    _append_bool_flag(
        cmd,
        bool(train.get("dataloader_pin_memory", True)),
        "--dataloader_pin_memory",
        "--no-dataloader_pin_memory",
    )
    _append_bool_flag(
        cmd,
        bool(train.get("dataloader_persistent_workers", True)),
        "--dataloader_persistent_workers",
        "--no-dataloader_persistent_workers",
    )
    _append_bool_flag(
        cmd,
        bool(train.get("wan_pad_to_compatible_frames", True)),
        "--wan_pad_to_compatible_frames",
        "--no-wan_pad_to_compatible_frames",
    )
    _append_bool_flag(cmd, bool(train["freeze_video_backbone"]), "--freeze_video_backbone", "--no-freeze_video_backbone")
    if str(train.get("resume_checkpoint", "")).strip():
        _append_flag(cmd, "--resume_checkpoint", train["resume_checkpoint"])
    if str(train.get("init_checkpoint", "")).strip():
        _append_flag(cmd, "--init_checkpoint", train["init_checkpoint"])
    if bool(train.get("resume_model_only", False)):
        cmd.append("--resume_model_only")
    if bool(train.get("init_checkpoint_strict", False)):
        cmd.append("--init_checkpoint_strict")
    return cmd


def _run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(cmd, env=env, check=True)


def _read_live_pid(pid_file: Path) -> int | None:
    if not pid_file.is_file():
        return None
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _launch_detached_h5_cache_builder(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    pid_path: Path,
) -> int:
    existing_pid = _read_live_pid(pid_path)
    if existing_pid is not None:
        print(
            f"[run_dual_branch_fm] H5 cache builder already running | pid={existing_pid} | log={log_path}",
            flush=True,
        )
        return existing_pid

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n")
    print(
        f"[run_dual_branch_fm] launched H5 cache builder | pid={process.pid} | log={log_path}",
        flush=True,
    )
    return process.pid


def _launch_detached_prepare(cmd: list[str], *, env: dict[str, str], status_dir: Path, python_bin: str) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = status_dir / f"stdout_{datetime.now():%Y%m%d_%H%M%S}.log"
    pid_file = status_dir / "preprocess.pid"
    with stdout_log.open("wb") as handle:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    pid_file.write_text(f"{process.pid}\n")
    (status_dir / "latest_stdout.log").write_text(f"{stdout_log}\n")
    (status_dir / "latest_pid").write_text(f"{process.pid}\n")
    print(f"[run_dual_branch_fm] Detached PID={process.pid}")
    print(f"[run_dual_branch_fm] Stdout log: {stdout_log}")
    print(f"[run_dual_branch_fm] Status dir: {status_dir}")
    print("[run_dual_branch_fm] Monitor with:")
    print(f"  {python_bin} {REPO_ROOT / 'scripts' / 'monitor_procigen_preprocess.py'} --status_dir {status_dir} --follow")


def parse_args() -> argparse.Namespace:
    h5_cache_max_sequences_env = _env_text("H5_CACHE_MAX_SEQUENCES")
    h5_cache_chunk_frames_env = _env_text("H5_CACHE_CHUNK_FRAMES")
    parser = argparse.ArgumentParser(description="Run ProciGen dual-branch FM training from the unified config.")
    parser.add_argument("--config", type=str, default=os.getenv("CONFIG_FILE", str(DEFAULT_CONFIG_PATH)))
    parser.add_argument("--dataset", type=str, default=_env_text("DATASET"))
    parser.add_argument("--python_bin", type=str, default=None)
    parser.add_argument("--run_name", "--task_name", dest="run_name", type=str, default=None)
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--dataloader_pin_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dataloader_persistent_workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=None)
    parser.add_argument(
        "--batch_sampler_mode",
        type=str,
        choices=("global_clip_shuffle", "sequence_grouped"),
        default=None,
    )
    parser.add_argument("--sequence_batch_interleave_window", type=int, default=None)
    parser.add_argument("--sequence_batch_burst", type=int, default=None)
    parser.add_argument("--raw_root", type=str, default=None)
    parser.add_argument("--prepared_root", "--data_root", dest="prepared_root", type=str, default=None)
    parser.add_argument("--split_file", "--split", dest="split_file", type=str, default=None)
    parser.add_argument("--split_key", type=str, default=None)
    parser.add_argument("--num_processes", type=int, default=None)
    parser.add_argument("--master_addr", type=str, default=None)
    parser.add_argument("--master_port", type=int, default=None)
    parser.add_argument("--nnodes", type=int, default=None)
    parser.add_argument("--node_rank", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--honest_val_every", type=int, default=None)
    parser.add_argument("--honest_val_num_ode_steps", type=int, default=None)
    parser.add_argument("--honest_val_sequence", type=str, default=None)
    parser.add_argument("--training_stage", type=str, choices=("stage1", "stage2"), default=None)
    parser.add_argument("--loss_preset", type=str, default=None)
    parser.add_argument("--wan_pad_to_compatible_frames", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--build_h5_cache", action=argparse.BooleanOptionalAction, default=_env_bool("BUILD_H5_CACHE"))
    parser.add_argument(
        "--h5_cache_continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("H5_CACHE_CONTINUE_ON_ERROR"),
    )
    parser.add_argument(
        "--h5_cache_overwrite",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("H5_CACHE_OVERWRITE"),
    )
    parser.add_argument(
        "--h5_cache_max_sequences",
        type=int,
        default=int(h5_cache_max_sequences_env) if h5_cache_max_sequences_env is not None else None,
    )
    parser.add_argument(
        "--h5_cache_chunk_frames",
        type=int,
        default=int(h5_cache_chunk_frames_env) if h5_cache_chunk_frames_env is not None else None,
    )
    parser.add_argument("--h5_cache_log", type=str, default=_env_text("H5_CACHE_LOG"))
    parser.add_argument("--prepare", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume_model_only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--init_checkpoint_strict", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze_video_backbone", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = load_dual_branch_fm_config(
        config_path=args.config,
        dataset=args.dataset,
    )
    _apply_env_overrides(resolved)
    _apply_cli_overrides(resolved, args)

    runtime = resolved["runtime"]
    dataset = resolved["dataset"]
    prepare = resolved["prepare"]
    dist = resolved["dist"]
    train = resolved["train"]
    python_bin = resolved["python_bin"]

    if not prepare["skip"] and not prepare["allow_in_place"] and dataset["raw_root"] == dataset["prepared_root"]:
        raise SystemExit(
            "[run_dual_branch_fm] Refusing to preprocess in-place into RAW_ROOT. Set a separate prepared_root in config, or ALLOW_PREPARE_IN_PLACE=1."
        )

    print("============================================================")
    print("  ProciGen Dual-Branch FM Training")
    print(f"  Config         : {resolved['config_path']}")
    print(f"  Dataset preset : {dataset['name']}")
    print(f"  Run name       : {runtime['run_name']}")
    print(f"  Prepared root  : {dataset['prepared_root']}")
    print(f"  Output dir     : {train['output_dir']}")
    print(f"  Logging        : {train['log_with']}")
    print(f"  Mixed precision: {train['mixed_precision']}")
    print(f"  Data workers   : {train['dataloader_num_workers']} / rank")
    print(f"  Sampler mode   : {train['batch_sampler_mode']}")
    print(f"  Max steps / LR : {train['max_steps']} / {train['lr']}")
    print(f"  Build H5 Cache : {int(bool(args.build_h5_cache))}")
    print("============================================================")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if train["log_with"] == "wandb":
        env.setdefault("WANDB_NAME", str(runtime["run_name"]))

    if not prepare["skip"]:
        prep_cmd = _prepare_command(resolved)
        if prepare["detach"]:
            if not prepare["only"]:
                raise SystemExit("[run_dual_branch_fm] DETACH_PREPARE=1 only supports PREPARE_ONLY=1.")
            print("[run_dual_branch_fm] Launching detached preprocessing...")
            _launch_detached_prepare(prep_cmd, env=env, status_dir=Path(prepare["status_dir"]), python_bin=python_bin)
            return
        print("[run_dual_branch_fm] Preparing GT assets...")
        _run_command(prep_cmd, env=env)

    if prepare["only"]:
        print("[run_dual_branch_fm] PREPARE_ONLY=1, exiting after preprocessing.")
        return

    build_h5_cache = bool(args.build_h5_cache)
    h5_cache_continue_on_error = True if args.h5_cache_continue_on_error is None else bool(args.h5_cache_continue_on_error)
    h5_cache_overwrite = bool(args.h5_cache_overwrite) if args.h5_cache_overwrite is not None else False
    if build_h5_cache:
        h5_log_path = Path(args.h5_cache_log).expanduser() if args.h5_cache_log else Path(train["output_dir"]) / "h5_cache_builder.log"
        if not h5_log_path.is_absolute():
            h5_log_path = (REPO_ROOT / h5_log_path).resolve()
        h5_pid_path = h5_log_path.parent / "h5_cache_builder.pid"
        h5_cache_cmd = _build_h5_cache_command(
            resolved,
            max_sequences=args.h5_cache_max_sequences,
            chunk_frames=args.h5_cache_chunk_frames,
            overwrite=h5_cache_overwrite,
            continue_on_error=h5_cache_continue_on_error,
        )
        _launch_detached_h5_cache_builder(
            h5_cache_cmd,
            env=env,
            log_path=h5_log_path,
            pid_path=h5_pid_path,
        )

    train_entry = _train_entry(resolved)
    visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    if int(dist["num_processes"]) > 1:
        print("[run_dual_branch_fm] Launching distributed training with torch.distributed.run...")
        print(f"[run_dual_branch_fm] CUDA_VISIBLE_DEVICES : {visible_devices or '<inherit all visible>'}")
        print(f"[run_dual_branch_fm] nproc/node    : {dist['num_processes']}")
        print(f"[run_dual_branch_fm] nnodes        : {dist['nnodes']}")
        print(f"[run_dual_branch_fm] node_rank     : {dist['node_rank']}")
        print(f"[run_dual_branch_fm] master        : {dist['master_addr']}:{dist['main_process_port']}")
        cmd = [
            python_bin,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node", str(dist["num_processes"]),
            "--nnodes", str(dist["nnodes"]),
            "--node_rank", str(dist["node_rank"]),
            "--master_addr", str(dist["master_addr"]),
            "--master_port", str(dist["main_process_port"]),
            *train_entry,
        ]
    else:
        print(f"[run_dual_branch_fm] Launching single-process training with CUDA_VISIBLE_DEVICES={visible_devices or '<inherit all visible>'}...")
        cmd = [python_bin, *train_entry]
    _run_command(cmd, env=env)


if __name__ == "__main__":
    main()
