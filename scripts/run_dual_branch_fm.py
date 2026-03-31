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

    for attr in ("batch_size", "max_steps", "lr"):
        value = getattr(args, attr)
        if value is not None:
            train[attr] = value

    for attr in ("honest_val_every", "honest_val_num_ode_steps"):
        value = getattr(args, attr)
        if value is not None:
            train[attr] = int(value)

    if args.honest_val_sequence is not None:
        train["honest_val_sequence"] = str(args.honest_val_sequence)

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
        "--background_value", str(train["background_value"]),
        "--image_height", str(train["image_height"]),
        "--image_width", str(train["image_width"]),
        "--patch_size", str(train["patch_size"]),
        "--hidden_dim", str(train["hidden_dim"]),
        "--depth", str(train["depth"]),
        "--num_heads", str(train["num_heads"]),
        "--mlp_ratio", str(train["mlp_ratio"]),
        "--dropout", str(train["dropout"]),
        "--video_channels", str(train["video_channels"]),
        "--num_human_gaussians", str(train["num_human_gaussians"]),
        "--num_object_gaussians", str(train["num_object_gaussians"]),
        "--num_joints", str(train["num_joints"]),
        "--contact_dim", str(train["contact_dim"]),
        "--loss_preset", train["loss_preset"],
        "--lambda_video_fm", str(train["lambda_video_fm"]),
        "--lambda_state_fm", str(train["lambda_state_fm"]),
        "--lambda_video_latent", str(train["lambda_video_latent"]),
        "--lambda_state_latent", str(train["lambda_state_latent"]),
        "--lambda_human_visible", str(train["lambda_human_visible"]),
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
    _append_bool_flag(cmd, bool(train["freeze_video_backbone"]), "--freeze_video_backbone", "--no-freeze_video_backbone")
    if str(train.get("resume_checkpoint", "")).strip():
        _append_flag(cmd, "--resume_checkpoint", train["resume_checkpoint"])
    if bool(train.get("resume_model_only", False)):
        cmd.append("--resume_model_only")
    return cmd


def _run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(cmd, env=env, check=True)


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
    parser.add_argument("--honest_val_every", type=int, default=None)
    parser.add_argument("--honest_val_num_ode_steps", type=int, default=None)
    parser.add_argument("--honest_val_sequence", type=str, default=None)
    parser.add_argument("--prepare", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume_model_only", action=argparse.BooleanOptionalAction, default=None)
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
    print(f"  Max steps / LR : {train['max_steps']} / {train['lr']}")
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
