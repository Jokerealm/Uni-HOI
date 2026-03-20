#!/usr/bin/env python3
"""
Read and display ProciGen preprocessing progress files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def format_seconds(seconds):
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    if seconds >= 3600:
        return f"{seconds / 3600.0:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60.0:.2f}m"
    return f"{seconds:.1f}s"


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tail_jsonl(path: Path, limit: int) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def read_recorded_pid(status_dir: Path, progress: Dict[str, object]) -> Optional[int]:
    pid = progress.get("pid")
    if pid is not None:
        try:
            return int(pid)
        except (TypeError, ValueError):
            pass

    pid_path = status_dir / "preprocess.pid"
    if not pid_path.is_file():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, TypeError, ValueError):
        return None


def probe_process(pid: Optional[int]) -> Tuple[Optional[bool], str]:
    if pid is None or pid <= 0:
        return None, ""

    proc_dir = Path("/proc") / str(pid)
    if not proc_dir.exists():
        return False, ""

    cmdline = ""
    cmdline_path = proc_dir / "cmdline"
    if cmdline_path.is_file():
        try:
            cmdline = cmdline_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", " ").strip()
        except OSError:
            cmdline = ""

    if not cmdline:
        try:
            os.kill(pid, 0)
        except OSError:
            return False, ""
    if cmdline and "preprocess_procigen_gt.py" not in cmdline:
        return False, cmdline
    return True, cmdline


def render_snapshot(status_dir: Path, recent_limit: int) -> str:
    progress = read_json(status_dir / "progress.json")
    if not progress:
        return f"No progress file found under {status_dir}"

    lines = []
    recorded_pid = read_recorded_pid(status_dir, progress)
    pid_alive, pid_cmdline = probe_process(recorded_pid)
    lines.append(f"status_dir: {status_dir}")
    lines.append(f"status: {progress.get('status', 'unknown')}")
    lines.append(f"timestamp: {progress.get('timestamp', 'n/a')}")
    if recorded_pid is not None:
        lines.append(
            "pid: "
            f"{recorded_pid} | alive="
            f"{'yes' if pid_alive is True else 'no' if pid_alive is False else 'unknown'}"
        )
    lines.append(
        "progress: "
        f"{progress.get('processed_total', 0)}/{progress.get('total_sequences', 0)} "
        f"(already_completed={progress.get('already_completed', 0)}, "
        f"prepared_new={progress.get('prepared_new', 0)}, "
        f"failed_new={progress.get('failed_new', 0)})"
    )
    lines.append(
        "runtime: "
        f"elapsed={format_seconds(progress.get('elapsed_seconds'))}, "
        f"avg_seq={format_seconds(progress.get('avg_sequence_seconds'))}, "
        f"eta={format_seconds(progress.get('eta_seconds'))}"
    )
    lines.append(
        "workers: "
        f"{progress.get('num_workers', 'n/a')} "
        f"| pending_remaining={progress.get('pending_remaining', 'n/a')}"
    )
    active_sequences = progress.get("active_sequences") or []
    if active_sequences:
        lines.append("active_sequences: " + ", ".join(str(item) for item in active_sequences))
    if progress.get("status") == "running" and pid_alive is False:
        lines.append(
            "warning: progress.json still says running, but the recorded PID is gone. "
            "This run has already stopped and the status file is stale."
        )
    if progress.get("error_type") or progress.get("error"):
        lines.append(
            f"error: {progress.get('error_type', 'Error')}: {progress.get('error', '')}".rstrip()
        )
    if progress.get("recent_sequence"):
        lines.append(
            "last_event: "
            f"{progress.get('last_event', 'n/a')} "
            f"| recent_sequence={progress.get('recent_sequence')}"
        )
    elif progress.get("last_event"):
        lines.append(f"last_event: {progress.get('last_event', 'n/a')}")
    if pid_cmdline:
        lines.append(f"cmd: {pid_cmdline}")

    recent_events = tail_jsonl(status_dir / "events.jsonl", recent_limit)
    if recent_events:
        lines.append("recent_events:")
        for event in recent_events:
            event_type = event.get("type", "event")
            sequence = event.get("sequence", "")
            status = event.get("status", "")
            elapsed = format_seconds(event.get("elapsed_seconds"))
            message = f"  {event.get('timestamp', 'n/a')} | {event_type}"
            if sequence:
                message += f" | {sequence}"
            if status:
                message += f" | {status}"
            if event_type == "sequence_result":
                message += f" | took={elapsed}"
            if status == "failed":
                message += f" | {event.get('error_type', 'Error')}: {event.get('error', '')}"
            lines.append(message)

    recent_failures = tail_jsonl(status_dir / "failures.jsonl", min(recent_limit, 5))
    if recent_failures:
        lines.append("recent_failures:")
        for failure in recent_failures:
            lines.append(
                f"  {failure.get('timestamp', 'n/a')} | {failure.get('sequence', 'n/a')} "
                f"| {failure.get('error_type', 'Error')}: {failure.get('error', '')}"
            )

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor ProciGen preprocessing progress.")
    parser.add_argument("--status_dir", type=str, required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--recent", type=int, default=10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    status_dir = Path(args.status_dir).expanduser().resolve()
    if not args.follow:
        print(render_snapshot(status_dir, args.recent))
        return

    while True:
        print("=" * 80)
        print(render_snapshot(status_dir, args.recent))
        progress = read_json(status_dir / "progress.json")
        state = str(progress.get("status", "unknown"))
        recorded_pid = read_recorded_pid(status_dir, progress)
        pid_alive, _ = probe_process(recorded_pid)
        if state in {"completed", "completed_with_failures", "failed", "crashed", "aborted"}:
            break
        if state == "running" and pid_alive is False:
            break
        time.sleep(max(int(args.interval), 1))


if __name__ == "__main__":
    main()
