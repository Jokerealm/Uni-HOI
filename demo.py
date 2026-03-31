"""Legacy demo placeholder.

The original single-image demo depended on the removed PVCNN dual-branch
diffusion stack. Keep this entrypoint as a clear redirect to the current
dual-branch Flow Matching workflow.
"""


def main() -> None:
    raise SystemExit(
        "The legacy PVCNN dual-branch demo has been removed. "
        "Use `python scripts/run_dual_branch_fm.py --help` for training or "
        "`python infer_dual_branch_fm.py --help` for current FM inference."
    )


if __name__ == "__main__":
    main()
