"""Run every experiment that may support the paper from one current source tree.

The command has no experiment-selection arguments. It first completes the
same-version NLM-10 Neural NP4G curriculum, then runs the dense-NLM reference
on the exact same raw splits, seeds, minibatches, and optimisation budget, and
finally builds machine-readable paper tables. It never reuses historical
artifacts.
"""

from __future__ import annotations

import subprocess
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import OFFICIAL_CONFIG, OfficialConfig
from .runtime import configure_logging
from .paper_evidence import (
    BASELINE_RELATIVE_PATH,
    build_paper_evidence,
)
from .run_official_experiments import (
    DEFAULT_ARTIFACT_ROOT,
    _atomic_json,
    _read_json,
    _runner_lock,
    _utc_now,
    initialise_artifact_root,
    run_official_battery,
)
from .tasks import (
    AUXILIARY_TASK_NAMES,
    CURRICULUM_EXECUTION_ORDER,
    PRIMARY_TASK_NAMES,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parent
PAPER_PROTOCOL_MANIFEST = "paper_protocol_manifest.json"
PAPER_DRIVER_STATE = "paper_driver_state.json"
logger = logging.getLogger(__name__)


def _baseline_command(report_path: Path, device: torch.device | str) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-u",
        "-m",
        "official_neural_np4g.nlm_baseline",
        "--device",
        str(torch.device(device)),
        "--report-out",
        str(report_path),
    ]


def _paper_protocol(config: OfficialConfig) -> dict[str, Any]:
    return {
        "protocol_version": config.protocol_version,
        "purpose": "complete reproducible evidence for paper tables",
        "historical_artifacts_allowed": False,
        "cross_protocol_bootstrap_allowed": False,
        "source_integrity": "source_hashes.json records canonical-file SHA-256 only",
        "stages": [
            {
                "name": "official_neural_np4g",
                "execution_order": list(CURRICULUM_EXECUTION_ORDER),
                "proof_obligations": len(PRIMARY_TASK_NAMES) + len(AUXILIARY_TASK_NAMES),
            },
            {
                "name": "dense_nlm_matched_finite_evaluation",
                "tasks": list(PRIMARY_TASK_NAMES),
                "seeds": list(config.attempt_seeds),
                "raw_split_identity": "exact same build_split values and SHA-256",
                "minibatch_identity": "exact same deterministic training_batch schedule",
                "maximum_steps": config.total_steps,
                "learned_adfs_supplied": False,
                "proof_claim": False,
            },
            {"name": "paper_evidence", "requires_all_previous_stages": True},
        ],
    }


def _initialise_paper_protocol(root: Path, config: OfficialConfig) -> None:
    path = root / PAPER_PROTOCOL_MANIFEST
    expected = _paper_protocol(config)
    if path.exists():
        if _read_json(path) != expected:
            raise RuntimeError("paper protocol manifest changed after the run began")
    else:
        _atomic_json(path, expected)


def _run_dense_nlm(root: Path, device: torch.device) -> None:
    report = root / BASELINE_RELATIVE_PATH
    if report.exists():
        logger.info("Dense NLM comparison already present: %s", report)
        return
    report.parent.mkdir(parents=True, exist_ok=True)
    command = _baseline_command(report, device)
    _atomic_json(
        root / PAPER_DRIVER_STATE,
        {
            "status": "RUNNING_DENSE_NLM",
            "report": BASELINE_RELATIVE_PATH.as_posix(),
            "started_at": _utc_now(),
        },
    )
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"dense NLM comparison failed with exit code {completed.returncode}")
    if not report.is_file():
        raise RuntimeError("dense NLM comparison returned without its fixed report")


def _run_paper_protocol_locked(
    root: Path | str = DEFAULT_ARTIFACT_ROOT,
    *,
    device: torch.device | str,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> dict[str, Any]:
    """Run or resume all current-source paper experiments and finalise evidence."""

    selected_device = torch.device(device)
    destination = initialise_artifact_root(root, device=selected_device, config=config)
    _initialise_paper_protocol(destination, config)
    try:
        _atomic_json(
            destination / PAPER_DRIVER_STATE,
            {"status": "RUNNING_OFFICIAL_CURRICULUM", "started_at": _utc_now()},
        )
        run_official_battery(destination, device=selected_device, config=config)
        _run_dense_nlm(destination, selected_device)
        evidence = build_paper_evidence(destination, config=config)
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            logger.warning("Paper experiment interrupted; saved state can be resumed")
        else:
            logger.exception("Paper experiment failed")
        _atomic_json(
            destination / PAPER_DRIVER_STATE,
            {
                "status": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                "error_type": type(error).__name__,
                "reason": str(error),
                "updated_at": _utc_now(),
            },
        )
        raise
    _atomic_json(
        destination / PAPER_DRIVER_STATE,
        {
            "status": "COMPLETE",
            "completed_at": _utc_now(),
            "paper_evidence": "paper_evidence.json",
        },
    )
    logger.info("Complete paper evidence written to %s", destination)
    return evidence


def run_paper_protocol(
    root: Path | str = DEFAULT_ARTIFACT_ROOT,
    *,
    device: torch.device | str,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> dict[str, Any]:
    """Hold one driver lock across training, NLM comparison, and aggregation."""

    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with _runner_lock(destination, filename="paper_driver.lock"):
        return _run_paper_protocol_locked(destination, device=device, config=config)


def main() -> None:
    configure_logging("paper-experiments", log_dir=DEFAULT_ARTIFACT_ROOT / "logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_paper_protocol(DEFAULT_ARTIFACT_ROOT, device=device, config=OFFICIAL_CONFIG)


def _command_entrypoint(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit(
            "usage: python -m official_neural_np4g.run_paper_experiments\n"
            "This canonical command is argument-free; no experiment was started."
        )
    main()


if __name__ == "__main__":
    _command_entrypoint()
