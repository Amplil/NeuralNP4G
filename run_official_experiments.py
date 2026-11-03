"""Run the preregistered official Neural NP4G experiment protocol.

Each NLM-10 curriculum task receives the complete fixed seed grid 0..9 in order.
The earliest strict equivalence proof is deterministically promoted for later
tasks, while the remaining seeds still run so proof rates are comparable with
the matched NLM baseline.  Later tasks receive only their preregistered,
semantically relevant ADF bank; declared dependencies are required acceptance
evidence.
Boundary/negative and ADF research tasks remain in the registry but are outside
this formal battery.  Failed attempts remain complete artifacts;
infrastructure failures pause the driver for an exact same-attempt resume.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

# This must be set before the first CUDA BLAS operation.  train.py repeats the
# fixed value so direct API users receive the same deterministic contract.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_existing_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _existing_cublas_config not in (None, _CUBLAS_WORKSPACE_CONFIG):
    raise RuntimeError(
        "formal Neural NP4G requires "
        f"CUBLAS_WORKSPACE_CONFIG={_CUBLAS_WORKSPACE_CONFIG}, got "
        f"{_existing_cublas_config!r}"
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG

import torch

from .adf import ADFEntry, ADFLibrary, VERIFIED_STATUS
from . import validated_numerics as vn
from .config import OFFICIAL_CONFIG, OfficialConfig, derive_seed
from .data import build_split
from .evaluate import evaluate_adf_causal_ablation, evaluate_split
from .model import NeuralNP4G
from .runtime import configure_logging, default_artifact_root
from .tasks import (
    CURRICULUM_EXECUTION_ORDER,
    EmpiricalTaskSpec,
    TaskSpec,
    fixed_adf_candidate_allowlist,
    get_task,
)
from .train import (
    TrainingResult,
    build_model,
    compression_training_results,
    load_checkpoint,
    semantic_compression_result,
    train_model,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = default_artifact_root(OFFICIAL_CONFIG.protocol_version)
logger = logging.getLogger(__name__)

# The formal runner executes the fixed staged curriculum.  Auxiliary predicates
# are proof obligations rather than paper comparison rows, but they must be
# certified and promoted before the downstream relational task is constructed.
FORMAL_EXECUTION_ORDER = CURRICULUM_EXECUTION_ORDER
FORMAL_TASKS = tuple(get_task(name) for name in FORMAL_EXECUTION_ORDER)
if len(FORMAL_TASKS) != len(FORMAL_EXECUTION_ORDER) or len(
    set(FORMAL_EXECUTION_ORDER)
) != len(FORMAL_TASKS):
    raise RuntimeError("formal staged curriculum registry is invalid")
TERMINAL_PROOF_FAILURES = frozenset(
    {
        "NOT_EQUIVALENT",
        "NOT_PROVED",
        "EXTRACTION_FAILED",
        "EXTRACTION_MISMATCH",
        "ADF_NOT_CAUSALLY_USED",
    }
)

# The completed paper protocol is deliberately self-contained.  Historical
# bootstrap artifacts are still retained on disk for audit, but no formal
# entry point may import, copy, or activate them.
FORBIDDEN_BOOTSTRAP_PATHS = (
    "bootstrap_manifest.json",
    "bootstrap_manifest.sha256",
    "bootstrap_in_progress.json",
    "bootstrap_source_v7",
    "bootstrap_reproof_v12",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(_json_value(value), **options) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    _replace_with_retry(temporary, path)


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """Tolerate short Windows scanner/read locks without dropping an update."""

    deadline = time.monotonic() + 30.0
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, _json_bytes(value, pretty=True))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_cross_protocol_artifacts(root: Path) -> None:
    present = [name for name in FORBIDDEN_BOOTSTRAP_PATHS if (root / name).exists()]
    if present:
        raise RuntimeError(
            "the official protocol forbids cross-protocol bootstrap artifacts: "
            + ", ".join(present)
        )


def _runtime_source_hashes() -> dict[str, str]:
    repository = PACKAGE_DIR.parent
    # Every executed model, baseline, and runtime helper lives in this package.
    files = list(PACKAGE_DIR.glob("*.py"))
    return {
        path.relative_to(repository).as_posix(): _file_sha256(path)
        for path in sorted(files)
    }


def _execution_contract(device: torch.device | str) -> dict[str, Any]:
    """Return only execution-critical fields that must match on resume."""

    requested = torch.device(device)
    contract: dict[str, Any] = {
        "selected_device_type": requested.type,
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "verification_contract": vn.metadata(),
    }
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        selected_index = (
            torch.cuda.current_device()
            if requested.index is None
            else int(requested.index)
        )
        selected = torch.device("cuda", selected_index)
        contract.update(
            {
                "selected_device": str(selected),
                "selected_device_index": selected_index,
                "selected_device_name": torch.cuda.get_device_name(selected),
                "selected_device_capability": list(
                    torch.cuda.get_device_capability(selected)
                ),
                "torch_cuda_version": torch.version.cuda,
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
            }
        )
    else:
        contract["selected_device"] = str(requested)
    return contract


def _environment_record(device: torch.device) -> dict[str, Any]:
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available() and device.type == "cuda":
        cuda.update(
            {
                "device_count": torch.cuda.device_count(),
                "selected_device": str(device),
                "selected_name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return {
        "recorded_at": _utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda": cuda,
        "execution_contract": _execution_contract(device),
    }


def _validate_environment_record(
    recorded: Mapping[str, Any],
    device: torch.device | str,
) -> None:
    """Reject a resume that would change its deterministic trajectory."""

    saved = recorded.get("execution_contract")
    if not isinstance(saved, Mapping):
        raise RuntimeError("environment record has no execution contract")
    current = _execution_contract(device)
    differing = sorted(
        key
        for key in set(saved) | set(current)
        if saved.get(key) != current.get(key)
    )
    if differing:
        details = ", ".join(
            f"{key}: recorded={saved.get(key)!r}, current={current.get(key)!r}"
            for key in differing
        )
        raise RuntimeError(f"execution environment contract mismatch ({details})")


def initialise_artifact_root(
    root: Path | str = DEFAULT_ARTIFACT_ROOT,
    *,
    device: torch.device | str,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> Path:
    """Create or validate the immutable root-level protocol records."""

    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    hashes_path = destination / "source_hashes.json"
    current_source_hashes = _runtime_source_hashes()
    if hashes_path.exists():
        source_hashes = _read_json(hashes_path)
        if source_hashes != current_source_hashes:
            raise RuntimeError(
                "runtime sources changed after the formal run began; "
                "use a new protocol version"
            )
    else:
        source_hashes = current_source_hashes
        _atomic_json(hashes_path, source_hashes)
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("source hash manifest is invalid")
    source_digest = hashlib.sha256(_json_bytes(source_hashes)).hexdigest()
    protocol = {
        "protocol_version": config.protocol_version,
        "config": config.as_dict(),
        "tasks": [asdict(task) for task in FORMAL_TASKS],
        "attempt_seeds": list(config.attempt_seeds),
        "execution_order": list(FORMAL_EXECUTION_ORDER),
        "source_digest": source_digest,
        "checkpoint_schedule": [
            {"stage": "phase1", "steps": config.phase1_steps},
            {"stage": "phase2", "steps": config.total_steps},
        ],
        "stage_order": [
            "phase1", "compression_portfolio", "phase2", "semantic_compression",
        ],
        "stage_decision": (
            "extract and strictly verify the immutable phase1 checkpoint; "
            "run phase2 on the same optimiser/RNG trajectory only if phase1 "
            "does not meet the proof/evidence acceptance rule"
        ),
        "proof_resource_budget_scope": "per_circuit_including_dependency_replay",
        "proof_resource_budget_policy": (
            "each replayed dependency circuit and the target circuit receive "
            "the fixed extraction time limit separately; the worker memory "
            "limit remains process-wide and the parent deadline covers every "
            "circuit plus 90 seconds of startup and reporting overhead"
        ),
        "repair_pruning": (
            "every scheduled prune is transactional and is rolled back if it "
            "would discard a zero-error solution; phase2 additionally applies "
            "fixed synthetic-domain discrete edge repair inside the same "
            "transaction"
        ),
        "prune_guard_policy": (
            f"at a scheduled prune whose post-optimizer event batch is zero-error, "
            f"the event batch must remain zero-error and the fixed TRAIN guard, "
            f"with {config.prune_guard_examples_per_size} examples per training "
            "size derived from prune_guard seed stream id 9, must have a "
            "non-increasing wrong count at every size versus its pre-prune "
            "baseline; guard examples need not themselves be zero-error and are "
            "not validation or test data"
        ),
        "partial_prune_policy": (
            f"if a full pruning proposal fails the fixed guard, evaluate at most "
            f"{config.partial_prune_max_trials} deterministic partial proposals; "
            "commit only a safe subset, and roll back the full proposal when zero "
            "safe kills remain"
        ),
        "exact_incumbent_policy": (
            f"after step_alpha reaches its fixed hard value 1, evaluate up to "
            f"{config.incumbent_guard_examples_per_size} materialised TRAIN "
            "examples per size; preserve every all-zero state with model, "
            "optimizer, RNG, and step_alpha, and restore it before pruning "
            "whenever intervening optimizer updates regress; do not create "
            "incumbents during the soft-to-hard homotopy"
        ),
        "compression_proof_policy": (
            f"freeze every all-TRAIN exact phase2 prune boundary with max fan-in "
            f"at most {config.compression_proof_max_fanin}; assess at most "
            f"{config.compression_proof_max_checkpoints} checkpoints ordered by "
            "(max fan-in, mean fan-in, step), then assess the phase2 boundary"
        ),
        "semantic_compression_policy": (
            "if the phase2 proof fails but the complete materialised TRAIN split "
            "is exact, try hidden nodes in circuit-index order and then live edges "
            "in (absolute-weight, destination, source) order; retain a removal only "
            "when the complete TRAIN split remains exact, expose neither held-out "
            "data nor the target formula to the decisions, freeze the resulting "
            "checkpoint, and assess it as the final preregistered stage"
        ),
        "dynamic_candidate_pruning": (
            "a temporary hard-OFF primitive or learned-ADF gate never makes "
            "its input columns permanently dead"
        ),
        "nlm10_curriculum_policy": (
            "the ten primary rows are exactly the NLM paper's Family Tree and "
            "General Graph tasks; is_sister, is_parent, is_brother, and "
            "connectivity_2 are additional certified helper obligations"
        ),
        "staged_adf_policy": (
            "has_sister consumes is_sister; is_grandparent consumes is_parent; "
            "is_uncle consumes is_brother with raw parent; is_mguncle consumes "
            "the certified is_uncle chain; connectivity_4 consumes "
            "connectivity_2; connectivity_6 consumes connectivity_2 and "
            "connectivity_4. Raw base channels are masked whenever the target "
            "is intended to be entirely reconstructible from those ADFs"
        ),
        "standalone_task_policy": (
            "has_father, adjacent_to_red, outdegree_1, and outdegree_2 receive "
            "an empty ADF bank because no earlier benchmark predicate is a "
            "sound sufficient decomposition of their target"
        ),
        "candidate_regularization": (
            "task-tied candidate banks use the fixed historical five-candidate "
            "reference width, so removing irrelevant candidates does not "
            "increase per-candidate OFF pressure"
        ),
        "training_metric_state": (
            "accuracy and losses describe the accepted post-optimizer, "
            "post-prune topology; pre-update values are separately labelled"
        ),
        "selection_policy": (
            "per task, run the complete fixed independent seed grid; select "
            "the earliest VERIFIED_EQUIVALENT seed as the downstream ADF, "
            "while dependent seeds additionally require preregistered ADF "
            "causal use"
        ),
        "attempt_policy": (
            "run every fixed seed 0..9 in order; a VERIFIED_EQUIVALENT proof "
            "may stop phase 2 only "
            "within that seed, never the remaining seed grid; report the full "
            "10-seed proof rate"
        ),
        "validation_feedback": False,
        "test_feedback": False,
        "held_out_reporting": (
            "validation and test are evaluated exactly once on the checkpoint "
            "selected by the preregistered proof-stage rule"
        ),
        "adf_causal_feedback": "dependent tasks only; fixed before the run",
        "adf_candidate_policy": (
            "every VERIFIED_EQUIVALENT task is promoted; each later attempt "
            "uses only its ordered task-specific allowlist, including the "
            "transitive support needed to evaluate a composite ADF, frozen by "
            "task, checkpoint filename, and checkpoint SHA256"
        ),
        "cross_protocol_bootstrap_policy": (
            "forbidden: every task and auxiliary ADF must be trained, extracted, "
            "and verified inside this protocol root from the fixed execution "
            "order; no checkpoint, source copy, or proof receipt from an older "
            "protocol may be imported"
        ),
        "adf_acceptance_policy": (
            "task.dependencies is a required subset of the frozen candidate "
            "prefix; causal and structural evidence apply only to that subset, "
            "and disabling each required ADF individually must add errors on "
            "the fixed independent adf_causal split"
        ),
        "attempt_decision": (
            "VERIFIED_EQUIVALENT for ordinary tasks; VERIFIED_EQUIVALENT plus "
            "preregistered ADF causal use for dependent tasks"
        ),
        "attempt_limit": len(config.attempt_seeds),
        "attempt_trajectory": (
            "one deterministic optimiser/RNG trajectory with exact-incumbent "
            "rollback and preregistered boundary/compression proof checkpoints"
        ),
        "proof_feedback": (
            "phase1 proof gives the preregistered stop-or-resume-phase2 branch; "
            "compression checkpoints and the phase2 boundary use a fixed "
            "TRAIN/structure-only order; validation and test are reporting outputs"
        ),
        "retry_policy": (
            "infrastructure failure pauses the driver and resumes the same attempt"
        ),
    }
    protocol_path = destination / "protocol_manifest.json"
    if protocol_path.exists():
        if _read_json(protocol_path) != _json_value(protocol):
            raise RuntimeError(
                "artifact root belongs to a different protocol or source tree; "
                "use a new protocol version"
            )
    else:
        _atomic_json(protocol_path, protocol)

    environment_path = destination / "environment.json"
    if environment_path.exists():
        recorded_environment = _read_json(environment_path)
        if not isinstance(recorded_environment, Mapping):
            raise RuntimeError("environment record is invalid")
        _validate_environment_record(recorded_environment, device)
    else:
        _atomic_json(environment_path, _environment_record(torch.device(device)))
    _reject_cross_protocol_artifacts(destination)
    return destination


@contextlib.contextmanager
def _runner_lock(root: Path, *, filename: str = "runner.lock") -> Iterator[None]:
    """Hold a process lock without leaving a stale-lock failure after a crash."""

    if filename not in {"runner.lock", "paper_driver.lock"}:
        raise ValueError("unsupported runner lock filename")
    lock_path = root / filename
    stream = lock_path.open("a+b")
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("another official experiment runner is active") from error
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("another official experiment runner is active") from error
        yield
    finally:
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()


def _attempt_dir(root: Path, task: TaskSpec, attempt_seed: int) -> Path:
    return root / "runs" / task.name / f"attempt_{attempt_seed:02d}"


def _stage_dir(attempt_dir: Path, stage: str) -> Path:
    if stage not in ("phase1", "phase2", "semantic_compression") and not (
        stage.startswith("compression_") and stage[12:].isdigit()
    ):
        raise ValueError(
            "stage must be phase1, phase2, semantic_compression, "
            "or compression_<step>"
        )
    return attempt_dir / "stages" / stage


def _global_library(root: Path) -> ADFLibrary:
    return ADFLibrary(root / "adf_library")


def _model_factory(
    record: Mapping[str, Any],
    dependencies: Sequence[ADFEntry],
) -> NeuralNP4G:
    return NeuralNP4G.from_config(record["build_config"], dependencies)


def _held_out_evaluation(
    *,
    task: TaskSpec,
    formal_seed: int,
    model: NeuralNP4G,
    dependencies: Sequence[ADFEntry],
    device: torch.device,
    config: OfficialConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifests: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    for split_name in ("validation", "test"):
        split = build_split(task, split_name, formal_seed, config=config)
        manifests[split_name] = split.manifest.as_dict()
        evaluations[split_name] = evaluate_split(
            model,
            split,
            learned_adfs=dependencies,
            device=device,
        )
    manifests["falsification"] = _falsification_manifest(
        task, formal_seed, config
    )
    return manifests, evaluations


def _falsification_manifest(
    task: TaskSpec,
    formal_seed: int,
    config: OfficialConfig,
) -> dict[str, Any]:
    return {
        "protocol_version": config.protocol_version,
        "task": task.name,
        "formal_seed": formal_seed,
        "data_seed": derive_seed(
            formal_seed,
            task.index,
            "falsification_data",
            config=config,
        ),
        "cells": [asdict(cell) for cell in config.falsification_cells],
        "kind": "post-extraction formula falsification",
    }


def _global_dependencies_are_certified(root: Path, task: TaskSpec) -> bool:
    """Whether the task's required subset exists in the certified library."""

    if not task.dependencies:
        return True
    records = _global_library(root).metadata()
    by_name = {str(record.get("task")): record for record in records}
    return all(
        dependency in by_name
        and by_name[dependency].get("verification_status") == VERIFIED_STATUS
        for dependency in task.dependencies
    )


def _load_global_candidates(
    root: Path,
    task: TaskSpec,
    device: torch.device,
) -> tuple[ADFEntry, ...]:
    """Load the preregistered earlier certified candidates for one task."""

    loaded = _global_library(root).load_entries(
        domain=task.domain,
        model_factory=_model_factory,
        device=device,
    )
    if task.name in FORMAL_EXECUTION_ORDER:
        position = FORMAL_EXECUTION_ORDER.index(task.name)
        allowed = set(FORMAL_EXECUTION_ORDER[:position])
        loaded = [entry for entry in loaded if entry.task in allowed]
    else:
        # Research API tasks are outside the formal staged curriculum.  Keep
        # their historical certified-prefix behaviour without allowing self
        # reuse when a task was already promoted before a crash/replay.
        loaded = [entry for entry in loaded if entry.task != task.name]

    allowlist = fixed_adf_candidate_allowlist(task)
    if allowlist is not None:
        by_name = {entry.task: entry for entry in loaded}
        unavailable = [name for name in allowlist if name not in by_name]
        if unavailable:
            raise RuntimeError(
                f"fixed ADF candidates are unavailable for {task.name}: {unavailable}"
            )
        loaded = [by_name[name] for name in allowlist]

    names = tuple(entry.task for entry in loaded)
    missing = [name for name in task.dependencies if name not in names]
    if missing:
        raise RuntimeError(
            f"required ADFs are absent from the certified prefix for {task.name}: {missing}"
        )
    return tuple(loaded)


# Compatibility name for research callers; its semantics are now the complete
# candidate prefix, not the task's exact required subset.
_load_global_dependencies = _load_global_candidates


def _adf_candidate_identity(entries: Sequence[ADFEntry]) -> list[dict[str, str]]:
    identity: list[dict[str, str]] = []
    for entry in entries:
        if not entry.checkpoint or not entry.checkpoint_sha256:
            raise RuntimeError(f"certified ADF '{entry.task}' has no immutable checkpoint identity")
        identity.append(
            {
                "task": entry.task,
                "checkpoint": entry.checkpoint,
                "checkpoint_sha256": entry.checkpoint_sha256,
            }
        )
    return identity


def _freeze_adf_candidate_manifest(
    attempt_dir: Path,
    *,
    task: TaskSpec,
    attempt_seed: int,
    entries: Sequence[ADFEntry],
    config: OfficialConfig,
) -> dict[str, Any]:
    """Create or validate one attempt's immutable ordered candidate manifest."""

    payload = {
        "protocol_version": config.protocol_version,
        "task": task.name,
        "attempt_seed": attempt_seed,
        "candidates": _adf_candidate_identity(entries),
    }
    path = attempt_dir / "adf_candidate_manifest.json"
    if path.exists():
        if _read_json(path) != payload:
            raise RuntimeError("ADF candidate manifest changed while resuming an attempt")
    else:
        _atomic_json(path, payload)
    return payload


def _attempt_terminal_result(
    attempt_dir: Path,
    task: TaskSpec,
    attempt_seed: int,
    config: OfficialConfig,
) -> dict[str, Any] | None:
    path = attempt_dir / "attempt_result.json"
    if not path.exists():
        return None
    result = _read_json(path)
    expected = {
        "protocol_version": config.protocol_version,
        "task": task.name,
        "attempt_seed": attempt_seed,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"attempt result has mismatched {key}")
    # Infrastructure failures are deliberately retryable.  train_model resumes
    # its saved optimizer/RNG/checkpoint state in this same directory.
    if result.get("status") == "INFRA_FAILURE":
        return None
    expected_index = config.attempt_seeds.index(attempt_seed)
    if "attempt_index" in result and result["attempt_index"] != expected_index:
        raise RuntimeError("attempt result has mismatched attempt_index")
    return result


def _next_attempt_seed(
    root: Path,
    task: TaskSpec,
    config: OfficialConfig,
) -> int | None:
    task_result = root / "runs" / task.name / "task_result.json"
    if task_result.exists():
        result = _read_json(task_result)
        if (
            result.get("status") != "BLOCKED_BY_UPSTREAM_ADF"
            or not _global_dependencies_are_certified(root, task)
        ):
            return None
    for attempt_seed in config.attempt_seeds:
        path = _attempt_dir(root, task, attempt_seed) / "attempt_result.json"
        if not path.exists():
            return attempt_seed
        attempt_status = _read_json(path).get("status")
        if attempt_status == "INFRA_FAILURE":
            # Infrastructure failures resume this exact attempt.  Every other
            # terminal outcome advances to the next seed, including a proof.
            return attempt_seed
    return None


def _write_blocked_task_result(
    root: Path,
    task: TaskSpec,
    config: OfficialConfig,
) -> dict[str, Any]:
    result = {
        "protocol_version": config.protocol_version,
        "task": task.name,
        "status": "BLOCKED_BY_UPSTREAM_ADF",
        "verification_status": "NOT_PROVED",
        "dependencies": list(task.dependencies),
        "successful_attempt": None,
        "successful_attempt_index": None,
        "attempts_completed": 0,
        "attempts_to_verified": None,
        "verified_attempts": 0,
        "verified_seeds": [],
        "proof_rate": None,
        "attempt_limit": len(config.attempt_seeds),
        "reason": "upstream ADF exhausted all fixed attempts without certification",
        "completed_at": _utc_now(),
    }
    task_dir = root / "runs" / task.name
    task_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(task_dir / "task_result.json", result)
    return result


def _dependency_task_is_exhausted(
    root: Path,
    task: TaskSpec,
    config: OfficialConfig,
) -> bool:
    """Whether every missing dependency has a terminal exhausted result."""

    certified = {str(record.get("task")) for record in _global_library(root).metadata()}
    for dependency in task.dependencies:
        if dependency in certified:
            continue
        path = root / "runs" / dependency / "task_result.json"
        if not path.exists():
            return False
        result = _read_json(path)
        if result.get("protocol_version") != config.protocol_version:
            raise RuntimeError("upstream task result belongs to another protocol")
        if result.get("status") not in (
            "ATTEMPTS_EXHAUSTED",
            "BLOCKED_BY_UPSTREAM_ADF",
        ):
            return False
    return True


def _promote_global_verified_adf(
    *,
    root: Path,
    task: TaskSpec,
    attempt_seed: int,
    model: NeuralNP4G,
    checkpoint_sha256: str,
    extraction: Mapping[str, Any],
) -> None:
    candidate_names = tuple(entry.task for entry in model.adf_input_specs)
    library = _global_library(root)
    if library.has(task.name):
        record = next(item for item in library.metadata() if item["task"] == task.name)
        if tuple(record.get("dependencies", ())) != candidate_names:
            raise RuntimeError("global certified ADF candidate prefix changed")
        metadata = record.get("extra_metadata", {})
        recorded_seed = int(metadata.get("attempt_seed"))
        if recorded_seed > attempt_seed:
            raise RuntimeError("global certified ADF is not the earliest verified seed")
        if recorded_seed == attempt_seed:
            recorded_hash = metadata.get("checkpoint_sha256")
            if recorded_hash != checkpoint_sha256:
                raise RuntimeError("global certified ADF checkpoint changed")
        # Later verified seeds are evidence for the proof rate, but the first
        # verified seed remains the immutable downstream ADF.
        return
    saved = library.save_verified(
        task=task.name,
        domain=task.domain,
        arity=task.target_arity,
        model=model,
        input_dims=tuple(model.input_dims),
        build_config=model.build_config(),
        dependencies=candidate_names,
        verification_status=VERIFIED_STATUS,
        extra_metadata={
            "attempt_seed": attempt_seed,
            "checkpoint_sha256": checkpoint_sha256,
            "formula": extraction.get("formula_raw_closed"),
        },
    )
    if not saved:
        raise RuntimeError("verified global ADF was not stored")


def _extract_attempt(
    *,
    root: Path,
    attempt_dir: Path,
    task: TaskSpec,
    attempt_seed: int,
    stage: str,
    checkpoint_steps: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    candidate_manifest: Path,
    config: OfficialConfig,
) -> dict[str, Any]:
    """Run the isolated proof worker and classify its fixed resource limits."""

    report_path = attempt_dir / "program_extraction.json"
    identity = {
        "protocol_version": config.protocol_version,
        "task": task.name,
        "attempt_seed": attempt_seed,
        "formal_seed": attempt_seed,
        "stage": stage,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_sha256": checkpoint_sha256,
        "adf_candidate_manifest_sha256": _file_sha256(candidate_manifest),
    }
    if report_path.exists():
        cached = _read_json(report_path)
        if not isinstance(cached, dict) or any(
            cached.get(key) != value for key, value in identity.items()
        ):
            raise RuntimeError("cached extraction does not match its stage checkpoint")
        return cached

    worker_report_path = attempt_dir / "program_extraction.worker.json"
    worker_report_path.unlink(missing_ok=True)
    log_path = attempt_dir / "extraction.log"
    manifest = _read_json(candidate_manifest)
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("candidates"), list):
        raise RuntimeError("ADF candidate manifest has no ordered candidate list")
    circuit_count = 1 + len(manifest["candidates"])
    command = [
        sys.executable,
        "-m",
        "official_neural_np4g.extraction_worker",
        "--task",
        task.name,
        "--formal-seed",
        str(attempt_seed),
        "--checkpoint",
        str(checkpoint),
        "--library-root",
        str(_global_library(root).root),
        "--candidate-manifest",
        str(candidate_manifest),
        "--report-out",
        str(worker_report_path),
        "--memory-gb",
        str(config.extraction_memory_gb),
        "--minimum-available-gb",
        str(config.extraction_min_available_gb),
        "--time-seconds",
        str(config.extraction_time_seconds),
    ]
    try:
        with log_path.open("ab") as log:
            completed = subprocess.run(
                command,
                cwd=PACKAGE_DIR.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=circuit_count * config.extraction_time_seconds + 90,
            )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "extraction worker exceeded its parent process deadline"
        ) from error

    raw: Any = (
        _read_json(worker_report_path) if worker_report_path.exists() else None
    )
    # The fixed in-worker guards hard-exit only after writing one explicit
    # resource-limit record.  This is a terminal proof failure, unlike a crash,
    # missing report, malformed report, or parent timeout (all infrastructure).
    if completed.returncode in (96, 97):
        if isinstance(raw, list) and raw and isinstance(raw[-1], dict):
            extraction = dict(raw[-1])
        elif isinstance(raw, dict):
            extraction = dict(raw)
        else:
            raise RuntimeError("proof resource guard exited without a valid report")
        reason = str(extraction.get("reason", ""))
        expected = "time cap:" if completed.returncode == 96 else "memory cap:"
        if extraction.get("status") != "EXTRACTION_FAILED" or not reason.startswith(expected):
            raise RuntimeError("proof resource guard report is inconsistent")
        extraction["resource_limit"] = "time" if completed.returncode == 96 else "memory"
    else:
        if not isinstance(raw, dict) or not raw:
            raise RuntimeError(
                f"extraction worker produced no valid report (exit={completed.returncode})"
            )
        extraction = dict(raw)
        if completed.returncode != 0:
            raise RuntimeError(
                f"extraction worker exited abnormally with code {completed.returncode}: "
                f"{extraction.get('reason', '')}"
            )

    extraction["worker_return_code"] = completed.returncode
    extraction.update(identity)
    _atomic_json(report_path, extraction)
    worker_report_path.unlink(missing_ok=True)
    return extraction


def _assess_checkpoint(
    *,
    root: Path,
    attempt_dir: Path,
    stage: str,
    task: TaskSpec,
    attempt_seed: int,
    training: TrainingResult,
    expected_steps: int,
    train_split: Any,
    dependencies: Sequence[ADFEntry],
    device: torch.device,
    config: OfficialConfig,
) -> tuple[dict[str, Any], NeuralNP4G, dict[str, Any]]:
    """Evaluate and prove one immutable preregistered stage checkpoint.

    Stage records make the phase-1 decision durable.  A resumed runner reuses
    a complete assessment only after validating its protocol, task, seed,
    checkpoint digest, and scheduled step boundary.
    """

    stage_dir = _stage_dir(attempt_dir, stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_result_path = stage_dir / "stage_result.json"
    extraction_path = stage_dir / "program_extraction.json"
    candidate_manifest_path = attempt_dir / "adf_candidate_manifest.json"
    candidate_manifest_sha256 = _file_sha256(candidate_manifest_path)

    frozen_model, checkpoint_record = load_checkpoint(
        training.checkpoint_path,
        dependencies,
        device=device,
    )
    if training.stage != stage:
        raise RuntimeError(
            f"training result stage mismatch: {training.stage!r} != {stage!r}"
        )
    if checkpoint_record.get("steps") != expected_steps:
        raise RuntimeError(
            f"{stage} checkpoint step mismatch: "
            f"{checkpoint_record.get('steps')} != {expected_steps}"
        )
    if _file_sha256(training.checkpoint_path) != training.checkpoint_sha256:
        raise RuntimeError(f"{stage} checkpoint changed before assessment")

    if stage_result_path.exists():
        cached = _read_json(stage_result_path)
        expected = {
            "protocol_version": config.protocol_version,
            "task": task.name,
            "attempt_seed": attempt_seed,
            "stage": stage,
            "steps": expected_steps,
            "checkpoint_sha256": training.checkpoint_sha256,
            "adf_candidate_manifest_sha256": candidate_manifest_sha256,
        }
        if any(cached.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"cached {stage} assessment does not match checkpoint")
        extraction = _read_json(extraction_path)
        if not isinstance(extraction, dict) or not extraction:
            raise RuntimeError(f"cached {stage} assessment is missing extraction")
        artifact_hashes = cached.get("artifact_sha256", {})
        if not isinstance(artifact_hashes, Mapping):
            raise RuntimeError(f"cached {stage} assessment has no artifact hashes")
        for name, digest in artifact_hashes.items():
            artifact_path = stage_dir / str(name)
            if not artifact_path.is_file() or _file_sha256(artifact_path) != digest:
                raise RuntimeError(f"cached {stage} artifact changed: {name}")
        return cached, frozen_model, extraction

    adf_causal_split = build_split(
        task,
        "adf_causal",
        attempt_seed,
        config=config,
    )
    proof_manifests = {
        "train": train_split.manifest.as_dict(),
        "adf_causal": adf_causal_split.manifest.as_dict(),
        "falsification": _falsification_manifest(task, attempt_seed, config),
    }
    _atomic_json(stage_dir / "proof_split_manifests.json", proof_manifests)

    causal: dict[str, Any] = {
        "applicable": False,
        "reason": "task has no required learned ADF dependency",
        "available_candidate_ids": [entry.candidate_id for entry in dependencies],
        "required_adfs": [],
    }
    if task.dependencies:
        causal = evaluate_adf_causal_ablation(
            frozen_model,
            adf_causal_split,
            learned_adfs=dependencies,
            required_adfs=task.dependencies,
            device=device,
        )
    _atomic_json(stage_dir / "adf_causal_ablation.json", causal)

    extraction = _extract_attempt(
        root=root,
        attempt_dir=stage_dir,
        task=task,
        attempt_seed=attempt_seed,
        stage=stage,
        checkpoint_steps=expected_steps,
        checkpoint=training.checkpoint_path,
        checkpoint_sha256=training.checkpoint_sha256,
        candidate_manifest=candidate_manifest_path,
        config=config,
    )
    verification = extraction.get("verification", {})
    if not isinstance(verification, Mapping):
        verification = {}
    engine_status = str(verification.get("status", "NOT_PROVED"))
    # The extractor can revoke a logical verdict after an independent
    # falsification check. Its final top-level status is authoritative.
    proof_status = str(extraction.get("status", "NOT_PROVED"))
    acceptance_failure: str | None = None
    if proof_status == VERIFIED_STATUS:
        if engine_status != VERIFIED_STATUS:
            proof_status = "NOT_PROVED"
            acceptance_failure = "final extraction success lacks a successful equivalence verdict"
        elif extraction.get("verification_contract") != vn.metadata():
            proof_status = "NOT_PROVED"
            acceptance_failure = "extraction has no proof under the current numerical contract"
    adf_atoms = set(extraction.get("pre_substitution_adf_atoms", []))
    causal_use = bool(causal.get("causal_use", False))
    adf_atom_evidence = (
        None if not task.dependencies else set(task.dependencies).issubset(adf_atoms)
    )
    adf_gates_on = (
        None
        if not task.dependencies
        else bool(causal.get("all_gates_on", False))
    )
    adf_structural_evidence = (
        None
        if not task.dependencies
        else bool(adf_atom_evidence and adf_gates_on)
    )
    adf_evidence = (
        not task.dependencies or bool(adf_structural_evidence and causal_use)
    )
    status = proof_status
    if proof_status == VERIFIED_STATUS and not adf_evidence:
        status = "ADF_NOT_CAUSALLY_USED"
    if status != VERIFIED_STATUS and status not in TERMINAL_PROOF_FAILURES:
        raise RuntimeError(f"unexpected proof worker status: {status!r}")

    if status == VERIFIED_STATUS:
        reason = "strict equivalence and required evidence verified"
    elif status == "ADF_NOT_CAUSALLY_USED":
        reason = (
            "strict equivalence passed, but the preregistered ADF-use "
            "requirements did not"
        )
    else:
        reason = acceptance_failure or str(extraction.get("reason", verification.get("reason", "")))
        if not reason:
            reason = (
                "verification produced a counterexample witness"
                if verification.get("witness") is not None
                else f"proof result: {proof_status}"
            )

    result = {
        "protocol_version": config.protocol_version,
        "task": task.name,
        "attempt_seed": attempt_seed,
        "stage": stage,
        "steps": expected_steps,
        "status": status,
        "verification_status": proof_status if status == proof_status else "NOT_PROVED",
        "proof_engine_status": engine_status,
        "reason": reason,
        "verification_witness": verification.get("witness"),
        "resource_limit": extraction.get("resource_limit"),
        "checkpoint_sha256": training.checkpoint_sha256,
        "checkpoint_path": str(training.checkpoint_path),
        "adf_candidate_manifest_sha256": candidate_manifest_sha256,
        "adf_candidates": _adf_candidate_identity(dependencies),
        "adf_evidence": adf_evidence,
        "adf_atom_evidence": adf_atom_evidence,
        "adf_gates_on": adf_gates_on,
        "adf_structural_evidence": adf_structural_evidence,
        "adf_causal_use": causal_use,
        "adf_causal_split_sha256": adf_causal_split.manifest.sha256(),
        "adf_causal_normal_wrong": causal.get("normal_wrong"),
        "adf_causal_disabled_wrong": causal.get("adf_disabled_wrong"),
        "adf_causal_additional_errors": causal.get("additional_errors"),
        "pre_substitution_adf_atoms": sorted(adf_atoms),
        "artifact_sha256": {
            name: _file_sha256(stage_dir / name)
            for name in (
                "proof_split_manifests.json",
                "adf_causal_ablation.json",
                "program_extraction.json",
            )
        },
        "assessed_at": _utc_now(),
    }
    _atomic_json(stage_result_path, result)
    return result, frozen_model, extraction


def _report_selected_checkpoint(
    *,
    attempt_dir: Path,
    task: TaskSpec,
    attempt_seed: int,
    stage: str,
    steps: int,
    checkpoint_sha256: str,
    model: NeuralNP4G,
    train_split: Any,
    dependencies: Sequence[ADFEntry],
    device: torch.device,
    config: OfficialConfig,
) -> dict[str, Any]:
    """Evaluate held-out splits once, after the proof-only stage decision."""

    result_path = attempt_dir / "reporting_result.json"
    identity = {
        "protocol_version": config.protocol_version,
        "task": task.name,
        "attempt_seed": attempt_seed,
        "selected_stage": stage,
        "checkpoint_steps": steps,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if result_path.exists():
        cached = _read_json(result_path)
        if not isinstance(cached, dict) or any(
            cached.get(key) != value for key, value in identity.items()
        ):
            raise RuntimeError("cached reporting result does not match selection")
        artifact_hashes = cached.get("artifact_sha256", {})
        if not isinstance(artifact_hashes, Mapping):
            raise RuntimeError("cached reporting result has no artifact hashes")
        for name, digest in artifact_hashes.items():
            path = attempt_dir / str(name)
            if not path.is_file() or _file_sha256(path) != digest:
                raise RuntimeError(f"cached reporting artifact changed: {name}")
        return cached

    manifests, evaluations = _held_out_evaluation(
        task=task,
        formal_seed=attempt_seed,
        model=model,
        dependencies=dependencies,
        device=device,
        config=config,
    )
    adf_causal_split = build_split(
        task, "adf_causal", attempt_seed, config=config
    )
    manifests["train"] = train_split.manifest.as_dict()
    manifests["adf_causal"] = adf_causal_split.manifest.as_dict()
    causal = _read_json(
        _stage_dir(attempt_dir, stage) / "adf_causal_ablation.json"
    )

    artifacts = {
        "split_manifests.json": manifests,
        "evaluation.json": evaluations,
        "adf_causal_ablation.json": causal,
    }
    for name, value in artifacts.items():
        path = attempt_dir / name
        if path.exists():
            if _read_json(path) != _json_value(value):
                raise RuntimeError(f"selected reporting artifact changed: {name}")
        else:
            _atomic_json(path, value)

    result = {
        **identity,
        "validation_zero_error": bool(
            evaluations["validation"]["total"]["zero_error"]
        ),
        "test_zero_error": bool(evaluations["test"]["total"]["zero_error"]),
        "validation_wrong": int(evaluations["validation"]["total"]["wrong"]),
        "test_wrong": int(evaluations["test"]["total"]["wrong"]),
        "artifact_sha256": {
            name: _file_sha256(attempt_dir / name) for name in artifacts
        },
        "reported_at": _utc_now(),
    }
    _atomic_json(result_path, result)
    return result


def run_attempt(
    task: str | TaskSpec,
    attempt_seed: int,
    root: Path | str,
    device: torch.device | str,
    *,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> dict[str, Any]:
    """Run one fixed initialisation through ordered proof checkpoints."""

    spec = get_task(task)
    if isinstance(spec, EmpiricalTaskSpec):
        raise ValueError(
            "empirical label tasks require the custom task runner, "
            "not formal proof execution"
        )
    if attempt_seed not in config.attempt_seeds:
        raise ValueError(f"attempt seed {attempt_seed} is outside the fixed protocol")
    attempt_index = config.attempt_seeds.index(attempt_seed)
    destination = Path(root).resolve()
    _reject_cross_protocol_artifacts(destination)
    attempt_dir = _attempt_dir(destination, spec, attempt_seed)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    if not _global_dependencies_are_certified(destination, spec):
        raise RuntimeError(
            f"global certified ADF dependencies are unavailable for {spec.name}"
        )
    dependencies = _load_global_candidates(destination, spec, device)
    candidate_manifest = _freeze_adf_candidate_manifest(
        attempt_dir,
        task=spec,
        attempt_seed=attempt_seed,
        entries=dependencies,
        config=config,
    )
    candidate_manifest_path = attempt_dir / "adf_candidate_manifest.json"
    candidate_manifest_sha256 = _file_sha256(candidate_manifest_path)

    terminal = _attempt_terminal_result(attempt_dir, spec, attempt_seed, config)
    if terminal is not None:
        return terminal

    started = time.monotonic()
    try:
        train_split = build_split(spec, "train", attempt_seed, config=config)
        model = build_model(spec, attempt_seed, dependencies, config=config)
        phase1_training = train_model(
            model,
            train_split,
            attempt_dir,
            device=device,
            config=config,
            stage="phase1",
        )
        phase1, frozen_model, extraction = _assess_checkpoint(
            root=destination,
            attempt_dir=attempt_dir,
            stage="phase1",
            task=spec,
            attempt_seed=attempt_seed,
            training=phase1_training,
            expected_steps=config.phase1_steps,
            train_split=train_split,
            dependencies=dependencies,
            device=device,
            config=config,
        )
        stage_results = {"phase1": phase1}
        selected = phase1
        selected_training = phase1_training
        selected_model = frozen_model
        selected_extraction = extraction

        if phase1["status"] != VERIFIED_STATUS:
            phase2_training = train_model(
                model,
                train_split,
                attempt_dir,
                device=device,
                config=config,
                stage="phase2",
            )
            compression_verified = False
            compression_results = compression_training_results(attempt_dir)
            for compression_training in compression_results[
                : config.compression_proof_max_checkpoints
            ]:
                compression, compression_model, compression_extraction = (
                    _assess_checkpoint(
                        root=destination,
                        attempt_dir=attempt_dir,
                        task=spec,
                        stage=compression_training.stage,
                        attempt_seed=attempt_seed,
                        training=compression_training,
                        expected_steps=compression_training.steps,
                        train_split=train_split,
                        dependencies=dependencies,
                        device=device,
                        config=config,
                    )
                )
                stage_results[compression_training.stage] = compression
                if compression["status"] == VERIFIED_STATUS:
                    selected = compression
                    selected_training = compression_training
                    selected_model = compression_model
                    selected_extraction = compression_extraction
                    compression_verified = True
                    break

            if not compression_verified:
                phase2, phase2_model, phase2_extraction = _assess_checkpoint(
                    root=destination,
                    attempt_dir=attempt_dir,
                    task=spec,
                    stage="phase2",
                    attempt_seed=attempt_seed,
                    training=phase2_training,
                    expected_steps=config.total_steps,
                    train_split=train_split,
                    dependencies=dependencies,
                    device=device,
                    config=config,
                )
                stage_results["phase2"] = phase2
                selected = phase2
                selected_training = phase2_training
                selected_model = phase2_model
                selected_extraction = phase2_extraction
                if phase2["status"] != VERIFIED_STATUS:
                    semantic_training = semantic_compression_result(
                        phase2_model,
                        train_split,
                        attempt_dir,
                        device=device,
                        config=config,
                    )
                    if semantic_training is not None:
                        semantic, semantic_model, semantic_extraction = (
                            _assess_checkpoint(
                                root=destination,
                                attempt_dir=attempt_dir,
                                task=spec,
                                stage=semantic_training.stage,
                                attempt_seed=attempt_seed,
                                training=semantic_training,
                                expected_steps=config.total_steps,
                                train_split=train_split,
                                dependencies=dependencies,
                                device=device,
                                config=config,
                            )
                        )
                        stage_results[semantic_training.stage] = semantic
                        selected = semantic
                        selected_training = semantic_training
                        selected_model = semantic_model
                        selected_extraction = semantic_extraction

        selected_pointer = {
            "protocol_version": config.protocol_version,
            "task": spec.name,
            "attempt_seed": attempt_seed,
            "selected_stage": selected["stage"],
            "checkpoint_steps": selected["steps"],
            "checkpoint_path": str(
                selected_training.checkpoint_path.relative_to(attempt_dir)
            ),
            "checkpoint_sha256": selected_training.checkpoint_sha256,
            "adf_candidate_manifest_sha256": candidate_manifest_sha256,
            "stage_result_sha256": _file_sha256(
                _stage_dir(attempt_dir, selected["stage"]) / "stage_result.json"
            ),
        }
        pointer_path = attempt_dir / "selected_checkpoint.json"
        if pointer_path.exists():
            if _read_json(pointer_path) != selected_pointer:
                raise RuntimeError("selected checkpoint pointer changed on resume")
        else:
            _atomic_json(pointer_path, selected_pointer)

        reporting = _report_selected_checkpoint(
            attempt_dir=attempt_dir,
            task=spec,
            attempt_seed=attempt_seed,
            stage=selected["stage"],
            steps=selected["steps"],
            checkpoint_sha256=selected_training.checkpoint_sha256,
            model=selected_model,
            train_split=train_split,
            dependencies=dependencies,
            device=device,
            config=config,
        )

        if selected["status"] == VERIFIED_STATUS:
            _promote_global_verified_adf(
                root=destination,
                task=spec,
                attempt_seed=attempt_seed,
                model=selected_model,
                checkpoint_sha256=selected_training.checkpoint_sha256,
                extraction=selected_extraction,
            )

        result = {
            "protocol_version": config.protocol_version,
            "task": spec.name,
            "task_index": spec.index,
            "domain": spec.domain,
            "attempt_seed": attempt_seed,
            "attempt_index": attempt_index,
            # Kept inside checkpoint/data records for compatibility; it means
            # this attempt's independent seed, not a second experiment axis.
            "formal_seed": attempt_seed,
            "dependencies": list(spec.dependencies),
            "adf_candidates": candidate_manifest["candidates"],
            "adf_candidate_manifest_sha256": candidate_manifest_sha256,
            "status": selected["status"],
            "verification_status": selected["verification_status"],
            "proof_engine_status": selected["proof_engine_status"],
            "reason": selected["reason"],
            "verification_witness": selected.get("verification_witness"),
            "error_type": None,
            "resource_limit": selected.get("resource_limit"),
            "trained": True,
            "selected_stage": selected["stage"],
            "selected_steps": selected["steps"],
            "stage_statuses": {
                name: stage_result["status"]
                for name, stage_result in stage_results.items()
            },
            "stage_checkpoints": {
                name: {
                    "steps": stage_result["steps"],
                    "checkpoint_sha256": stage_result["checkpoint_sha256"],
                }
                for name, stage_result in stage_results.items()
            },
            "checkpoint_sha256": selected_training.checkpoint_sha256,
            "checkpoint_path": str(selected_training.checkpoint_path),
            "validation_zero_error": reporting["validation_zero_error"],
            "test_zero_error": reporting["test_zero_error"],
            "validation_wrong": reporting["validation_wrong"],
            "test_wrong": reporting["test_wrong"],
            "adf_evidence": selected["adf_evidence"],
            "adf_atom_evidence": selected["adf_atom_evidence"],
            "adf_gates_on": selected["adf_gates_on"],
            "adf_structural_evidence": selected["adf_structural_evidence"],
            "adf_causal_use": selected["adf_causal_use"],
            "adf_causal_split_sha256": selected["adf_causal_split_sha256"],
            "adf_causal_normal_wrong": selected["adf_causal_normal_wrong"],
            "adf_causal_disabled_wrong": selected["adf_causal_disabled_wrong"],
            "adf_causal_additional_errors": selected["adf_causal_additional_errors"],
            "pre_substitution_adf_atoms": selected["pre_substitution_adf_atoms"],
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": _utc_now(),
        }
        _atomic_json(attempt_dir / "attempt_result.json", result)
        return result
    except Exception as error:
        failure = {
            "protocol_version": config.protocol_version,
            "task": spec.name,
            "attempt_seed": attempt_seed,
            "attempt_index": attempt_index,
            "formal_seed": attempt_seed,
            "dependencies": list(spec.dependencies),
            "adf_candidates": candidate_manifest["candidates"],
            "adf_candidate_manifest_sha256": candidate_manifest_sha256,
            "status": "INFRA_FAILURE",
            "error_type": type(error).__name__,
            "reason": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": _utc_now(),
        }
        _atomic_json(attempt_dir / "failure.json", failure)
        _atomic_json(attempt_dir / "attempt_result.json", failure)
        return failure


def run_task(
    task: str | TaskSpec,
    root: Path | str,
    device: torch.device | str,
    *,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> dict[str, Any]:
    """Run the complete fixed seed grid and select its earliest strict proof."""

    spec = get_task(task)
    destination = Path(root).resolve()
    _reject_cross_protocol_artifacts(destination)
    task_dir = destination / "runs" / spec.name
    task_dir.mkdir(parents=True, exist_ok=True)
    result_path = task_dir / "task_result.json"
    if result_path.exists():
        existing = _read_json(result_path)
        if existing.get("protocol_version") != config.protocol_version:
            raise RuntimeError("task result belongs to another protocol")
        if existing.get("status") != "BLOCKED_BY_UPSTREAM_ADF":
            return existing
        # A blocked result is reconsidered if a dependency was certified by a
        # later repair/replay.  It is returned only while dependencies remain
        # terminally exhausted.
        if not _global_dependencies_are_certified(destination, spec):
            return existing

    if not _global_dependencies_are_certified(destination, spec):
        if _dependency_task_is_exhausted(destination, spec, config):
            return _write_blocked_task_result(destination, spec, config)
        raise RuntimeError(
            f"upstream ADF task is not yet terminal for {spec.name}; "
            "run it first in the fixed execution order"
        )

    completed: list[dict[str, Any]] = []
    for attempt_seed in config.attempt_seeds:
        attempt_index = config.attempt_seeds.index(attempt_seed)
        _atomic_json(
            destination / "driver_state.json",
            {
                "status": "RUNNING",
                "task": spec.name,
                "current_attempt_seed": attempt_seed,
                "current_attempt_index": attempt_index,
                "attempt_limit": len(config.attempt_seeds),
                "updated_at": _utc_now(),
            },
        )
        attempt = run_attempt(
            spec,
            attempt_seed,
            destination,
            device,
            config=config,
        )
        if attempt.get("status") == "INFRA_FAILURE":
            # Do not advance to another initialisation.  Re-running the driver
            # resumes this exact attempt via train_model's saved state.
            raise RuntimeError(
                f"infrastructure failure in {spec.name} attempt {attempt_seed}: "
                f"{attempt.get('reason', '')}"
            )
        completed.append(attempt)

    verified = [item for item in completed if item.get("status") == VERIFIED_STATUS]
    if verified:
        first = verified[0]
        successful_attempt = int(first["attempt_seed"])
        successful_index = config.attempt_seeds.index(successful_attempt)
        result = {
            "protocol_version": config.protocol_version,
            "task": spec.name,
            "status": VERIFIED_STATUS,
            "verification_status": VERIFIED_STATUS,
            "successful_attempt": successful_attempt,
            "successful_attempt_index": successful_index,
            "attempts_completed": len(completed),
            "attempts_to_verified": successful_index + 1,
            "verified_attempts": len(verified),
            "verified_seeds": [int(item["attempt_seed"]) for item in verified],
            "proof_rate": len(verified) / len(completed),
            "attempt_limit": len(config.attempt_seeds),
            "attempt_statuses": [item["status"] for item in completed],
            "checkpoint_sha256": first.get("checkpoint_sha256"),
            "selected_stage": first.get("selected_stage"),
            "selected_steps": first.get("selected_steps"),
            "reason": (
                f"{len(verified)}/{len(completed)} fixed seeds verified; "
                f"earliest seed {successful_attempt} selected for downstream ADF"
            ),
            "error_type": None,
            "completed_at": _utc_now(),
        }
        _atomic_json(result_path, result)
        return result

    result = {
        "protocol_version": config.protocol_version,
        "task": spec.name,
        "status": "ATTEMPTS_EXHAUSTED",
        "verification_status": "NOT_PROVED",
        "successful_attempt": None,
        "successful_attempt_index": None,
        "attempts_completed": len(completed),
        "attempts_to_verified": None,
        "verified_attempts": 0,
        "verified_seeds": [],
        "proof_rate": 0.0,
        "attempt_limit": len(config.attempt_seeds),
        "attempt_statuses": [item["status"] for item in completed],
        "reason": "all fixed attempts ended without a verified result",
        "completed_at": _utc_now(),
    }
    _atomic_json(result_path, result)
    return result


def build_summary(
    root: Path | str,
    *,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> dict[str, Any]:
    """Aggregate official task outcomes and every attempted initialisation."""

    destination = Path(root).resolve()
    tasks: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    terminal_tasks = 0
    verified_tasks = 0
    for task in FORMAL_TASKS:
        task_path = destination / "runs" / task.name / "task_result.json"
        result = _read_json(task_path) if task_path.exists() else {}
        status = str(result.get("status", "PENDING"))
        if status != "PENDING":
            terminal_tasks += 1
        if status == VERIFIED_STATUS:
            verified_tasks += 1

        attempts: list[dict[str, Any]] = []
        for attempt_seed in config.attempt_seeds:
            attempt_dir = _attempt_dir(destination, task, attempt_seed)
            path = attempt_dir / "attempt_result.json"
            attempt = _read_json(path) if path.exists() else {}
            if attempt:
                attempt_status = str(attempt.get("status", "PENDING"))
            elif status == "BLOCKED_BY_UPSTREAM_ADF":
                attempt_status = "NOT_RUN_BLOCKED"
            else:
                attempt_status = "PENDING"
            selected_stage = attempt.get("selected_stage")
            report_dir = (
                _stage_dir(attempt_dir, str(selected_stage))
                if selected_stage
                and (
                    selected_stage in ("phase1", "phase2", "semantic_compression")
                    or str(selected_stage).startswith("compression_")
                )
                else attempt_dir
            )
            extraction_path = report_dir / "program_extraction.json"
            extraction = (
                _read_json(extraction_path) if extraction_path.exists() else {}
            )
            if not isinstance(extraction, dict):
                extraction = {}
            verification = extraction.get("verification", {})
            if not isinstance(verification, Mapping):
                verification = {}
            causal_path = attempt_dir / "adf_causal_ablation.json"
            causal = _read_json(causal_path) if causal_path.exists() else {}
            if not isinstance(causal, dict):
                causal = {}
            evaluation_path = attempt_dir / "evaluation.json"
            evaluations = (
                _read_json(evaluation_path) if evaluation_path.exists() else {}
            )
            if not isinstance(evaluations, dict):
                evaluations = {}

            def _reported_wrong(split_name: str) -> Any:
                split = evaluations.get(split_name, {})
                if not isinstance(split, Mapping):
                    return ""
                total = split.get("total", {})
                if not isinstance(total, Mapping):
                    return ""
                return total.get("wrong", "")

            attempt_record = {
                "attempt_index": config.attempt_seeds.index(attempt_seed),
                "attempt_seed": attempt_seed,
                "status": attempt_status,
                "selected_stage": selected_stage or "",
                "selected_steps": attempt.get("selected_steps", ""),
                "phase1_status": attempt.get("stage_statuses", {}).get(
                    "phase1", ""
                ),
                "phase2_status": attempt.get("stage_statuses", {}).get(
                    "phase2", ""
                ),
                "reason": attempt.get(
                    "reason",
                    verification.get("reason", extraction.get("reason", "")),
                ),
                "error_type": attempt.get("error_type", ""),
                "verification_status": attempt.get("verification_status", ""),
                "proof_engine_status": attempt.get(
                    "proof_engine_status",
                    verification.get("status", extraction.get("status", "")),
                ),
                "verification_witness": attempt.get(
                    "verification_witness", verification.get("witness")
                ),
                "resource_limit": extraction.get("resource_limit", ""),
                "validation_wrong": attempt.get(
                    "validation_wrong", _reported_wrong("validation")
                ),
                "test_wrong": attempt.get("test_wrong", _reported_wrong("test")),
                "adf_evidence": attempt.get("adf_evidence", ""),
                "adf_atom_evidence": attempt.get("adf_atom_evidence", ""),
                "adf_gates_on": attempt.get(
                    "adf_gates_on", causal.get("all_gates_on", "")
                ),
                "adf_structural_evidence": attempt.get(
                    "adf_structural_evidence", ""
                ),
                "adf_causal_use": attempt.get(
                    "adf_causal_use", causal.get("causal_use", "")
                ),
                "adf_causal_split_sha256": attempt.get(
                    "adf_causal_split_sha256", ""
                ),
                "adf_causal_normal_wrong": attempt.get(
                    "adf_causal_normal_wrong", causal.get("normal_wrong", "")
                ),
                "adf_causal_disabled_wrong": attempt.get(
                    "adf_causal_disabled_wrong",
                    causal.get("adf_disabled_wrong", ""),
                ),
                "adf_causal_additional_errors": attempt.get(
                    "adf_causal_additional_errors",
                    causal.get("additional_errors", ""),
                ),
                "checkpoint_sha256": attempt.get("checkpoint_sha256", ""),
            }
            attempts.append(attempt_record)
            rows.append(
                {
                    "task": task.name,
                    "task_status": status,
                    "task_reason": result.get("reason", ""),
                    "task_error_type": result.get("error_type", ""),
                    "attempt_index": attempt_record["attempt_index"],
                    "attempt_seed": attempt_record["attempt_seed"],
                    **{
                        f"attempt_{key}": value
                        for key, value in attempt_record.items()
                        if key not in ("attempt_index", "attempt_seed")
                    },
                }
            )
        tasks[task.name] = {
            "status": status,
            "reason": result.get("reason", ""),
            "error_type": result.get("error_type", ""),
            "verification_status": result.get("verification_status", ""),
            "successful_attempt": result.get("successful_attempt"),
            "successful_attempt_index": result.get("successful_attempt_index"),
            "selected_stage": result.get("selected_stage"),
            "selected_steps": result.get("selected_steps"),
            "attempts_completed": result.get(
                "attempts_completed",
                sum(
                    item["status"] == VERIFIED_STATUS
                    or item["status"] in TERMINAL_PROOF_FAILURES
                    for item in attempts
                ),
            ),
            "attempts_to_verified": result.get("attempts_to_verified"),
            "verified_attempts": result.get("verified_attempts", 0),
            "verified_seeds": result.get("verified_seeds", []),
            "proof_rate": result.get("proof_rate"),
            "attempt_limit": len(config.attempt_seeds),
            "attempts": attempts,
        }

    summary = {
        "protocol_version": config.protocol_version,
        "updated_at": _utc_now(),
        "planned_tasks": len(FORMAL_TASKS),
        "terminal_tasks": terminal_tasks,
        "verified_tasks": verified_tasks,
        "complete": terminal_tasks == len(FORMAL_TASKS),
        "tasks": tasks,
        "adf_library": _global_library(destination).metadata(),
    }
    _atomic_json(destination / "summary.json", summary)
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=list(rows[0]) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)
    _atomic_write(destination / "summary.csv", csv_stream.getvalue().encode("utf-8"))

    lines = [
        f"# {config.protocol_version} progress",
        "",
        f"Terminal tasks: **{terminal_tasks} / {len(FORMAL_TASKS)}**",
        "",
        "| task | status | selected seed | verified seeds | proof rate | attempts completed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in FORMAL_TASKS:
        item = tasks[task.name]
        successful = item["successful_attempt"]
        lines.append(
            f"| {task.name} | {item['status']} | "
            f"{'' if successful is None else successful} | "
            f"{item['verified_attempts']} | "
            f"{'' if item['proof_rate'] is None else item['proof_rate']} | "
            f"{item['attempts_completed']} |"
        )
    _atomic_write(destination / "summary.md", ("\n".join(lines) + "\n").encode("utf-8"))
    return summary


def run_official_battery(
    root: Path | str = DEFAULT_ARTIFACT_ROOT,
    *,
    device: torch.device | str,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> dict[str, Any]:
    """Run or resume the complete same-protocol curriculum.

    This is the programmatic entry point used by the paper-wide driver.  It
    always begins at the first formal task, resumes only artifacts bound to the
    same source/config/environment manifests, and never accepts an imported
    checkpoint or ADF.
    """

    device = torch.device(device)
    root = initialise_artifact_root(
        root,
        device=device,
        config=config,
    )
    _reject_cross_protocol_artifacts(root)
    tasks = {task.name: task for task in FORMAL_TASKS}
    with _runner_lock(root):
        logger.info("Official protocol=%s device=%s root=%s", config.protocol_version, device, root)
        try:
            for task_name in FORMAL_EXECUTION_ORDER:
                task = tasks[task_name]
                current_attempt_seed = _next_attempt_seed(
                    root, task, config
                )
                _atomic_json(
                    root / "driver_state.json",
                    {
                        "status": "RUNNING",
                        "task": task.name,
                        "attempt_seeds": list(config.attempt_seeds),
                        "current_attempt_seed": current_attempt_seed,
                        "current_attempt_index": (
                            None
                            if current_attempt_seed is None
                            else config.attempt_seeds.index(
                                current_attempt_seed
                            )
                        ),
                        "started_at": _utc_now(),
                    },
                )
                logger.info("Starting task=%s", task.name)
                result = run_task(
                    task,
                    root,
                    device,
                    config=config,
                )
                logger.info("Task=%s status=%s attempt=%s", task.name,
                            result.get("status"), result.get("successful_attempt"))
                build_summary(root, config=config)
        except KeyboardInterrupt:
            logger.warning("Official experiment interrupted; saved state can be resumed")
            state_path = root / "driver_state.json"
            previous = _read_json(state_path) if state_path.exists() else {}
            _atomic_json(
                state_path,
                {**previous, "status": "INTERRUPTED", "updated_at": _utc_now()},
            )
            build_summary(root, config=config)
            raise
        except Exception as error:
            logger.exception("Official experiment failed")
            state_path = root / "driver_state.json"
            previous = _read_json(state_path) if state_path.exists() else {}
            _atomic_json(
                state_path,
                {
                    **previous,
                    "status": "INFRA_FAILURE",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "updated_at": _utc_now(),
                },
            )
            build_summary(root, config=config)
            raise
        summary = build_summary(root, config=config)
        _atomic_json(
            root / "driver_state.json",
            {
                "status": "COMPLETE",
                "completed_at": _utc_now(),
                "terminal_tasks": summary["terminal_tasks"],
            },
        )
        logger.info("Official experiment battery complete")
        return summary


def main() -> None:
    configure_logging("official-experiments", log_dir=DEFAULT_ARTIFACT_ROOT / "logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_official_battery(
        DEFAULT_ARTIFACT_ROOT,
        device=device,
        config=OFFICIAL_CONFIG,
    )


def _command_entrypoint(argv: Sequence[str] | None = None) -> None:
    """Keep the argument-free benchmark; route explicit task requests safely."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        from .task_execution import command
        command(arguments)
    else:
        main()


if __name__ == "__main__":
    _command_entrypoint()
