"""Isolated program-extraction worker used by the formal runner.

Extraction can be combinatorial.  Running it in a child process lets the
formal runner enforce the preregistered memory and time budgets without
risking the training trajectory or the remaining experiment cells.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from .adf import ADFEntry, ADFLibrary
from .config import derive_seed
from .extract import (
    _start_memory_guard,
    _start_time_guard,
    extract_task,
)
from .model import NeuralNP4G
from .runtime import configure_logging
from .tasks import get_task
from .train import load_checkpoint


logger = logging.getLogger("official_neural_np4g.extraction_worker")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    deadline = time.monotonic() + 30.0
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def _model_factory(
    record: Mapping[str, Any],
    dependencies: list[ADFEntry],
) -> NeuralNP4G:
    return NeuralNP4G.from_config(record["build_config"], dependencies)


def run_worker(
    *,
    task_name: str,
    formal_seed: int,
    checkpoint: Path,
    library_root: Path,
    candidate_manifest: Path,
    report_out: Path,
    memory_gb: float,
    minimum_available_gb: float,
    time_seconds: float,
) -> dict[str, Any]:
    """Extract and verify one frozen checkpoint on CPU."""

    logger.info(
        "Extraction started task=%s seed=%s checkpoint=%s memory_gb=%s time_seconds=%s",
        task_name, formal_seed, checkpoint, memory_gb, time_seconds,
    )
    task = get_task(task_name)
    sanity_seed = derive_seed(
        formal_seed,
        task.index,
        "test_data",
        91,
    )
    falsification_seed = derive_seed(
        formal_seed,
        task.index,
        "falsification_data",
        91,
    )
    reports: list[dict[str, Any]] = []
    current_task: list[str | None] = [task.name]
    # Setup has its own bounded parent deadline; each circuit below starts a
    # fresh proof budget, including its independent finite checks.
    task_started: list[float | None] = [None]
    _start_memory_guard(
        cap_gb=memory_gb,
        min_avail_gb=minimum_available_gb,
        report_out=report_out,
        reports=reports,
        current_task=current_task,
    )
    _start_time_guard(
        per_task_sec=time_seconds,
        report_out=report_out,
        reports=reports,
        current_task=current_task,
        task_started=task_started,
    )

    library = ADFLibrary(library_root)
    available = library.load_entries(
        domain=task.domain,
        model_factory=_model_factory,
        device="cpu",
    )
    by_name = {entry.task: entry for entry in available}
    try:
        manifest = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read ADF candidate manifest: {candidate_manifest}") from error
    if not isinstance(manifest, Mapping):
        raise RuntimeError("ADF candidate manifest must be a JSON object")
    if manifest.get("task") != task.name or manifest.get("attempt_seed") != formal_seed:
        raise RuntimeError("ADF candidate manifest belongs to another task or attempt")
    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_candidates, list):
        raise RuntimeError("ADF candidate manifest has no ordered candidate list")

    dependencies: list[ADFEntry] = []
    seen: set[str] = set()
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            raise RuntimeError("ADF candidate manifest contains a malformed entry")
        dependency = str(raw_candidate.get("task", ""))
        expected_checkpoint = raw_candidate.get("checkpoint")
        expected_hash = raw_candidate.get("checkpoint_sha256")
        if not dependency or dependency in seen:
            raise RuntimeError("ADF candidate manifest names must be unique and non-empty")
        seen.add(dependency)
        if dependency not in by_name:
            raise RuntimeError(
                f"certified ADF '{dependency}' is missing for task '{task.name}'"
            )
        entry = by_name[dependency]
        if entry.checkpoint != expected_checkpoint:
            raise RuntimeError(
                f"certified ADF '{dependency}' checkpoint differs from the frozen manifest"
            )
        if entry.checkpoint_sha256 != expected_hash:
            raise RuntimeError(
                f"certified ADF '{dependency}' hash differs from the frozen manifest"
            )
        dependencies.append(entry)
    missing_required = [name for name in task.dependencies if name not in seen]
    if missing_required:
        raise RuntimeError(
            f"required ADFs are missing from the frozen candidate manifest: {missing_required}"
        )

    model, checkpoint_record = load_checkpoint(
        checkpoint,
        dependencies,
        device="cpu",
    )
    entry = ADFEntry(
        task=task.name,
        domain=task.domain,
        arity=task.target_arity,
        input_dims=tuple(model.input_dims),
        model=model,
        dependencies=tuple(dependency.task for dependency in dependencies),
    )

    formulas: dict[str, dict[str, Any]] = {}
    dependency_reports: dict[str, dict[str, Any]] = {}
    for dependency in dependencies:
        dependency_spec = get_task(dependency.task)
        dependency_sanity_seed = derive_seed(
            formal_seed,
            dependency_spec.index,
            "test_data",
            91,
        )
        dependency_falsification_seed = derive_seed(
            formal_seed,
            dependency_spec.index,
            "falsification_data",
            91,
        )
        upstream = [by_name[name] for name in dependency.dependencies]
        current_task[0] = dependency.task
        task_started[0] = time.time()
        dependency_report = extract_task(
            dependency.task,
            dependency,
            upstream,
            formulas,
            nominal=NeuralNP4G.SYN_STATE_LO,
            do_sanity=True,
            sanity_seed=dependency_sanity_seed,
            falsification_seed=dependency_falsification_seed,
        )
        expression = dependency_report.pop("expr_raw_obj", None)
        dependency_reports[dependency.task] = dependency_report
        if dependency_report.get("status") != "VERIFIED_EQUIVALENT" or expression is None:
            raise RuntimeError(
                f"certified ADF '{dependency.task}' could not be replayed: "
                f"{dependency_report.get('status')}"
            )
        formulas[dependency.task] = {
            "expr_raw": expression,
            "report": dependency_report,
        }

    current_task[0] = task.name
    task_started[0] = time.time()
    report = extract_task(
        task.name,
        entry,
        dependencies,
        formulas,
        nominal=NeuralNP4G.SYN_STATE_LO,
        do_sanity=True,
        sanity_seed=sanity_seed,
        falsification_seed=falsification_seed,
    )
    report.pop("expr_raw_obj", None)
    report["checkpoint_path"] = str(checkpoint.resolve())
    report["formal_seed"] = formal_seed
    report["sanity_seed"] = sanity_seed
    report["falsification_seed"] = falsification_seed
    report["dependency_extractions"] = dependency_reports
    report["pre_substitution_adf_atoms"] = [
        dependency.task
        for dependency in dependencies
        if f"ADF:{dependency.task}" in str(report.get("formula_extracted", ""))
        and model.candidate_weight(dependency.candidate_id).detach().cpu().item()
        == 1.0
    ]
    report["learned_adf_gate_bits"] = {
        dependency.task: float(
            model.candidate_weight(dependency.candidate_id).detach().cpu().item()
        )
        for dependency in dependencies
    }
    report["adf_candidates"] = [
        {
            "task": dependency.task,
            "checkpoint": dependency.checkpoint,
            "checkpoint_sha256": dependency.checkpoint_sha256,
        }
        for dependency in dependencies
    ]
    _write_report(report_out, report)
    logger.info("Extraction completed task=%s status=%s report=%s", task.name, report.get("status"), report_out)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True)
    parser.add_argument("--formal-seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--memory-gb", type=float, required=True)
    parser.add_argument("--minimum-available-gb", type=float, required=True)
    parser.add_argument("--time-seconds", type=float, required=True)
    arguments = parser.parse_args()
    configure_logging("extraction-worker", log_dir=arguments.report_out.parent / "logs")
    try:
        run_worker(
            task_name=arguments.task,
            formal_seed=arguments.formal_seed,
            checkpoint=arguments.checkpoint,
            library_root=arguments.library_root,
            candidate_manifest=arguments.candidate_manifest,
            report_out=arguments.report_out,
            memory_gb=arguments.memory_gb,
            minimum_available_gb=arguments.minimum_available_gb,
            time_seconds=arguments.time_seconds,
        )
    except BaseException as error:
        logger.exception("Extraction failed task=%s checkpoint=%s", arguments.task, arguments.checkpoint)
        _write_report(
            arguments.report_out,
            {
                "task": arguments.task,
                "status": "EXTRACTION_FAILED",
                "reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
