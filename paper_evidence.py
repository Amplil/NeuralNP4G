"""Build the paper's tables from one complete, current protocol root only.

The builder never consults historical artifact directories and cannot fill
missing rows from an older run. A final report is emitted only after all
fourteen NLM-10 curriculum proof obligations and all matched dense-NLM runs
are terminal.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any, Mapping

from .config import OFFICIAL_CONFIG, OfficialConfig
from .nlm_baseline import MATCHED_BASELINE_NAME, NLM_PAPER_SPECS, protocol_record
from .run_official_experiments import (
    TERMINAL_PROOF_FAILURES,
    VERIFIED_STATUS,
    _atomic_json,
    _atomic_write,
    _file_sha256,
    _json_bytes,
    _read_json,
    _reject_cross_protocol_artifacts,
    _runtime_source_hashes,
)
from .tasks import AUXILIARY_TASK_NAMES, PRIMARY_TASK_NAMES


BASELINE_SEEDS = OFFICIAL_CONFIG.attempt_seeds
BASELINE_SIZES = tuple(cell.objects for cell in OFFICIAL_CONFIG.test_cells)
BASELINE_RELATIVE_PATH = Path("baselines") / "dense_nlm" / "nlm_matched_evaluation.json"
EVIDENCE_JSON = "paper_evidence.json"
EVIDENCE_CSV = "paper_primary_results.csv"
EVIDENCE_MD = "paper_evidence.md"


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is missing or malformed")
    return value


def _load_complete_official_summary(
    root: Path,
    config: OfficialConfig,
) -> Mapping[str, Any]:
    path = root / "summary.json"
    if not path.is_file():
        raise RuntimeError("official summary.json is missing; run the full curriculum first")
    summary = _require_mapping(_read_json(path), "official summary")
    if summary.get("protocol_version") != config.protocol_version:
        raise RuntimeError("official summary belongs to a different protocol")
    expected = len(PRIMARY_TASK_NAMES) + len(AUXILIARY_TASK_NAMES)
    if summary.get("complete") is not True or int(summary.get("terminal_tasks", -1)) != expected:
        raise RuntimeError(
            f"official curriculum is incomplete: expected {expected} terminal obligations"
        )
    tasks = _require_mapping(summary.get("tasks"), "official task summary")
    expected_names = set(PRIMARY_TASK_NAMES) | set(AUXILIARY_TASK_NAMES)
    if set(tasks) != expected_names:
        raise RuntimeError("official summary task set is not the fixed NLM-10 curriculum")
    terminal = set(TERMINAL_PROOF_FAILURES) | {VERIFIED_STATUS}
    for name in (*PRIMARY_TASK_NAMES, *AUXILIARY_TASK_NAMES):
        task = _require_mapping(tasks[name], f"official task {name}")
        status = task.get("status")
        attempts = task.get("attempts")
        if (
            not isinstance(attempts, list)
            or len(attempts) != len(config.attempt_seeds)
            or any(not isinstance(attempt, Mapping) for attempt in attempts)
        ):
            raise RuntimeError(f"official task {name} does not contain the full seed grid")
        seeds = tuple(int(attempt.get("attempt_seed", -1)) for attempt in attempts)
        if seeds != config.attempt_seeds:
            raise RuntimeError(f"official task {name} seed order differs from the contract")
        if status == "BLOCKED_BY_UPSTREAM_ADF":
            if int(task.get("attempts_completed", -1)) != 0 or any(
                attempt.get("status") != "NOT_RUN_BLOCKED" for attempt in attempts
            ):
                raise RuntimeError(f"blocked official task {name} has attempted seeds")
            continue
        if status not in (VERIFIED_STATUS, "ATTEMPTS_EXHAUSTED"):
            raise RuntimeError(f"official task {name} has a non-terminal status")
        if int(task.get("attempts_completed", -1)) != len(config.attempt_seeds) or any(
            attempt.get("status") not in terminal for attempt in attempts
        ):
            raise RuntimeError(f"official task {name} did not complete every fixed seed")
        verified_seeds = [
            int(attempt["attempt_seed"])
            for attempt in attempts
            if attempt.get("status") == VERIFIED_STATUS
        ]
        if int(task.get("verified_attempts", -1)) != len(verified_seeds):
            raise RuntimeError(f"official task {name} has inconsistent proof counts")
        if list(task.get("verified_seeds", ())) != verified_seeds:
            raise RuntimeError(f"official task {name} has inconsistent verified seeds")
        expected_rate = len(verified_seeds) / len(config.attempt_seeds)
        if float(task.get("proof_rate", -1.0)) != expected_rate:
            raise RuntimeError(f"official task {name} has an inconsistent proof rate")
        selected = verified_seeds[0] if verified_seeds else None
        if task.get("successful_attempt") != selected:
            raise RuntimeError(f"official task {name} did not select its earliest proof")
        if (status == VERIFIED_STATUS) != bool(verified_seeds):
            raise RuntimeError(f"official task {name} status disagrees with its proofs")
    return summary


def _validate_root_contract(root: Path, config: OfficialConfig) -> None:
    _reject_cross_protocol_artifacts(root)
    source_path = root / "source_hashes.json"
    protocol_path = root / "protocol_manifest.json"
    paper_protocol_path = root / "paper_protocol_manifest.json"
    for path in (source_path, protocol_path, paper_protocol_path):
        if not path.is_file():
            raise RuntimeError(f"required root contract is missing: {path.name}")
    source_hashes = _require_mapping(_read_json(source_path), "source hash manifest")
    if not source_hashes or any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        for name, digest in source_hashes.items()
    ):
        raise RuntimeError("source hash manifest is malformed")
    if dict(source_hashes) != _runtime_source_hashes():
        raise RuntimeError("source hash manifest does not match canonical source files")
    protocol = _require_mapping(_read_json(protocol_path), "official protocol manifest")
    expected_source_digest = hashlib.sha256(_json_bytes(source_hashes)).hexdigest()
    if (
        protocol.get("protocol_version") != config.protocol_version
        or protocol.get("source_digest") != expected_source_digest
    ):
        raise RuntimeError("official protocol is not bound to this source hash manifest")
    paper_protocol = _require_mapping(
        _read_json(paper_protocol_path), "paper protocol manifest"
    )
    if (
        paper_protocol.get("protocol_version") != config.protocol_version
        or paper_protocol.get("historical_artifacts_allowed") is not False
        or paper_protocol.get("cross_protocol_bootstrap_allowed") is not False
    ):
        raise RuntimeError("paper protocol does not enforce current-source-only evidence")


def _load_complete_baseline(
    root: Path,
    config: OfficialConfig,
) -> Mapping[str, Any]:
    path = root / BASELINE_RELATIVE_PATH
    if not path.is_file():
        raise RuntimeError("dense NLM report is missing; run the fixed comparison first")
    report = _require_mapping(_read_json(path), "dense NLM report")
    protocol = _require_mapping(report.get("protocol"), "dense NLM protocol")
    if protocol.get("name") != MATCHED_BASELINE_NAME:
        raise RuntimeError("dense NLM report has the wrong protocol identity")
    hidden_dim = int(protocol.get("hidden_dim", -1))
    if hidden_dim <= 0 or dict(protocol) != protocol_record(
        config, hidden_dim=hidden_dim
    ):
        raise RuntimeError("dense NLM protocol record differs from the fixed contract")
    if protocol.get("protocol_version") != config.protocol_version:
        raise RuntimeError("dense NLM report belongs to another official protocol")
    if tuple(protocol.get("tasks", ())) != PRIMARY_TASK_NAMES:
        raise RuntimeError("dense NLM task order is not the fixed NLM-10 order")
    if tuple(int(value) for value in protocol.get("seeds", ())) != config.attempt_seeds:
        raise RuntimeError("dense NLM seeds differ from the Neural NP4G seed grid")
    training = _require_mapping(protocol.get("training"), "dense NLM training protocol")
    expected_training = {
        "phase1_steps": config.phase1_steps,
        "phase2_steps": config.phase2_steps,
        "total_steps": config.total_steps,
        "batch_size": config.batch_size,
        "phase1_learning_rate": config.phase1_learning_rate,
        "phase2_learning_rate": config.phase2_learning_rate,
        "gradient_clip_norm": config.gradient_clip_norm,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise RuntimeError("dense NLM optimisation grid differs from Neural NP4G")
    if (
        protocol.get("shared_raw_split_contract") is not True
        or protocol.get("shared_minibatch_contract") is not True
        or protocol.get("learned_adfs_supplied") is not False
        or protocol.get("selection_feedback") is not False
    ):
        raise RuntimeError("dense NLM shared-protocol contract is incomplete")
    results = _require_mapping(report.get("results"), "dense NLM results")
    if report.get("complete") is not True or set(results) != set(PRIMARY_TASK_NAMES):
        raise RuntimeError("dense NLM comparison is incomplete")
    for task in PRIMARY_TASK_NAMES:
        rows = results.get(task)
        if not isinstance(rows, list) or tuple(
            int(row.get("seed", -1)) for row in rows
        ) != config.attempt_seeds:
            raise RuntimeError(f"dense NLM task {task} does not contain the fixed seed grid")
        for row in rows:
            manifests = _require_mapping(
                row.get("split_manifests"), f"dense NLM {task} split manifests"
            )
            for split in ("train", "validation", "test", "falsification"):
                manifest = _require_mapping(
                    manifests.get(split), f"dense NLM {task} {split} manifest"
                )
                if (
                    manifest.get("protocol_version") != config.protocol_version
                    or manifest.get("task") != task
                    or int(manifest.get("formal_seed", -1)) != int(row["seed"])
                    or not isinstance(manifest.get("sha256"), str)
                ):
                    raise RuntimeError(
                        f"dense NLM {task} seed {row['seed']} has a mismatched {split} split"
                    )
    return report


def _primary_rows(tasks: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in PRIMARY_TASK_NAMES:
        task = _require_mapping(tasks[name], f"task {name}")
        rows.append(
            {
                "task": name,
                "status": task.get("status"),
                "attempts_completed": task.get("attempts_completed"),
                "verified_attempts": task.get("verified_attempts"),
                "proof_rate": task.get("proof_rate"),
                "successful_attempt": task.get("successful_attempt"),
                "selected_stage": task.get("selected_stage"),
                "selected_steps": task.get("selected_steps"),
            }
        )
    return rows


def _validate_cross_model_split_identity(
    root: Path,
    tasks: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> int:
    """Prove value-level raw split identity for every attempted primary seed."""

    baseline_results = _require_mapping(
        baseline.get("results"), "dense NLM results"
    )
    matches = 0
    for task_name in PRIMARY_TASK_NAMES:
        task = _require_mapping(tasks[task_name], f"task {task_name}")
        nlm_by_seed = {
            int(row["seed"]): row for row in baseline_results[task_name]
        }
        attempts = task.get("attempts")
        if not isinstance(attempts, list):
            raise RuntimeError(f"task {task_name} has no attempt records")
        for attempt in attempts:
            if not isinstance(attempt, Mapping) or attempt.get("status") not in (
                set(TERMINAL_PROOF_FAILURES) | {VERIFIED_STATUS}
            ):
                continue
            seed = int(attempt["attempt_seed"])
            manifest_path = (
                root
                / "runs"
                / task_name
                / f"attempt_{seed:02d}"
                / "split_manifests.json"
            )
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"Neural NP4G split manifest is missing for {task_name} seed {seed}"
                )
            np4g_manifests = _require_mapping(
                _read_json(manifest_path),
                f"Neural NP4G {task_name} seed {seed} split manifests",
            )
            nlm_manifests = _require_mapping(
                nlm_by_seed[seed].get("split_manifests"),
                f"dense NLM {task_name} seed {seed} split manifests",
            )
            for split in ("train", "validation", "test"):
                np4g = _require_mapping(
                    np4g_manifests.get(split), f"Neural NP4G {split} manifest"
                )
                nlm = _require_mapping(
                    nlm_manifests.get(split), f"dense NLM {split} manifest"
                )
                if np4g.get("sha256") != nlm.get("sha256"):
                    raise RuntimeError(
                        f"raw split mismatch for {task_name} seed {seed} {split}"
                    )
                matches += 1
    return matches


def _terminal_attempts(tasks: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    terminal = set(TERMINAL_PROOF_FAILURES) | {VERIFIED_STATUS}
    for task_name in (*PRIMARY_TASK_NAMES, *AUXILIARY_TASK_NAMES):
        task = _require_mapping(tasks[task_name], f"task {task_name}")
        attempts = task.get("attempts")
        if not isinstance(attempts, list):
            raise RuntimeError(f"task {task_name} has no attempt records")
        for attempt in attempts:
            if not isinstance(attempt, Mapping) or attempt.get("status") not in terminal:
                continue
            records.append({"task": task_name, **dict(attempt)})
    return records


def _finite_perfect_rejections(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        if attempt.get("status") == VERIFIED_STATUS:
            continue
        if attempt.get("validation_wrong") != 0 or attempt.get("test_wrong") != 0:
            continue
        rows.append(
            {
                "task": attempt["task"],
                "attempt_seed": attempt.get("attempt_seed"),
                "selected_stage": attempt.get("selected_stage"),
                "selected_steps": attempt.get("selected_steps"),
                "proof_status": attempt.get("status"),
                "proof_engine_status": attempt.get("proof_engine_status"),
                "validation_wrong": 0,
                "test_wrong": 0,
                "reason": attempt.get("reason", ""),
                "checkpoint_sha256": attempt.get("checkpoint_sha256", ""),
            }
        )
    return rows


def _nlm_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = _require_mapping(report["results"], "dense NLM results")
    rows: list[dict[str, Any]] = []
    for task in PRIMARY_TASK_NAMES:
        runs = results[task]
        passing = 0
        wrong = 0
        cells = 0
        failed_sizes = 0
        for run in runs:
            evaluations = _require_mapping(run.get("evaluations"), f"NLM {task} run")
            test = _require_mapping(evaluations.get("test"), f"NLM {task} test")
            per_size = _require_mapping(test.get("per_size"), f"NLM {task} test sizes")
            run_pass = True
            for size in BASELINE_SIZES:
                cell = _require_mapping(per_size.get(f"n{size}"), f"NLM {task} n={size}")
                if cell.get("error") or cell.get("complete") is not True:
                    failed_sizes += 1
                    run_pass = False
                cell_wrong = int(cell.get("wrong", 0))
                wrong += cell_wrong
                cells += int(cell.get("cells", 0))
                if cell_wrong:
                    run_pass = False
            passing += int(run_pass)
        rows.append(
            {
                "task": task,
                "passing_runs": passing,
                "runs": len(BASELINE_SEEDS),
                "wrong": wrong,
                "cells": cells,
                "failed_or_oom_sizes": failed_sizes,
                "proof_object": False,
                "architecture_fidelity": NLM_PAPER_SPECS[
                    task
                ].local_architecture_fidelity,
            }
        )
    return rows


def build_paper_evidence(
    root: Path | str,
    *,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> dict[str, Any]:
    """Validate and write the final, single-root paper evidence bundle."""

    destination = Path(root).resolve()
    _validate_root_contract(destination, config)
    summary = _load_complete_official_summary(destination, config)
    baseline = _load_complete_baseline(destination, config)
    tasks = _require_mapping(summary["tasks"], "official task summary")
    shared_split_hash_matches = _validate_cross_model_split_identity(
        destination, tasks, baseline
    )
    primary = _primary_rows(tasks)
    auxiliary = [
        {
            "task": name,
            "status": _require_mapping(tasks[name], f"task {name}").get("status"),
            "attempts_completed": _require_mapping(tasks[name], f"task {name}").get(
                "attempts_completed"
            ),
            "verified_attempts": _require_mapping(tasks[name], f"task {name}").get(
                "verified_attempts"
            ),
            "proof_rate": _require_mapping(tasks[name], f"task {name}").get(
                "proof_rate"
            ),
        }
        for name in AUXILIARY_TASK_NAMES
    ]
    attempts = _terminal_attempts(tasks)
    finite_perfect = _finite_perfect_rejections(attempts)
    nlm = _nlm_rows(baseline)
    verified_attempts = sum(item.get("status") == VERIFIED_STATUS for item in attempts)
    extracted_attempts = sum(
        item.get("proof_engine_status") not in ("", None, "EXTRACTION_FAILED")
        for item in attempts
    )
    evidence = {
        "schema_version": 1,
        "protocol_version": config.protocol_version,
        "ready_for_paper": True,
        "historical_evidence_used": False,
        "source_integrity": "SHA-256 manifest only; canonical source files are not copied",
        "inputs": {
            "source_hashes": "source_hashes.json",
            "source_hashes_sha256": _file_sha256(destination / "source_hashes.json"),
            "protocol_manifest": "protocol_manifest.json",
            "protocol_manifest_sha256": _file_sha256(
                destination / "protocol_manifest.json"
            ),
            "paper_protocol_manifest": "paper_protocol_manifest.json",
            "paper_protocol_manifest_sha256": _file_sha256(
                destination / "paper_protocol_manifest.json"
            ),
            "official_summary": "summary.json",
            "official_summary_sha256": _file_sha256(destination / "summary.json"),
            "dense_nlm_report": BASELINE_RELATIVE_PATH.as_posix(),
            "dense_nlm_report_sha256": _file_sha256(destination / BASELINE_RELATIVE_PATH),
        },
        "official": {
            "proof_obligations": len(PRIMARY_TASK_NAMES) + len(AUXILIARY_TASK_NAMES),
            "terminal_obligations": summary["terminal_tasks"],
            "verified_obligations": summary["verified_tasks"],
            "primary_tasks": primary,
            "auxiliary_tasks": auxiliary,
            "attempt_pipeline": {
                "terminal_attempts": len(attempts),
                "finite_perfect_attempts": sum(
                    item.get("validation_wrong") == 0 and item.get("test_wrong") == 0
                    for item in attempts
                ),
                "extracted_attempts": extracted_attempts,
                "verified_attempts": verified_attempts,
            },
            "finite_perfect_rejections": finite_perfect,
        },
        "dense_nlm": {
            "scope": (
                "matched finite evaluation only; no NP4G extraction or equivalence proof; "
                "2-OutDegree is a labelled breadth-3 approximation of the paper's breadth-4 model"
            ),
            "shared_protocol": {
                "raw_splits": True,
                "seed_grid": list(config.attempt_seeds),
                "minibatch_schedule": True,
                "maximum_steps": config.total_steps,
                "held_out_feedback": False,
                "verified_split_hash_matches": shared_split_hash_matches,
            },
            "rows": nlm,
        },
        "published_nlm_reference": {
            "source": "https://arxiv.org/abs/1904.11694",
            "not_a_local_reproduction": True,
            "rows": [
                {
                    "task": name,
                    "reported_accuracy": list(spec.paper_accuracy),
                    "test_objects": list(spec.paper_test_objects),
                    "depth": spec.depth,
                    "breadth": spec.breadth,
                    "residual": spec.residual,
                    "training_examples": spec.paper_examples,
                }
                for name, spec in NLM_PAPER_SPECS.items()
            ],
        },
    }
    _atomic_json(destination / EVIDENCE_JSON, evidence)

    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=list(primary[0]))
    writer.writeheader()
    writer.writerows(primary)
    _atomic_write(destination / EVIDENCE_CSV, csv_stream.getvalue().encode("utf-8"))

    lines = [
        f"# {config.protocol_version} paper evidence",
        "",
        "This report uses no historical run and copies no source into artifacts.",
        "",
        "## Primary tasks",
        "",
        "| task | status | verified seeds | proof rate | selected seed | selected checkpoint |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in primary:
        checkpoint = (
            "--"
            if not row["selected_stage"]
            else f"{row['selected_stage']} / {row['selected_steps']}"
        )
        lines.append(
            f"| {row['task']} | {row['status']} | "
            f"{row['verified_attempts']}/{row['attempts_completed']} | "
            f"{row['proof_rate']:.3f} | {row['successful_attempt']} | {checkpoint} |"
        )
    lines.extend(
        [
            "",
            "## Finite-perfect but proof-rejected checkpoints",
            "",
            f"Count: **{len(finite_perfect)}**",
            "",
            "## Dense NLM matched finite evaluation",
            "",
            "The 2-OutDegree row is a breadth-3 local approximation; the NLM paper uses breadth 4.",
            "",
            "| task | passing runs | wrong / cells | failed/OOM sizes |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in nlm:
        lines.append(
            f"| {row['task']} | {row['passing_runs']}/{row['runs']} | "
            f"{row['wrong']} / {row['cells']} | {row['failed_or_oom_sizes']} |"
        )
    _atomic_write(destination / EVIDENCE_MD, ("\n".join(lines) + "\n").encode("utf-8"))
    return evidence


__all__ = [
    "BASELINE_RELATIVE_PATH",
    "BASELINE_SEEDS",
    "BASELINE_SIZES",
    "EVIDENCE_CSV",
    "EVIDENCE_JSON",
    "EVIDENCE_MD",
    "build_paper_evidence",
]
