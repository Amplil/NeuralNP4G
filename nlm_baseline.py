"""Matched finite NLM baseline for the official NLM-10 comparison.

The baseline consumes the exact materialised raw splits, seed identities,
minibatch schedule, optimiser, batch size, learning-rate phases, and maximum
step budget used by the official Neural NP4G runner.  Validation, test, and
falsification data are reporting-only.  Learned ADFs are deliberately not
given to the baseline because they are the proposed method, not an input in
the original NLM model.

The local dense implementation has breadth 3.  This exactly matches nine of
the ten NLM-paper task specifications.  The paper used breadth 4 for
2-OutDegree; that row is explicitly labelled as a local approximation and the
published NLM number remains a separate reference, never a reproduction claim.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
import numpy as np

from .config import OFFICIAL_CONFIG, OfficialConfig, derive_seed
from .data import FixedSplit, GraphBatch, build_split
from .run_official_experiments import _atomic_json, _read_json, _utc_now
from .runtime import configure_logging
from .tasks import PRIMARY_TASK_NAMES, TaskSpec, get_task


NLM_PAPER_URL = "https://arxiv.org/abs/1904.11694"
MATCHED_BASELINE_NAME = "dense-nlm-nlm10-matched"
logger = logging.getLogger("official_neural_np4g.nlm_baseline")


@dataclass(frozen=True, slots=True)
class NLMPaperSpec:
    depth: int
    breadth: int
    residual: bool
    paper_train_objects: int
    paper_test_objects: tuple[int, int]
    paper_examples: int
    paper_accuracy: tuple[float, float] = (1.0, 1.0)

    @property
    def local_architecture_fidelity(self) -> str:
        return "exact" if self.breadth == 3 else "approx_breadth3"


NLM_PAPER_SPECS: Mapping[str, NLMPaperSpec] = {
    "has_father": NLMPaperSpec(4, 3, False, 20, (20, 100), 50_000),
    "has_sister": NLMPaperSpec(4, 3, False, 20, (20, 100), 50_000),
    "is_grandparent": NLMPaperSpec(4, 3, False, 20, (20, 100), 100_000),
    "is_uncle": NLMPaperSpec(4, 3, False, 20, (20, 100), 100_000),
    "is_mguncle": NLMPaperSpec(4, 3, False, 20, (20, 100), 200_000),
    "adjacent_to_red": NLMPaperSpec(4, 3, False, 10, (10, 50), 100_000),
    "connectivity_4": NLMPaperSpec(4, 3, False, 10, (10, 50), 50_000),
    "connectivity_6": NLMPaperSpec(8, 3, True, 10, (10, 50), 50_000),
    "outdegree_1": NLMPaperSpec(4, 3, False, 10, (10, 50), 50_000),
    "outdegree_2": NLMPaperSpec(5, 4, True, 10, (10, 50), 100_000),
}


if tuple(NLM_PAPER_SPECS) != PRIMARY_TASK_NAMES:
    raise RuntimeError("NLM paper specifications must match the primary task order")


def _dense_module() -> Any:
    """Load the comparison implementation only when the baseline is executed."""

    return importlib.import_module(
        ".dense_nlm", package=__package__
    )


def _set_nlm_seed(seed: int) -> None:
    """Seed every backend while retaining the official 63-bit seed identity."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _release_device_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _full_input3(batch: GraphBatch, device: torch.device) -> GraphBatch:
    """Expand the compact dead NP4G ternary channel for a breadth-3 NLM."""

    moved = batch.to(device)
    objects = moved.num_objects
    channels = int(moved.input3.shape[-1])
    input3 = torch.zeros(
        (moved.batch_size, objects, objects, objects, channels),
        dtype=moved.input3.dtype,
        device=device,
    )
    return GraphBatch(
        moved.input0,
        moved.input1,
        moved.input2,
        input3,
        moved.target,
        moved.pair_mask,
    )


def _build_model(
    task: TaskSpec,
    formal_seed: int,
    device: torch.device,
    *,
    config: OfficialConfig,
    hidden_dim: int,
) -> Any:
    dense = _dense_module()
    seed = derive_seed(formal_seed, task.index, "model", config=config)
    _set_nlm_seed(seed)
    spec = NLM_PAPER_SPECS[task.name]
    model = dense.GraphReasoningModel(
        kind="nlm",
        num_objects=config.train_cells[0].objects,
        hidden_dim=hidden_dim,
        depth=spec.depth,
        residual=spec.residual,
        np4g_hidden=config.hidden_nodes,
        np4g_ticks=config.circuit_ticks,
        target_arity_value=task.target_arity,
        input_dims=task.base_input_dims,
    )
    return model.to(device)


def _backward_training_microbatches(
    model: Any,
    compact: GraphBatch,
    device: torch.device,
) -> float:
    """Accumulate the full balanced-BCE gradient one structure at a time.

    Both the positive class weight and the mask denominator are computed over
    the entire scheduled batch.  In particular, a structure with no positives
    must not reset the class weight for the other structures.  The dense NLM
    has no operations coupling different batch examples, so summing these
    gradients gives the same mathematical objective as the full batch (with
    ordinary floating-point differences from the changed reduction order).
    The caller zeros gradients, clips them, and updates parameters once.
    """

    target = compact.target.to(device)
    mask = compact.pair_mask.to(device)
    positives = (target * mask).sum()
    negatives = ((1.0 - target) * mask).sum()
    positive_weight = negatives / positives.clamp_min(1.0)
    denominator = mask.sum().clamp_min(1.0)
    total_loss = target.new_zeros(())
    model.set_num_objects(compact.num_objects)
    for example in range(compact.batch_size):
        selected = compact.index_select(torch.tensor([example]))
        batch = _full_input3(selected, device)
        logits = model(batch)
        cell_loss = F.binary_cross_entropy_with_logits(
            logits,
            batch.target,
            reduction="none",
            pos_weight=positive_weight,
        )
        loss = (cell_loss * batch.pair_mask).sum() / denominator
        loss.backward()
        total_loss += loss.detach()
        # Release this structure's graph before constructing the next one.
        del selected, batch, logits, cell_loss, loss
    return float(total_loss.item())


def train_matched_nlm(
    model: Any,
    split: FixedSplit,
    *,
    device: torch.device,
    config: OfficialConfig,
) -> dict[str, Any]:
    """Train without held-out feedback under the NP4G optimisation grid."""

    logger.info("NLM training started task=%s seed=%s device=%s", split.task.name, split.formal_seed, device)
    optimisation_seed = derive_seed(
        split.formal_seed,
        split.task.index,
        "optimisation",
        config=config,
    )
    _set_nlm_seed(optimisation_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.phase1_learning_rate)
    started = time.monotonic()
    model.train()
    final_loss = float("nan")
    for step in range(config.total_steps):
        learning_rate = (
            config.phase1_learning_rate
            if step < config.phase1_steps
            else config.phase2_learning_rate
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        compact = split.training_batch(step, config.batch_size, "cpu")
        optimizer.zero_grad(set_to_none=True)
        final_loss = _backward_training_microbatches(model, compact, device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        if (step + 1) % config.state_checkpoint_interval == 0 or step + 1 == config.total_steps:
            logger.info(
                "NLM training task=%s seed=%s step=%s/%s loss=%.6g",
                split.task.name, split.formal_seed, step + 1, config.total_steps, final_loss,
            )
    elapsed = time.monotonic() - started
    logger.info("NLM training completed task=%s seed=%s elapsed=%.1fs", split.task.name, split.formal_seed, elapsed)
    return {
        "steps": config.total_steps,
        "phase1_steps": config.phase1_steps,
        "phase2_steps": config.phase2_steps,
        "final_train_batch_loss": final_loss,
        "elapsed_seconds": elapsed,
    }


@torch.no_grad()
def evaluate_matched_nlm(
    model: Any,
    split: FixedSplit,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate every fixed example, using microbatches to limit NLM tensors."""

    model.eval()
    per_size: dict[str, Any] = {}
    total_wrong = total_cells = total_positives = 0
    total_loss = 0.0
    for objects, source in split.batches:
        wrong = cells = positives = 0
        loss_sum = 0.0
        error: str | None = None
        model.set_num_objects(objects)
        try:
            for example in range(source.batch_size):
                compact = source.index_select(torch.tensor([example]))
                batch = _full_input3(compact, device)
                logits = model(batch)
                selected = batch.pair_mask > 0.5
                selected_logits = logits[selected]
                target = batch.target[selected]
                loss_sum += float(
                    F.binary_cross_entropy_with_logits(
                        selected_logits, target, reduction="sum"
                    ).item()
                )
                prediction = selected_logits >= 0.0
                truth = target >= 0.5
                wrong += int((prediction != truth).sum().item())
                cells += int(truth.numel())
                positives += int(truth.sum().item())
        except Exception as exc:  # Record OOM/runtime limits as data, not accuracy.
            error = f"{type(exc).__name__}: {str(exc)[:240]}"
            logger.exception(
                "NLM evaluation incomplete task=%s seed=%s split=%s objects=%s",
                split.task.name, split.formal_seed, split.name, objects,
            )
            _release_device_memory()
        row: dict[str, Any] = {
            "wrong": wrong,
            "cells": cells,
            "positives": positives,
            "accuracy": None if cells == 0 else 1.0 - wrong / cells,
            "mean_loss": None if cells == 0 else loss_sum / cells,
            "complete": error is None and cells == int(
                split.manifest.cells[split.sizes.index(objects)].evaluated_cells
            ),
        }
        if error is not None:
            row["error"] = error
        per_size[f"n{objects}"] = row
        total_wrong += wrong
        total_cells += cells
        total_positives += positives
        total_loss += loss_sum
    logger.info(
        "NLM evaluation task=%s seed=%s split=%s wrong=%s cells=%s complete=%s",
        split.task.name, split.formal_seed, split.name, total_wrong, total_cells,
        all(row["complete"] for row in per_size.values()),
    )
    return {
        "split": split.name,
        "task": split.task.name,
        "formal_seed": split.formal_seed,
        "checkpoint_only": True,
        "total": {
            "wrong": total_wrong,
            "cells": total_cells,
            "positives": total_positives,
            "accuracy": None if total_cells == 0 else 1.0 - total_wrong / total_cells,
            "mean_loss": None if total_cells == 0 else total_loss / total_cells,
            "complete": all(row["complete"] for row in per_size.values()),
        },
        "per_size": per_size,
    }


def _split_bundle(
    task: TaskSpec,
    seed: int,
    config: OfficialConfig,
) -> dict[str, FixedSplit]:
    return {
        name: build_split(task, name, seed, config=config)
        for name in ("train", "validation", "test", "falsification")
    }


def protocol_record(config: OfficialConfig, *, hidden_dim: int) -> dict[str, Any]:
    return {
        "name": MATCHED_BASELINE_NAME,
        "protocol_version": config.protocol_version,
        "paper_source": NLM_PAPER_URL,
        "tasks": list(PRIMARY_TASK_NAMES),
        "seeds": list(config.attempt_seeds),
        "shared_raw_split_contract": True,
        "shared_minibatch_contract": True,
        "learned_adfs_supplied": False,
        "selection_feedback": False,
        "hidden_dim": hidden_dim,
        "training": {
            "phase1_steps": config.phase1_steps,
            "phase2_steps": config.phase2_steps,
            "total_steps": config.total_steps,
            "batch_size": config.batch_size,
            "effective_batch_size": config.batch_size,
            "microbatch_size": 1,
            "phase1_learning_rate": config.phase1_learning_rate,
            "phase2_learning_rate": config.phase2_learning_rate,
            "gradient_clip_norm": config.gradient_clip_norm,
            "optimizer": "Adam",
            "loss": "balanced binary cross entropy",
            "loss_normalization": (
                "positive weight = full-batch masked negatives / max(full-batch "
                "masked positives, 1); each microbatch masked loss sum divided "
                "by max(full-batch mask sum, 1); gradients summed across every "
                "scheduled example before one clip and one optimizer update"
            ),
            "optimizer_updates_per_effective_batch": 1,
            "deterministic_algorithms": True,
        },
        "local_model_scope": (
            "breadth-3 dense NLM; exact paper architecture for 9/10 rows; "
            "2-OutDegree is explicitly approximate because the paper uses breadth 4"
        ),
        "task_specs": {
            name: {
                "depth": spec.depth,
                "paper_breadth": spec.breadth,
                "local_breadth": 3,
                "residual": spec.residual,
                "paper_train_objects": spec.paper_train_objects,
                "paper_test_objects": list(spec.paper_test_objects),
                "paper_examples": spec.paper_examples,
                "paper_accuracy": list(spec.paper_accuracy),
                "local_architecture_fidelity": spec.local_architecture_fidelity,
            }
            for name, spec in NLM_PAPER_SPECS.items()
        },
    }


def run_matched_nlm_baseline(
    report_path: Path | str,
    *,
    device: torch.device | str,
    config: OfficialConfig = OFFICIAL_CONFIG,
    hidden_dim: int = 16,
) -> dict[str, Any]:
    """Run/resume the complete matched baseline and atomically write its report."""

    destination = Path(report_path).resolve()
    selected_device = torch.device(device)
    logger.info("Matched NLM baseline started device=%s report=%s", selected_device, destination)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    expected_protocol = protocol_record(config, hidden_dim=hidden_dim)
    if destination.is_file():
        completed_report = _read_json(destination)
        if completed_report.get("protocol") != expected_protocol:
            raise RuntimeError("completed dense-NLM report belongs to another protocol")
        rows = completed_report.get("results", {})
        if completed_report.get("complete") is not True or not all(
            tuple(int(row["seed"]) for row in rows.get(name, [])) == config.attempt_seeds
            for name in PRIMARY_TASK_NAMES
        ):
            raise RuntimeError("final dense-NLM report is incomplete")
        logger.info("Reused completed NLM baseline report=%s", destination)
        return completed_report
    dense = _dense_module()
    dense.warmup_cuda_torch(selected_device)
    partial_path = destination.with_name(f"{destination.stem}.partial.json")
    results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PRIMARY_TASK_NAMES
    }
    if partial_path.is_file():
        partial = _read_json(partial_path)
        if partial.get("protocol") != expected_protocol:
            raise RuntimeError("partial dense-NLM report belongs to another protocol")
        results = {name: list(partial["results"].get(name, [])) for name in PRIMARY_TASK_NAMES}
        logger.info("Resumed NLM baseline from partial report=%s", partial_path)

    for task_name in PRIMARY_TASK_NAMES:
        task = get_task(task_name)
        completed = {int(row["seed"]) for row in results[task_name]}
        for seed in config.attempt_seeds:
            if seed in completed:
                continue
            logger.info("NLM attempt started task=%s seed=%s", task_name, seed)
            splits = _split_bundle(task, seed, config)
            model = _build_model(
                task,
                seed,
                selected_device,
                config=config,
                hidden_dim=hidden_dim,
            )
            training = train_matched_nlm(
                model,
                splits["train"],
                device=selected_device,
                config=config,
            )
            evaluations = {
                name: evaluate_matched_nlm(model, splits[name], device=selected_device)
                for name in ("validation", "test", "falsification")
            }
            spec = NLM_PAPER_SPECS[task_name]
            results[task_name].append(
                {
                    "seed": seed,
                    "architecture_fidelity": spec.local_architecture_fidelity,
                    "training": training,
                    "split_manifests": {
                        name: split.manifest.as_dict() for name, split in splits.items()
                    },
                    "evaluations": evaluations,
                    "completed_at": _utc_now(),
                }
            )
            payload = {
                "protocol": expected_protocol,
                "results": results,
                "complete": False,
            }
            _atomic_json(partial_path, payload)
            logger.info("Saved NLM baseline partial report=%s task=%s seed=%s", partial_path, task_name, seed)
            del model, splits
            _release_device_memory()

    payload = {
        "protocol": expected_protocol,
        "results": results,
        "complete": all(
            tuple(int(row["seed"]) for row in results[name]) == config.attempt_seeds
            for name in PRIMARY_TASK_NAMES
        ),
        "completed_at": _utc_now(),
    }
    _atomic_json(destination, payload)
    logger.info("Matched NLM baseline completed complete=%s report=%s", payload["complete"], destination)
    if payload["complete"] and partial_path.is_file():
        partial = _read_json(partial_path)
        if partial.get("protocol") == expected_protocol and partial.get("results") == results:
            partial_path.unlink()
    return payload


def _command_entrypoint(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for diagnostic logs.")
    args = parser.parse_args(argv)
    configure_logging("nlm-baseline", log_dir=args.log_dir)
    try:
        run_matched_nlm_baseline(args.report_out, device=args.device)
    except Exception:
        logger.exception("Matched NLM baseline command failed")
        raise


if __name__ == "__main__":
    _command_entrypoint()


__all__ = [
    "MATCHED_BASELINE_NAME",
    "NLM_PAPER_SPECS",
    "NLMPaperSpec",
    "evaluate_matched_nlm",
    "protocol_record",
    "run_matched_nlm_baseline",
    "train_matched_nlm",
]
