"""Read-only validation and test evaluation for the official protocol."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .adf import ADFEntry, augment_batch
from .data import FixedSplit
from .model import NeuralNP4G


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationTotals:
    """Exact aggregate counts for one frozen checkpoint and one split."""

    wrong: int
    cells: int
    positives: int
    loss_sum: float

    @property
    def accuracy(self) -> float:
        return 1.0 - self.wrong / max(1, self.cells)

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / max(1, self.cells)

    @property
    def zero_error(self) -> bool:
        return self.wrong == 0

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "wrong": self.wrong,
            "cells": self.cells,
            "positives": self.positives,
            "negatives": self.cells - self.positives,
            "accuracy": self.accuracy,
            "mean_loss": self.mean_loss,
            "zero_error": self.zero_error,
        }


def _evaluate_batch(
    model: NeuralNP4G,
    batch: Any,
) -> EvaluationTotals:
    logits = model(batch)
    if logits.shape != batch.target.shape:
        raise RuntimeError(
            f"model returned {tuple(logits.shape)}, expected {tuple(batch.target.shape)}"
        )
    selected = batch.pair_mask > 0.5
    targets = batch.target[selected]
    selected_logits = logits[selected]
    losses = F.binary_cross_entropy_with_logits(
        selected_logits,
        targets,
        reduction="sum",
    )
    predictions = selected_logits >= 0.0
    truth = targets >= 0.5
    return EvaluationTotals(
        wrong=int((predictions != truth).sum().item()),
        cells=int(truth.numel()),
        positives=int(truth.sum().item()),
        loss_sum=float(losses.item()),
    )


@torch.no_grad()
def evaluate_split(
    model: NeuralNP4G,
    split: FixedSplit,
    *,
    learned_adfs: Sequence[ADFEntry] = (),
    device: torch.device | str,
) -> dict[str, Any]:
    """Evaluate every preregistered cell without changing model state.

    The returned counts are exact over the split.  They are reporting outputs
    only: callers must never feed them back into optimisation or checkpoint
    selection.
    """

    model.eval()
    device = torch.device(device)
    per_size: dict[str, dict[str, int | float | bool]] = {}
    wrong = cells = positives = 0
    loss_sum = 0.0
    for objects, cpu_batch in split.batches:
        batch = augment_batch(cpu_batch.to(device), learned_adfs)
        model.set_num_objects(objects)
        result = _evaluate_batch(model, batch)
        per_size[f"n{objects}"] = result.as_dict()
        wrong += result.wrong
        cells += result.cells
        positives += result.positives
        loss_sum += result.loss_sum

    total = EvaluationTotals(wrong, cells, positives, loss_sum)
    logger.info(
        "Evaluation task=%s seed=%s split=%s wrong=%s cells=%s accuracy=%.6f",
        split.task.name, split.formal_seed, split.name, total.wrong, total.cells,
        total.accuracy,
    )
    return {
        "split": split.name,
        "task": split.task.name,
        "formal_seed": split.formal_seed,
        "checkpoint_only": True,
        "total": total.as_dict(),
        "per_size": per_size,
    }


@torch.no_grad()
def evaluate_adf_causal_ablation(
    model: NeuralNP4G,
    split: FixedSplit,
    *,
    learned_adfs: Sequence[ADFEntry],
    required_adfs: Sequence[str] | None = None,
    device: torch.device | str,
) -> dict[str, Any]:
    """Disable the preregistered required ADF subset and measure errors.

    ``learned_adfs`` is the complete ordered candidate prefix required to
    evaluate the checkpoint.  ``required_adfs`` selects only the task-declared
    subset whose causal use is an acceptance condition; optional candidates
    remain unchanged during the ablation.  Passing ``None`` preserves the
    research-API behaviour of ablating every supplied candidate.  Parameters
    are restored exactly before returning.
    """

    entries = tuple(learned_adfs)
    required_names = (
        tuple(entry.task for entry in entries)
        if required_adfs is None
        else tuple(str(name) for name in required_adfs)
    )
    if len(set(required_names)) != len(required_names):
        raise ValueError("required ADF names must be unique")
    if not required_names:
        return {
            "applicable": False,
            "reason": "task has no required learned ADF dependency",
            "available_candidate_ids": [entry.candidate_id for entry in entries],
            "required_adfs": [],
        }

    by_name = {entry.task: entry for entry in entries}
    missing = [name for name in required_names if name not in by_name]
    if missing:
        raise ValueError(f"required ADFs are absent from the candidate prefix: {missing}")

    candidate_ids = model.candidate_gates.candidate_ids
    indices: list[int] = []
    identifiers: list[str] = []
    for name in required_names:
        entry = by_name[name]
        identifier = entry.candidate_id
        try:
            indices.append(candidate_ids.index(identifier))
        except ValueError as error:
            raise RuntimeError(f"model has no gate for {identifier}") from error
        identifiers.append(identifier)

    logits = model.candidate_gates.logits
    saved = logits.detach().clone()
    gate_bits = model.candidate_gates().detach()
    gates_on = all(float(gate_bits[index].item()) == 1.0 for index in indices)
    normal = evaluate_split(
        model,
        split,
        learned_adfs=entries,
        device=device,
    )
    individual: dict[str, dict[str, Any]] = {}
    try:
        logits[indices] = -20.0
        disabled = evaluate_split(
            model,
            split,
            learned_adfs=entries,
            device=device,
        )
        joint_disabled_wrong = int(disabled["total"]["wrong"])
        if len(indices) == 1:
            name = required_names[0]
            individual[name] = {
                "gate_bit": float(gate_bits[indices[0]].item()),
                "adf_disabled_wrong": joint_disabled_wrong,
                "additional_errors": joint_disabled_wrong
                - int(normal["total"]["wrong"]),
            }
        else:
            for name, index in zip(required_names, indices):
                logits.copy_(saved)
                logits[index] = -20.0
                one_disabled = evaluate_split(
                    model,
                    split,
                    learned_adfs=entries,
                    device=device,
                )
                one_wrong = int(one_disabled["total"]["wrong"])
                individual[name] = {
                    "gate_bit": float(gate_bits[index].item()),
                    "adf_disabled_wrong": one_wrong,
                    "additional_errors": one_wrong
                    - int(normal["total"]["wrong"]),
                }
    finally:
        logits.copy_(saved)

    normal_wrong = int(normal["total"]["wrong"])
    disabled_wrong = int(disabled["total"]["wrong"])
    for record in individual.values():
        record["causal_use"] = (
            float(record["gate_bit"]) == 1.0
            and int(record["additional_errors"]) > 0
        )
    individual_causal_use = bool(individual) and all(
        bool(record["causal_use"]) for record in individual.values()
    )
    return {
        "applicable": True,
        "candidate_ids": identifiers,
        "available_candidate_ids": [entry.candidate_id for entry in entries],
        "required_adfs": list(required_names),
        "required_gate_bits": {
            name: float(gate_bits[index].item())
            for name, index in zip(required_names, indices)
        },
        "all_gates_on": gates_on,
        "normal_wrong": normal_wrong,
        "adf_disabled_wrong": disabled_wrong,
        "additional_errors": disabled_wrong - normal_wrong,
        "individual_ablation": individual,
        "individual_causal_use": individual_causal_use,
        "causal_use": gates_on and individual_causal_use,
        "split": split.name,
    }


__all__ = [
    "EvaluationTotals",
    "evaluate_adf_causal_ablation",
    "evaluate_split",
]
