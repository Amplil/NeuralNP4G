"""The single fixed optimisation trajectory used by the formal experiments.

Only a materialised ``train`` split enters this module.  Evaluation, program
extraction, and proof are deliberately separate commands run after each fixed
stage checkpoint hash has been fixed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# PyTorch requires this fixed workspace before the first CUDA matrix
# multiplication when deterministic algorithms are enabled.  It is a
# reproducibility constant of the official trajectory, not a user option.
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
import torch.nn.functional as F

from .adf import ADFEntry, augment_batch
from .config import OFFICIAL_CONFIG, OfficialConfig, derive_seed
from .data import FixedSplit, GraphBatch
from .model import NeuralNP4G
from .tasks import TaskSpec, get_task, input_dims_for


TRAINING_STATE_FORMAT_VERSION = 8
CHECKPOINT_FORMAT_VERSION = 8
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    task: str
    formal_seed: int
    stage: str
    steps: int
    elapsed_seconds: float
    checkpoint_path: Path
    checkpoint_sha256: str
    train_split_sha256: str
    metrics: tuple[dict[str, Any], ...]
    structure: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "formal_seed": self.formal_seed,
            "stage": self.stage,
            "steps": self.steps,
            "elapsed_seconds": self.elapsed_seconds,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "train_split_sha256": self.train_split_sha256,
            "metrics": list(self.metrics),
            "structure": self.structure,
        }


@dataclass(frozen=True, slots=True)
class _PruneGuardCell:
    """One immutable TRAIN-only cell used to protect pruning decisions."""

    objects: int
    indices: tuple[int, ...]
    raw_sha256: str
    evaluated_cells: int
    batch: GraphBatch


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _protocol_digest(config: OfficialConfig) -> str:
    return hashlib.sha256(_json_bytes(config.as_dict())).hexdigest()


def _adf_candidate_identity(model: NeuralNP4G) -> list[dict[str, str | None]]:
    """Immutable ordered identity of the frozen candidate prefix."""

    return [
        {
            "task": entry.task,
            "checkpoint": entry.checkpoint,
            "checkpoint_sha256": entry.checkpoint_sha256,
        }
        for entry in model.adf_input_specs
    ]


def _build_train_guard_panel(
    split: FixedSplit,
    config: OfficialConfig,
    *,
    examples_per_size: int,
    seed_stream: str,
    policy: str,
) -> tuple[tuple[_PruneGuardCell, ...], dict[str, Any]]:
    """Select a deterministic all-size panel exclusively from TRAIN."""

    if split.name != "train":
        raise ValueError("the prune guard can only be built from a train split")
    requested = int(examples_per_size)
    cells: list[_PruneGuardCell] = []
    records: list[dict[str, Any]] = []
    for objects, source in sorted(split.batches, key=lambda item: int(item[0])):
        examples = min(requested, source.batch_size)
        seed = derive_seed(
            split.formal_seed,
            split.task.index,
            seed_stream,
            int(objects),
            source.batch_size,
            examples,
            config=config,
        )
        generator = np.random.default_rng(seed)
        selected = np.sort(
            generator.choice(source.batch_size, size=examples, replace=False)
        )
        indices = tuple(int(value) for value in selected.tolist())
        raw_batch = source.index_select(torch.tensor(indices, dtype=torch.long))
        evaluated_cells = int((raw_batch.pair_mask > 0.5).sum().item())
        raw_sha256 = raw_batch.sha256()
        cells.append(
            _PruneGuardCell(
                objects=int(objects),
                indices=indices,
                raw_sha256=raw_sha256,
                evaluated_cells=evaluated_cells,
                batch=raw_batch,
            )
        )
        records.append(
            {
                "objects": int(objects),
                "source_examples": source.batch_size,
                "examples": examples,
                "indices": list(indices),
                "selection_seed": int(seed),
                "raw_batch_sha256": raw_sha256,
                "evaluated_cells": evaluated_cells,
            }
        )

    core = {
        "policy": policy,
        "train_split_sha256": split.manifest.sha256(),
        "examples_per_size": requested,
        "cells": records,
    }
    manifest = {
        **core,
        "sha256": hashlib.sha256(_json_bytes(core)).hexdigest(),
    }
    return tuple(cells), manifest


def _build_prune_guard_panel(
    split: FixedSplit,
    config: OfficialConfig,
) -> tuple[tuple[_PruneGuardCell, ...], dict[str, Any]]:
    """Small fixed TRAIN panel used by bounded one-edge prune trials."""

    return _build_train_guard_panel(
        split,
        config,
        examples_per_size=config.prune_guard_examples_per_size,
        seed_stream="prune_guard",
        policy="fixed_train_multisize_v1",
    )


def _build_incumbent_guard_panel(
    split: FixedSplit,
    config: OfficialConfig,
) -> tuple[tuple[_PruneGuardCell, ...], dict[str, Any]]:
    """Full TRAIN panel used to preserve exact optimizer/pruning incumbents."""

    return _build_train_guard_panel(
        split,
        config,
        examples_per_size=config.incumbent_guard_examples_per_size,
        seed_stream="incumbent_guard",
        policy="full_train_incumbent_v1",
    )


def _prepare_prune_guard_panel(
    panel: Sequence[_PruneGuardCell],
    model: NeuralNP4G,
    device: torch.device,
) -> tuple[_PruneGuardCell, ...]:
    """Move and ADF-augment the fixed panel once for repeated guard probes."""

    prepared: list[_PruneGuardCell] = []
    for cell in panel:
        batch = cell.batch.to(device)
        if model.adf_input_specs:
            batch = augment_batch(batch, model.adf_input_specs)
        prepared.append(
            _PruneGuardCell(
                objects=cell.objects,
                indices=cell.indices,
                raw_sha256=cell.raw_sha256,
                evaluated_cells=cell.evaluated_cells,
                batch=batch,
            )
        )
    return tuple(prepared)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8"))


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location=device)


def build_model(
    task: str | TaskSpec,
    formal_seed: int,
    learned_adfs: Sequence[ADFEntry] = (),
    *,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> NeuralNP4G:
    """Construct the official model under an isolated deterministic RNG state."""

    spec = get_task(task)
    entries = tuple(learned_adfs)
    dimensions = input_dims_for(spec, entries)
    model_seed = derive_seed(
        formal_seed,
        spec.index,
        "model",
        config=config,
    )
    # Initialisation happens on CPU and does not perturb the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = NeuralNP4G(
            num_objects=config.train_cells[0].objects,
            target_arity=spec.target_arity,
            input_dims=dimensions,
            message_ticks=spec.message_ticks,
            channels=config.channels,
            hidden_nodes=config.hidden_nodes,
            circuit_ticks=config.circuit_ticks,
            learned_adfs=entries,
            preprocess=spec.preprocess_ops,
            base_input_policy=spec.base_input_policy,
            edge_sparsity=config.edge_sparsity,
            neuron_sparsity=config.neuron_sparsity,
        )
    return model


def masked_balanced_bce(logits: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
    """Binary cross entropy with equal positive/negative class contribution."""

    if logits.shape != batch.target.shape:
        raise ValueError(
            f"logit shape {tuple(logits.shape)} differs from target {tuple(batch.target.shape)}"
        )
    selected = batch.pair_mask > 0.5
    if not bool(selected.any()):
        raise ValueError("training batch has no evaluated cells")
    values = F.binary_cross_entropy_with_logits(
        logits[selected], batch.target[selected], reduction="none"
    )
    labels = batch.target[selected] > 0.5
    positive = labels
    negative = ~labels
    if bool(positive.any()) and bool(negative.any()):
        return 0.5 * values[positive].mean() + 0.5 * values[negative].mean()
    return values.mean()


def _batch_accuracy(logits: torch.Tensor, batch: GraphBatch) -> tuple[float, int, int]:
    selected = batch.pair_mask > 0.5
    predicted = logits >= 0.0
    expected = batch.target >= 0.5
    wrong = int(((predicted != expected) & selected).sum().item())
    cells = int(selected.sum().item())
    return (1.0 - wrong / cells), wrong, cells


def _accepted_prune_stats(model: NeuralNP4G) -> dict[str, int | float]:
    """Describe the mask that is actually retained after a prune proposal."""

    fanins = model.core.unpruned_fanin()
    return {
        "max_fanin": int(fanins.max().item()) if fanins.numel() else 0,
        "mean_fanin": float(fanins.float().mean().item()) if fanins.numel() else 0.0,
    }


@torch.no_grad()
def _post_update_metrics(
    model: NeuralNP4G,
    batch: GraphBatch,
    *,
    synthetic_weight: float,
    config: OfficialConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Measure the accepted post-optimiser/post-prune model without RNG drift."""

    saved_rng = _rng_state(device)
    try:
        logits = model(batch)
        data_loss = masked_balanced_bce(logits, batch)
        structure_loss = model.structure_loss()
        candidate_loss = model.candidate_loss()
        binarization_loss = model.binarization_loss()
        saturation_loss = model.last_saturation_loss()
        synthetic_loss = logits.new_tensor(0.0)
        if synthetic_weight > 0.0:
            synthetic_loss = model.synthetic_saturation_loss(
                samples=config.synthetic_samples
            )
        total_loss = (
            data_loss
            + config.structure_loss_weight * structure_loss
            + config.candidate_loss_weight * candidate_loss
            + config.binarization_loss_weight * binarization_loss
            + config.saturation_loss_weight * saturation_loss
            + synthetic_weight * synthetic_loss
        )
        values = {
            "loss": float(total_loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "structure_loss": float(structure_loss.detach().cpu()),
            "candidate_loss": float(candidate_loss.detach().cpu()),
            "binarization_loss": float(binarization_loss.detach().cpu()),
            "saturation_loss": float(saturation_loss.detach().cpu()),
            "synthetic_saturation_loss": float(synthetic_loss.detach().cpu()),
        }
        return logits.detach(), values
    finally:
        _restore_rng_state(saved_rng, device)


def _phase(
    step: int,
    config: OfficialConfig,
) -> tuple[int, int, int, float, float, int, float, int]:
    """Return the preregistered schedule values for a zero-based global step."""

    if not 0 <= step < config.total_steps:
        raise ValueError("step outside the official schedule")
    if step < config.phase1_steps:
        return (
            1,
            step,
            config.phase1_steps,
            config.phase1_learning_rate,
            0.0,
            config.phase1_prune_interval,
            config.phase1_prune_fraction,
            config.phase1_fanin,
        )
    phase_step = step - config.phase1_steps
    return (
        2,
        phase_step,
        config.phase2_steps,
        config.phase2_learning_rate,
        config.synthetic_saturation_loss_weight,
        config.phase2_prune_interval,
        config.phase2_prune_fraction,
        config.phase2_fanin,
    )


def _scheduled_fanin(
    model: NeuralNP4G,
    phase: int,
    phase_step: int,
    phase_steps: int,
    prune_fraction: float,
    target: int,
    config: OfficialConfig,
) -> int:
    if phase == 1:
        start = int(model.core.edge_mask.sum(dim=1).max().item())
    else:
        start = config.phase1_fanin
    ramp_steps = max(1, int(math.ceil(phase_steps * prune_fraction)))
    progress = min(1.0, (phase_step + 1) / ramp_steps)
    value = int(round(start + progress * (target - start)))
    return max(target, value)


def _rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda" and "torch_cuda" in state:
        # ``torch.load(..., map_location=device)`` also moves these saved byte
        # tensors to CUDA.  The CUDA RNG API deliberately accepts CPU byte
        # tensors only, so restore that representation explicitly.
        torch.cuda.set_rng_state_all(
            [generator_state.cpu() for generator_state in state["torch_cuda"]]
        )


def _guard_accuracy_record(logits: torch.Tensor, batch: GraphBatch) -> dict[str, int]:
    _, wrong, cells = _batch_accuracy(logits, batch)
    return {"wrong": wrong, "cells": cells}


@torch.no_grad()
def _evaluate_prune_guard(
    model: NeuralNP4G,
    panel: Sequence[_PruneGuardCell],
    *,
    event_batch: GraphBatch,
    device: torch.device,
) -> dict[str, Any]:
    """Measure an event batch and every fixed TRAIN guard size without RNG drift."""

    saved_rng = _rng_state(device)
    saved_objects = int(model.num_objects)
    try:
        model.set_num_objects(event_batch.num_objects)
        event = _guard_accuracy_record(model(event_batch), event_batch)
        sizes: list[dict[str, int]] = []
        for cell in panel:
            model.set_num_objects(cell.objects)
            record = _guard_accuracy_record(model(cell.batch), cell.batch)
            sizes.append({"objects": cell.objects, **record})
        return {"event": event, "sizes": sizes}
    finally:
        model.set_num_objects(saved_objects)
        _restore_rng_state(saved_rng, device)


def _guard_non_regressing(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    """Require exactness on the prune event and no per-size TRAIN regression."""

    baseline_event = baseline.get("event")
    candidate_event = candidate.get("event")
    if not isinstance(baseline_event, dict) or not isinstance(candidate_event, dict):
        raise ValueError("prune guard reports must contain event records")
    if int(baseline_event["cells"]) != int(candidate_event["cells"]):
        raise RuntimeError("prune guard changed the event evaluated-cell mask")
    if int(candidate_event["wrong"]) != 0:
        return False

    baseline_sizes = list(baseline.get("sizes", ()))
    candidate_sizes = list(candidate.get("sizes", ()))
    if len(baseline_sizes) != len(candidate_sizes):
        raise RuntimeError("prune guard size panel changed during pruning")
    for before, after in zip(baseline_sizes, candidate_sizes):
        identity_before = (int(before["objects"]), int(before["cells"]))
        identity_after = (int(after["objects"]), int(after["cells"]))
        if identity_before != identity_after:
            raise RuntimeError("prune guard cell identity changed during pruning")
        if int(after["wrong"]) > int(before["wrong"]):
            return False
    return True


def _guard_all_zero(report: dict[str, Any]) -> bool:
    """Whether the event and every size cell are exactly classified."""

    event = report.get("event", {})
    sizes = report.get("sizes", ())
    return (
        isinstance(event, dict)
        and int(event.get("wrong", -1)) == 0
        and bool(sizes)
        and all(int(record.get("wrong", -1)) == 0 for record in sizes)
    )


@torch.no_grad()
def _prune_guard_allows(
    model: NeuralNP4G,
    panel: Sequence[_PruneGuardCell],
    baseline: dict[str, Any],
    *,
    event_batch: GraphBatch,
    device: torch.device,
) -> bool:
    """Fast event-first/per-size probe used for one-edge partial commits."""

    saved_rng = _rng_state(device)
    saved_objects = int(model.num_objects)
    try:
        baseline_event = baseline["event"]
        model.set_num_objects(event_batch.num_objects)
        event = _guard_accuracy_record(model(event_batch), event_batch)
        if int(event["cells"]) != int(baseline_event["cells"]):
            raise RuntimeError("prune guard changed the event evaluated-cell mask")
        if int(event["wrong"]) != 0:
            return False

        baseline_sizes = list(baseline["sizes"])
        if len(baseline_sizes) != len(panel):
            raise RuntimeError("prune guard baseline has the wrong number of sizes")
        for cell, before in zip(panel, baseline_sizes):
            identity = (cell.objects, cell.evaluated_cells)
            expected = (int(before["objects"]), int(before["cells"]))
            if identity != expected:
                raise RuntimeError("prune guard baseline cell identity mismatch")
            model.set_num_objects(cell.objects)
            after = _guard_accuracy_record(model(cell.batch), cell.batch)
            if int(after["cells"]) != cell.evaluated_cells:
                raise RuntimeError("prune guard changed a size-cell mask")
            if int(after["wrong"]) > int(before["wrong"]):
                return False
        return True
    finally:
        model.set_num_objects(saved_objects)
        _restore_rng_state(saved_rng, device)


def _normalise_prune_kill_masks(
    raw_masks: Any,
    mask_before: torch.Tensor,
    mask_after: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Validate model-reported kill categories against the actual proposal."""

    category_names = (
        "killed_dead_inputs",
        "killed_topk",
        "killed_count_cap",
    )
    actual = (mask_before > 0) & (mask_after <= 0)
    if raw_masks is None:
        zero = torch.zeros_like(actual)
        return {
            "killed_dead_inputs": zero.clone(),
            "killed_topk": actual.clone(),
            "killed_count_cap": zero.clone(),
        }
    if not isinstance(raw_masks, dict):
        raise TypeError("prune-step kill masks must be a dictionary")

    result: dict[str, torch.Tensor] = {}
    covered = torch.zeros_like(actual)
    for name in category_names:
        value = raw_masks.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != actual.shape:
            raise ValueError(f"prune-step kill mask {name!r} has the wrong shape")
        selected = value.to(device=actual.device, dtype=torch.bool)
        if bool((selected & ~actual).any()):
            raise RuntimeError(f"prune-step kill mask {name!r} reports an unpruned edge")
        if bool((selected & covered).any()):
            raise RuntimeError("prune-step kill categories overlap")
        result[name] = selected
        covered |= selected
    if not torch.equal(covered, actual):
        raise RuntimeError("prune-step kill categories do not cover the proposal")
    return result


def _ordered_prune_edges(
    kill_mask: torch.Tensor,
    magnitude_before: torch.Tensor,
) -> list[tuple[int, int]]:
    if kill_mask.shape != magnitude_before.shape:
        raise ValueError("prune kill mask and magnitude snapshot shapes differ")
    coordinates = [
        (int(row), int(column))
        for row, column in torch.nonzero(kill_mask, as_tuple=False).cpu().tolist()
    ]
    magnitude = magnitude_before.detach().cpu()
    return sorted(
        coordinates,
        key=lambda item: (float(magnitude[item[0], item[1]]), item[0], item[1]),
    )


def _prune_edge_order_sha256(edges: Sequence[tuple[int, int]]) -> str:
    return hashlib.sha256(
        _json_bytes([[int(row), int(column)] for row, column in edges])
    ).hexdigest()


@torch.no_grad()
def _partial_commit_prune(
    model: NeuralNP4G,
    panel: Sequence[_PruneGuardCell],
    baseline: dict[str, Any],
    *,
    event_batch: GraphBatch,
    mask_before: torch.Tensor,
    proposal_mask: torch.Tensor,
    magnitude_before: torch.Tensor,
    max_trials: int,
    device: torch.device,
) -> dict[str, Any]:
    """Replay proposed kills deterministically, retaining only guarded edges."""

    proposed_kills = (mask_before > 0) & (proposal_mask <= 0)
    ordered = _ordered_prune_edges(proposed_kills, magnitude_before)
    trials = min(len(ordered), int(max_trials))
    accepted = torch.zeros_like(proposed_kills)
    model.core.prune_mask.copy_(mask_before)

    for row, column in ordered[:trials]:
        model.core.prune_mask[row, column] = proposal_mask[row, column]
        if _prune_guard_allows(
            model,
            panel,
            baseline,
            event_batch=event_batch,
            device=device,
        ):
            accepted[row, column] = True
        else:
            model.core.prune_mask[row, column] = mask_before[row, column]

    guard_after = _evaluate_prune_guard(
        model,
        panel,
        event_batch=event_batch,
        device=device,
    )
    if not _guard_non_regressing(baseline, guard_after):
        raise RuntimeError("partial pruning committed a TRAIN guard regression")
    committed = int(accepted.sum().item())
    return {
        "accepted_mask": accepted,
        "guard_after": guard_after,
        "proposal_edge_order_sha256": _prune_edge_order_sha256(ordered),
        "proposal_kills_total": len(ordered),
        "partial_trials": trials,
        "partial_committed_kills": committed,
        "partial_rejected_kills": trials - committed,
        "partial_truncated": trials < len(ordered),
    }


def _seed_optimisation(
    formal_seed: int,
    task: TaskSpec,
    device: torch.device,
    config: OfficialConfig,
) -> int:
    seed = derive_seed(
        formal_seed,
        task.index,
        "optimisation",
        config=config,
    )
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed


def _validate_training_inputs(
    model: NeuralNP4G,
    split: FixedSplit,
    config: OfficialConfig,
) -> None:
    if split.name != "train":
        raise ValueError("train_model accepts only a train split")
    if split.manifest.protocol_version != config.protocol_version:
        raise ValueError("train split was generated under another protocol")
    if model.target_arity != split.task.target_arity:
        raise ValueError("model and task target arity differ")
    if model.num_ticks != split.task.message_ticks:
        raise ValueError("model and task message ticks differ")
    if model.base_input_policy != split.task.base_input_policy:
        raise ValueError("model and task base-input policies differ")
    candidate_names = tuple(entry.task for entry in model.adf_input_specs)
    missing = [name for name in split.task.dependencies if name not in candidate_names]
    if missing:
        raise ValueError(
            f"task '{split.task.name}' is missing required ADFs {tuple(missing)}; "
            f"model candidates are {candidate_names}"
        )
    expected_dims = input_dims_for(split.task, model.adf_input_specs)
    if tuple(model.input_dims) != expected_dims:
        raise ValueError("model input widths do not match task and ADF registry")


def _write_metrics(path: Path, metrics: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in metrics
    )
    _atomic_write(path, content.encode("utf-8"))


def _save_resume_state(
    path: Path,
    *,
    model: NeuralNP4G,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    elapsed_seconds: float,
    metrics: list[dict[str, Any]],
    split: FixedSplit,
    prune_guard_sha256: str,
    incumbent_guard_sha256: str,
    config: OfficialConfig,
    device: torch.device,
) -> None:
    _atomic_torch_save(
        path,
        {
            "format_version": TRAINING_STATE_FORMAT_VERSION,
            "protocol_version": config.protocol_version,
            "protocol_sha256": _protocol_digest(config),
            "task": split.task.name,
            "formal_seed": split.formal_seed,
            "train_split_sha256": split.manifest.sha256(),
            "prune_guard_sha256": prune_guard_sha256,
            "incumbent_guard_sha256": incumbent_guard_sha256,
            "adf_candidates": _adf_candidate_identity(model),
            "next_step": int(next_step),
            "elapsed_seconds": float(elapsed_seconds),
            "metrics": metrics,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _rng_state(device),
        },
    )


def _load_resume_state(
    path: Path,
    *,
    model: NeuralNP4G,
    optimizer: torch.optim.Optimizer,
    split: FixedSplit,
    prune_guard_sha256: str,
    incumbent_guard_sha256: str,
    config: OfficialConfig,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]]]:
    payload = _torch_load(path, device)
    expected = {
        "format_version": TRAINING_STATE_FORMAT_VERSION,
        "protocol_version": config.protocol_version,
        "protocol_sha256": _protocol_digest(config),
        "task": split.task.name,
        "formal_seed": split.formal_seed,
        "train_split_sha256": split.manifest.sha256(),
        "prune_guard_sha256": prune_guard_sha256,
        "incumbent_guard_sha256": incumbent_guard_sha256,
        "adf_candidates": _adf_candidate_identity(model),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"resume state {key} mismatch: {payload.get(key)!r} != {value!r}"
            )
    next_step = int(payload["next_step"])
    if not 0 <= next_step <= config.total_steps:
        raise ValueError("resume state step lies outside the fixed trajectory")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng_state(payload["rng_state"], device)
    return (
        next_step,
        float(payload.get("elapsed_seconds", 0.0)),
        list(payload.get("metrics", [])),
    )


def _incumbent_identity(
    model: NeuralNP4G,
    split: FixedSplit,
    config: OfficialConfig,
    *,
    prune_guard_sha256: str,
    incumbent_guard_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": TRAINING_STATE_FORMAT_VERSION,
        "protocol_version": config.protocol_version,
        "protocol_sha256": _protocol_digest(config),
        "task": split.task.name,
        "formal_seed": split.formal_seed,
        "train_split_sha256": split.manifest.sha256(),
        "prune_guard_sha256": prune_guard_sha256,
        "incumbent_guard_sha256": incumbent_guard_sha256,
        "adf_candidates": _adf_candidate_identity(model),
    }


def _save_exact_incumbent(
    path: Path,
    *,
    model: NeuralNP4G,
    optimizer: torch.optim.Optimizer,
    step: int,
    guard_report: dict[str, Any],
    split: FixedSplit,
    prune_guard_sha256: str,
    incumbent_guard_sha256: str,
    config: OfficialConfig,
    device: torch.device,
) -> None:
    """Atomically preserve an all-TRAIN exact model and optimizer state."""

    if not _guard_all_zero(guard_report):
        raise ValueError("only an all-TRAIN exact state can become incumbent")
    if float(model.step_alpha) != 1.0:
        raise ValueError("only a fully hard step state can become incumbent")
    _atomic_torch_save(
        path,
        {
            **_incumbent_identity(
                model,
                split,
                config,
                prune_guard_sha256=prune_guard_sha256,
                incumbent_guard_sha256=incumbent_guard_sha256,
            ),
            "step": int(step),
            "step_alpha": float(model.step_alpha),
            "guard_report": guard_report,
            "structure": _accepted_prune_stats(model),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _rng_state(device),
        },
    )


def _restore_exact_incumbent(
    path: Path,
    *,
    model: NeuralNP4G,
    optimizer: torch.optim.Optimizer,
    split: FixedSplit,
    prune_guard_sha256: str,
    incumbent_guard_sha256: str,
    config: OfficialConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Restore and identity-check the last exact optimizer/pruning incumbent."""

    payload = _torch_load(path, device)
    expected = _incumbent_identity(
        model,
        split,
        config,
        prune_guard_sha256=prune_guard_sha256,
        incumbent_guard_sha256=incumbent_guard_sha256,
    )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"exact incumbent {key} mismatch: {payload.get(key)!r} != {value!r}"
            )
    if not _guard_all_zero(dict(payload.get("guard_report", {}))):
        raise ValueError("exact incumbent has no all-TRAIN zero-error witness")
    step_alpha = float(payload.get("step_alpha", -1.0))
    if step_alpha != 1.0:
        raise ValueError("exact incumbent is not a fully hard step state")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    model.step_alpha = step_alpha
    _restore_rng_state(payload["rng_state"], device)
    return payload


def _checkpoint_payload(
    model: NeuralNP4G,
    split: FixedSplit,
    config: OfficialConfig,
    *,
    stage: str,
    steps: int,
    prune_guard_sha256: str,
    incumbent_guard_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "protocol_version": config.protocol_version,
        "protocol_sha256": _protocol_digest(config),
        "task": split.task.name,
        "task_index": split.task.index,
        "domain": split.task.domain,
        "formal_seed": split.formal_seed,
        "train_split_sha256": split.manifest.sha256(),
        "prune_guard_sha256": prune_guard_sha256,
        "incumbent_guard_sha256": incumbent_guard_sha256,
        "stage": stage,
        "steps": steps,
        "adf_candidates": _adf_candidate_identity(model),
        "model_config": model.build_config(),
        "model_state": model.state_dict(),
    }


def _freeze_compression_checkpoint(
    destination: Path,
    *,
    model: NeuralNP4G,
    split: FixedSplit,
    config: OfficialConfig,
    step: int,
    elapsed_seconds: float,
    prune_guard_sha256: str,
    incumbent_guard_sha256: str,
) -> TrainingResult:
    """Freeze one all-TRAIN exact, proof-friendly intermediate checkpoint."""

    stage = f"compression_{int(step):05d}"
    stage_dir = destination / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = stage_dir / "checkpoint.pt"
    digest_path = stage_dir / "checkpoint.sha256"
    summary_path = stage_dir / "training_summary.json"
    structure = model.structure_summary()
    structure["compression"] = _accepted_prune_stats(model)

    if not checkpoint_path.exists():
        _atomic_torch_save(
            checkpoint_path,
            _checkpoint_payload(
                model,
                split,
                config,
                stage=stage,
                steps=step,
                prune_guard_sha256=prune_guard_sha256,
                incumbent_guard_sha256=incumbent_guard_sha256,
            ),
        )
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if digest_path.exists():
        fields = digest_path.read_text(encoding="ascii").split()
        if not fields or fields[0] != checkpoint_sha256:
            raise ValueError(f"{stage} checkpoint digest changed")
    else:
        _atomic_write(
            digest_path,
            f"{checkpoint_sha256}  {checkpoint_path.name}\n".encode("ascii"),
        )
    result = TrainingResult(
        task=split.task.name,
        formal_seed=split.formal_seed,
        stage=stage,
        steps=int(step),
        elapsed_seconds=float(elapsed_seconds),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        train_split_sha256=split.manifest.sha256(),
        metrics=(),
        structure=structure,
    )
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_summary = json.loads(_json_bytes(result.as_dict()).decode("utf-8"))
        # A crash after freezing the immutable compression checkpoint but
        # before the next resume-state write can replay this boundary.  Wall
        # time is not part of the checkpoint identity.
        expected_summary["elapsed_seconds"] = existing.get("elapsed_seconds")
        if existing != expected_summary:
            raise RuntimeError(f"{stage} training summary changed")
    else:
        _atomic_json(summary_path, result.as_dict())
    return result


def compression_training_results(output_dir: Path | str) -> tuple[TrainingResult, ...]:
    """Load frozen compression checkpoints in the fixed proof-priority order."""

    destination = Path(output_dir).resolve()
    results: list[TrainingResult] = []
    for summary_path in sorted(
        (destination / "stages").glob("compression_*/training_summary.json")
    ):
        record = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint_path = Path(record["checkpoint_path"])
        digest = _file_sha256(checkpoint_path)
        if digest != record.get("checkpoint_sha256"):
            raise ValueError(f"compression checkpoint changed: {checkpoint_path}")
        results.append(
            TrainingResult(
                task=str(record["task"]),
                formal_seed=int(record["formal_seed"]),
                stage=str(record["stage"]),
                steps=int(record["steps"]),
                elapsed_seconds=float(record["elapsed_seconds"]),
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=digest,
                train_split_sha256=str(record["train_split_sha256"]),
                metrics=tuple(record.get("metrics", ())),
                structure=dict(record["structure"]),
            )
        )

    def priority(result: TrainingResult) -> tuple[int, float, int]:
        compression = result.structure.get("compression", {})
        return (
            int(compression.get("max_fanin", 1 << 30)),
            float(compression.get("mean_fanin", float("inf"))),
            int(result.steps),
        )

    return tuple(sorted(results, key=priority))


@torch.no_grad()
def _semantic_compress_exact_model(
    model: NeuralNP4G,
    panel: Sequence[_PruneGuardCell],
    *,
    device: torch.device,
    config: OfficialConfig,
) -> dict[str, Any] | None:
    """Greedily shrink an exact model using only the complete TRAIN split.

    The order is fixed before any probe: hidden nodes by circuit index, then
    live edges by ``(absolute effective weight, destination, source)``.  A
    proposal is committed only when every materialised TRAIN cell remains
    exact.  No proof verdict, target formula, validation cell, or test cell is
    visible here.
    """

    if not panel:
        raise ValueError("semantic compression requires a non-empty TRAIN panel")
    event_batch = panel[0].batch
    baseline = _evaluate_prune_guard(
        model, panel, event_batch=event_batch, device=device
    )
    if not _guard_all_zero(baseline):
        return None

    core = model.core
    _, node_gates_before = core.gates()
    node_order = [
        int(index)
        for index in range(core.hidden_start, core.n_nodes)
        if float(node_gates_before[index].item()) >= 0.5
    ]
    accepted_nodes: list[int] = []
    rejected_nodes: list[int] = []
    for index in node_order:
        previous = core.node_logits[index].detach().clone()
        core.node_logits[index] = float(config.semantic_compression_off_logit)
        accepted = False
        try:
            accepted = _prune_guard_allows(
                model, panel, baseline, event_batch=event_batch, device=device
            )
        finally:
            if not accepted:
                core.node_logits[index] = previous
        if accepted:
            accepted_nodes.append(index)
        else:
            rejected_nodes.append(index)

    magnitude = core.effective_weight().detach().abs()
    alive = (
        (core.edge_mask * core.prune_mask) > 0
    ) & (magnitude > 0)
    edge_order = sorted(
        (
            (int(row), int(column))
            for row, column in torch.nonzero(alive, as_tuple=False).tolist()
        ),
        key=lambda item: (
            float(magnitude[item[0], item[1]].item()), item[0], item[1]
        ),
    )
    edge_trials = edge_order[: int(config.semantic_compression_max_edge_trials)]
    accepted_edges: list[tuple[int, int]] = []
    rejected_edges: list[tuple[int, int]] = []
    for row, column in edge_trials:
        previous = core.prune_mask[row, column].detach().clone()
        core.prune_mask[row, column] = 0.0
        accepted = False
        try:
            accepted = _prune_guard_allows(
                model, panel, baseline, event_batch=event_batch, device=device
            )
        finally:
            if not accepted:
                core.prune_mask[row, column] = previous
        if accepted:
            accepted_edges.append((row, column))
        else:
            rejected_edges.append((row, column))

    final_guard = _evaluate_prune_guard(
        model, panel, event_batch=event_batch, device=device
    )
    if not _guard_all_zero(final_guard):
        raise RuntimeError("semantic compression lost complete-TRAIN exactness")
    final_effective = core.effective_weight().detach().abs()
    final_live_edges = int(
        (
            ((core.edge_mask * core.prune_mask) > 0)
            & (final_effective > 0)
        ).sum().item()
    )
    return {
        "policy": "complete_train_greedy_nodes_then_edges_v1",
        "baseline_guard": baseline,
        "final_guard": final_guard,
        "node_order": node_order,
        "accepted_node_indices": accepted_nodes,
        "rejected_node_indices": rejected_nodes,
        "edge_order_sha256": _prune_edge_order_sha256(edge_order),
        "edge_candidates": len(edge_order),
        "edge_trials": len(edge_trials),
        "edge_trials_truncated": len(edge_trials) < len(edge_order),
        "accepted_edges": [[row, column] for row, column in accepted_edges],
        "rejected_edges": [[row, column] for row, column in rejected_edges],
        "final_live_edges": final_live_edges,
    }


def semantic_compression_result(
    model: NeuralNP4G,
    train_split: FixedSplit,
    output_dir: Path | str,
    *,
    device: torch.device | str,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> TrainingResult | None:
    """Freeze the preregistered post-phase-2 TRAIN-only compression stage.

    ``None`` means that the phase-2 checkpoint was not exact on the complete
    TRAIN split, so no semantic-compression stage is eligible.

    The caller supplies the frozen phase-2 model on each fresh attempt. Once
    written, the compression checkpoint is immutable; missing sidecars after
    an interrupted save are rebuilt from its own evidence without pruning
    again. Existing sidecars must agree with the entire frozen result.
    """

    _validate_training_inputs(model, train_split, config)
    if not config.semantic_compression:
        raise ValueError("semantic compression is disabled by the protocol")
    destination = Path(output_dir).resolve()
    stage = "semantic_compression"
    stage_dir = destination / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = stage_dir / "checkpoint.pt"
    digest_path = stage_dir / "checkpoint.sha256"
    summary_path = stage_dir / "training_summary.json"
    device = torch.device(device)
    model.to(device)
    model.eval()
    for entry in model.adf_input_specs:
        entry.model.to(device)
        entry.model.eval()

    _, prune_manifest = _build_prune_guard_panel(train_split, config)
    raw_full_guard, full_manifest = _build_incumbent_guard_panel(train_split, config)
    if any(
        cell["examples"] != cell["source_examples"]
        for cell in full_manifest["cells"]
    ):
        raise ValueError("semantic compression requires the complete materialised TRAIN split")
    prune_sha256 = str(prune_manifest["sha256"])
    full_sha256 = str(full_manifest["sha256"])
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "protocol_version": config.protocol_version,
        "protocol_sha256": _protocol_digest(config),
        "task": train_split.task.name,
        "formal_seed": train_split.formal_seed,
        "train_split_sha256": train_split.manifest.sha256(),
        "prune_guard_sha256": prune_sha256,
        "incumbent_guard_sha256": full_sha256,
        "stage": stage,
        "steps": config.total_steps,
        "adf_candidates": _adf_candidate_identity(model),
        "model_config": model.build_config(),
    }
    if checkpoint_path.exists():
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        if digest_path.exists():
            fields = digest_path.read_text(encoding="ascii").split()
            if fields != [checkpoint_sha256, checkpoint_path.name]:
                raise ValueError("semantic-compression checkpoint digest changed")
        payload = _torch_load(checkpoint_path, device)
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"semantic-compression checkpoint {key} mismatch: "
                    f"{payload.get(key)!r} != {value!r}"
                )
        model.load_state_dict(payload["model_state"], strict=True)
        compression = payload["semantic_compression"]
        elapsed = float(payload["semantic_compression_elapsed_seconds"])
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("semantic-compression elapsed time is invalid")
        if not all(
            _guard_all_zero(compression.get(key, {}))
            for key in ("baseline_guard", "final_guard")
        ):
            raise ValueError("semantic-compression checkpoint lacks TRAIN exactness evidence")
    else:
        if summary_path.exists() or digest_path.exists():
            raise FileExistsError(
                "semantic-compression metadata exists without its immutable checkpoint"
            )
        panel = _prepare_prune_guard_panel(raw_full_guard, model, device)
        started = time.monotonic()
        compression = _semantic_compress_exact_model(
            model, panel, device=device, config=config
        )
        if compression is None:
            logger.info(
                "Semantic compression ineligible task=%s seed=%s: phase2 is not exact on TRAIN",
                train_split.task.name, train_split.formal_seed,
            )
            return None
        elapsed = time.monotonic() - started
        payload = _checkpoint_payload(
            model,
            train_split,
            config,
            stage=stage,
            steps=config.total_steps,
            prune_guard_sha256=prune_sha256,
            incumbent_guard_sha256=full_sha256,
        )
        payload["semantic_compression"] = compression
        payload["semantic_compression_elapsed_seconds"] = elapsed
        _atomic_torch_save(checkpoint_path, payload)
        checkpoint_sha256 = _file_sha256(checkpoint_path)
    structure = model.structure_summary()
    structure["semantic_compression"] = compression
    result = TrainingResult(
        task=train_split.task.name,
        formal_seed=train_split.formal_seed,
        stage=stage,
        steps=config.total_steps,
        elapsed_seconds=elapsed,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        train_split_sha256=train_split.manifest.sha256(),
        metrics=(),
        structure=structure,
    )
    if summary_path.exists():
        record = json.loads(summary_path.read_text(encoding="utf-8"))
        if record != json.loads(_json_bytes(result.as_dict())):
            raise ValueError("semantic-compression training summary changed")
    if not digest_path.exists():
        _atomic_write(
            digest_path,
            f"{checkpoint_sha256}  {checkpoint_path.name}\n".encode("ascii"),
        )
    if not summary_path.exists():
        _atomic_json(summary_path, result.as_dict())
    logger.info(
        "Semantic compression checkpoint task=%s seed=%s elapsed=%.1fs path=%s sha256=%s",
        train_split.task.name, train_split.formal_seed, elapsed, checkpoint_path,
        checkpoint_sha256,
    )
    return result


def train_model(
    model: NeuralNP4G,
    train_split: FixedSplit,
    output_dir: Path | str,
    *,
    device: torch.device | str,
    stage: str,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> TrainingResult:
    """Run one fixed stage of the trajectory and freeze its checkpoint.

    The function has no arguments for any other data split and no accuracy
    threshold.  ``stage`` accepts only the preregistered ``phase1`` and
    ``phase2`` boundaries.  Phase 2 can only resume the optimiser/model/RNG
    state saved at or after phase 1; it cannot bypass the phase-1 checkpoint.
    """

    _validate_training_inputs(model, train_split, config)
    stage_steps = {
        "phase1": config.phase1_steps,
        "phase2": config.total_steps,
    }
    if stage not in stage_steps:
        raise ValueError("stage must be 'phase1' or 'phase2'")
    target_steps = stage_steps[stage]
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stage_dir = destination / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    logger.info(
        "Training task=%s seed=%s stage=%s device=%s output=%s",
        train_split.task.name, train_split.formal_seed, stage, device, destination,
    )
    raw_prune_guard, prune_guard_manifest = _build_prune_guard_panel(
        train_split, config
    )
    prune_guard_sha256 = str(prune_guard_manifest["sha256"])
    raw_incumbent_guard, incumbent_guard_manifest = _build_incumbent_guard_panel(
        train_split, config
    )
    incumbent_guard_sha256 = str(incumbent_guard_manifest["sha256"])

    protocol_path = destination / "protocol.json"
    # Convert tuples to their JSON representation before comparing with a
    # previously written record.  This makes a resumed invocation identical
    # to the first invocation instead of treating JSON lists as a mismatch.
    protocol_record = json.loads(
        _json_bytes(
            {
                "config": config.as_dict(),
                "config_sha256": _protocol_digest(config),
                "task": asdict(train_split.task),
                "formal_seed": train_split.formal_seed,
                "adf_candidates": _adf_candidate_identity(model),
                "prune_guard": prune_guard_manifest,
                "incumbent_guard": incumbent_guard_manifest,
            }
        ).decode("utf-8")
    )
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol_record:
            raise ValueError("output directory belongs to another formal protocol cell")
    else:
        _atomic_json(protocol_path, protocol_record)
    _atomic_json(destination / "split_manifest.json", train_split.manifest.as_dict())

    model.to(device)
    for entry in model.adf_input_specs:
        entry.model.to(device)
        entry.model.eval()
    prune_guard = _prepare_prune_guard_panel(raw_prune_guard, model, device)
    incumbent_guard = _prepare_prune_guard_panel(
        raw_incumbent_guard, model, device
    )
    model.sat_state_margin = config.state_saturation_margin
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.phase1_learning_rate,
    )

    resume_path = destination / "training_state.pt"
    checkpoint_path = stage_dir / "checkpoint.pt"
    digest_path = stage_dir / "checkpoint.sha256"
    summary_path = stage_dir / "training_summary.json"
    metric_path = destination / "train_metrics.jsonl"
    incumbent_path = destination / "exact_incumbent.pt"
    if checkpoint_path.exists() and not resume_path.exists():
        raise FileExistsError(
            f"{stage} checkpoint exists without its matching resume state; "
            "refusing to overwrite"
        )
    phase1_checkpoint = destination / "stages" / "phase1" / "checkpoint.pt"
    if stage == "phase2" and not phase1_checkpoint.exists():
        raise FileNotFoundError(
            "phase2 cannot start before the immutable phase1 checkpoint exists"
        )

    optimisation_seed = _seed_optimisation(
        train_split.formal_seed, train_split.task, device, config
    )
    start_step = 0
    elapsed_before = 0.0
    metrics: list[dict[str, Any]] = []
    if resume_path.exists():
        start_step, elapsed_before, metrics = _load_resume_state(
            resume_path,
            model=model,
            optimizer=optimizer,
            split=train_split,
            prune_guard_sha256=prune_guard_sha256,
            incumbent_guard_sha256=incumbent_guard_sha256,
            config=config,
            device=device,
        )

    if stage == "phase2" and start_step < config.phase1_steps:
        raise RuntimeError("phase2 resume state is earlier than the phase1 boundary")

    if resume_path.exists():
        logger.info("Resumed training at step=%s from %s", start_step, resume_path)

    stage_metrics = [
        row for row in metrics if int(row.get("step", target_steps + 1)) <= target_steps
    ]
    if checkpoint_path.exists():
        if start_step < target_steps:
            raise FileExistsError(
                f"{stage} checkpoint exists before its fixed boundary completed; "
                "refusing to overwrite it"
            )
        payload = _torch_load(checkpoint_path, device)
        expected_checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "protocol_version": config.protocol_version,
            "protocol_sha256": _protocol_digest(config),
            "task": train_split.task.name,
            "formal_seed": train_split.formal_seed,
            "train_split_sha256": train_split.manifest.sha256(),
            "prune_guard_sha256": prune_guard_sha256,
            "incumbent_guard_sha256": incumbent_guard_sha256,
            "stage": stage,
            "steps": target_steps,
        }
        for key, value in expected_checkpoint.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"{stage} checkpoint {key} mismatch: "
                    f"{payload.get(key)!r} != {value!r}"
                )
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        if digest_path.exists():
            fields = digest_path.read_text(encoding="ascii").split()
            if not fields or fields[0] != checkpoint_sha256:
                raise ValueError(f"{stage} checkpoint does not match checkpoint.sha256")
        else:
            _atomic_write(
                digest_path,
                f"{checkpoint_sha256}  {checkpoint_path.name}\n".encode("ascii"),
            )

        # A crash may happen after the immutable checkpoint was created but
        # before one of these derived reports was written.  Reconstruct them
        # deterministically; never rerun a step or rewrite checkpoint.pt.
        model.load_state_dict(payload["model_state"], strict=True)
        result = TrainingResult(
            task=train_split.task.name,
            formal_seed=train_split.formal_seed,
            stage=stage,
            steps=target_steps,
            elapsed_seconds=elapsed_before,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            train_split_sha256=train_split.manifest.sha256(),
            metrics=tuple(stage_metrics),
            structure=model.structure_summary(),
        )
        if not metric_path.exists():
            _write_metrics(metric_path, metrics)
        if not summary_path.exists():
            summary = result.as_dict()
            summary["optimisation_seed"] = optimisation_seed
            _atomic_json(summary_path, summary)
        logger.info("Reused frozen checkpoint=%s sha256=%s", checkpoint_path, checkpoint_sha256)
        return result

    if start_step > target_steps:
        raise FileNotFoundError(
            f"{stage} checkpoint is missing but resume state has already advanced "
            f"past step {target_steps}"
        )

    started = time.monotonic()
    last_prune: dict[str, Any] | None = (
        metrics[-1].get("last_prune") if metrics else None
    )
    model.train()
    for step in range(start_step, target_steps):
        (
            phase,
            phase_step,
            phase_steps,
            learning_rate,
            synthetic_weight,
            prune_interval,
            prune_fraction,
            target_fanin,
        ) = _phase(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        alpha_horizon = max(
            1,
            int(math.ceil(config.total_steps * config.step_alpha_ramp_fraction)),
        )
        model.step_alpha = min(1.0, (step + 1) / alpha_horizon)

        batch = train_split.training_batch(step, config.batch_size, device)
        if model.adf_input_specs:
            batch = augment_batch(batch, model.adf_input_specs)
        model.set_num_objects(batch.num_objects)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        data_loss = masked_balanced_bce(logits, batch)
        structure_loss = model.structure_loss()
        candidate_loss = model.candidate_loss()
        binarization_loss = model.binarization_loss()
        saturation_loss = model.last_saturation_loss()
        total_loss = (
            data_loss
            + config.structure_loss_weight * structure_loss
            + config.candidate_loss_weight * candidate_loss
            + config.binarization_loss_weight * binarization_loss
            + config.saturation_loss_weight * saturation_loss
        )
        synthetic_loss = total_loss.new_tensor(0.0)
        if synthetic_weight > 0.0:
            synthetic_loss = model.synthetic_saturation_loss(
                samples=config.synthetic_samples
            )
            total_loss = total_loss + synthetic_weight * synthetic_loss
        pre_update_accuracy, pre_update_wrong, pre_update_cells = _batch_accuracy(
            logits.detach(), batch
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        optimizer.step()

        phase_number = phase_step + 1
        ramp_steps = max(1, int(math.ceil(phase_steps * prune_fraction)))
        should_prune = (
            phase_number % prune_interval == 0
            or phase_number == ramp_steps
            or phase_number == phase_steps
        )
        if should_prune:
            fanin = _scheduled_fanin(
                model,
                phase,
                phase_step,
                phase_steps,
                prune_fraction,
                target_fanin,
                config,
            )
            with torch.no_grad():
                pre_prune_logits = model(batch)
            _, pre_prune_wrong, pre_prune_cells = _batch_accuracy(
                pre_prune_logits, batch
            )
            incumbent_enabled = float(model.step_alpha) == 1.0
            full_guard_before = (
                _evaluate_prune_guard(
                    model,
                    incumbent_guard,
                    event_batch=batch,
                    device=device,
                )
                if incumbent_enabled
                else None
            )
            optimizer_rollback = False
            incumbent_source_step: int | None = None
            if (
                full_guard_before is not None
                and not _guard_all_zero(full_guard_before)
                and incumbent_path.exists()
            ):
                incumbent = _restore_exact_incumbent(
                    incumbent_path,
                    model=model,
                    optimizer=optimizer,
                    split=train_split,
                    prune_guard_sha256=prune_guard_sha256,
                    incumbent_guard_sha256=incumbent_guard_sha256,
                    config=config,
                    device=device,
                )
                optimizer_rollback = True
                incumbent_source_step = int(incumbent["step"])
                with torch.no_grad():
                    pre_prune_logits = model(batch)
                _, pre_prune_wrong, pre_prune_cells = _batch_accuracy(
                    pre_prune_logits, batch
                )
                full_guard_before = _evaluate_prune_guard(
                    model,
                    incumbent_guard,
                    event_batch=batch,
                    device=device,
                )
                if not _guard_all_zero(full_guard_before):
                    raise RuntimeError("restored incumbent is not all-TRAIN exact")

            if full_guard_before is not None and _guard_all_zero(full_guard_before):
                _save_exact_incumbent(
                    incumbent_path,
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,
                    guard_report=full_guard_before,
                    split=train_split,
                    prune_guard_sha256=prune_guard_sha256,
                    incumbent_guard_sha256=incumbent_guard_sha256,
                    config=config,
                    device=device,
                )
            protected = pre_prune_wrong == 0
            guard_before = (
                _evaluate_prune_guard(
                    model,
                    prune_guard,
                    event_batch=batch,
                    device=device,
                )
                if protected
                else None
            )
            if protected and int(guard_before["event"]["wrong"]) != 0:
                raise RuntimeError("zero-error prune event was not reproducible")

            prune_mask_before = model.core.prune_mask.detach().clone()
            with torch.no_grad():
                magnitude_before = model.core.effective_weight().abs().detach().clone()
            proposed_stats: dict[str, Any] = dict(
                model.prune_step(fanin, return_kill_masks=True)
            )
            mask_after_standard = model.core.prune_mask.detach().clone()
            raw_kill_masks = proposed_stats.pop("_kill_masks", None)
            kill_masks = _normalise_prune_kill_masks(
                raw_kill_masks,
                prune_mask_before,
                mask_after_standard,
            )
            for name, mask in kill_masks.items():
                if int(proposed_stats.get(name, -1)) != int(mask.sum().item()):
                    raise RuntimeError(f"prune-step statistic {name!r} is not truthful")

            repair_before = model.core.prune_mask.detach().clone()
            repair_stats: dict[str, Any] = {}
            repair_ran = synthetic_weight > 0.0 and config.repair_pruning
            if repair_ran:
                repair_stats = dict(
                    model.prune_synthetic_repair(
                        n=config.repair_samples,
                        max_kills=config.repair_max_kills,
                    )
                )
            proposal_mask = model.core.prune_mask.detach().clone()
            repair_kill_mask = (repair_before > 0) & (proposal_mask <= 0)
            if repair_ran:
                if int(repair_stats.get("repair_kills", -1)) != int(
                    repair_kill_mask.sum().item()
                ):
                    raise RuntimeError("repair-pruning kill statistic is not truthful")
                kill_masks["repair_kills"] = repair_kill_mask
            elif bool(repair_kill_mask.any()):
                raise RuntimeError("prune mask changed although repair pruning did not run")
            proposed_stats.update(repair_stats)
            proposed_stats.update(_accepted_prune_stats(model))

            proposed_kills = (prune_mask_before > 0) & (proposal_mask <= 0)
            ordered_proposal = _ordered_prune_edges(
                proposed_kills, magnitude_before
            )
            order_sha256 = _prune_edge_order_sha256(ordered_proposal)
            guard_proposed: dict[str, Any] | None = None
            if protected:
                guard_proposed = _evaluate_prune_guard(
                    model,
                    prune_guard,
                    event_batch=batch,
                    device=device,
                )
                proposed_wrong = int(guard_proposed["event"]["wrong"])
                proposed_cells = int(guard_proposed["event"]["cells"])
                proposal_safe = _guard_non_regressing(
                    guard_before, guard_proposed
                )
            else:
                with torch.no_grad():
                    proposed_logits = model(batch)
                _, proposed_wrong, proposed_cells = _batch_accuracy(
                    proposed_logits, batch
                )
                proposal_safe = True
            if pre_prune_cells != proposed_cells or pre_update_cells != proposed_cells:
                raise RuntimeError("pruning changed the evaluated-cell mask")

            proposal_kills_total = len(ordered_proposal)
            partial_trials = 0
            partial_committed = 0
            partial_rejected = 0
            partial_truncated = False
            if protected and not proposal_safe:
                partial = _partial_commit_prune(
                    model,
                    prune_guard,
                    guard_before,
                    event_batch=batch,
                    mask_before=prune_mask_before,
                    proposal_mask=proposal_mask,
                    magnitude_before=magnitude_before,
                    max_trials=config.partial_prune_max_trials,
                    device=device,
                )
                accepted_mask = partial.pop("accepted_mask")
                guard_after = partial.pop("guard_after")
                order_sha256 = str(partial.pop("proposal_edge_order_sha256"))
                proposal_kills_total = int(partial.pop("proposal_kills_total"))
                partial_trials = int(partial.pop("partial_trials"))
                partial_committed = int(partial.pop("partial_committed_kills"))
                partial_rejected = int(partial.pop("partial_rejected_kills"))
                partial_truncated = bool(partial.pop("partial_truncated"))
                if partial:
                    raise RuntimeError("partial pruning returned unknown fields")
                commit_mode = "partial" if partial_committed else "rollback"
            else:
                accepted_mask = proposed_kills
                guard_after = guard_proposed
                commit_mode = "full" if protected else "unprotected"

            full_guard_after = (
                _evaluate_prune_guard(
                    model,
                    incumbent_guard,
                    event_batch=batch,
                    device=device,
                )
                if incumbent_enabled
                else None
            )
            if (
                full_guard_before is not None
                and full_guard_after is not None
                and incumbent_path.exists()
                and not _guard_all_zero(full_guard_after)
            ):
                incumbent_partial = _partial_commit_prune(
                    model=model,
                    panel=incumbent_guard,
                    baseline=full_guard_before,
                    event_batch=batch,
                    mask_before=prune_mask_before,
                    proposal_mask=proposal_mask,
                    magnitude_before=magnitude_before,
                    max_trials=config.partial_prune_max_trials,
                    device=device,
                )
                accepted_mask = incumbent_partial.pop("accepted_mask")
                full_guard_after = incumbent_partial.pop("guard_after")
                order_sha256 = str(
                    incumbent_partial.pop("proposal_edge_order_sha256")
                )
                proposal_kills_total = int(
                    incumbent_partial.pop("proposal_kills_total")
                )
                partial_trials = int(incumbent_partial.pop("partial_trials"))
                partial_committed = int(
                    incumbent_partial.pop("partial_committed_kills")
                )
                partial_rejected = int(
                    incumbent_partial.pop("partial_rejected_kills")
                )
                partial_truncated = bool(
                    incumbent_partial.pop("partial_truncated")
                )
                if incumbent_partial:
                    raise RuntimeError(
                        "full-TRAIN partial pruning returned unknown fields"
                    )
                guard_after = _evaluate_prune_guard(
                    model,
                    prune_guard,
                    event_batch=batch,
                    device=device,
                )
                if not _guard_all_zero(full_guard_after):
                    raise RuntimeError("full-TRAIN partial pruning lost exactness")
                commit_mode = (
                    "incumbent_partial"
                    if partial_committed
                    else "incumbent_rollback"
                )
            if full_guard_after is not None and _guard_all_zero(full_guard_after):
                _save_exact_incumbent(
                    incumbent_path,
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,
                    guard_report=full_guard_after,
                    split=train_split,
                    prune_guard_sha256=prune_guard_sha256,
                    incumbent_guard_sha256=incumbent_guard_sha256,
                    config=config,
                    device=device,
                )

            rollback = commit_mode in ("rollback", "incumbent_rollback")
            accepted_stats: dict[str, Any] = dict(proposed_stats)
            for name, mask in kill_masks.items():
                accepted_stats[name] = int((mask & accepted_mask).sum().item())
            if "repair_rows" in proposed_stats:
                accepted_repair_rows: list[dict[str, Any]] = []
                for repair_row in proposed_stats["repair_rows"]:
                    row = int(repair_row["row"])
                    column = int(repair_row["edge_src"])
                    if bool(accepted_mask[row, column]):
                        accepted_repair_rows.append(repair_row)
                accepted_stats["repair_rows"] = accepted_repair_rows
            accepted_stats.update(_accepted_prune_stats(model))

            accepted_post_wrong = (
                int(guard_after["event"]["wrong"])
                if guard_after is not None
                else proposed_wrong
            )

            last_prune = {
                "step": step + 1,
                "phase": phase,
                "fanin": fanin,
                **accepted_stats,
                "commit_mode": commit_mode,
                "rolled_back": rollback,
                "rollback_reason": (
                    (
                        "full_train_exact_incumbent_would_be_lost"
                        if commit_mode == "incumbent_rollback"
                        else "zero_error_solution_would_be_lost"
                    )
                    if rollback
                    else None
                ),
                "optimizer_updates_rolled_back": optimizer_rollback,
                "incumbent_source_step": incumbent_source_step,
                "pre_update_wrong": pre_update_wrong,
                "pre_prune_wrong": pre_prune_wrong,
                "proposed_post_prune_wrong": proposed_wrong,
                "accepted_post_prune_wrong": accepted_post_wrong,
                "cells": proposed_cells,
                "guard_policy": "fixed_train_multisize_v1",
                "guard_before": guard_before,
                "guard_proposed": guard_proposed,
                "guard_after": guard_after,
                "incumbent_guard_before": full_guard_before,
                "incumbent_guard_after": full_guard_after,
                "proposal_kills_total": proposal_kills_total,
                "accepted_kills_total": int(accepted_mask.sum().item()),
                "partial_trials": partial_trials,
                "partial_committed_kills": partial_committed,
                "partial_rejected_kills": partial_rejected,
                "partial_truncated": partial_truncated,
                "proposal_edge_order_sha256": order_sha256,
                "proposed": proposed_stats,
            }

            compression_stats = _accepted_prune_stats(model)
            if (
                step + 1 > config.phase1_steps
                and step + 1 < config.total_steps
                and full_guard_after is not None
                and _guard_all_zero(full_guard_after)
                and int(compression_stats["max_fanin"])
                <= config.compression_proof_max_fanin
            ):
                _freeze_compression_checkpoint(
                    destination,
                    model=model,
                    split=train_split,
                    config=config,
                    step=step + 1,
                    elapsed_seconds=(
                        elapsed_before + (time.monotonic() - started)
                    ),
                    prune_guard_sha256=prune_guard_sha256,
                    incumbent_guard_sha256=incumbent_guard_sha256,
                )

        if (
            step == start_step
            or (step + 1) % config.metric_interval == 0
            or step + 1 == target_steps
        ):
            reported_logits, reported_losses = _post_update_metrics(
                model,
                batch,
                synthetic_weight=synthetic_weight,
                config=config,
                device=device,
            )
            accuracy, wrong, cells = _batch_accuracy(reported_logits, batch)
            metrics.append(
                {
                    "step": step + 1,
                    "phase": phase,
                    "objects": batch.num_objects,
                    "learning_rate": learning_rate,
                    "step_alpha": float(model.step_alpha),
                    **reported_losses,
                    "optimisation_loss": float(total_loss.detach().cpu()),
                    "pre_update_accuracy": pre_update_accuracy,
                    "pre_update_wrong": pre_update_wrong,
                    "accuracy": accuracy,
                    "wrong": wrong,
                    "cells": cells,
                    "last_prune": last_prune,
                }
            )

        should_save = (
            (step + 1) % config.state_checkpoint_interval == 0
            or step + 1 == config.phase1_steps
            or step + 1 == config.total_steps
        )
        if should_save:
            elapsed = elapsed_before + (time.monotonic() - started)
            _save_resume_state(
                resume_path,
                model=model,
                optimizer=optimizer,
                next_step=step + 1,
                elapsed_seconds=elapsed,
                metrics=metrics,
                split=train_split,
                prune_guard_sha256=prune_guard_sha256,
                incumbent_guard_sha256=incumbent_guard_sha256,
                config=config,
                device=device,
            )
            _write_metrics(metric_path, metrics)
            logger.info(
                "Saved training state task=%s seed=%s stage=%s step=%s/%s elapsed=%.1fs",
                train_split.task.name, train_split.formal_seed, stage, step + 1,
                target_steps, elapsed,
            )

    elapsed = elapsed_before + (time.monotonic() - started)
    # The stage checkpoint is created only after its scheduled boundary.  Its
    # digest is therefore fixed before any extraction or evaluation can begin.
    _atomic_torch_save(
        checkpoint_path,
        _checkpoint_payload(
            model,
            train_split,
            config,
            stage=stage,
            steps=target_steps,
            prune_guard_sha256=prune_guard_sha256,
            incumbent_guard_sha256=incumbent_guard_sha256,
        ),
    )
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    _atomic_write(
        digest_path,
        f"{checkpoint_sha256}  {checkpoint_path.name}\n".encode("ascii"),
    )
    stage_metrics = [
        row for row in metrics if int(row.get("step", target_steps + 1)) <= target_steps
    ]
    result = TrainingResult(
        task=train_split.task.name,
        formal_seed=train_split.formal_seed,
        stage=stage,
        steps=target_steps,
        elapsed_seconds=elapsed,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        train_split_sha256=train_split.manifest.sha256(),
        metrics=tuple(stage_metrics),
        structure=model.structure_summary(),
    )
    summary = result.as_dict()
    summary["optimisation_seed"] = optimisation_seed
    _atomic_json(summary_path, summary)
    logger.info(
        "Training completed task=%s seed=%s stage=%s steps=%s elapsed=%.1fs checkpoint=%s sha256=%s",
        train_split.task.name, train_split.formal_seed, stage, target_steps, elapsed,
        checkpoint_path, checkpoint_sha256,
    )
    return result


def load_checkpoint(
    checkpoint_path: Path | str,
    learned_adfs: Sequence[ADFEntry] = (),
    *,
    device: torch.device | str = "cpu",
) -> tuple[NeuralNP4G, dict[str, Any]]:
    """Rebuild one frozen formal checkpoint for evaluation or extraction."""

    path = Path(checkpoint_path)
    device = torch.device(device)
    payload = _torch_load(path, device)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported stage checkpoint format")
    entries = tuple(learned_adfs)
    expected_candidates = tuple(
        str(item.get("task")) for item in payload.get("adf_candidates", ())
    )
    actual_candidates = tuple(entry.task for entry in entries)
    if actual_candidates != expected_candidates:
        raise ValueError(
            f"checkpoint requires ADF candidates {expected_candidates}, "
            f"got {actual_candidates}"
        )
    expected_checkpoints = tuple(
        item.get("checkpoint") for item in payload.get("adf_candidates", ())
    )
    actual_checkpoints = tuple(entry.checkpoint for entry in entries)
    if actual_checkpoints != expected_checkpoints:
        raise ValueError("checkpoint ADF candidate filenames do not match the library")
    expected_hashes = tuple(
        item.get("checkpoint_sha256") for item in payload.get("adf_candidates", ())
    )
    actual_hashes = tuple(entry.checkpoint_sha256 for entry in entries)
    if actual_hashes != expected_hashes:
        raise ValueError("checkpoint ADF candidate hashes do not match the library")
    model = NeuralNP4G.from_config(payload["model_config"], entries)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    logger.debug("Loaded checkpoint=%s task=%s stage=%s", path, payload.get("task"), payload.get("stage"))
    return model, payload


__all__ = [
    "TrainingResult",
    "build_model",
    "compression_training_results",
    "load_checkpoint",
    "masked_balanced_bce",
    "semantic_compression_result",
    "train_model",
]
