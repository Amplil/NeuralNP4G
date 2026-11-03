"""Certified, ordered ADF storage and fixed downstream reuse semantics."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn

from .preprocess import PREPROCESS_VERSION


ADF_LIBRARY_FORMAT_VERSION = 2
VERIFIED_STATUS = "VERIFIED_EQUIVALENT"
_SAFE_TASK = re.compile(r"^[A-Za-z0-9_.-]+$")

# The historical candidate bank contained five method-level candidates.  A
# task-tied bank must not silently multiply the regularisation applied to each
# remaining candidate merely because irrelevant candidates were omitted.  In
# particular, averaging a singleton bank made the required ``cexists`` gate
# receive five times the pressure it received in the previously verified
# recipe.  Keep the per-candidate scale tied to this fixed method constant.
CANDIDATE_REGULARIZATION_REFERENCE_WIDTH = 5


class CandidateGateBank(nn.Module):
    """Independent exact-forward ON/OFF gates for primitives and learned ADFs."""

    def __init__(
        self,
        candidate_ids: Sequence[str],
        *,
        initial_logit: float = 2.0,
    ) -> None:
        super().__init__()
        identifiers = tuple(str(item) for item in candidate_ids)
        if any(not item for item in identifiers):
            raise ValueError("candidate identifiers must be non-empty")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate identifiers must be unique")
        self._candidate_ids = identifiers
        self._index = {item: index for index, item in enumerate(identifiers)}
        self.logits = nn.Parameter(
            torch.full((len(identifiers),), float(initial_logit))
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return self._candidate_ids

    def probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def forward(self) -> torch.Tensor:
        probability = self.probabilities()
        bit = (probability > 0.5).to(probability.dtype)
        return bit.detach() + (probability - probability.detach())

    def weight(self, candidate_id: str) -> torch.Tensor:
        try:
            index = self._index[candidate_id]
        except KeyError as error:
            raise KeyError(f"unknown candidate: {candidate_id}") from error
        return self()[index]

    def regularization_loss(self, *, activity_weight: float = 3.0) -> torch.Tensor:
        probability = self.probabilities()
        if probability.numel() == 0:
            return self.logits.sum() * 0.0
        normalizer = max(
            CANDIDATE_REGULARIZATION_REFERENCE_WIDTH,
            int(probability.numel()),
        )
        uncertainty = (probability * (1.0 - probability)).sum() / normalizer
        activity = probability.sum() / normalizer
        return uncertainty + float(activity_weight) * activity

    def summary(self) -> dict[str, float]:
        values = self().detach().cpu().tolist()
        return {
            identifier: float(value)
            for identifier, value in zip(self._candidate_ids, values)
        }

    def get_extra_state(self) -> dict[str, list[str]]:
        return {"candidate_ids": list(self._candidate_ids)}

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        loaded = tuple(str(item) for item in state.get("candidate_ids", []))
        if loaded != self._candidate_ids:
            raise RuntimeError(
                "candidate order mismatch: "
                f"checkpoint={loaded}, model={self._candidate_ids}"
            )


@dataclass
class ADFEntry:
    """One certified learned predicate and the input prefix it consumes."""

    task: str
    domain: str
    arity: int
    input_dims: tuple[int, int, int, int]
    model: nn.Module
    dependencies: tuple[str, ...] = ()
    checkpoint_sha256: str | None = None
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("task must be non-empty")
        if not self.domain:
            raise ValueError("domain must be non-empty")
        if int(self.arity) not in (1, 2):
            raise ValueError("arity must be 1 or 2")
        self.arity = int(self.arity)
        dims = tuple(int(value) for value in self.input_dims)
        if len(dims) != 4 or any(value < 0 for value in dims):
            raise ValueError("input_dims must contain four non-negative widths")
        self.input_dims = dims
        self.dependencies = tuple(str(item) for item in self.dependencies)
        if self.task in self.dependencies:
            raise ValueError("an ADF cannot depend on itself")
        if self.checkpoint_sha256 is not None:
            self.checkpoint_sha256 = str(self.checkpoint_sha256)
            if len(self.checkpoint_sha256) != 64:
                raise ValueError("checkpoint_sha256 must be a 64-character digest")
        if self.checkpoint is not None:
            self.checkpoint = str(self.checkpoint)
            if not self.checkpoint:
                raise ValueError("checkpoint must be a non-empty relative name")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def candidate_id(self) -> str:
        return f"adf:{self.task}"

    @property
    def extra_unary(self) -> int:
        return int(self.arity == 1)

    @property
    def extra_binary(self) -> int:
        return int(self.arity == 2)


def candidate_ids(entries: Sequence[ADFEntry]) -> tuple[str, ...]:
    """Build the ordered learned-ADF candidate list used by a model."""

    result = tuple(entry.candidate_id for entry in entries)
    if len(set(result)) != len(result):
        raise ValueError("learned candidate identifiers must be unique")
    return result


def adf_extra_dims(entries: Sequence[ADFEntry]) -> tuple[int, int]:
    return (
        sum(entry.extra_unary for entry in entries),
        sum(entry.extra_binary for entry in entries),
    )


def _replace_batch_inputs(
    batch: Any,
    input1: torch.Tensor,
    input2: torch.Tensor,
) -> Any:
    if dataclasses.is_dataclass(batch):
        return dataclasses.replace(batch, input1=input1, input2=input2)
    if hasattr(batch, "_replace"):
        return batch._replace(input1=input1, input2=input2)
    clone = copy.copy(batch)
    try:
        clone.input1 = input1
        clone.input2 = input2
    except (AttributeError, TypeError) as error:
        raise TypeError(
            "batch must be a dataclass, named tuple, or mutable object with input1/input2"
        ) from error
    return clone


def augment_batch(batch: Any, entries: Sequence[ADFEntry]) -> Any:
    """Append frozen, hard ADF outputs in dependency order.

    A unary result is appended to ``input1``.  A binary result is appended to
    ``input2`` after its diagonal is cleared.  Entry ``k`` sees only the input
    prefix recorded when it was trained.
    """

    if not entries:
        return batch
    if not hasattr(batch, "input1") or not hasattr(batch, "input2"):
        raise TypeError("batch must provide input1 and input2 tensors")

    input1 = batch.input1
    input2 = batch.input2
    if not isinstance(input1, torch.Tensor) or not isinstance(input2, torch.Tensor):
        raise TypeError("batch input1 and input2 must be tensors")
    if input1.ndim != 3:
        raise ValueError("input1 must have shape [B, M, D1]")
    if input2.ndim != 4:
        raise ValueError("input2 must have shape [B, M, M, D2]")
    if input2.shape[1] != input2.shape[2] or input1.shape[:2] != input2.shape[:2]:
        raise ValueError("input1 and input2 object dimensions differ")

    batch_size, objects = int(input1.shape[0]), int(input1.shape[1])
    available: list[str] = []
    # ``no_grad`` keeps the frozen callees out of autograd while returning
    # ordinary tensors that a downstream trainable model may safely consume.
    with torch.no_grad():
        for entry in entries:
            missing = [item for item in entry.dependencies if item not in available]
            if missing:
                raise ValueError(
                    f"ADF '{entry.task}' has unavailable dependencies {missing}"
                )
            d1 = int(entry.input_dims[1])
            d2 = int(entry.input_dims[2])
            if input1.shape[-1] < d1 or input2.shape[-1] < d2:
                raise ValueError(
                    f"ADF '{entry.task}' expects input widths ({d1}, {d2}), "
                    f"but only ({input1.shape[-1]}, {input2.shape[-1]}) are available"
                )
            sub_batch = _replace_batch_inputs(
                batch,
                input1[..., :d1],
                input2[..., :d2],
            )
            entry.model.eval()
            logits = entry.model(sub_batch)
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"ADF '{entry.task}' did not return a tensor")

            if entry.arity == 1:
                expected = (batch_size, objects)
                if tuple(logits.shape) != expected:
                    raise ValueError(
                        f"ADF '{entry.task}' returned {tuple(logits.shape)}, expected {expected}"
                    )
                value = (logits >= 0).to(dtype=input1.dtype, device=input1.device)
                input1 = torch.cat([input1, value.unsqueeze(-1)], dim=-1)
            else:
                expected = (batch_size, objects, objects)
                if tuple(logits.shape) != expected:
                    raise ValueError(
                        f"ADF '{entry.task}' returned {tuple(logits.shape)}, expected {expected}"
                    )
                value = (logits >= 0).to(dtype=input2.dtype, device=input2.device)
                eye = torch.eye(objects, device=input2.device, dtype=input2.dtype)
                value = value * (1.0 - eye).unsqueeze(0)
                input2 = torch.cat([input2, value.unsqueeze(-1)], dim=-1)
            available.append(entry.task)

    return _replace_batch_inputs(batch, input1, input2)


class ADFAugmentedData:
    """Apply the one official ADF reuse rule to every sampled batch."""

    def __init__(self, inner: Any, entries: Sequence[ADFEntry]) -> None:
        self.inner = inner
        self.entries = tuple(entries)

    @property
    def backend(self) -> str:
        return str(getattr(self.inner, "backend", "unknown"))

    def sample(self, batch: int, n: int, device: torch.device) -> Any:
        return augment_batch(self.inner.sample(batch, n, device), self.entries)


ModelFactory = Callable[[Mapping[str, Any], Sequence[ADFEntry]], nn.Module]


class ADFLibrary:
    """Directory-backed ordered store containing certified learned predicates."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def _empty_index(self) -> dict[str, Any]:
        return {
            "format_version": ADF_LIBRARY_FORMAT_VERSION,
            "primitive_registry_version": PREPROCESS_VERSION,
            "entries": [],
        }

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return self._empty_index()
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if int(payload.get("format_version", -1)) != ADF_LIBRARY_FORMAT_VERSION:
            raise ValueError("unsupported ADF library format")
        if int(payload.get("primitive_registry_version", -1)) != PREPROCESS_VERSION:
            raise ValueError("ADF library preprocess ops do not match this model")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("ADF library index has no entries list")
        return payload

    def tasks(self, *, domain: str | None = None) -> list[str]:
        records = self._read_index()["entries"]
        return [
            str(record["task"])
            for record in records
            if domain is None or str(record["domain"]) == domain
        ]

    def has(self, task: str) -> bool:
        return task in self.tasks()

    def metadata(self) -> list[dict[str, Any]]:
        records = [dict(record) for record in self._read_index()["entries"]]
        for record in records:
            self._verify_checkpoint(record)
        return records

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _verify_checkpoint(self, record: Mapping[str, Any]) -> Path:
        checkpoint = self.root / str(record["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing ADF checkpoint: {checkpoint}")
        expected = record.get("checkpoint_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(
                f"ADF '{record.get('task')}' has no valid checkpoint digest"
            )
        actual = self._file_sha256(checkpoint)
        if actual != expected:
            raise ValueError(
                f"ADF '{record.get('task')}' checkpoint SHA256 mismatch"
            )
        return checkpoint

    def save_verified(
        self,
        *,
        task: str,
        domain: str,
        arity: int,
        model: nn.Module,
        input_dims: tuple[int, int, int, int],
        build_config: Mapping[str, Any],
        dependencies: Sequence[str] = (),
        verification_status: str,
        extra_metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> bool:
        """Store a model only after the caller supplies the certified result."""

        if verification_status != VERIFIED_STATUS:
            raise ValueError(
                f"ADF promotion requires {VERIFIED_STATUS}, got {verification_status!r}"
            )
        if not _SAFE_TASK.fullmatch(task):
            raise ValueError("task contains characters unsafe for a checkpoint name")
        if not domain:
            raise ValueError("domain must be non-empty")
        arity = int(arity)
        if arity not in (1, 2):
            raise ValueError("arity must be 1 or 2")
        dims = tuple(int(value) for value in input_dims)
        if len(dims) != 4 or any(value < 0 for value in dims):
            raise ValueError("input_dims must contain four non-negative widths")
        dependency_names = tuple(str(item) for item in dependencies)
        if len(set(dependency_names)) != len(dependency_names):
            raise ValueError("dependencies must be unique")
        if task in dependency_names:
            raise ValueError("an ADF cannot depend on itself")

        index = self._read_index()
        records: list[dict[str, Any]] = index["entries"]
        existing_position = next(
            (position for position, item in enumerate(records) if item["task"] == task),
            None,
        )
        if existing_position is not None and not overwrite:
            return False

        earlier = records if existing_position is None else records[:existing_position]
        earlier_by_task = {str(item["task"]): item for item in earlier}
        missing = [item for item in dependency_names if item not in earlier_by_task]
        if missing:
            raise ValueError(f"dependencies must precede the new ADF: {missing}")
        wrong_domain = [
            item
            for item in dependency_names
            if str(earlier_by_task[item]["domain"]) != domain
        ]
        if wrong_domain:
            raise ValueError(f"dependencies belong to another domain: {wrong_domain}")

        config = dict(build_config)
        extra = dict(extra_metadata or {})
        json.dumps({"build_config": config, "extra_metadata": extra})
        checkpoint_name = f"{task}.pt"
        record = {
            "task": task,
            "domain": domain,
            "arity": arity,
            "input_dims": list(dims),
            "dependencies": list(dependency_names),
            "build_config": config,
            "extra_metadata": extra,
            "verification_status": VERIFIED_STATUS,
            "primitive_registry_version": PREPROCESS_VERSION,
            "checkpoint": checkpoint_name,
        }

        state = {
            key: (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else copy.deepcopy(value)
            )
            for key, value in model.state_dict().items()
        }
        self.root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.root / checkpoint_name
        checkpoint_tmp = self.root / f".{checkpoint_name}.tmp"
        torch.save({"state_dict": state}, checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint_path)
        record["checkpoint_sha256"] = self._file_sha256(checkpoint_path)

        if existing_position is None:
            records.append(record)
        else:
            records[existing_position] = record
        self._write_index(index)
        return True

    def _write_index(self, payload: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / ".index.json.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.index_path)

    def load_entries(
        self,
        *,
        domain: str,
        model_factory: ModelFactory,
        before_task: str | None = None,
        device: torch.device | str = "cpu",
    ) -> list[ADFEntry]:
        """Rebuild one domain's certified prefix in stored dependency order."""

        records = self._read_index()["entries"]
        loaded: list[ADFEntry] = []
        by_task: dict[str, ADFEntry] = {}
        for raw_record in records:
            record = dict(raw_record)
            task = str(record["task"])
            if before_task is not None and task == before_task:
                break
            if str(record["domain"]) != domain:
                continue
            if record.get("verification_status") != VERIFIED_STATUS:
                raise ValueError(f"ADF '{task}' is not certified")
            if int(record.get("primitive_registry_version", -1)) != PREPROCESS_VERSION:
                raise ValueError(f"ADF '{task}' uses another preprocess op set")

            dependency_names = tuple(str(item) for item in record.get("dependencies", []))
            missing = [item for item in dependency_names if item not in by_task]
            if missing:
                raise ValueError(f"ADF '{task}' has missing earlier dependencies {missing}")
            dependency_entries = [by_task[item] for item in dependency_names]
            model = model_factory(record, dependency_entries)
            if not isinstance(model, nn.Module):
                raise TypeError("model_factory must return torch.nn.Module")

            checkpoint = self._verify_checkpoint(record)
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            if not isinstance(payload, Mapping) or "state_dict" not in payload:
                raise ValueError(f"invalid ADF checkpoint: {checkpoint}")
            model.load_state_dict(payload["state_dict"], strict=True)
            model.to(device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

            dims = tuple(int(value) for value in record["input_dims"])
            entry = ADFEntry(
                task=task,
                domain=domain,
                arity=int(record["arity"]),
                input_dims=dims,  # type: ignore[arg-type]
                model=model,
                dependencies=dependency_names,
                checkpoint_sha256=str(record["checkpoint_sha256"]),
                checkpoint=str(record["checkpoint"]),
            )
            loaded.append(entry)
            by_task[task] = entry
        return loaded


__all__ = [
    "ADFEntry",
    "ADFAugmentedData",
    "ADFLibrary",
    "ADF_LIBRARY_FORMAT_VERSION",
    "CandidateGateBank",
    "VERIFIED_STATUS",
    "adf_extra_dims",
    "augment_batch",
    "candidate_ids",
]
