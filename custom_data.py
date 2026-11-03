"""Strict, bounded JSON input for empirical custom-task experiments.

Schema version 1 contains ``task`` and nonempty ``train``, ``validation``, and
``test`` lists under ``splits``. Each record is one graph, with ``input1`` of
shape [M, 1], ``input2`` of shape [M, M, 2 or 4], and unary [M] or binary [M, M]
``target``. An optional ``mask`` has the target's shape. All values must be
Boolean or numeric 0/1. No formulas, Python, pickle, or task dependencies are
accepted. Labels describe a finite empirical dataset, never a symbolic proof.
Omitted unary preprocessing enables the registered aggregation candidates;
an explicitly empty list requests node-only predictions and emits a warning.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import CellManifest, FixedSplit, GraphBatch, SplitManifest
from .preprocess import ops_for_site
from .tasks import TASKS, EmpiricalTaskSpec


CUSTOM_SCHEMA_VERSION = 1
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_OBJECTS = 256
MAX_EXAMPLES = 10_000
MAX_ARRAY_VALUES = 10_000_000
MAX_MESSAGE_TICKS = 32
_SPLIT_NAMES = ("train", "validation", "test")
_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


@dataclass(frozen=True)
class CustomTaskData:
    """Validated task, fixed CPU splits, and exact source-file provenance."""

    task: EmpiricalTaskSpec
    splits: dict[str, FixedSplit]
    document: dict[str, Any]
    source_sha256: str


def _object(
    value: Any, *, required: set[str], optional: set[str], location: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"{location} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{location} has unknown fields: {', '.join(sorted(unknown))}")
    return value


def _integer(value: Any, *, minimum: int, maximum: int, location: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{location} must be an integer in [{minimum}, {maximum}]")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _boolean_array(value: Any, shape: tuple[int, ...], location: str) -> np.ndarray:
    # Validate nested list lengths and leaf types before NumPy allocates an
    # array. In particular, numeric strings and ragged/object arrays fail.
    def validate(node: Any, dimensions: tuple[int, ...], path: str) -> None:
        if not dimensions:
            if type(node) not in (bool, int, float) or node not in (0, 1):
                raise ValueError(f"{path} must contain only Boolean or finite 0/1 values")
            return
        if not isinstance(node, list) or len(node) != dimensions[0]:
            raise ValueError(f"{path} has the wrong shape; {location} must have shape {shape}")
        for index, child in enumerate(node):
            validate(child, dimensions[1:], f"{path}[{index}]")

    validate(value, shape, location)
    # Canonical Boolean representation also makes 0, false, and -0.0 hash
    # identically when detecting leakage across splits.
    return np.asarray(value, dtype=np.bool_)


def _read_document(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        info = source.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("custom dataset must be a regular JSON file")
        if info.st_size > MAX_FILE_BYTES:
            raise ValueError(f"custom dataset exceeds the {MAX_FILE_BYTES}-byte file limit")
        with source.open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
    except OSError as error:
        raise ValueError(f"could not read custom dataset: {error}") from error
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"custom dataset exceeds the {MAX_FILE_BYTES}-byte file limit")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError(f"invalid custom dataset JSON: {error}") from error
    return (
        _object(
            document,
            required={"schema_version", "task", "splits"},
            optional=set(),
            location="dataset",
        ),
        hashlib.sha256(raw).hexdigest(),
    )


def _task_spec(value: Any) -> EmpiricalTaskSpec:
    fields = _object(
        value,
        required={"name", "domain", "target_arity"},
        optional={"message_ticks", "preprocess"},
        location="task",
    )
    name = fields["name"]
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError("task.name must match [a-z][a-z0-9_]{0,63}")
    if fields["domain"] not in ("graph", "family"):
        raise ValueError("task.domain must be 'graph' or 'family'")
    arity = _integer(fields["target_arity"], minimum=1, maximum=2, location="task.target_arity")
    ticks = _integer(
        fields.get("message_ticks", 1),
        minimum=1,
        maximum=MAX_MESSAGE_TICKS,
        location="task.message_ticks",
    )
    default_preprocess = (
        [op.name for op in ops_for_site("aggregate")] if arity == 1 else []
    )
    preprocess = fields.get("preprocess", default_preprocess)
    if not isinstance(preprocess, list) or any(not isinstance(op, str) for op in preprocess):
        raise ValueError("task.preprocess must be a list of primitive names")
    if arity == 1 and not preprocess:
        warnings.warn(
            "task.preprocess=[] for a unary task ignores all binary relations "
            "(including adjacency and family relations); omit preprocess to "
            "enable the default aggregation candidates",
            UserWarning,
            stacklevel=3,
        )
    return EmpiricalTaskSpec(
        index=len(TASKS),
        name=f"custom_{name}",
        domain=fields["domain"],
        target_arity=arity,
        message_ticks=ticks,
        formula="empirical labels supplied by the custom dataset",
        preprocess=tuple(preprocess),
    )


def _input_fingerprint(input1: np.ndarray, input2: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (input1, input2):
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _make_batch(records: list[tuple[np.ndarray, ...]]) -> GraphBatch:
    tensors = [
        torch.from_numpy(np.stack(column).astype(np.float32))
        for column in zip(*records)
    ]
    return GraphBatch(
        input0=torch.zeros((len(records), 1), dtype=torch.float32),
        input1=tensors[0],
        input2=tensors[1],
        input3=torch.zeros((len(records), 1, 1, 1, 1), dtype=torch.float32),
        target=tensors[2],
        pair_mask=tensors[3],
    )


def load_custom_task(
    path: str | Path, formal_seed: int, protocol_version: str
) -> CustomTaskData:
    """Load JSON as disjoint, fixed, empirical splits without executing code.

    Object counts are grouped in ascending order and example order is retained
    within each group. Binary targets must have zero diagonals; default masks
    exclude those diagonals, and supplied masks must do the same. Repeated
    relational inputs across splits are rejected even if their labels differ.

    ``CellManifest.data_seed`` is zero because no data are generated. The
    formal seed controls subsequent minibatch sampling, never input contents.
    """

    _integer(formal_seed, minimum=0, maximum=(1 << 63) - 1, location="formal_seed")
    if not isinstance(protocol_version, str) or not protocol_version or len(protocol_version) > 256:
        raise ValueError("protocol_version must be a nonempty string of at most 256 characters")
    document, source_sha256 = _read_document(path)
    _integer(
        document["schema_version"],
        minimum=CUSTOM_SCHEMA_VERSION,
        maximum=CUSTOM_SCHEMA_VERSION,
        location="schema_version",
    )
    task = _task_spec(document["task"])
    split_documents = _object(
        document["splits"],
        required=set(_SPLIT_NAMES),
        optional=set(),
        location="splits",
    )
    examples = 0
    for split_name in _SPLIT_NAMES:
        records = split_documents[split_name]
        if not isinstance(records, list) or not records:
            raise ValueError(f"splits.{split_name} must be a nonempty list")
        examples += len(records)
    if examples > MAX_EXAMPLES:
        raise ValueError(f"custom dataset exceeds the {MAX_EXAMPLES}-example limit")

    splits: dict[str, FixedSplit] = {}
    seen_inputs: dict[str, tuple[str, int]] = {}
    values = 0
    for split_name in _SPLIT_NAMES:
        grouped: dict[int, list[tuple[np.ndarray, ...]]] = {}
        for index, record in enumerate(split_documents[split_name]):
            location = f"splits.{split_name}[{index}]"
            record = _object(
                record,
                required={"input1", "input2", "target"},
                optional={"mask"},
                location=location,
            )
            if not isinstance(record["input1"], list):
                raise ValueError(f"{location}.input1 must have shape [M, 1]")
            objects = _integer(
                len(record["input1"]),
                minimum=1,
                maximum=MAX_OBJECTS,
                location=f"{location} object count",
            )
            target_shape = (objects,) * task.target_arity
            width = task.base_input_dims[2]
            values += 2 + objects + objects * objects * width + 2 * objects ** task.target_arity
            if values > MAX_ARRAY_VALUES:
                raise ValueError(f"custom dataset exceeds the {MAX_ARRAY_VALUES}-value tensor limit")
            input1 = _boolean_array(record["input1"], (objects, 1), f"{location}.input1")
            input2 = _boolean_array(record["input2"], (objects, objects, width), f"{location}.input2")
            target = _boolean_array(record["target"], target_shape, f"{location}.target")
            if "mask" in record:
                mask = _boolean_array(record["mask"], target_shape, f"{location}.mask")
            else:
                mask = np.ones(target_shape, dtype=np.bool_)
                if task.target_arity == 2:
                    np.fill_diagonal(mask, False)
            if task.domain == "graph":
                if bool(np.diag(input2[..., 0]).any()):
                    raise ValueError(f"{location}.input2 adjacency diagonal must be zero")
                if not np.array_equal(input2[..., 1], np.eye(objects, dtype=np.bool_)):
                    raise ValueError(f"{location}.input2 identity channel must equal the identity matrix")
            if task.target_arity == 2:
                if bool(np.diag(target).any()):
                    raise ValueError(f"{location}.target binary diagonal must be zero")
                if bool(np.diag(mask).any()):
                    raise ValueError(f"{location}.mask must exclude the binary diagonal")
            if not bool(mask.any()):
                raise ValueError(f"{location}.mask must select at least one evaluated cell")
            fingerprint = _input_fingerprint(input1, input2)
            earlier = seen_inputs.get(fingerprint)
            if earlier is not None and earlier[0] != split_name:
                raise ValueError(
                    f"duplicate relational inputs across splits: {location} and "
                    f"splits.{earlier[0]}[{earlier[1]}]"
                )
            seen_inputs.setdefault(fingerprint, (split_name, index))
            grouped.setdefault(objects, []).append((input1, input2, target, mask))

        batches: list[tuple[int, GraphBatch]] = []
        cells: list[CellManifest] = []
        for objects, records in sorted(grouped.items()):
            batch = _make_batch(records)
            evaluated = batch.target[batch.pair_mask > 0.5]
            positives = int(evaluated.sum().item())
            cells.append(
                CellManifest(
                    objects=objects,
                    examples=batch.batch_size,
                    data_seed=0,
                    sha256=batch.sha256(),
                    evaluated_cells=evaluated.numel(),
                    positives=positives,
                    negatives=evaluated.numel() - positives,
                )
            )
            batches.append((objects, batch))
        manifest = SplitManifest(
            protocol_version=protocol_version,
            task=task.name,
            task_index=task.index,
            split=split_name,
            formal_seed=formal_seed,
            cells=tuple(cells),
        )
        splits[split_name] = FixedSplit(task, split_name, formal_seed, tuple(batches), manifest)
    return CustomTaskData(task, splits, document, source_sha256)


__all__ = ["CUSTOM_SCHEMA_VERSION", "CustomTaskData", "load_custom_task"]
