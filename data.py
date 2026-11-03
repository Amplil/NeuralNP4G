"""Deterministic, disjoint data splits for the official experiment suite."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import OFFICIAL_CONFIG, OfficialConfig, derive_seed
from .tasks import TaskSpec, get_task


@dataclass(frozen=True)
class GraphBatch:
    """The standalone relational batch consumed by :class:`NeuralNP4G`."""

    input0: torch.Tensor
    input1: torch.Tensor
    input2: torch.Tensor
    input3: torch.Tensor
    target: torch.Tensor
    pair_mask: torch.Tensor

    def __post_init__(self) -> None:
        batch = int(self.input1.shape[0])
        objects = int(self.input1.shape[1])
        if self.input0.ndim != 2 or self.input0.shape[0] != batch:
            raise ValueError("input0 must have shape [B, D0]")
        if self.input1.ndim != 3:
            raise ValueError("input1 must have shape [B, M, D1]")
        if self.input2.ndim != 4:
            raise ValueError("input2 must have shape [B, M, M, D2]")
        if self.input2.shape[:3] != (batch, objects, objects):
            raise ValueError("input1/input2 object dimensions differ")
        if self.input3.ndim != 5 or self.input3.shape[0] != batch:
            raise ValueError("input3 must have shape [B, ..., D3]")
        if self.target.shape != self.pair_mask.shape:
            raise ValueError("target and pair_mask shapes differ")
        if self.target.shape not in ((batch, objects), (batch, objects, objects)):
            raise ValueError("target must be unary [B,M] or binary [B,M,M]")

    @property
    def batch_size(self) -> int:
        return int(self.input1.shape[0])

    @property
    def num_objects(self) -> int:
        return int(self.input1.shape[1])

    @property
    def target_arity(self) -> int:
        return self.target.ndim - 1

    def index_select(self, indices: torch.Tensor) -> "GraphBatch":
        indices = indices.to(device=self.input1.device, dtype=torch.long)
        return GraphBatch(
            *(tensor.index_select(0, indices) for tensor in self.tensors())
        )

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(*(tensor.to(device) for tensor in self.tensors()))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input0,
            self.input1,
            self.input2,
            self.input3,
            self.target,
            self.pair_mask,
        )

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in zip(
            ("input0", "input1", "input2", "input3", "target", "pair_mask"),
            self.tensors(),
        ):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(name.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CellManifest:
    objects: int
    examples: int
    data_seed: int
    sha256: str
    evaluated_cells: int
    positives: int
    negatives: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "objects": self.objects,
            "examples": self.examples,
            "data_seed": self.data_seed,
            "sha256": self.sha256,
            "evaluated_cells": self.evaluated_cells,
            "positives": self.positives,
            "negatives": self.negatives,
        }


@dataclass(frozen=True, slots=True)
class SplitManifest:
    protocol_version: str
    task: str
    task_index: int
    split: str
    formal_seed: int
    cells: tuple[CellManifest, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "task": self.task,
            "task_index": self.task_index,
            "split": self.split,
            "formal_seed": self.formal_seed,
            "cells": [cell.as_dict() for cell in self.cells],
            "sha256": self.sha256(),
        }

    def sha256(self) -> str:
        core = {
            "protocol_version": self.protocol_version,
            "task": self.task,
            "task_index": self.task_index,
            "split": self.split,
            "formal_seed": self.formal_seed,
            "cells": [cell.as_dict() for cell in self.cells],
        }
        encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FixedSplit:
    """Materialised CPU tensors and their immutable provenance manifest."""

    task: TaskSpec
    name: str
    formal_seed: int
    batches: tuple[tuple[int, GraphBatch], ...]
    manifest: SplitManifest

    def __post_init__(self) -> None:
        if self.name != self.manifest.split:
            raise ValueError("split name and manifest disagree")
        if self.task.name != self.manifest.task:
            raise ValueError("task and manifest disagree")
        if self.formal_seed != self.manifest.formal_seed:
            raise ValueError("formal seed and manifest disagree")
        sizes = [objects for objects, _ in self.batches]
        if len(sizes) != len(set(sizes)):
            raise ValueError("a split cannot contain a repeated object count")
        if tuple(sizes) != tuple(cell.objects for cell in self.manifest.cells):
            raise ValueError("batch cells and manifest cells disagree")

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(objects for objects, _ in self.batches)

    def batch_for_size(self, objects: int) -> GraphBatch:
        for size, batch in self.batches:
            if size == int(objects):
                return batch
        raise ValueError(f"split has no n={objects} cell")

    def training_batch(
        self,
        step: int,
        batch_size: int,
        device: torch.device | str,
    ) -> GraphBatch:
        """Return the deterministic minibatch assigned to one formal step."""

        if self.name != "train":
            raise ValueError("only the train split can supply optimisation batches")
        if step < 0 or batch_size <= 0:
            raise ValueError("step must be non-negative and batch_size positive")
        cell_index = int(step) % len(self.batches)
        objects, source = self.batches[cell_index]
        seed = derive_seed(
            self.formal_seed,
            self.task.index,
            "train_batch",
            int(step),
            int(objects),
        )
        generator = np.random.default_rng(seed)
        selected = generator.integers(
            0, source.batch_size, size=int(batch_size), dtype=np.int64
        )
        indices = torch.from_numpy(selected)
        return source.index_select(indices).to(device)


def _torch(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array.astype(np.float32, copy=False)))


def _compact_input3(batch: int) -> np.ndarray:
    # The official unary/binary model intentionally does not consume ternary
    # facts.  Retaining only the channel axis avoids O(M^3) dead storage.
    return np.zeros((batch, 1, 1, 1, 1), dtype=np.float32)


def _clear_diagonal(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    index = np.arange(values.shape[1])
    result[:, index, index] = False
    return result


def _boolean_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # float32 matrix multiplication is exact for the domain sizes in the
    # protocol (counts <= 500) and uses the available BLAS implementation.
    return np.matmul(left.astype(np.float32), right.astype(np.float32)) > 0.0


def _within_hops(adjacency: np.ndarray, hops: int) -> np.ndarray:
    reach = adjacency.copy()
    frontier = adjacency.copy()
    for _ in range(2, int(hops) + 1):
        frontier = _boolean_product(frontier, adjacency)
        reach |= frontier
        # Dense sampled graphs often saturate after two rounds.  Stopping when
        # every cell is reached preserves the exact Boolean result.
        if bool(reach.all()):
            break
    return reach


def _graph_batch(
    task: TaskSpec,
    examples: int,
    objects: int,
    rng: np.random.Generator,
    config: OfficialConfig,
) -> GraphBatch:
    probability = rng.uniform(
        config.graph_edge_probability_min,
        config.graph_edge_probability_max,
        size=(examples, 1, 1),
    )
    adjacency = rng.random((examples, objects, objects)) < probability
    diagonal = np.arange(objects)
    adjacency[:, diagonal, diagonal] = False
    red = rng.random((examples, objects)) < config.graph_red_probability

    if task.name == "adjacent_to_red":
        target = (adjacency & red[:, None, :]).any(axis=-1)
    elif task.name == "outdegree_1":
        target = adjacency.sum(axis=-1) == 1
    elif task.name == "outdegree_2":
        target = adjacency.sum(axis=-1) == 2
    elif task.name == "inverse_edge":
        target = adjacency.swapaxes(1, 2)
    elif task.name == "undirected_edge":
        target = adjacency | adjacency.swapaxes(1, 2)
    elif task.name == "graph_colouring":
        same_colour = red[:, :, None] == red[:, None, :]
        target = (adjacency & same_colour).any(axis=-1)
    elif task.name in ("connectivity_2", "connectivity_4", "connectivity_6"):
        target = _within_hops(adjacency, int(task.name.rsplit("_", 1)[1]))
    elif task.name == "two_hop":
        target = _clear_diagonal(_boolean_product(adjacency, adjacency))
    elif task.name == "four_hop_via_adf":
        # Binary ADF outputs have a zero diagonal by contract.  Build the
        # target from that same certified interface, not from an unmasked A^4.
        two_hop = _clear_diagonal(_boolean_product(adjacency, adjacency))
        target = _clear_diagonal(_boolean_product(two_hop, two_hop))
    else:
        raise ValueError(f"task '{task.name}' is not a graph-generator task")

    eye = np.eye(objects, dtype=np.float32)
    input2 = np.stack(
        [adjacency.astype(np.float32), np.broadcast_to(eye, adjacency.shape)],
        axis=-1,
    )
    if task.target_arity == 1:
        mask = np.ones((examples, objects), dtype=np.float32)
    else:
        target = _clear_diagonal(target)
        mask = np.ones((examples, objects, objects), dtype=np.float32)
        mask[:, diagonal, diagonal] = 0.0
    return GraphBatch(
        input0=torch.zeros((examples, 1), dtype=torch.float32),
        input1=_torch(red[..., None]),
        input2=_torch(input2),
        input3=_torch(_compact_input3(examples)),
        target=_torch(target),
        pair_mask=_torch(mask),
    )


def _family_relations(
    examples: int,
    objects: int,
    rng: np.random.Generator,
    parent_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    male = rng.random((examples, objects)) < 0.5
    father = np.zeros((examples, objects, objects), dtype=bool)
    mother = np.zeros_like(father)
    for child in range(1, objects):
        candidates = np.arange(child)
        for example in range(examples):
            older_male = candidates[male[example, :child]]
            older_female = candidates[~male[example, :child]]
            if older_male.size and rng.random() < parent_probability:
                father[example, child, int(rng.choice(older_male))] = True
            if older_female.size and rng.random() < parent_probability:
                mother[example, child, int(rng.choice(older_female))] = True

    parent = father | mother
    son = parent.swapaxes(1, 2) & male[:, None, :]
    daughter = parent.swapaxes(1, 2) & (~male[:, None, :])
    return male, father, mother, son, daughter


def _husband_counterfactual_batch(
    examples: int,
    objects: int,
    rng: np.random.Generator,
) -> GraphBatch:
    """Build a focused truth-table cell for the three required husband ADFs.

    Natural family structures make a co-parent pair consist of exactly one
    father and one mother.  That makes either role predicate redundant on the
    generator support.  This TRAIN/adf_causal-only cell instead realizes four
    relation patterns: the one positive conjunction and one counterexample for
    dropping each required atom.  Pair masks focus optimisation on the two
    ordered endpoints.  Optional has_father(x/y) values are independently
    varied so they cannot become replacement shortcuts.
    """

    if objects < 6:
        raise ValueError("husband counterfactual cells require at least 6 objects")
    male = rng.random((examples, objects)) < 0.5
    father = np.zeros((examples, objects, objects), dtype=bool)
    mother = np.zeros_like(father)
    mask = np.zeros((examples, objects, objects), dtype=np.float32)

    for example in range(examples):
        x, y, shared_child, other_child, x_parent, y_parent = (
            int(value) for value in rng.permutation(objects)[:6]
        )
        pattern = example % 4
        nuisance = (example // 4) % 4

        if pattern == 0:
            # co_parent=1, is_father(x)=1, is_mother(y)=1 -> husband=1
            father[example, shared_child, x] = True
            mother[example, shared_child, y] = True
            male[example, x] = True
            male[example, y] = False
        elif pattern == 1:
            # co_parent=1, is_father(x)=0, is_mother(y)=1 -> husband=0
            mother[example, shared_child, x] = True
            mother[example, shared_child, y] = True
            male[example, x] = False
            male[example, y] = False
        elif pattern == 2:
            # co_parent=1, is_father(x)=1, is_mother(y)=0 -> husband=0
            father[example, shared_child, x] = True
            father[example, shared_child, y] = True
            male[example, x] = True
            male[example, y] = True
        else:
            # is_father(x)=1 and is_mother(y)=1, but co_parent=0 -> husband=0
            father[example, shared_child, x] = True
            mother[example, other_child, y] = True
            male[example, x] = True
            male[example, y] = False

        if nuisance & 1:
            father[example, x, x_parent] = True
        if nuisance & 2:
            father[example, y, y_parent] = True
        mask[example, x, y] = 1.0
        mask[example, y, x] = 1.0

    parent = father | mother
    son = parent.swapaxes(1, 2) & male[:, None, :]
    daughter = parent.swapaxes(1, 2) & (~male[:, None, :])
    target = _clear_diagonal(_boolean_product(father.swapaxes(1, 2), mother))
    input2 = np.stack(
        [
            father.astype(np.float32),
            mother.astype(np.float32),
            son.astype(np.float32),
            daughter.astype(np.float32),
        ],
        axis=-1,
    )
    return GraphBatch(
        input0=torch.zeros((examples, 1), dtype=torch.float32),
        input1=torch.zeros((examples, objects, 1), dtype=torch.float32),
        input2=_torch(input2),
        input3=_torch(_compact_input3(examples)),
        target=_torch(target),
        pair_mask=_torch(mask),
    )


def _family_batch(
    task: TaskSpec,
    examples: int,
    objects: int,
    rng: np.random.Generator,
    config: OfficialConfig,
) -> GraphBatch:
    male, father, mother, son, daughter = _family_relations(
        examples,
        objects,
        rng,
        config.family_parent_probability,
    )
    parent = father | mother
    eye = np.eye(objects, dtype=bool)[None, :, :]
    sister = _boolean_product(mother, daughter) & ~eye
    child_of_parent = (son | daughter).swapaxes(1, 2)
    brother = _boolean_product(child_of_parent, son) & ~eye
    grandparent = _boolean_product(parent, parent)
    uncle = _boolean_product(parent, brother)

    if task.name == "has_father":
        target = father.any(axis=-1)
    elif task.name == "is_sister":
        target = sister
    elif task.name == "is_parent":
        target = parent
    elif task.name == "is_father":
        target = father.any(axis=1)
    elif task.name == "is_mother":
        target = mother.any(axis=1)
    elif task.name == "co_parent":
        target = _boolean_product(parent.swapaxes(1, 2), parent)
    elif task.name == "has_sister":
        target = sister.any(axis=-1)
    elif task.name == "is_brother":
        target = brother
    elif task.name == "is_grandparent":
        target = grandparent
    elif task.name == "is_uncle":
        target = uncle
    elif task.name == "is_mguncle":
        target = _boolean_product(mother, uncle)
    elif task.name == "husband":
        target = _boolean_product(father.swapaxes(1, 2), mother)
    else:
        raise ValueError(f"task '{task.name}' is not a family-generator task")

    input2 = np.stack(
        [
            father.astype(np.float32),
            mother.astype(np.float32),
            son.astype(np.float32),
            daughter.astype(np.float32),
        ],
        axis=-1,
    )
    if task.target_arity == 1:
        mask = np.ones((examples, objects), dtype=np.float32)
    else:
        target = _clear_diagonal(target)
        mask = np.ones((examples, objects, objects), dtype=np.float32)
        diagonal = np.arange(objects)
        mask[:, diagonal, diagonal] = 0.0
    return GraphBatch(
        input0=torch.zeros((examples, 1), dtype=torch.float32),
        input1=torch.zeros((examples, objects, 1), dtype=torch.float32),
        input2=_torch(input2),
        input3=_torch(_compact_input3(examples)),
        target=_torch(target),
        pair_mask=_torch(mask),
    )


def _cell_manifest(objects: int, seed: int, batch: GraphBatch) -> CellManifest:
    selected = batch.pair_mask > 0.5
    labels = batch.target[selected]
    positives = int((labels > 0.5).sum().item())
    cells = int(labels.numel())
    return CellManifest(
        objects=int(objects),
        examples=batch.batch_size,
        data_seed=int(seed),
        sha256=batch.sha256(),
        evaluated_cells=cells,
        positives=positives,
        negatives=cells - positives,
    )


def build_split(
    task: str | TaskSpec,
    split: str,
    formal_seed: int,
    *,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> FixedSplit:
    """Materialise one split from its independent preregistered seed stream."""

    spec = get_task(task)
    if formal_seed < 0:
        raise ValueError("formal_seed must be non-negative")
    natural_cells = config.cells_for(split)
    counterfactual_cell = None
    if spec.name == "husband" and split == "train":
        counterfactual_cell = config.husband_counterfactual_train_cell
    elif spec.name == "husband" and split == "adf_causal":
        counterfactual_cell = config.husband_counterfactual_adf_causal_cell
    cells = (
        (*natural_cells, counterfactual_cell)
        if counterfactual_cell is not None
        else natural_cells
    )
    batches: list[tuple[int, GraphBatch]] = []
    manifests: list[CellManifest] = []
    for cell_index, cell in enumerate(cells):
        is_counterfactual = (
            counterfactual_cell is not None and cell_index == len(natural_cells)
        )
        stream = (
            f"{split}_husband_counterfactual_data"
            if is_counterfactual
            else f"{split}_data"
        )
        data_seed = derive_seed(
            formal_seed,
            spec.index,
            stream,
            cell_index,
            cell.objects,
            cell.examples,
            config=config,
        )
        generator = np.random.default_rng(data_seed)
        if is_counterfactual:
            batch = _husband_counterfactual_batch(
                cell.examples, cell.objects, generator
            )
        elif spec.domain == "graph":
            batch = _graph_batch(spec, cell.examples, cell.objects, generator, config)
        else:
            batch = _family_batch(spec, cell.examples, cell.objects, generator, config)
        batches.append((cell.objects, batch))
        manifests.append(_cell_manifest(cell.objects, data_seed, batch))

    manifest = SplitManifest(
        protocol_version=config.protocol_version,
        task=spec.name,
        task_index=spec.index,
        split=split,
        formal_seed=int(formal_seed),
        cells=tuple(manifests),
    )
    return FixedSplit(spec, split, int(formal_seed), tuple(batches), manifest)


def build_all_splits(
    task: str | TaskSpec,
    formal_seed: int,
    *,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> dict[str, FixedSplit]:
    """Build all five independent splits without sharing generator state."""

    return {
        split: build_split(task, split, formal_seed, config=config)
        for split in (
            "train",
            "validation",
            "adf_causal",
            "test",
            "falsification",
        )
    }


__all__ = [
    "CellManifest",
    "FixedSplit",
    "GraphBatch",
    "SplitManifest",
    "build_all_splits",
    "build_split",
]
