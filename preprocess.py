"""Candidate-gated relational primitives for the official Neural NP4G model.

Each official task names an ordered subset of these ops.  Every listed op is
registered in the same hard ON/OFF candidate bank as learned ADFs and is
applied at every message tick only when its learned gate is ON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import torch


PreprocessSite = Literal["aggregate", "join"]
WidthFunction = Callable[[int], int]
PreprocessFunction = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]

CRISP_SCALE = 4.0
EXISTS_MARGIN = 0.5
COUNT_CAP = 8.0
PAIR_JOIN_BLOCK = 2


@dataclass(frozen=True, slots=True)
class PreprocessOp:
    """One aggregation or join used as circuit-input preprocessing.

    ``aggregate`` ops consume ``(relation, state)`` with shapes ``[B, M, M]``
    and ``[B, M, C]``.  ``join`` ops consume a pair state ``[B, M, M, C]`` as
    the first argument and require the second argument to be ``None``.
    """

    name: str
    site: PreprocessSite
    abstract_kind: str
    _width: WidthFunction = field(repr=False, compare=False)
    _function: PreprocessFunction = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name or ":" in self.name:
            raise ValueError("preprocess name must be a non-empty simple name")
        if self.site not in ("aggregate", "join"):
            raise ValueError(f"unknown preprocess site: {self.site}")
        if not self.abstract_kind:
            raise ValueError("abstract_kind must be non-empty")

    @property
    def op_id(self) -> str:
        """Stable identifier used by checkpoints and task preprocessing."""

        return f"primitive:{self.site}:{self.name}"

    def output_width(self, channels: int) -> int:
        """Number of output channels contributed for state width ``channels``."""

        channels = int(channels)
        if channels <= 0:
            raise ValueError("channels must be positive")
        width = int(self._width(channels))
        if width <= 0:
            raise ValueError(f"preprocess {self.op_id} produced invalid width {width}")
        return width

    def apply(
        self,
        first: torch.Tensor,
        second: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the op to tensors appropriate for its site."""

        return self._function(first, second)

    def __call__(
        self,
        first: torch.Tensor,
        second: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.apply(first, second)


def _same_width(channels: int) -> int:
    return channels


def _pair_width(channels: int) -> int:
    block = min(PAIR_JOIN_BLOCK, channels)
    return block * block


def _validate_aggregate_inputs(
    relation: torch.Tensor,
    state: torch.Tensor | None,
) -> torch.Tensor:
    if state is None:
        raise ValueError("aggregate preprocess requires a state tensor")
    if relation.ndim != 3:
        raise ValueError("relation must have shape [B, M, M]")
    if state.ndim != 3:
        raise ValueError("state must have shape [B, M, C]")
    if relation.shape[0] != state.shape[0]:
        raise ValueError("relation and state batch sizes differ")
    if relation.shape[1] != relation.shape[2] or relation.shape[1] != state.shape[1]:
        raise ValueError("relation and state object dimensions differ")
    if state.shape[1] == 0 or state.shape[2] == 0:
        raise ValueError("object and channel dimensions must be positive")
    return state


def _validate_join_input(pair: torch.Tensor, unused: torch.Tensor | None) -> torch.Tensor:
    if unused is not None:
        raise ValueError("join preprocess accepts only the pair-state tensor")
    if pair.ndim != 4:
        raise ValueError("pair state must have shape [B, M, M, C]")
    if pair.shape[1] != pair.shape[2]:
        raise ValueError("pair-state object dimensions differ")
    if pair.shape[1] == 0 or pair.shape[3] == 0:
        raise ValueError("object and channel dimensions must be positive")
    return pair


def _aggregate_exists(relation: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
    state = _validate_aggregate_inputs(relation, state)
    truth = torch.sigmoid(state * CRISP_SCALE)
    contribution = torch.relu(
        relation.unsqueeze(-1) * truth.unsqueeze(1) - EXISTS_MARGIN
    )
    return contribution.amax(dim=2)


def _aggregate_forall(relation: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
    state = _validate_aggregate_inputs(relation, state)
    truth = torch.sigmoid(state * CRISP_SCALE)
    violation = torch.relu(
        relation.unsqueeze(-1) * (1.0 - truth).unsqueeze(1) - EXISTS_MARGIN
    )
    return 1.0 - violation.amax(dim=2)


def _aggregate_count(relation: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
    state = _validate_aggregate_inputs(relation, state)
    truth = torch.sigmoid(state * CRISP_SCALE)
    contribution = torch.relu(
        relation.unsqueeze(-1) * truth.unsqueeze(1) - EXISTS_MARGIN
    )
    return contribution.sum(dim=2).clamp(max=COUNT_CAP)


def _join_exists(pair: torch.Tensor, unused: torch.Tensor | None) -> torch.Tensor:
    pair = _validate_join_input(pair, unused)
    channels = int(pair.shape[-1])
    truth = torch.sigmoid(pair * CRISP_SCALE)
    outputs: list[torch.Tensor] = []
    for channel in range(channels):
        values = truth[..., channel]
        left = values.unsqueeze(2)
        right = values.transpose(1, 2).unsqueeze(1)
        contribution = torch.relu(left * right - EXISTS_MARGIN)
        outputs.append(contribution.amax(dim=3))
    return torch.stack(outputs, dim=-1)


def _join_pair_exists(pair: torch.Tensor, unused: torch.Tensor | None) -> torch.Tensor:
    pair = _validate_join_input(pair, unused)
    block = min(PAIR_JOIN_BLOCK, int(pair.shape[-1]))
    truth = torch.sigmoid(pair * CRISP_SCALE)
    outputs: list[torch.Tensor] = []
    for left_channel in range(block):
        left = truth[..., left_channel].unsqueeze(2)
        for right_channel in range(block):
            right = truth[..., right_channel].transpose(1, 2).unsqueeze(1)
            contribution = torch.relu(left * right - EXISTS_MARGIN)
            outputs.append(contribution.amax(dim=3))
    return torch.stack(outputs, dim=-1)


PREPROCESS_VERSION = 3

PREPROCESS_OPS: tuple[PreprocessOp, ...] = (
    PreprocessOp("cexists", "aggregate", "exists", _same_width, _aggregate_exists),
    PreprocessOp("cforall", "aggregate", "forall", _same_width, _aggregate_forall),
    PreprocessOp("satcount", "aggregate", "count", _same_width, _aggregate_count),
    PreprocessOp("cexists", "join", "exists_join", _same_width, _join_exists),
    PreprocessOp(
        "cexistsp_r",
        "join",
        "pair_exists_join",
        _pair_width,
        _join_pair_exists,
    ),
)


def ops_for_site(
    site: PreprocessSite,
    registry: Sequence[PreprocessOp] = PREPROCESS_OPS,
) -> tuple[PreprocessOp, ...]:
    """Return the ordered preprocess ops available at one relational site."""

    if site not in ("aggregate", "join"):
        raise ValueError(f"unknown preprocess site: {site}")
    return tuple(op for op in registry if op.site == site)


def site_for_arity(arity: int) -> PreprocessSite:
    """Map a task arity onto the preprocess site it uses."""

    if int(arity) == 1:
        return "aggregate"
    if int(arity) == 2:
        return "join"
    raise ValueError("arity must be 1 or 2")


def lookup_ops(
    names: Sequence[str],
    site: PreprocessSite,
    registry: Sequence[PreprocessOp] = PREPROCESS_OPS,
) -> tuple[PreprocessOp, ...]:
    """Resolve an ordered task preprocess list against one site."""

    available = ops_for_site(site, registry)
    by_name = {op.name: op for op in available}
    if len(by_name) != len(available):
        raise ValueError(f"preprocess ops have duplicate {site} names")
    result: list[PreprocessOp] = []
    seen: set[str] = set()
    for name in names:
        if not name:
            raise ValueError("preprocess name must be non-empty")
        if name in seen:
            raise ValueError(f"task preprocess repeats {name}")
        seen.add(name)
        try:
            result.append(by_name[name])
        except KeyError as error:
            raise ValueError(f"unknown {site} preprocess op: {name}") from error
    return tuple(result)


def ops_from_ids(
    identifiers: Sequence[str],
    registry: Sequence[PreprocessOp] = PREPROCESS_OPS,
) -> tuple[PreprocessOp, ...]:
    """Rebuild preprocessing from checkpoint identifiers."""

    by_id = {op.op_id: op for op in registry}
    if len(by_id) != len(registry):
        raise ValueError("preprocess ops contain duplicate identifiers")
    result: list[PreprocessOp] = []
    seen: set[str] = set()
    for identifier in identifiers:
        if identifier in seen:
            raise ValueError(f"duplicate preprocess id {identifier}")
        seen.add(identifier)
        try:
            result.append(by_id[identifier])
        except KeyError as error:
            raise ValueError(f"unknown preprocess op {identifier}") from error
    return tuple(result)


def _validate_ops(registry: Sequence[PreprocessOp]) -> None:
    identifiers = [op.op_id for op in registry]
    if not identifiers:
        raise ValueError("preprocess ops must not be empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("preprocess ops contain duplicate identifiers")
    for site in ("aggregate", "join"):
        if not any(op.site == site for op in registry):
            raise ValueError(f"preprocess ops have no {site} entry")


_validate_ops(PREPROCESS_OPS)


__all__ = [
    "COUNT_CAP",
    "CRISP_SCALE",
    "EXISTS_MARGIN",
    "PAIR_JOIN_BLOCK",
    "PREPROCESS_OPS",
    "PREPROCESS_VERSION",
    "PreprocessOp",
    "PreprocessSite",
    "lookup_ops",
    "ops_for_site",
    "ops_from_ids",
    "site_for_arity",
]
