"""Immutable protocol constants for the official Neural NP4G experiments.

The formal runner imports :data:`OFFICIAL_CONFIG` directly.  There is no
command-line surface for choosing another learning method: a research variant
is made explicitly in Python and receives a different protocol version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


OFFICIAL_PROTOCOL_VERSION = "neural-np4g-full-20260919"
# Every task receives these independent initialisation seeds in this exact
# order.  The complete task-by-seed grid is executed; the earliest strictly
# verified seed is the deterministic ADF selected for downstream tasks.
OFFICIAL_ATTEMPT_SEEDS = tuple(range(10))
# Compatibility alias for the seed argument used throughout data/training.
OFFICIAL_SEEDS = OFFICIAL_ATTEMPT_SEEDS


@dataclass(frozen=True, slots=True)
class SizeCell:
    """Number of objects and independently generated structures in one cell."""

    objects: int
    examples: int

    def __post_init__(self) -> None:
        if self.objects <= 0:
            raise ValueError("objects must be positive")
        if self.examples <= 0:
            raise ValueError("examples must be positive")


TRAIN_CELLS = tuple(SizeCell(n, 256) for n in (10, 20, 30, 40))
VALIDATION_CELLS = tuple(SizeCell(n, 128) for n in (10, 20, 30, 40))
# A downstream model's dependency-use claim is evaluated on this dedicated
# split.  Its cell schedule intentionally mirrors validation while its seed
# stream is independent, so neither validation nor test observations can
# affect the proof-driven attempt decision.
ADF_CAUSAL_CELLS = tuple(SizeCell(n, 128) for n in (10, 20, 30, 40))
TEST_CELLS = (
    *(SizeCell(n, 128) for n in (10, 20, 30, 40)),
    SizeCell(50, 64),
    SizeCell(100, 32),
    SizeCell(200, 16),
    SizeCell(500, 4),
)
FALSIFICATION_CELLS = tuple(SizeCell(n, 6) for n in (50, 100, 200, 500))


@dataclass(frozen=True, slots=True)
class OfficialConfig:
    """The one preregistered model, data, and optimisation recipe.

    Two phases are scheduled parts of one trajectory.  The immutable phase-1
    boundary is always proved; only a failed proof resumes the same optimiser
    and RNG state into phase 2.  No held-out split selects this branch.
    """

    protocol_version: str = OFFICIAL_PROTOCOL_VERSION
    seeds: tuple[int, ...] = OFFICIAL_ATTEMPT_SEEDS

    train_cells: tuple[SizeCell, ...] = TRAIN_CELLS
    validation_cells: tuple[SizeCell, ...] = VALIDATION_CELLS
    adf_causal_cells: tuple[SizeCell, ...] = ADF_CAUSAL_CELLS
    test_cells: tuple[SizeCell, ...] = TEST_CELLS
    falsification_cells: tuple[SizeCell, ...] = FALSIFICATION_CELLS

    graph_edge_probability_min: float = 0.0
    graph_edge_probability_max: float = 0.3
    graph_red_probability: float = 0.25
    family_parent_probability: float = 0.95

    # The natural family generator makes the two parent roles perfectly
    # correlated once co_parent is true.  These extra TRAIN/adf_causal cells
    # enumerate relation structures that independently vary the three husband
    # ADF inputs, so a finite-data shortcut cannot omit is_father or is_mother.
    husband_counterfactual_train_cell: SizeCell = SizeCell(8, 64)
    husband_counterfactual_adf_causal_cell: SizeCell = SizeCell(8, 64)

    # Four discrete state channels are fixed for every NLM-10 task and helper.
    channels: int = 4
    hidden_nodes: int = 12
    circuit_ticks: int = 2
    edge_sparsity: float = 1e-4
    neuron_sparsity: float = 1e-3

    phase1_steps: int = 25_000
    phase2_steps: int = 25_000
    batch_size: int = 8
    phase1_learning_rate: float = 5e-3
    phase2_learning_rate: float = 2e-3
    gradient_clip_norm: float = 1.0
    structure_loss_weight: float = 0.01
    candidate_loss_weight: float = 0.01
    binarization_loss_weight: float = 0.02
    saturation_loss_weight: float = 0.05
    synthetic_saturation_loss_weight: float = 0.3
    synthetic_samples: int = 512
    state_saturation_margin: float = 0.99
    # During optimisation the exact sign state is introduced continuously;
    # evaluation and every immutable proof checkpoint always use the fully
    # discrete state.  This is a schedule constant, not a method switch.
    step_alpha_ramp_fraction: float = 0.4

    # The phase-1 boundary is a preregistered proof checkpoint, not a
    # validation-selected checkpoint.  A strict proof stops the attempt before
    # phase 2; otherwise the exact same optimiser trajectory resumes.
    phase1_boundary_proof: bool = True

    # Phase 2 combines ordinary top-k pruning with the discrete synthetic-
    # domain repair used by the historically verified recipe.
    repair_pruning: bool = True
    repair_samples: int = 512
    repair_max_kills: int = 4

    # When the post-optimiser event batch is exact, pruning must keep it exact
    # and must not increase the wrong count on the fixed, independently seeded
    # TRAIN guard sample for any training size.  A deterministic bounded search
    # may retain only a guard-safe subset of an otherwise destructive proposal.
    prune_guard_examples_per_size: int = 16
    partial_prune_max_trials: int = 128

    # A second TRAIN-only panel protects a discovered exact incumbent across
    # both optimizer updates and pruning.  The official TRAIN cells contain
    # 256 materialised structures per size, so this covers the full split.
    incumbent_guard_examples_per_size: int = 256
    compression_proof_max_fanin: int = 6
    compression_proof_max_checkpoints: int = 4

    # One uniform final stage for every task: remove nodes/edges only when
    # the complete materialised TRAIN split stays exact, then verify again.
    semantic_compression: bool = True
    semantic_compression_max_edge_trials: int = 128
    semantic_compression_off_logit: float = -30.0

    phase1_fanin: int = 10
    phase2_fanin: int = 3
    phase1_prune_interval: int = 500
    phase2_prune_interval: int = 1_000
    phase1_prune_fraction: float = 0.5
    phase2_prune_fraction: float = 0.6

    metric_interval: int = 250
    state_checkpoint_interval: int = 1_000
    extraction_memory_gb: float = 6.0
    extraction_min_available_gb: float = 3.0
    extraction_time_seconds: int = 1_800

    seed_namespace: int = 0x4E503447

    def __post_init__(self) -> None:
        if not self.protocol_version:
            raise ValueError("protocol_version must be non-empty")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique tuple")
        if len(self.seeds) > len(OFFICIAL_ATTEMPT_SEEDS):
            raise ValueError("the formal protocol permits at most 10 attempts")
        if self.seeds != tuple(range(len(self.seeds))):
            raise ValueError("attempt seeds must be the ordered prefix 0..N-1")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("formal seeds must be non-negative")
        if self.graph_edge_probability_min < 0.0:
            raise ValueError("graph_edge_probability_min must be non-negative")
        if self.graph_edge_probability_max > 1.0:
            raise ValueError("graph_edge_probability_max must not exceed one")
        if self.graph_edge_probability_min > self.graph_edge_probability_max:
            raise ValueError("graph probability bounds are reversed")
        if not 0.0 <= self.graph_red_probability <= 1.0:
            raise ValueError("graph_red_probability must lie in [0, 1]")
        if not 0.0 <= self.family_parent_probability <= 1.0:
            raise ValueError("family_parent_probability must lie in [0, 1]")
        if self.husband_counterfactual_train_cell.objects in {
            cell.objects for cell in self.train_cells
        }:
            raise ValueError(
                "husband counterfactual TRAIN size must be distinct from natural TRAIN sizes"
            )
        if self.husband_counterfactual_adf_causal_cell.objects in {
            cell.objects for cell in self.adf_causal_cells
        }:
            raise ValueError(
                "husband counterfactual causal size must be distinct from natural causal sizes"
            )
        positive_ints = (
            self.channels,
            self.hidden_nodes,
            self.circuit_ticks,
            self.phase1_steps,
            self.phase2_steps,
            self.batch_size,
            self.synthetic_samples,
            self.repair_samples,
            self.repair_max_kills,
            self.prune_guard_examples_per_size,
            self.partial_prune_max_trials,
            self.incumbent_guard_examples_per_size,
            self.compression_proof_max_fanin,
            self.compression_proof_max_checkpoints,
            self.semantic_compression_max_edge_trials,
            self.phase1_fanin,
            self.phase2_fanin,
            self.phase1_prune_interval,
            self.phase2_prune_interval,
            self.metric_interval,
            self.state_checkpoint_interval,
            self.extraction_time_seconds,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("positive protocol constants must be positive")
        if not self.phase1_boundary_proof:
            raise ValueError("the official protocol requires phase-1 boundary proof")
        if not self.repair_pruning:
            raise ValueError("the official protocol requires repair pruning")
        if not self.semantic_compression:
            raise ValueError("the official protocol requires semantic compression")
        if not np.isfinite(self.semantic_compression_off_logit) or self.semantic_compression_off_logit >= 0.0:
            raise ValueError("semantic_compression_off_logit must be finite and negative")
        if self.phase2_fanin > self.phase1_fanin:
            raise ValueError("phase2_fanin must not exceed phase1_fanin")
        if not 0.0 < self.phase1_prune_fraction <= 1.0:
            raise ValueError("phase1_prune_fraction must lie in (0, 1]")
        if not 0.0 < self.phase2_prune_fraction <= 1.0:
            raise ValueError("phase2_prune_fraction must lie in (0, 1]")
        if not 0.0 < self.step_alpha_ramp_fraction <= 1.0:
            raise ValueError("step_alpha_ramp_fraction must lie in (0, 1]")

    @property
    def total_steps(self) -> int:
        return self.phase1_steps + self.phase2_steps

    @property
    def attempt_seeds(self) -> tuple[int, ...]:
        """Fixed per-task initialisation attempts, evaluated in order."""

        return self.seeds

    def cells_for(self, split: str) -> tuple[SizeCell, ...]:
        try:
            return {
                "train": self.train_cells,
                "validation": self.validation_cells,
                "adf_causal": self.adf_causal_cells,
                "test": self.test_cells,
                "falsification": self.falsification_cells,
            }[split]
        except KeyError as error:
            raise ValueError(f"unknown split: {split}") from error

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


OFFICIAL_CONFIG = OfficialConfig()


_STREAM_IDS = {
    "model": 1,
    "train_data": 2,
    "validation_data": 3,
    "test_data": 4,
    "falsification_data": 5,
    "train_batch": 6,
    "optimisation": 7,
    "adf_causal_data": 8,
    "prune_guard": 9,
    "incumbent_guard": 10,
    "train_husband_counterfactual_data": 11,
    "adf_causal_husband_counterfactual_data": 12,
}


def derive_seed(
    formal_seed: int,
    task_index: int,
    stream: str,
    *coordinates: int,
    config: OfficialConfig = OFFICIAL_CONFIG,
) -> int:
    """Derive a stable independent 63-bit seed without Python's salted hash."""

    if formal_seed < 0 or task_index < 0 or any(value < 0 for value in coordinates):
        raise ValueError("seed coordinates must be non-negative")
    try:
        stream_id = _STREAM_IDS[stream]
    except KeyError as error:
        raise ValueError(f"unknown seed stream: {stream}") from error
    sequence = np.random.SeedSequence(
        [
            int(config.seed_namespace),
            int(formal_seed),
            int(task_index),
            int(stream_id),
            *(int(value) for value in coordinates),
        ]
    )
    words = sequence.generate_state(2, dtype=np.uint32)
    value = (int(words[0]) << 32) | int(words[1])
    return value & ((1 << 63) - 1)


__all__ = [
    "ADF_CAUSAL_CELLS",
    "FALSIFICATION_CELLS",
    "OFFICIAL_CONFIG",
    "OFFICIAL_ATTEMPT_SEEDS",
    "OFFICIAL_PROTOCOL_VERSION",
    "OFFICIAL_SEEDS",
    "OfficialConfig",
    "SizeCell",
    "TEST_CELLS",
    "TRAIN_CELLS",
    "VALIDATION_CELLS",
    "derive_seed",
]
