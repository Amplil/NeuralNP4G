"""The official Neural NP4G relational model.

The public model has one execution path.  Research variants are made by
passing different task-tied preprocessing in Python; the formal runner
always uses the ops registered on the task.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adf import (
    ADFEntry,
    CANDIDATE_REGULARIZATION_REFERENCE_WIDTH,
    CandidateGateBank,
    candidate_ids,
)
from .circuit import NP4GCircuit
from .preprocess import (
    COUNT_CAP,
    CRISP_SCALE,
    EXISTS_MARGIN,
    PAIR_JOIN_BLOCK,
    PREPROCESS_VERSION,
    PreprocessOp,
    ops_for_site,
    ops_from_ids,
    site_for_arity,
)


class NeuralNP4G(nn.Module):
    """A single learned free circuit reused across relational message ticks."""

    SYN_STATE_LO = 0.98
    SYN_OUT_PRE_MARGIN = 3.5
    SYN_HID_PRE_MARGIN = 4.0
    SYN_OUT_PRE_MAX = 8.0
    SYN_HID_LO = 0.95
    SYN_SLACK_MAX = 1.0
    MAX_COUNT_FANIN = 2
    CEXISTSP_R_BLOCK = PAIR_JOIN_BLOCK

    def __init__(
        self,
        *,
        num_objects: int,
        target_arity: int,
        input_dims: tuple[int, int, int, int],
        message_ticks: int,
        channels: int = 8,
        hidden_nodes: int = 12,
        circuit_ticks: int = 2,
        learned_adfs: Sequence[ADFEntry] = (),
        preprocess: Sequence[PreprocessOp] | None = None,
        base_input_policy: str = "all",
        edge_sparsity: float = 1e-4,
        neuron_sparsity: float = 1e-3,
        edge_logit_init: float = 0.0,
        hidden_logit_init: float = 2.0,
        edge_weight_std: float = 0.5,
        feedforward_prior: bool = True,
        ff_prior_strength: float = 2.0,
    ) -> None:
        super().__init__()
        if int(target_arity) not in (1, 2):
            raise ValueError("target_arity must be 1 or 2")
        if len(input_dims) != 4 or any(int(value) < 0 for value in input_dims):
            raise ValueError("input_dims must contain four non-negative widths")
        if int(message_ticks) <= 0:
            raise ValueError("message_ticks must be positive")
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        if str(base_input_policy) not in ("all", "adf_only"):
            raise ValueError("base_input_policy must be 'all' or 'adf_only'")
        if str(base_input_policy) == "adf_only" and not learned_adfs:
            raise ValueError("adf_only models require learned ADF inputs")

        self.kind = "neural_np4g"
        self.num_objects = int(num_objects)
        self.target_arity_value = int(target_arity)
        self.num_ticks = int(message_ticks)
        self.channels = int(channels)
        self.input_dims = tuple(int(value) for value in input_dims)
        self.d_unary = self.input_dims[1]
        self.d_binary = self.input_dims[2]
        self.bidirectional = True
        # v12 has one proof-friendly discrete state update.  The forward value
        # is an exact {-1,+1} bit at evaluation/proof time; training uses the
        # fixed homotopy schedule set by train.py while retaining tanh
        # gradients through a straight-through estimator.
        self.state_update = "step"
        self.step_alpha = 1.0
        self.hard_forward = True
        self.kill_global_mean = True
        self.sat_state_margin = 0.0
        self._last_sat_loss: torch.Tensor | None = None

        site = site_for_arity(self.target_arity_value)
        if preprocess is None:
            preprocess = ops_for_site(site)
        self.preprocess_ops = tuple(preprocess)
        preprocess_ids = [op.op_id for op in self.preprocess_ops]
        if len(set(preprocess_ids)) != len(preprocess_ids):
            raise ValueError("preprocess contains duplicate ops")
        if any(op.site != site for op in self.preprocess_ops):
            raise ValueError(
                f"preprocess for arity {self.target_arity_value} must use {site} ops"
            )
        self.aggregate_specs = tuple(
            op for op in self.preprocess_ops if op.site == "aggregate"
        )
        self.join_specs = tuple(
            op for op in self.preprocess_ops if op.site == "join"
        )
        self.aggregation_ops = tuple(spec.name for spec in self.aggregate_specs)
        self.join_aggregation_ops = tuple(spec.name for spec in self.join_specs)

        self.adf_input_specs = tuple(learned_adfs)
        self.base_input_policy = str(base_input_policy)
        self.adf_call_specs: tuple[()] = ()
        extra_unary = sum(entry.arity == 1 for entry in self.adf_input_specs)
        extra_binary = sum(entry.arity == 2 for entry in self.adf_input_specs)
        if self.d_unary < extra_unary or self.d_binary < extra_binary:
            raise ValueError("learned ADF entries exceed the declared input widths")
        self.base_input_dims = (
            self.input_dims[0],
            self.d_unary - extra_unary,
            self.d_binary - extra_binary,
            self.input_dims[3],
        )

        primitive_candidate_ids = tuple(op.op_id for op in self.preprocess_ops)
        self.candidate_gates = CandidateGateBank(
            (*primitive_candidate_ids, *candidate_ids(self.adf_input_specs))
        )

        channels_i = self.channels
        self.global_embed = nn.Linear(self.input_dims[0], channels_i)
        if self.target_arity_value == 2:
            # Ordered endpoint embeddings are essential for asymmetric binary
            # predicates assembled from unary ADFs (for example husband(x,y)
            # from is_father(x), is_mother(y), and co_parent(x,y)).
            self.source_node_embed = nn.Linear(self.d_unary, channels_i)
            self.target_node_embed = nn.Linear(self.d_unary, channels_i)
            self.edge_embed = nn.Linear(self.d_binary, channels_i)
        else:
            self.node_embed = nn.Linear(self.d_unary, channels_i)

        if self.target_arity_value == 1:
            aggregate_width = sum(spec.output_width(channels_i) for spec in self.aggregate_specs)
            core_in = channels_i + 2 * self.d_binary * aggregate_width + channels_i
        else:
            join_width = sum(spec.output_width(channels_i) for spec in self.join_specs)
            core_in = 2 * channels_i + join_width + 2 * self.d_binary
        self.core = NP4GCircuit(
            d_in=core_in,
            d_out=channels_i,
            n_hidden=int(hidden_nodes),
            num_ticks=int(circuit_ticks),
            edge_sparsity=float(edge_sparsity),
            neuron_sparsity=float(neuron_sparsity),
            edge_logit_init=float(edge_logit_init),
            hidden_logit_init=float(hidden_logit_init),
            edge_weight_std=float(edge_weight_std),
            feedforward_prior=bool(feedforward_prior),
            ff_prior_strength=float(ff_prior_strength),
        )
        self.core_in = core_in
        self.head = nn.Linear(channels_i, 1)

    @property
    def target_arity(self) -> int:
        return self.target_arity_value

    def set_num_objects(self, num_objects: int) -> None:
        self.num_objects = int(num_objects)

    def build_config(self) -> dict[str, Any]:
        """Serializable constructor values required to replay this circuit."""

        return {
            "num_objects": self.num_objects,
            "target_arity": self.target_arity_value,
            "input_dims": list(self.input_dims),
            "message_ticks": self.num_ticks,
            "channels": self.channels,
            "hidden_nodes": self.core.n_hidden,
            "circuit_ticks": self.core.num_ticks,
            "edge_sparsity": self.core.edge_sparsity,
            "neuron_sparsity": self.core.neuron_sparsity,
            "primitive_registry_version": PREPROCESS_VERSION,
            "primitive_ids": [op.op_id for op in self.preprocess_ops],
            "candidate_ids": list(self.candidate_gates.candidate_ids),
            "candidate_regularization_reference_width": (
                CANDIDATE_REGULARIZATION_REFERENCE_WIDTH
            ),
            "state_update": self.state_update,
            "ordered_binary_endpoints": self.target_arity_value == 2,
            "base_input_policy": self.base_input_policy,
        }

    def checkpoint_payload(self) -> dict[str, Any]:
        """Return the standalone replay payload for an end-of-training state."""

        own_state = {
            key: (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else copy.deepcopy(value)
            )
            for key, value in self.state_dict().items()
            if not key.startswith("adf_input_specs.")
        }
        return {"build_config": self.build_config(), "state_dict": own_state}

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        learned_adfs: Sequence[ADFEntry] = (),
    ) -> "NeuralNP4G":
        if int(config.get("primitive_registry_version", PREPROCESS_VERSION)) != PREPROCESS_VERSION:
            raise ValueError("checkpoint uses another preprocess op set")
        if str(config.get("state_update", "")) != "step":
            raise ValueError("checkpoint does not use the fixed v12 step state")
        if int(config.get("candidate_regularization_reference_width", -1)) != (
            CANDIDATE_REGULARIZATION_REFERENCE_WIDTH
        ):
            raise ValueError(
                "checkpoint uses another candidate regularization scale"
            )
        if int(config["target_arity"]) == 2 and not bool(
            config.get("ordered_binary_endpoints", False)
        ):
            raise ValueError("binary checkpoint lacks ordered endpoint embeddings")
        primitive_ids = config.get("primitive_ids")
        if primitive_ids is None:
            raise ValueError("checkpoint is missing task preprocessing")
        model = cls(
            num_objects=int(config.get("num_objects", 10)),
            target_arity=int(config["target_arity"]),
            input_dims=tuple(int(value) for value in config["input_dims"]),
            message_ticks=int(config["message_ticks"]),
            channels=int(config.get("channels", 8)),
            hidden_nodes=int(config.get("hidden_nodes", 12)),
            circuit_ticks=int(config.get("circuit_ticks", 2)),
            learned_adfs=learned_adfs,
            preprocess=ops_from_ids(tuple(str(item) for item in primitive_ids)),
            base_input_policy=str(config.get("base_input_policy", "all")),
            edge_sparsity=float(config.get("edge_sparsity", 1e-4)),
            neuron_sparsity=float(config.get("neuron_sparsity", 1e-3)),
        )
        expected = tuple(config.get("candidate_ids", model.candidate_gates.candidate_ids))
        if expected != model.candidate_gates.candidate_ids:
            raise ValueError("checkpoint candidate registry does not match the model")
        return model

    def candidate_weight(self, candidate_id: str) -> torch.Tensor:
        return self.candidate_gates.weight(candidate_id)

    def _primitive_gate_weights(
        self,
        attr: str,
        ops: Sequence[str],
        device: torch.device,
    ) -> torch.Tensor:
        if attr == "aggregation_gate_logits":
            specs = self.aggregate_specs
        elif attr == "join_aggregation_gate_logits":
            specs = self.join_specs
        else:
            raise ValueError(f"unknown primitive gate bank: {attr}")
        if tuple(op.name for op in specs) != tuple(str(item) for item in ops):
            raise ValueError("primitive gate request does not match model registry")
        if not specs:
            return torch.empty(0, device=device)
        return torch.stack(
            [self.candidate_weight(op.op_id).to(device) for op in specs]
        )

    def _adf_candidate_weight(self, entry: ADFEntry, device: torch.device) -> torch.Tensor:
        return self.candidate_weight(entry.candidate_id).to(device)

    def input_channel_gate_weights(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unary = torch.ones(self.d_unary, device=device, dtype=dtype)
        binary = torch.ones(self.d_binary, device=device, dtype=dtype)
        if self.base_input_policy == "adf_only":
            unary[: self.base_input_dims[1]] = 0.0
            binary[: self.base_input_dims[2]] = 0.0
        unary_entries = [entry for entry in self.adf_input_specs if entry.arity == 1]
        binary_entries = [entry for entry in self.adf_input_specs if entry.arity == 2]
        for offset, entry in enumerate(unary_entries, self.d_unary - len(unary_entries)):
            unary[offset] = self._adf_candidate_weight(entry, device).to(dtype)
        for offset, entry in enumerate(binary_entries, self.d_binary - len(binary_entries)):
            binary[offset] = self._adf_candidate_weight(entry, device).to(dtype)
        return unary, binary

    def _gate_inputs(
        self,
        unary: torch.Tensor,
        binary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unary_weight, binary_weight = self.input_channel_gate_weights(
            unary.device, unary.dtype
        )
        return unary * unary_weight, binary * binary_weight

    def _aggregate(self, relation: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if not self.aggregate_specs:
            return state.new_zeros(state.shape[0], state.shape[1], 0)
        weights = self._primitive_gate_weights(
            "aggregation_gate_logits", self.aggregation_ops, state.device
        )
        return torch.cat(
            [
                spec(relation, state) * weights[index].to(state.dtype)
                for index, spec in enumerate(self.aggregate_specs)
            ],
            dim=-1,
        )

    def _join(self, pair: torch.Tensor) -> torch.Tensor:
        if not self.join_specs:
            return pair.new_zeros(*pair.shape[:3], 0)
        weights = self._primitive_gate_weights(
            "join_aggregation_gate_logits", self.join_aggregation_ops, pair.device
        )
        return torch.cat(
            [
                spec(pair) * weights[index].to(pair.dtype)
                for index, spec in enumerate(self.join_specs)
            ],
            dim=-1,
        )

    def _snap(self, preactivation: torch.Tensor) -> torch.Tensor:
        """Exact sign forward with a tanh surrogate gradient."""

        soft = torch.tanh(preactivation)
        hard = torch.where(
            preactivation >= 0,
            torch.ones_like(soft),
            -torch.ones_like(soft),
        )
        alpha = float(self.step_alpha) if self.training else 1.0
        if alpha >= 1.0:
            return hard.detach() + (soft - soft.detach())
        mixed = alpha * hard + (1.0 - alpha) * soft
        return mixed.detach() + (soft - soft.detach())

    def _init_state(self, preactivation: torch.Tensor) -> torch.Tensor:
        return self._snap(preactivation)

    def _step(self, core_output: torch.Tensor) -> torch.Tensor:
        return self._snap(core_output)

    def _begin_saturation(self) -> list[torch.Tensor] | None:
        self._last_sat_loss = None
        return [] if self.sat_state_margin > 0 else None

    def _record_state(
        self, terms: list[torch.Tensor] | None, state: torch.Tensor
    ) -> None:
        if terms is not None:
            terms.append(torch.relu(self.sat_state_margin - state.abs()).mean())

    def _record_core(self, terms: list[torch.Tensor] | None) -> None:
        if terms is not None:
            terms.append(self.core.last_hidden_saturation())

    def _finish_saturation(self, terms: list[torch.Tensor] | None) -> None:
        if terms:
            self._last_sat_loss = torch.stack(terms).mean()

    def forward(self, batch: Any) -> torch.Tensor:
        unary, binary = self._gate_inputs(batch.input1, batch.input2)
        if self.target_arity_value == 1:
            return self._forward_unary(batch.input0, unary, binary)
        return self._forward_binary(unary, binary)

    def _forward_unary(
        self,
        global_input: torch.Tensor,
        unary: torch.Tensor,
        binary: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, objects, _ = unary.shape
        global_state = self.global_embed(global_input).unsqueeze(1)
        state = self._init_state(self.node_embed(unary) + global_state)
        terms = self._begin_saturation()
        self._record_state(terms, state)
        for _ in range(self.num_ticks):
            messages: list[torch.Tensor] = []
            for relation_index in range(self.d_binary):
                relation = binary[..., relation_index]
                messages.append(self._aggregate(relation, state))
                messages.append(self._aggregate(relation.transpose(1, 2), state))
            global_max = state.amax(dim=1, keepdim=True).expand(
                batch_size, objects, self.channels
            )
            core_input = torch.cat([state, *messages, global_max], dim=-1)
            if core_input.shape[-1] != self.core_in:
                raise RuntimeError("unary core input layout does not match the circuit")
            state = self._step(self.core(core_input))
            self._record_core(terms)
            self._record_state(terms, state)
        self._finish_saturation(terms)
        return self.head(state).squeeze(-1)

    def _forward_binary(
        self,
        unary: torch.Tensor,
        binary: torch.Tensor,
    ) -> torch.Tensor:
        source = self.source_node_embed(unary)
        target = self.target_node_embed(unary)
        pair = self._init_state(
            self.edge_embed(binary)
            + source.unsqueeze(2)
            + target.unsqueeze(1)
        )
        binary_t = binary.transpose(1, 2)
        terms = self._begin_saturation()
        self._record_state(terms, pair)
        for _ in range(self.num_ticks):
            core_input = torch.cat(
                [pair, pair.transpose(1, 2), self._join(pair), binary, binary_t],
                dim=-1,
            )
            if core_input.shape[-1] != self.core_in:
                raise RuntimeError("binary core input layout does not match the circuit")
            pair = self._step(self.core(core_input))
            self._record_core(terms)
            self._record_state(terms, pair)
        self._finish_saturation(terms)
        return self.head(pair).squeeze(-1)

    def structure_loss(self) -> torch.Tensor:
        return self.core.structure_loss()

    def binarization_loss(self) -> torch.Tensor:
        probability = self.candidate_gates.probabilities()
        candidate_term = (
            (probability * (1.0 - probability)).mean()
            if probability.numel()
            else probability.sum()
        )
        return self.core.binarization_loss() + candidate_term

    def candidate_loss(self) -> torch.Tensor:
        return self.candidate_gates.regularization_loss()

    def last_saturation_loss(self) -> torch.Tensor:
        if self._last_sat_loss is None:
            return self.head.weight.new_tensor(0.0)
        return self._last_sat_loss

    def _random_state(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        sign = torch.where(
            torch.rand(shape, device=device) < 0.5,
            -torch.ones(shape, device=device),
            torch.ones(shape, device=device),
        )
        magnitude = self.SYN_STATE_LO + (
            1.0 - self.SYN_STATE_LO
        ) * torch.rand(shape, device=device)
        return sign * magnitude

    def _synthetic_core_inputs(self, samples: int, device: torch.device) -> torch.Tensor:
        channels = self.channels
        q_low = float(torch.sigmoid(torch.tensor(CRISP_SCALE * self.SYN_STATE_LO)))
        q_high = float(torch.sigmoid(torch.tensor(CRISP_SCALE)))

        def uniform(low: float, high: float, width: int) -> torch.Tensor:
            return low + (high - low) * torch.rand(samples, width, device=device)

        def bits(width: int) -> torch.Tensor:
            return (torch.rand(samples, width, device=device) < 0.5).float()

        def preprocess_block(op: PreprocessOp) -> torch.Tensor:
            width = op.output_width(channels)
            if op.abstract_kind in ("exists", "exists_join", "pair_exists_join"):
                return bits(width) * uniform(
                    max(0.0, q_low * q_low - EXISTS_MARGIN)
                    if "join" in op.abstract_kind
                    else max(0.0, q_low - EXISTS_MARGIN),
                    max(0.0, q_high * q_high - EXISTS_MARGIN)
                    if "join" in op.abstract_kind
                    else max(0.0, q_high - EXISTS_MARGIN),
                    width,
                )
            if op.abstract_kind == "forall":
                active = bits(width)
                false_value = uniform(0.5 + 1.0 - q_high, 0.5 + 1.0 - q_low, width)
                return active + (1.0 - active) * false_value
            if op.abstract_kind == "count":
                count = torch.floor(torch.rand(samples, width, device=device) ** 2 * 18)
                contribution = uniform(q_low - EXISTS_MARGIN, q_high - EXISTS_MARGIN, width)
                return torch.clamp(count * contribution, max=COUNT_CAP)
            raise ValueError(f"no abstract sampler for {op.abstract_kind}")

        pieces: list[torch.Tensor] = []
        _, binary_gate = self.input_channel_gate_weights(device)
        if self.target_arity_value == 2:
            primitive_gate = self._primitive_gate_weights(
                "join_aggregation_gate_logits",
                self.join_aggregation_ops,
                device,
            ).detach()
            pieces.extend(
                [
                    self._random_state((samples, channels), device),
                    self._random_state((samples, channels), device),
                ]
            )
            for index, spec in enumerate(self.join_specs):
                pieces.append(preprocess_block(spec) * primitive_gate[index])
            pieces.append(bits(self.d_binary) * binary_gate.detach())
            pieces.append(bits(self.d_binary) * binary_gate.detach())
        else:
            primitive_gate = self._primitive_gate_weights(
                "aggregation_gate_logits",
                self.aggregation_ops,
                device,
            ).detach()
            pieces.append(self._random_state((samples, channels), device))
            for relation_index in range(self.d_binary):
                for _ in range(2):
                    for index, spec in enumerate(self.aggregate_specs):
                        pieces.append(
                            preprocess_block(spec)
                            * binary_gate[relation_index].detach()
                            * primitive_gate[index]
                        )
            pieces.append(self._random_state((samples, channels), device))
        result = torch.cat(pieces, dim=-1)
        if result.shape[-1] != self.core_in:
            raise RuntimeError("synthetic input layout does not match the circuit")
        return result

    def _input_width_vector(self, device: torch.device) -> torch.Tensor:
        widths = torch.zeros(self.core.n_nodes, device=device)
        state_width = 1.0 - self.SYN_STATE_LO
        position = 0

        def fill(width: int, value: float) -> None:
            nonlocal position
            widths[position : position + width] = value
            position += width

        _, binary_gate = self.input_channel_gate_weights(device)
        if self.target_arity_value == 2:
            primitive_gate = self._primitive_gate_weights(
                "join_aggregation_gate_logits",
                self.join_aggregation_ops,
                device,
            ).detach()
            fill(2 * self.channels, state_width)
            for index, spec in enumerate(self.join_specs):
                fill(
                    spec.output_width(self.channels),
                    0.05 * float(primitive_gate[index]),
                )
            for _ in range(2):
                for index in range(self.d_binary):
                    fill(1, 0.0 if float(binary_gate[index]) == 0.0 else 0.0)
        else:
            primitive_gate = self._primitive_gate_weights(
                "aggregation_gate_logits",
                self.aggregation_ops,
                device,
            ).detach()
            fill(self.channels, state_width)
            for relation_index in range(self.d_binary):
                for _ in range(2):
                    for index, spec in enumerate(self.aggregate_specs):
                        base = 0.06 if spec.abstract_kind == "count" else 0.05
                        fill(
                            spec.output_width(self.channels),
                            base
                            * float(binary_gate[relation_index].detach())
                            * float(primitive_gate[index]),
                        )
            fill(self.channels, state_width)
        if position != self.core.d_in:
            raise RuntimeError("input-width layout does not match the circuit")
        if self.core.n_hidden:
            widths[self.core.hidden_start :] = 1.0 - self.SYN_HID_LO
        return widths

    def synthetic_saturation_loss(self, samples: int = 512) -> torch.Tensor:
        device = self.head.weight.device
        core = self.core
        x = self._synthetic_core_inputs(int(samples), device)
        values = x.new_zeros(x.shape[0], core.n_nodes)
        values[:, : core.d_in] = x
        effective = core.effective_weight()
        _, node_gate = core.gates()
        live_hidden = node_gate[core.hidden_start :].detach().unsqueeze(0)
        terms: list[torch.Tensor] = []

        def cvar(value: torch.Tensor) -> torch.Tensor:
            flat = value.reshape(-1)
            count = max(1, flat.numel() // 16)
            return torch.topk(flat, count).values.mean()

        def penalties(pre: torch.Tensor) -> None:
            output = pre[:, core.out_start : core.out_end]
            terms.append(cvar(torch.relu(self.SYN_OUT_PRE_MARGIN - output.abs())))
            terms.append(cvar(torch.relu(output.abs() - self.SYN_OUT_PRE_MAX)))
            if core.n_hidden:
                hidden = pre[:, core.hidden_start :]
                terms.append(
                    cvar(
                        torch.relu(self.SYN_HID_PRE_MARGIN - hidden.abs())
                        * live_hidden
                    )
                )

        for _ in range(core.num_ticks):
            pre = F.linear(values, effective, core.bias)
            penalties(pre)
            proposed = torch.where(core.hidden_mask.unsqueeze(0), torch.sigmoid(pre), pre)
            values = torch.where(core.non_input_mask.unsqueeze(0), proposed, values)
            values = values * node_gate.unsqueeze(0)

        final_state = self._random_state((int(samples), self.channels), device)
        terms.append(cvar(torch.relu(1.0 - self.head(final_state).squeeze(-1).abs())))

        unary_gate, binary_gate = self.input_channel_gate_weights(device)
        unary = (torch.rand(int(samples), self.d_unary, device=device) < 0.5).float()
        unary = unary * unary_gate.detach()
        if self.target_arity_value == 2:
            other = (torch.rand_like(unary) < 0.5).float() * unary_gate.detach()
            binary = (torch.rand(int(samples), self.d_binary, device=device) < 0.5).float()
            binary = binary * binary_gate.detach()
            init_pre = (
                self.edge_embed(binary)
                + self.source_node_embed(unary)
                + self.target_node_embed(other)
            )
        else:
            global_zero = torch.zeros(
                int(samples), self.input_dims[0], device=device
            )
            init_pre = self.node_embed(unary) + self.global_embed(global_zero)
        terms.append(cvar(torch.relu(math.atanh(self.SYN_STATE_LO) - init_pre.abs())))

        slack = effective.abs() @ self._input_width_vector(device)
        terms.append(
            torch.relu(slack[core.out_start :] - self.SYN_SLACK_MAX).mean()
        )
        return torch.stack(terms).sum()

    def dead_input_indices(self) -> list[int]:
        """Return inputs that are structurally, rather than temporarily, dead.

        Every allocated input column is structurally valid.  Primitive and ADF
        candidate gates are trainable hard gates: an OFF bit in one optimiser
        step is not evidence that the corresponding column can be deleted.
        Permanently pruning such a column destroys the STE recovery path and
        made the v6 ``has_father`` trajectory irrecoverable.  Width-zero or
        absent inputs are never allocated, so the safe result is empty.
        """

        return []

    def count_input_indices(self) -> list[int]:
        if self.target_arity_value == 2:
            return []
        result: list[int] = []
        position = self.channels
        for _relation in range(self.d_binary):
            for _direction in range(2):
                for spec in self.aggregate_specs:
                    width = spec.output_width(self.channels)
                    if spec.abstract_kind == "count":
                        result.extend(range(position, position + width))
                    position += width
        return result

    def prune_synthetic_repair(
        self,
        n: int = 512,
        max_kills: int = 4,
    ) -> dict[str, Any]:
        """Surgically remove edges that cause abstract-domain violations.

        Some circuits are correct on sampled graph data but contain structures
        such as ``8 - 2 * count`` that become indecisive inside the extractor's
        abstract input bands.  Moving the offending weight continuously toward
        zero can cross a much larger violation, so the synthetic saturation
        loss alone may not repair it.  For each of the worst violating rows,
        evaluate every live edge as a discrete one-edge removal and permanently
        prune an edge only when that move reduces the violation by at least 75%.

        This is the official-model adaptation of the legacy
        ``prune_synthetic_repair`` operation.  It mutates only ``prune_mask``;
        weights, logits, and all sampled training data remain unchanged.
        """

        core = self.core
        device = self.head.weight.device
        stats: dict[str, Any] = {"repair_kills": 0, "repair_rows": []}
        with torch.no_grad():
            effective = core.effective_weight()
            _, node_gate = core.soft_structure()
            synthetic = self._synthetic_core_inputs(int(n), device)
            values = synthetic.new_zeros(int(n), core.n_nodes)
            values[:, : core.d_in] = synthetic
            traces: list[tuple[torch.Tensor, torch.Tensor]] = []
            for _ in range(core.num_ticks):
                pre = F.linear(values, effective, core.bias)
                traces.append((values.clone(), pre))
                proposed = torch.where(
                    core.hidden_mask.unsqueeze(0),
                    torch.sigmoid(pre),
                    pre,
                )
                values = torch.where(
                    core.non_input_mask.unsqueeze(0), proposed, values
                )
                values = values * node_gate.unsqueeze(0)

            def row_violation(
                row: int,
                pre_per_tick: list[torch.Tensor],
            ) -> float:
                violation = 0.0
                is_output = core.out_start <= row < core.out_end
                margin = (
                    self.SYN_OUT_PRE_MARGIN
                    if is_output
                    else self.SYN_HID_PRE_MARGIN
                )
                for pre_row in pre_per_tick:
                    current = float(
                        torch.relu(margin - pre_row.abs()).max().item()
                    )
                    if is_output:
                        current = max(
                            current,
                            float(
                                torch.relu(
                                    pre_row.abs() - self.SYN_OUT_PRE_MAX
                                )
                                .max()
                                .item()
                            ),
                        )
                    violation = max(violation, current)
                return violation

            widths = self._input_width_vector(device)
            slack = effective.abs() @ widths
            candidates: list[tuple[float, int]] = []
            for row in range(core.out_start, core.n_nodes):
                if (
                    row >= core.hidden_start
                    and float(node_gate[row].item()) <= 0.5
                ):
                    continue
                violation = row_violation(
                    row, [trace[1][:, row] for trace in traces]
                )
                violation = max(
                    violation,
                    float(
                        torch.relu(slack[row] - self.SYN_SLACK_MAX).item()
                    ),
                )
                if violation > 0.2:
                    candidates.append((violation, row))

            candidates.sort(reverse=True)
            for violation_before, row in candidates[: max(1, int(max_kills))]:
                alive = (core.edge_mask * core.prune_mask)[row] > 0
                best: tuple[float, int] | None = None
                for source in torch.nonzero(alive).flatten().tolist():
                    pre_without_edge = [
                        trace[1][:, row]
                        - effective[row, source] * trace[0][:, source]
                        for trace in traces
                    ]
                    violation_without_edge = row_violation(
                        row, pre_without_edge
                    )
                    violation_without_edge = max(
                        violation_without_edge,
                        float(
                            torch.relu(
                                slack[row]
                                - effective[row, source].abs() * widths[source]
                                - self.SYN_SLACK_MAX
                            ).item()
                        ),
                    )
                    if best is None or violation_without_edge < best[0]:
                        best = (violation_without_edge, int(source))

                if best is not None and best[0] <= 0.25 * violation_before:
                    kill = torch.zeros_like(core.prune_mask, dtype=torch.bool)
                    kill[row, best[1]] = True
                    core.prune_edges(kill)
                    stats["repair_kills"] += 1
                    stats["repair_rows"].append(
                        {
                            "row": int(row),
                            "edge_src": int(best[1]),
                            "viol_before": round(violation_before, 3),
                            "viol_after": round(best[0], 3),
                        }
                    )
        return stats

    def prune_step(
        self,
        fanin: int,
        *,
        return_kill_masks: bool = False,
    ) -> dict[str, Any]:
        before_dead = (
            self.core.prune_mask.detach().clone()
            if return_kill_masks
            else None
        )
        dead_columns = torch.zeros(
            self.core.n_nodes,
            dtype=torch.bool,
            device=self.core.prune_mask.device,
        )
        for index in self.dead_input_indices():
            dead_columns[index] = True
        dead_columns[self.core.out_start : self.core.out_end] = True
        kill = dead_columns.unsqueeze(0).expand(self.core.n_nodes, -1)
        kill = kill & (self.core.edge_mask > 0)
        killed_dead = self.core.prune_edges(kill)
        after_dead = (
            self.core.prune_mask.detach().clone()
            if return_kill_masks
            else None
        )
        killed_topk = self.core.prune_to_topk(int(fanin))
        after_topk = (
            self.core.prune_mask.detach().clone()
            if return_kill_masks
            else None
        )

        killed_count = 0
        count_columns = self.count_input_indices()
        if count_columns:
            with torch.no_grad():
                magnitude = self.core.effective_weight().abs()
                alive = (self.core.edge_mask * self.core.prune_mask) > 0
                count_mask = torch.zeros(
                    self.core.n_nodes, dtype=torch.bool, device=alive.device
                )
                count_mask[count_columns] = True
                extra_kill = torch.zeros_like(alive)
                for row in range(self.core.out_start, self.core.n_nodes):
                    candidates = alive[row] & count_mask
                    if int(candidates.sum()) <= self.MAX_COUNT_FANIN:
                        continue
                    score = torch.where(
                        candidates,
                        magnitude[row],
                        torch.full_like(magnitude[row], -1.0),
                    )
                    keep = torch.topk(score, self.MAX_COUNT_FANIN).indices
                    row_kill = candidates.clone()
                    row_kill[keep] = False
                    extra_kill[row] = row_kill
            killed_count = self.core.prune_edges(extra_kill)

        fanins = self.core.unpruned_fanin()
        result: dict[str, Any] = {
            "killed_dead_inputs": killed_dead,
            "killed_topk": killed_topk,
            "killed_count_cap": killed_count,
            "max_fanin": int(fanins.max().item()),
            "mean_fanin": float(fanins.float().mean().item()),
        }
        if return_kill_masks:
            if before_dead is None or after_dead is None or after_topk is None:
                raise RuntimeError("pruning kill-mask snapshots were not created")
            after_count = self.core.prune_mask.detach().clone()
            result["_kill_masks"] = {
                "killed_dead_inputs": (before_dead > 0) & (after_dead <= 0),
                "killed_topk": (after_dead > 0) & (after_topk <= 0),
                "killed_count_cap": (after_topk > 0) & (after_count <= 0),
            }
        return result

    def structure_summary(self) -> dict[str, Any]:
        return {
            "primitive_registry_version": PREPROCESS_VERSION,
            "preprocess": [op.op_id for op in self.preprocess_ops],
            "candidate_gates": self.candidate_gates.summary(),
            "circuit": self.core.structure_summary(),
        }


__all__ = ["NeuralNP4G"]
