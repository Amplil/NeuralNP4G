"""Shared free neuron circuit for the official Neural NP4G model."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _binary_gate(probability: torch.Tensor) -> torch.Tensor:
    """Use exact bits in the forward pass and the smooth derivative backward."""

    bit = (probability > 0.5).to(probability.dtype)
    return bit.detach() + (probability - probability.detach())


class NP4GCircuit(nn.Module):
    """One recurrent complete directed graph over input, output, and hidden nodes.

    Input nodes are clamped at every circuit tick.  Output nodes are linear and
    hidden nodes use a sigmoid.  Structural gates always use exact forward
    values; there is no alternate execution mode.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_hidden: int = 12,
        num_ticks: int = 2,
        *,
        edge_sparsity: float = 1e-4,
        neuron_sparsity: float = 1e-3,
        edge_logit_init: float = 0.0,
        hidden_logit_init: float = 2.0,
        edge_weight_std: float = 0.5,
        feedforward_prior: bool = True,
        ff_prior_strength: float = 2.0,
    ) -> None:
        super().__init__()
        if d_in <= 0:
            raise ValueError("d_in must be positive")
        if d_out <= 0:
            raise ValueError("d_out must be positive")
        if n_hidden < 0:
            raise ValueError("n_hidden must be non-negative")
        if num_ticks <= 0:
            raise ValueError("num_ticks must be positive")
        if edge_weight_std < 0:
            raise ValueError("edge_weight_std must be non-negative")

        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.n_hidden = int(n_hidden)
        self.num_ticks = int(num_ticks)
        self.n_nodes = self.d_in + self.d_out + self.n_hidden

        self.in_start = 0
        self.in_end = self.d_in
        self.out_start = self.in_end
        self.out_end = self.out_start + self.d_out
        self.hidden_start = self.out_end

        self.edge_sparsity = float(edge_sparsity)
        self.neuron_sparsity = float(neuron_sparsity)
        self.activation_name = "sigmoid"

        edge_logits = torch.full(
            (self.n_nodes, self.n_nodes), float(edge_logit_init)
        )
        edge_weights = torch.randn(self.n_nodes, self.n_nodes) * float(edge_weight_std)

        if feedforward_prior and self.n_hidden > 0:
            strength = float(ff_prior_strength)
            edge_logits.fill_(-strength)
            edge_logits[self.hidden_start :, self.in_start : self.in_end] = strength
            edge_logits[self.out_start : self.out_end, self.hidden_start :] = strength
            edge_logits[self.out_start : self.out_end, self.in_start : self.in_end] = 0.0
            edge_logits[self.hidden_start :, self.hidden_start :] = 0.0

            edge_weights.mul_(0.02)
            input_scale = 1.0 / math.sqrt(max(self.d_in, 1))
            hidden_scale = 1.0 / math.sqrt(max(self.n_hidden, 1))
            edge_weights[self.hidden_start :, self.in_start : self.in_end] = (
                torch.randn(self.n_hidden, self.d_in) * input_scale
            )
            edge_weights[self.out_start : self.out_end, self.hidden_start :] = (
                torch.randn(self.d_out, self.n_hidden) * hidden_scale
            )
            edge_weights[self.out_start : self.out_end, self.in_start : self.in_end] = (
                torch.randn(self.d_out, self.d_in) * input_scale
            )

        self.edge_logits = nn.Parameter(edge_logits)
        self.edge_weight = nn.Parameter(edge_weights)
        self.node_logits = nn.Parameter(
            torch.full((self.n_nodes,), float(hidden_logit_init))
        )
        self.bias = nn.Parameter(torch.zeros(self.n_nodes))

        edge_mask = torch.zeros(self.n_nodes, self.n_nodes)
        edge_mask[self.out_start :, :] = 1.0
        hidden_gate_mask = torch.zeros(self.n_nodes)
        hidden_gate_mask[self.hidden_start :] = 1.0
        input_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        input_mask[self.in_start : self.in_end] = True
        hidden_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        hidden_mask[self.hidden_start :] = True

        self.register_buffer("edge_mask", edge_mask)
        self.register_buffer("hidden_gate_mask", hidden_gate_mask)
        self.register_buffer("input_mask", input_mask)
        self.register_buffer("non_input_mask", ~input_mask)
        self.register_buffer("hidden_mask", hidden_mask)
        self.register_buffer("prune_mask", torch.ones(self.n_nodes, self.n_nodes))

        self.sat_hidden_margin = 0.0
        self._last_hidden_sat: torch.Tensor | None = None

    def gate_probabilities(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return smooth edge and node probabilities for losses and reports."""

        keep = (self.edge_mask * self.prune_mask).to(self.edge_logits.dtype)
        edge = torch.sigmoid(self.edge_logits) * keep
        hidden = torch.sigmoid(self.node_logits)
        node = (
            1.0
            - self.hidden_gate_mask
            + self.hidden_gate_mask * hidden
        )
        return edge, node

    def gates(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact-forward edge and node gates with smooth derivatives."""

        keep = (self.edge_mask * self.prune_mask).to(self.edge_logits.dtype)
        edge = _binary_gate(torch.sigmoid(self.edge_logits)) * keep
        hidden = _binary_gate(torch.sigmoid(self.node_logits))
        node = (
            1.0
            - self.hidden_gate_mask
            + self.hidden_gate_mask * hidden
        )
        return edge, node

    def soft_structure(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the circuit gates used by forward execution."""

        return self.gates()

    def effective_weight(self) -> torch.Tensor:
        edge_gate, node_gate = self.gates()
        return (
            edge_gate
            * self.edge_weight
            * node_gate.unsqueeze(0)
            * node_gate.unsqueeze(1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 0 or inputs.shape[-1] != self.d_in:
            raise ValueError(f"inputs must end with width {self.d_in}")
        leading_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.d_in)
        values = flat.new_zeros(flat.shape[0], self.n_nodes)
        values[:, self.in_start : self.in_end] = flat

        effective = self.effective_weight()
        _, node_gate = self.gates()
        non_input = self.non_input_mask.unsqueeze(0)
        hidden_mask = self.hidden_mask.unsqueeze(0)

        collect = self.sat_hidden_margin > 0 and self.n_hidden > 0
        penalties: list[torch.Tensor] = []
        self._last_hidden_sat = None

        for _ in range(self.num_ticks):
            pre = F.linear(values, effective, self.bias)
            if collect:
                hidden_pre = pre[:, self.hidden_start :]
                live = node_gate[self.hidden_start :].detach().unsqueeze(0)
                penalties.append(
                    (torch.relu(self.sat_hidden_margin - hidden_pre.abs()) * live).mean()
                )
            proposed = torch.where(hidden_mask, torch.sigmoid(pre), pre)
            values = torch.where(non_input, proposed, values)
            values = values * node_gate.unsqueeze(0)

        if penalties:
            self._last_hidden_sat = torch.stack(penalties).mean()

        outputs = values[:, self.out_start : self.out_end]
        return outputs.reshape(*leading_shape, self.d_out)

    def structure_loss(self) -> torch.Tensor:
        edge_gate, node_gate = self.gates()
        keep = self.edge_mask * self.prune_mask
        edge_count = keep.sum().clamp_min(1.0)
        hidden_count = self.hidden_gate_mask.sum().clamp_min(1.0)
        edge_density = edge_gate.sum() / edge_count
        hidden_density = (
            node_gate * self.hidden_gate_mask
        ).sum() / hidden_count
        return (
            self.edge_sparsity * edge_density
            + self.neuron_sparsity * hidden_density
        )

    def sparsity_loss(self) -> torch.Tensor:
        return self.structure_loss()

    def binarization_loss(self) -> torch.Tensor:
        edge_probability, node_probability = self.gate_probabilities()
        keep = self.edge_mask * self.prune_mask
        edge_count = keep.sum().clamp_min(1.0)
        hidden_count = self.hidden_gate_mask.sum().clamp_min(1.0)
        edge_term = (
            edge_probability * (1.0 - edge_probability) * keep
        ).sum() / edge_count
        hidden_term = (
            node_probability
            * (1.0 - node_probability)
            * self.hidden_gate_mask
        ).sum() / hidden_count
        return edge_term + hidden_term

    def last_hidden_saturation(self) -> torch.Tensor:
        if self._last_hidden_sat is None:
            return self.edge_weight.new_tensor(0.0)
        return self._last_hidden_sat

    def prune_edges(self, kill: torch.Tensor) -> int:
        """Permanently remove selected valid edges and return the new count."""

        if kill.shape != self.prune_mask.shape:
            raise ValueError(
                f"kill mask must have shape {tuple(self.prune_mask.shape)}"
            )
        with torch.no_grad():
            selected = kill.to(device=self.prune_mask.device, dtype=torch.bool)
            selected &= self.edge_mask.to(torch.bool)
            selected &= self.prune_mask.to(torch.bool)
            self.prune_mask[selected] = 0.0
        return int(selected.sum().item())

    def prune_to_topk(
        self,
        k: int,
        protect_cols: torch.Tensor | None = None,
    ) -> int:
        """Keep at most ``k`` strongest unpruned incoming edges per live row."""

        k = int(k)
        if k < 0:
            raise ValueError("k must be non-negative")
        if protect_cols is not None and protect_cols.shape != (self.n_nodes,):
            raise ValueError(f"protect_cols must have shape ({self.n_nodes},)")

        with torch.no_grad():
            magnitude = self.effective_weight().abs()
            alive = (self.edge_mask * self.prune_mask) > 0
            masked = torch.where(alive, magnitude, torch.full_like(magnitude, -1.0))
            kill = torch.zeros_like(alive)
            protected = (
                protect_cols.to(device=alive.device, dtype=torch.bool)
                if protect_cols is not None
                else None
            )
            for row in range(self.out_start, self.n_nodes):
                row_alive = alive[row]
                if int(row_alive.sum().item()) <= k:
                    continue
                row_kill = row_alive.clone()
                if k > 0:
                    keep_indices = torch.topk(masked[row], k=k).indices
                    row_kill[keep_indices] = False
                if protected is not None:
                    row_kill &= ~protected
                kill[row] = row_kill
        return self.prune_edges(kill)

    def unpruned_fanin(self) -> torch.Tensor:
        alive = (self.edge_mask * self.prune_mask) > 0
        return alive[self.out_start :].sum(dim=1)

    def structure_summary(
        self,
        *,
        weight_threshold: float = 0.05,
        top_k: int = 20,
    ) -> dict[str, Any]:
        edge_gate, node_gate = self.gates()
        effective = self.effective_weight()
        alive = (
            (self.edge_mask > 0)
            & (self.prune_mask > 0)
            & (edge_gate >= 0.5)
            & (effective.abs() >= float(weight_threshold))
        )
        indices = torch.nonzero(alive, as_tuple=False).detach().cpu()
        edge_gate_cpu = edge_gate.detach().cpu()
        effective_cpu = effective.detach().cpu()
        node_gate_cpu = node_gate.detach().cpu()

        edges: list[dict[str, Any]] = []
        for destination, source in indices.tolist():
            edges.append(
                {
                    "source": self._node_label(int(source)),
                    "target": self._node_label(int(destination)),
                    "weight": float(effective_cpu[destination, source]),
                    "edge_gate": float(edge_gate_cpu[destination, source]),
                }
            )
        edges.sort(key=lambda item: abs(float(item["weight"])), reverse=True)

        hidden = [
            {
                "node": self._node_label(index),
                "gate": float(node_gate_cpu[index]),
            }
            for index in range(self.hidden_start, self.n_nodes)
        ]
        return {
            "nodes": self.n_nodes,
            "active_edges": len(edges),
            "active_hidden_neurons": sum(item["gate"] >= 0.5 for item in hidden),
            "hidden_neurons": hidden,
            "edges": edges[: max(0, int(top_k))],
        }

    def _node_label(self, index: int) -> str:
        if index < self.in_end:
            return f"in_{index}"
        if index < self.out_end:
            return f"out_{index - self.out_start}"
        return f"hidden_{index - self.hidden_start}"


__all__ = ["NP4GCircuit"]
