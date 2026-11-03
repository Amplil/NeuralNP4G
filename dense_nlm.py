"""Self-contained dense breadth-3 Neural Logic Machine comparison model.

The expand, reduce, permutation, and dense logic operations preserve the
repository's original PyTorch baseline, including its parameter names and
initialization order. This module requires only PyTorch and supports the
official graph batches without importing the older experimental package.
"""

from __future__ import annotations

from itertools import permutations
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import GraphBatch


def expand(input_tensor: torch.Tensor, num_objects: int) -> torch.Tensor:
    """Replicate an input over a new second-last object dimension."""
    ndims = input_tensor.dim() + 1
    multiples = [num_objects if i == ndims - 2 else 1 for i in range(ndims)]
    return input_tensor.unsqueeze(-2).repeat(*multiples)


def _reduce_mask(input_tensor: torch.Tensor, num_objects: int) -> torch.Tensor:
    dimension = input_tensor.dim() - 2
    base = 1.0 - torch.eye(num_objects, device=input_tensor.device, dtype=input_tensor.dtype)
    if dimension < 2:
        return torch.ones((), device=input_tensor.device, dtype=input_tensor.dtype)
    if dimension == 2:
        return base
    if dimension == 3:
        return base.unsqueeze(2) * base.unsqueeze(1) * base.unsqueeze(0)
    raise NotImplementedError("Only NLM breadth <= 3 is implemented.")


def reduce(input_tensor: torch.Tensor, num_objects: int) -> torch.Tensor:
    """Concatenate masked max/min reductions over the last object axis."""
    mask = _reduce_mask(input_tensor, num_objects)
    mask = mask.reshape((1,) + tuple(mask.shape) + (1,))
    mask = mask.to(device=input_tensor.device, dtype=input_tensor.dtype)
    max_part = torch.amax(input_tensor * mask, dim=-2)
    min_part = torch.amin(input_tensor * mask + (1.0 - mask), dim=-2)
    return torch.cat([max_part, min_part], dim=-1)


def input_permutations(input_tensor: torch.Tensor) -> torch.Tensor:
    """Concatenate every permutation of the input's object axes."""
    dimension = input_tensor.dim() - 2
    if dimension < 2:
        return input_tensor
    pieces = []
    for perm in permutations(range(1, 1 + dimension)):
        pieces.append(input_tensor.permute(0, *perm, 1 + dimension))
    return torch.cat(pieces, dim=-1)


def _activation_from_name(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    activations: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": F.relu,
        "gelu": F.gelu,
        "linear": lambda x: x,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]


class TorchNeuralLogic(nn.Module):
    """A dense neural logic block applied to all object-axis permutations."""

    def __init__(self, hidden_dim: int, activation: str = "sigmoid") -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.linear = nn.LazyLinear(hidden_dim)
        self.activation_name = activation
        self.activation_fn = _activation_from_name(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(self.linear(input_permutations(x)))


class TorchNLMNeuralLogicLayerBreadth3(nn.Module):
    """One breadth-3 layer using dense neural logic for arities 0 through 3."""

    def __init__(
        self,
        num_objects: int,
        hidden_dim: int,
        residual: bool = False,
        input_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_objects = int(num_objects)
        self.hidden_dim = int(hidden_dim)
        self.residual = bool(residual)
        self.input_dims = tuple(input_dims or (hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.logic_layers = nn.ModuleList(
            [TorchNeuralLogic(hidden_dim) for _ in range(4)]
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        input0, input1, input2, input3 = inputs
        agg0 = torch.cat([input0, reduce(input1, self.num_objects)], dim=-1)
        agg1 = torch.cat([input1, expand(input0, self.num_objects), reduce(input2, self.num_objects)], dim=-1)
        agg2 = torch.cat([input2, expand(input1, self.num_objects), reduce(input3, self.num_objects)], dim=-1)
        agg3 = torch.cat([input3, expand(input2, self.num_objects)], dim=-1)
        outputs = [
            logic(agg)
            for logic, agg in zip(self.logic_layers, [agg0, agg1, agg2, agg3])
        ]
        if self.residual:
            outputs = [torch.cat([out, src], dim=-1) for out, src in zip(outputs, inputs)]
        return outputs

    def weight_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def architecture_parameters(self) -> list[nn.Parameter]:
        return []


class GraphReasoningModel(nn.Module):
    """Stacked dense breadth-3 NLM with a unary or binary prediction head.

    The unused NP4G configuration arguments keep the baseline construction
    interface compatible with existing experiment configurations. The model
    supports the dense ``nlm`` kind only.
    """

    def __init__(
        self,
        kind: str,
        num_objects: int,
        hidden_dim: int,
        depth: int,
        np4g_hidden: int,
        np4g_ticks: int,
        residual: bool = False,
        edge_sparsity: float = 1e-4,
        neuron_sparsity: float = 1e-3,
        edge_logit_init: float = 0.0,
        sign_logit_init: float = 2.0,
        target_arity_value: int = 2,
        input_dims: tuple[int, int, int, int] = (1, 1, 2, 1),
    ) -> None:
        super().__init__()
        if kind != "nlm":
            raise ValueError(f"Unknown model kind: {kind}; this module provides dense NLM only.")
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if target_arity_value not in (1, 2):
            raise ValueError("target_arity_value must be 1 or 2.")
        self.kind = kind
        self.num_objects = int(num_objects)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.residual = bool(residual)
        self.target_arity_value = int(target_arity_value)

        current_input_dims = input_dims
        layers: list[nn.Module] = []
        for _layer_idx in range(depth):
            layer = TorchNLMNeuralLogicLayerBreadth3(
                num_objects=self.num_objects,
                hidden_dim=hidden_dim,
                residual=residual,
                input_dims=current_input_dims,
            )
            layers.append(layer)
            if residual:
                current_input_dims = tuple(dim + hidden_dim for dim in current_input_dims)
            else:
                current_input_dims = (hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(layers)
        self.output_dims = current_input_dims
        self.head = nn.Linear(current_input_dims[self.target_arity_value], 1)

    def set_num_objects(self, num_objects: int) -> None:
        self.num_objects = int(num_objects)
        for layer in self.layers:
            layer.num_objects = self.num_objects

    def structure_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_tensor(0.0)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        inputs: list[torch.Tensor] = [batch.input0, batch.input1, batch.input2, batch.input3]
        for layer in self.layers:
            inputs = layer(inputs)
        return self.head(inputs[self.target_arity_value]).squeeze(-1)

    def structure_summary(self) -> None:
        return None


def masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    balanced: bool = True,
) -> torch.Tensor:
    """Compute the balanced binary cross entropy over the selected cells."""
    pos_weight = None
    if balanced:
        positives = (target * mask).sum()
        negatives = ((1.0 - target) * mask).sum()
        pos_weight = negatives / positives.clamp_min(1.0)
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight,
    )
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def warmup_cuda_torch(device: torch.device) -> None:
    """Initialize PyTorch CUDA and cuBLAS before the first dense forward."""
    if device.type != "cuda":
        return
    idx = device.index if device.index is not None else 0
    torch.cuda.set_device(idx)
    torch.cuda.init()
    x = torch.randn(64, 64, device=device, dtype=torch.float32)
    torch.matmul(x, x.t()).sum().item()
    lin = nn.Linear(64, 64, device=device)
    lin(x[:16]).sum().item()
    torch.cuda.synchronize(device)
