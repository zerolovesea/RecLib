"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Ma X, Zhao L, Huang G, et al. Top-K sparsely activated experts for multi-task learning.
        (STAR) is widely used in large scale recommendation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from recforge.basic.model import BaseModel
from recforge.basic.layers import EmbeddingLayer, MLP, PredictionLayer
from recforge.basic.features import DenseFeature, SparseFeature, SequenceFeature


class STAR(BaseModel):
    """Sparse-activated Router (STAR) for multi-task learning.

    STAR extends MMOE by introducing a sparse routing gate that only activates
    a small portion of experts for each task. This reduces negative transfer
    and stabilizes training when the number of tasks or experts is large.
    """

    @property
    def model_name(self) -> str:
        return "STAR"

    @property
    def task_type(self):
        return self.task if isinstance(self.task, list) else [self.task]

    def __init__(
        self,
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        expert_params: dict,
        num_experts: int,
        tower_params_list: list[dict],
        target: list[str],
        top_k: int = 2,
        gate_hidden_dim: int | None = None,
        temperature: float = 1.0,
        task: str | list[str] = "binary",
        optimizer: str = "adam",
        optimizer_params: dict | None = None,
        loss: str | nn.Module | list[str | nn.Module] | None = "bce",
        device: str = "cpu",
        model_id: str = "baseline",
        embedding_l1_reg: float = 1e-6,
        dense_l1_reg: float = 1e-5,
        embedding_l2_reg: float = 1e-5,
        dense_l2_reg: float = 1e-4,
    ):
        if optimizer_params is None:
            optimizer_params = {}

        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            target=target,
            task=task,
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            early_stop_patience=20,
            model_id=model_id,
        )

        self.loss = loss or "bce"
        self.num_tasks = len(target)
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.temperature = max(1e-3, float(temperature))

        if len(tower_params_list) != self.num_tasks:
            raise ValueError(
                f"Number of tower params ({len(tower_params_list)}) must match number of tasks ({self.num_tasks})"
            )

        self.all_features = dense_features + sparse_features + sequence_features
        self.embedding = EmbeddingLayer(features=self.all_features)

        emb_dim_total = sum(f.embedding_dim for f in self.all_features if not isinstance(f, DenseFeature))
        dense_input_dim = sum((getattr(f, "embedding_dim", 1) or 1) for f in dense_features)
        input_dim = emb_dim_total + dense_input_dim

        # Expert networks shared among tasks
        self.experts = nn.ModuleList(
            [MLP(input_dim=input_dim, output_layer=False, **expert_params) for _ in range(self.num_experts)]
        )

        if expert_params.get("dims"):
            expert_output_dim = expert_params["dims"][-1]
        else:
            expert_output_dim = input_dim

        # Task specific sparse gates
        self.gates = nn.ModuleList()
        for _ in range(self.num_tasks):
            if gate_hidden_dim and gate_hidden_dim > 0:
                gate = nn.Sequential(
                    nn.Linear(input_dim, gate_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(gate_hidden_dim, self.num_experts),
                )
            else:
                gate = nn.Linear(input_dim, self.num_experts)
            self.gates.append(gate)

        self.towers = nn.ModuleList(
            [MLP(input_dim=expert_output_dim, output_layer=True, **params) for params in tower_params_list]
        )
        self.prediction_layer = PredictionLayer(task_type=self.task_type, task_dims=[1] * self.num_tasks)

        self._register_regularization_weights(
            embedding_attr="embedding", include_modules=["experts", "gates", "towers"]
        )

        self.compile(optimizer=optimizer, optimizer_params=optimizer_params, loss=loss)

    def forward(self, x):
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)

        expert_outputs = torch.stack([expert(input_flat) for expert in self.experts], dim=0)
        expert_outputs_t = expert_outputs.permute(1, 0, 2)  # [B, num_experts, expert_dim]

        task_outputs: list[torch.Tensor] = []
        for task_idx in range(self.num_tasks):
            gate_logits = self.gates[task_idx](input_flat)
            gate_weights = self._sparse_softmax(gate_logits)
            gate_weights = gate_weights.unsqueeze(2)
            routed = torch.sum(gate_weights * expert_outputs_t, dim=1)
            task_output = self.towers[task_idx](routed)
            task_outputs.append(task_output)

        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)

    def _sparse_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        if self.top_k >= self.num_experts:
            return torch.softmax(logits / self.temperature, dim=1)

        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(1, topk_idx, topk_vals)
        return torch.softmax(mask / self.temperature, dim=1)
