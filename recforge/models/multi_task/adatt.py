"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Zheng H, Xu T, Chen B, et al. AdaTT builds adaptive task-to-task communication
        on top of a shared representation for multi-task recommendation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from recforge.basic.model import BaseModel
from recforge.basic.layers import EmbeddingLayer, MLP, PredictionLayer
from recforge.basic.features import DenseFeature, SparseFeature, SequenceFeature


class AdaTT(BaseModel):
    """Adaptive Task-to-Task communication network.

    AdaTT first builds a shared representation and then performs self-attention
    across task-specific projections to adaptively transfer knowledge.
    """

    @property
    def model_name(self) -> str:
        return "AdaTT"

    @property
    def task_type(self):
        return self.task if isinstance(self.task, list) else [self.task]

    def __init__(
        self,
        dense_features: list[DenseFeature],
        sparse_features: list[SparseFeature],
        sequence_features: list[SequenceFeature],
        shared_bottom_params: dict,
        tower_params_list: list[dict],
        target: list[str],
        transfer_dim: int | None = None,
        num_heads: int = 2,
        attention_dropout: float = 0.0,
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
        if len(tower_params_list) != self.num_tasks:
            raise ValueError(
                f"Number of tower params ({len(tower_params_list)}) must match number of tasks ({self.num_tasks})"
            )

        self.all_features = dense_features + sparse_features + sequence_features
        self.embedding = EmbeddingLayer(features=self.all_features)

        emb_dim_total = sum(f.embedding_dim for f in self.all_features if not isinstance(f, DenseFeature))
        dense_input_dim = sum((getattr(f, "embedding_dim", 1) or 1) for f in dense_features)
        input_dim = emb_dim_total + dense_input_dim

        self.shared_bottom = MLP(input_dim=input_dim, output_layer=False, **shared_bottom_params)
        if shared_bottom_params.get("dims"):
            shared_dim = shared_bottom_params["dims"][-1]
        else:
            shared_dim = input_dim

        self.transfer_dim = int(transfer_dim or shared_dim)
        self.num_heads = max(1, int(num_heads))
        if self.transfer_dim % self.num_heads != 0:
            raise ValueError("transfer_dim must be divisible by num_heads")

        self.task_projections = nn.ModuleList(
            [nn.Linear(shared_dim, self.transfer_dim) for _ in range(self.num_tasks)]
        )

        self.q_proj = nn.Linear(self.transfer_dim, self.transfer_dim)
        self.k_proj = nn.Linear(self.transfer_dim, self.transfer_dim)
        self.v_proj = nn.Linear(self.transfer_dim, self.transfer_dim)
        self.o_proj = nn.Linear(self.transfer_dim, self.transfer_dim)
        self.attn_dropout = nn.Dropout(attention_dropout)

        self.residual_gates = nn.Parameter(torch.zeros(self.num_tasks, 2))
        self.layer_norm = nn.LayerNorm(self.transfer_dim)

        self.towers = nn.ModuleList(
            [MLP(input_dim=self.transfer_dim, output_layer=True, **params) for params in tower_params_list]
        )
        self.prediction_layer = PredictionLayer(task_type=self.task_type, task_dims=[1] * self.num_tasks)

        self._register_regularization_weights(
            embedding_attr="embedding",
            include_modules=[
                "shared_bottom",
                "task_projections",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "towers",
            ],
        )

        self.compile(optimizer=optimizer, optimizer_params=optimizer_params, loss=loss)

    def forward(self, x):
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        shared_feat = self.shared_bottom(input_flat)
        task_states = torch.stack([proj(shared_feat) for proj in self.task_projections], dim=1)

        attn_out = self._task_attention(task_states)
        task_outputs: list[torch.Tensor] = []
        for task_idx in range(self.num_tasks):
            alpha_beta = torch.sigmoid(self.residual_gates[task_idx])
            fused = alpha_beta[0] * task_states[:, task_idx, :] + alpha_beta[1] * attn_out[:, task_idx, :]
            fused = self.layer_norm(fused)
            task_output = self.towers[task_idx](fused)
            task_outputs.append(task_output)

        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)

    def _task_attention(self, states: torch.Tensor) -> torch.Tensor:
        # states: [B, T, D]
        q = self.q_proj(states)
        k = self.k_proj(states)
        v = self.v_proj(states)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.transfer_dim // self.num_heads)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = self._merge_heads(context)
        return self.o_proj(context)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        head_dim = self.transfer_dim // self.num_heads
        x = x.view(bsz, seq_len, self.num_heads, head_dim)
        return x.permute(0, 2, 1, 3)  # [B, H, T, head_dim]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, head_dim]
        bsz, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bsz, seq_len, num_heads * head_dim)
