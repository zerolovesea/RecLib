"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Li X, Chen H, Li J, et al. PEPNet: Parameter and Embedding Personalized Network for large-scale
        multi-task recommendation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from recforge.basic.model import BaseModel
from recforge.basic.layers import EmbeddingLayer, MLP, PredictionLayer
from recforge.basic.features import DenseFeature, SparseFeature, SequenceFeature


class PEPNet(BaseModel):
    """Parameter & Embedding Personalized Network.

    A shared backbone learns a holistic representation while task-aware adapters
    modulate this representation per sample via a lightweight hyper-network
    conditioned on both inputs and learnable task embeddings.
    """

    @property
    def model_name(self) -> str:
        return "PEPNet"

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
        task_emb_dim: int = 32,
        adapter_hidden_dim: int = 64,
        context_dropout: float = 0.0,
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

        self.task_emb_dim = int(task_emb_dim)
        self.context_dropout = nn.Dropout(context_dropout)

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

        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, self.task_emb_dim),
            nn.ReLU(),
        )

        if adapter_hidden_dim and adapter_hidden_dim > 0:
            self.adapter_generator = nn.Sequential(
                nn.Linear(self.task_emb_dim, adapter_hidden_dim),
                nn.ReLU(),
                nn.Linear(adapter_hidden_dim, shared_dim * 2),
            )
        else:
            self.adapter_generator = nn.Linear(self.task_emb_dim, shared_dim * 2)

        self.task_embeddings = nn.Parameter(torch.randn(self.num_tasks, self.task_emb_dim))

        self.towers = nn.ModuleList(
            [MLP(input_dim=shared_dim, output_layer=True, **params) for params in tower_params_list]
        )
        self.prediction_layer = PredictionLayer(task_type=self.task_type, task_dims=[1] * self.num_tasks)

        self._register_regularization_weights(
            embedding_attr="embedding",
            include_modules=["shared_bottom", "context_encoder", "adapter_generator", "towers"],
        )

        self.compile(optimizer=optimizer, optimizer_params=optimizer_params, loss=loss)

    def forward(self, x):
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        shared_feat = self.shared_bottom(input_flat)
        context = self.context_dropout(self.context_encoder(input_flat))

        task_outputs: list[torch.Tensor] = []
        for task_idx in range(self.num_tasks):
            task_context = context + self.task_embeddings[task_idx]
            adapter_params = self.adapter_generator(task_context)
            scale, shift = adapter_params.chunk(2, dim=1)
            scale = torch.sigmoid(scale)
            personalized = shared_feat * (1.0 + scale) + shift
            task_output = self.towers[task_idx](personalized)
            task_outputs.append(task_output)

        y = torch.cat(task_outputs, dim=1)
        return self.prediction_layer(y)
