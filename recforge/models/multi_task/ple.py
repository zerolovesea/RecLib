"""
Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
Reference:
    [1] Tang H, Liu J, Zhao M, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations[C]//RecSys. 2020: 269-278.
"""

import torch
import torch.nn as nn

from recforge.basic.model import BaseModel
from recforge.basic.layers import EmbeddingLayer, MLP
from recforge.basic.features import DenseFeature, SparseFeature, SequenceFeature


class PLE(BaseModel):
    """
    Progressive Layered Extraction
    
    PLE is an advanced multi-task learning model that extends MMOE by introducing
    both task-specific experts and shared experts at each level. It uses a progressive
    routing mechanism where experts from level k feed into gates at level k+1.
    This design better captures task-specific and shared information progressively.
    """
    
    @property
    def model_name(self):
        return "PLE"

    @property
    def task_type(self):
        return self.task if isinstance(self.task, list) else [self.task]
    
    def __init__(self,
                 dense_features: list[DenseFeature],
                 sparse_features: list[SparseFeature],
                 sequence_features: list[SequenceFeature],
                 shared_expert_params: dict,
                 specific_expert_params: dict,
                 num_shared_experts: int,
                 num_specific_experts: int,
                 num_levels: int,
                 tower_params_list: list[dict],
                 target: list[str],
                 task: str | list[str] = 'binary',
                 optimizer: str = "adam",
                 optimizer_params: dict = {},
                 loss: str | nn.Module | list[str | nn.Module] | None = "bce",
                 device: str = 'cpu',
                 model_id: str = "baseline",
                 embedding_l1_reg=1e-6,
                 dense_l1_reg=1e-5,
                 embedding_l2_reg=1e-5,
                 dense_l2_reg=1e-4):
        
        super(PLE, self).__init__(
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
            model_id=model_id
        )

        self.loss = loss
        if self.loss is None:
            self.loss = "bce"
        
        # Number of tasks, experts, and levels
        self.num_tasks = len(target)
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_levels = num_levels
        
        if len(tower_params_list) != self.num_tasks:
            raise ValueError(f"Number of tower params ({len(tower_params_list)}) must match number of tasks ({self.num_tasks})")
            
        # All features
        self.all_features = dense_features + sparse_features + sequence_features

        # Embedding layer
        self.embedding = EmbeddingLayer(features=self.all_features)

        # Calculate input dimension
        emb_dim_total = sum([f.embedding_dim for f in self.all_features if not isinstance(f, DenseFeature)])
        dense_input_dim = sum([getattr(f, "embedding_dim", 1) or 1 for f in dense_features])
        input_dim = emb_dim_total + dense_input_dim
        
        # Get expert output dimension
        if 'dims' in shared_expert_params and len(shared_expert_params['dims']) > 0:
            expert_output_dim = shared_expert_params['dims'][-1]
        else:
            expert_output_dim = input_dim
        
        # Build extraction layers (CGC layers)
        self.shared_experts_layers = nn.ModuleList()  # [num_levels]
        self.specific_experts_layers = nn.ModuleList()  # [num_levels, num_tasks]
        self.gates_layers = nn.ModuleList()  # [num_levels, num_tasks + 1] (+1 for shared gate)
        
        for level in range(num_levels):
            # Input dimension for this level
            level_input_dim = input_dim if level == 0 else expert_output_dim
            
            # Shared experts for this level
            shared_experts = nn.ModuleList()
            for _ in range(num_shared_experts):
                expert = MLP(input_dim=level_input_dim, output_layer=False, **shared_expert_params)
                shared_experts.append(expert)
            self.shared_experts_layers.append(shared_experts)
            
            # Task-specific experts for this level
            specific_experts_for_tasks = nn.ModuleList()
            for _ in range(self.num_tasks):
                task_experts = nn.ModuleList()
                for _ in range(num_specific_experts):
                    expert = MLP(input_dim=level_input_dim, output_layer=False, **specific_expert_params)
                    task_experts.append(expert)
                specific_experts_for_tasks.append(task_experts)
            self.specific_experts_layers.append(specific_experts_for_tasks)
            
            # Gates for this level (num_tasks task gates + 1 shared gate)
            gates = nn.ModuleList()
            # Task-specific gates
            for _ in range(self.num_tasks):
                num_experts_for_gate = num_shared_experts + num_specific_experts
                gate = nn.Sequential(
                    nn.Linear(level_input_dim, num_experts_for_gate),
                    nn.Softmax(dim=1)
                )
                gates.append(gate)
            # Shared gate
            gate = nn.Sequential(
                nn.Linear(level_input_dim, num_shared_experts),
                nn.Softmax(dim=1)
            )
            gates.append(gate)
            self.gates_layers.append(gates)
        
        # Task-specific towers
        self.towers = nn.ModuleList()
        for tower_params in tower_params_list:
            tower = MLP(input_dim=expert_output_dim, output_layer=True, **tower_params)
            self.towers.append(tower)

        # Register regularization weights
        self._register_regularization_weights(
            embedding_attr='embedding',
            include_modules=['shared_experts_layers', 'specific_experts_layers', 'gates_layers', 'towers']
        )

        self.compile(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss=loss
        )

    def forward(self, x):
        # Get all embeddings and flatten
        input_flat = self.embedding(x=x, features=self.all_features, squeeze_dim=True)
        
        # Progressive extraction through levels
        task_fea = [input_flat] * self.num_tasks  # Initialize task features
        shared_fea = input_flat  # Initialize shared feature
        
        for level in range(self.num_levels):
            # Get experts for this level
            shared_experts = self.shared_experts_layers[level]
            specific_experts = self.specific_experts_layers[level]
            gates = self.gates_layers[level]
            
            # Compute shared expert outputs
            shared_expert_outputs = [expert(shared_fea) for expert in shared_experts]
            shared_expert_outputs = torch.stack(shared_expert_outputs, dim=0)  # [num_shared, B, expert_dim]
            
            # Process each task
            new_task_fea = []
            for task_idx in range(self.num_tasks):
                # Task-specific expert outputs
                task_experts = specific_experts[task_idx]
                task_expert_outputs = [expert(task_fea[task_idx]) for expert in task_experts]
                task_expert_outputs = torch.stack(task_expert_outputs, dim=0)  # [num_specific, B, expert_dim]
                
                # Combine shared and task-specific experts
                all_expert_outputs = torch.cat([shared_expert_outputs, task_expert_outputs], dim=0)  # [num_shared + num_specific, B, expert_dim]
                
                # Gate for this task
                gate_weights = gates[task_idx](task_fea[task_idx])  # [B, num_shared + num_specific]
                gate_weights = gate_weights.unsqueeze(2)  # [B, num_shared + num_specific, 1]
                all_expert_outputs_t = all_expert_outputs.permute(1, 0, 2)  # [B, num_experts, expert_dim]
                
                # Gated output for this task
                gated_output = torch.sum(gate_weights * all_expert_outputs_t, dim=1)  # [B, expert_dim]
                new_task_fea.append(gated_output)
            
            # Shared gate (only uses shared experts)
            shared_gate_weights = gates[self.num_tasks](shared_fea)  # [B, num_shared]
            shared_gate_weights = shared_gate_weights.unsqueeze(2)  # [B, num_shared, 1]
            shared_expert_outputs_t = shared_expert_outputs.permute(1, 0, 2)  # [B, num_shared, expert_dim]
            new_shared_fea = torch.sum(shared_gate_weights * shared_expert_outputs_t, dim=1)  # [B, expert_dim]
            
            # Update for next level
            task_fea = new_task_fea
            shared_fea = new_shared_fea
        
        # Task-specific towers
        task_outputs = []
        for task_idx in range(self.num_tasks):
            tower_output = self.towers[task_idx](task_fea[task_idx])  # [B, 1]
            task_outputs.append(tower_output)
        
        # Stack outputs: [B, num_tasks]
        y = torch.cat(task_outputs, dim=1)
        y = torch.sigmoid(y)
        
        return y  # [B, num_tasks]
