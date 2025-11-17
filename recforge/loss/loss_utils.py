"""
Loss utilities for RecLib

Date: create on 09/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
"""
import torch
import torch.nn as nn
from typing import Literal

from recforge.loss.match_losses import (
    BPRLoss, 
    HingeLoss, 
    TripletLoss, 
    SampledSoftmaxLoss,
    CosineContrastiveLoss, 
    InfoNCELoss
)

# Valid task types for validation
VALID_TASK_TYPES = ['binary', 'multiclass', 'regression', 'multivariate_regression', 'match', 'ranking', 'multitask', 'multilabel']


def get_loss_fn(
    task_type: str = "binary",
    training_mode: str | None = None,
    loss: str | nn.Module | None = None,
    **loss_kwargs
) -> nn.Module:
    """
    Get loss function based on task type and training mode.
    
    Examples:
        # Ranking task (binary classification)
        >>> loss_fn = get_loss_fn(task_type="binary", loss="bce")
        
        # Match task with pointwise training
        >>> loss_fn = get_loss_fn(task_type="match", training_mode="pointwise")
        
        # Match task with pairwise training
        >>> loss_fn = get_loss_fn(task_type="match", training_mode="pairwise", loss="bpr")
        
        # Match task with listwise training
        >>> loss_fn = get_loss_fn(task_type="match", training_mode="listwise", loss="sampled_softmax")
    """

    if isinstance(loss, nn.Module):
        return loss

    if task_type == "match":
        if training_mode == "pointwise":
            # Pointwise training uses binary cross entropy
            if loss is None or loss == "bce" or loss == "binary_crossentropy":
                return nn.BCELoss(**loss_kwargs)
            elif loss == "cosine_contrastive":
                return CosineContrastiveLoss(**loss_kwargs)
            elif isinstance(loss, str):
                raise ValueError(f"Unsupported pointwise loss: {loss}")
        
        elif training_mode == "pairwise":
            if loss is None or loss == "bpr":
                return BPRLoss(**loss_kwargs)
            elif loss == "hinge":
                return HingeLoss(**loss_kwargs)
            elif loss == "triplet":
                return TripletLoss(**loss_kwargs)
            elif isinstance(loss, str):
                raise ValueError(f"Unsupported pairwise loss: {loss}")
        
        elif training_mode == "listwise":
            if loss is None or loss == "sampled_softmax" or loss == "softmax":
                return SampledSoftmaxLoss(**loss_kwargs)
            elif loss == "infonce":
                return InfoNCELoss(**loss_kwargs)
            elif loss == "crossentropy" or loss == "ce":
                return nn.CrossEntropyLoss(**loss_kwargs)
            elif isinstance(loss, str):
                raise ValueError(f"Unsupported listwise loss: {loss}")
        
        else:
            raise ValueError(f"Unknown training_mode: {training_mode}")

    elif task_type in ["ranking", "multitask", "binary"]:
        if loss is None or loss == "bce" or loss == "binary_crossentropy":
            return nn.BCELoss(**loss_kwargs)
        elif loss == "mse":
            return nn.MSELoss(**loss_kwargs)
        elif loss == "mae":
            return nn.L1Loss(**loss_kwargs)
        elif loss == "crossentropy" or loss == "ce":
            return nn.CrossEntropyLoss(**loss_kwargs)
        elif isinstance(loss, str):
            raise ValueError(f"Unsupported loss function: {loss}")

    elif task_type == "multiclass":
        if loss is None or loss == "crossentropy" or loss == "ce":
            return nn.CrossEntropyLoss(**loss_kwargs)
        elif isinstance(loss, str):
            raise ValueError(f"Unsupported multiclass loss: {loss}")
    
    elif task_type == "regression":
        if loss is None or loss == "mse":
            return nn.MSELoss(**loss_kwargs)
        elif loss == "mae":
            return nn.L1Loss(**loss_kwargs)
        elif isinstance(loss, str):
            raise ValueError(f"Unsupported regression loss: {loss}")
    
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
    
    return loss


def validate_training_mode(
    training_mode: str,
    support_training_modes: list[str],
    model_name: str = "Model"
) -> None:
    """
    Validate that the requested training mode is supported by the model.
    
    Args:
        training_mode: Requested training mode
        support_training_modes: List of supported training modes
        model_name: Name of the model (for error messages)
    
    Raises:
        ValueError: If training mode is not supported
    """
    if training_mode not in support_training_modes:
        raise ValueError(
            f"{model_name} does not support training_mode='{training_mode}'. "
            f"Supported modes: {support_training_modes}"
        )
