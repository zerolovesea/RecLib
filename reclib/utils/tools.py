import numpy as np
import torch
import torch.nn as nn
from typing import Iterator, Iterable
import pandas as pd

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from collections import OrderedDict

def get_auto_embedding_dim(num_classes: int) -> int:
    """ Calculate the dim of embedding vector according to number of classes in the category
    emb_dim = [6 * (num_classes)^(1/4)]
    reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    Args:
        num_classes: number of classes in the category

    Returns:
        the dim of embedding vector
    """
    return int(np.floor(6 * np.power(num_classes, 0.25)))


def get_task_type(model) -> str:
    return model.task_type

def get_initializer_fn(init_type='normal', activation='linear', param=None):
    param = param or {}

    try:
        gain = param.get('gain', nn.init.calculate_gain(activation, param.get('param', None)))
    except ValueError:
        gain = 1.0  # for custom activations like 'dice'
    
    def initializer_fn(tensor):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(tensor, gain=gain)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(tensor, gain=gain)
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(tensor, a=param.get('a', 0), nonlinearity=activation)
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(tensor, a=param.get('a', 0), nonlinearity=activation)
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(tensor, gain=gain)
        elif init_type == 'normal':
            nn.init.normal_(tensor, mean=param.get('mean', 0.0), std=param.get('std', 0.0001))
        elif init_type == 'uniform':
            nn.init.uniform_(tensor, a=param.get('a', -0.05), b=param.get('b', 0.05))
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        return tensor

    return initializer_fn


def get_optimizer_fn(
                optimizer: str = "adam",
                params: Iterable[torch.nn.Parameter] | None = None, 
                **optimizer_params): # type: ignore
    
    if params is None:
        raise ValueError("params cannot be None. Please provide model parameters.")

    if 'lr' not in optimizer_params:
        optimizer_params['lr'] = 1e-3
    
    if isinstance(optimizer, str):
        opt_name = optimizer.lower()
        if opt_name == "adam":
            opt_class = torch.optim.Adam
        elif opt_name == "sgd":
            opt_class = torch.optim.SGD
        elif opt_name == "adamw":
            opt_class = torch.optim.AdamW
        elif opt_name == "adagrad":
            opt_class = torch.optim.Adagrad
        elif opt_name == "rmsprop":
            opt_class = torch.optim.RMSprop
        else:
            raise NotImplementedError(f"Unsupported optimizer: {optimizer}")
        optimizer_fn = opt_class(params=params, **optimizer_params)

    elif isinstance(optimizer, torch.optim.Optimizer):
        optimizer_fn = optimizer
    return optimizer_fn

def get_loss_fn(task_type: str = "binary",
        loss: str | nn.Module | None = "bce",):
    
    if loss is None:
        if task_type == "binary":
            loss_fn = nn.BCELoss()
        elif task_type == "multiclass":
            loss_fn = nn.CrossEntropyLoss()
        elif task_type == "regression":
            loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
    elif isinstance(loss, str):
        if loss == "bce" or loss == "binary_crossentropy":
            loss_fn = nn.BCELoss()
        elif loss == "mse":
            loss_fn = nn.MSELoss()
        elif loss == "mae":
            loss_fn = nn.L1Loss()
        elif loss == "crossentropy" or loss == "ce":
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Unsupported loss function: {loss}")
    elif isinstance(loss, nn.Module):
        loss_fn = loss
    else:
        raise TypeError(f"Invalid loss type: {type(loss)}")
    return loss_fn

def get_scheduler_fn(scheduler, optimizer, **scheduler_params):
    if isinstance(scheduler, str):
        if scheduler == "step":
            scheduler_fn = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler == "cosine":
            scheduler_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        else:
            raise NotImplementedError(f"Unsupported scheduler: {scheduler}")
    elif isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
        scheduler_fn = scheduler
    else:
        raise TypeError(f"Invalid scheduler type: {type(scheduler)}")
    return scheduler_fn


