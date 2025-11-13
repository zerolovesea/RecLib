"""
Base Model & Base Match Model Class

Date: create on 27/10/2025
Author:
    Yang Zhou,zyaztec@gmail.com
"""

import os
import tqdm
import torch
import logging
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Literal
from torch.utils.data import DataLoader, TensorDataset

from reclib.basic.callback import EarlyStopper
from reclib.basic.features import DenseFeature, SparseFeature, SequenceFeature
from reclib.basic.metrics import configure_metrics, evaluate_metrics

from reclib.data.utils import get_column_data
from reclib.basic.loggers import setup_logger, colorize
from reclib.utils.tools import get_optimizer_fn, get_scheduler_fn
from reclib.loss import get_loss_fn


class BaseModel(nn.Module):
    @property
    def model_name(self) -> str:
        raise NotImplementedError
    
    @property
    def task_type(self) -> str:
        raise NotImplementedError

    def __init__(self, 
                 dense_features: list[DenseFeature] | None = None, 
                 sparse_features: list[SparseFeature] | None = None, 
                 sequence_features: list[SequenceFeature] | None = None,
                 target: list[str] | str | None = None,
                 task: str|list[str] = 'binary',
                 device: str = 'cpu',
                 embedding_l1_reg: float = 0.0,
                 dense_l1_reg: float = 0.0,
                 embedding_l2_reg: float = 0.0, 
                 dense_l2_reg: float = 0.0,
                 early_stop_patience: int = 20, 
                 model_id: str = 'baseline'):
        
        super(BaseModel, self).__init__()

        self.device = torch.device(device)
        self.dense_features = list(dense_features) if dense_features is not None else []
        self.sparse_features = list(sparse_features) if sparse_features is not None else []
        self.sequence_features = list(sequence_features) if sequence_features is not None else []
        
        if isinstance(target, str):
            self.target = [target]
        else:
            self.target = list(target) if target is not None else []
        
        self.target_index = {target_name: idx for idx, target_name in enumerate(self.target)}

        self.task = task
        self.nums_task = len(task) if isinstance(task, list) else 1

        self._embedding_l1_reg = embedding_l1_reg
        self._dense_l1_reg = dense_l1_reg
        self._embedding_l2_reg = embedding_l2_reg
        self._dense_l2_reg = dense_l2_reg

        self._regularization_weights = [] # list of dense weights for regularization, used to compute reg loss
        self._embedding_params = [] # list of embedding weights for regularization, used to compute reg loss

        self.early_stop_patience = early_stop_patience
        self._max_gradient_norm = 1.0   # Maximum gradient norm for gradient clipping

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_id = model_id
        
        checkpoint_dir = os.path.abspath(os.path.join(project_root, "..", "checkpoints"))
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.checkpoint = os.path.join(
            checkpoint_dir,
            f"{self.model_name}_{self.model_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.model"
        )

        self.best = os.path.join(
            checkpoint_dir,
            f"{self.model_name}_{self.model_id}_best.model"
        )


        self._logger_initialized = False
        self._verbose = 1
        self.best = os.path.join(
            checkpoint_dir,
            f"{self.model_name}_{self.model_id}_best.model"
        )

    # register regularization weights to lists
    def _register_regularization_weights(self, 
                                        embedding_attr: str = 'embedding',
                                        exclude_modules: list[str] | None = [], # modules wont add regularization, example: ['fm', 'lr'] / ['fm.fc'] / etc.
                                        include_modules: list[str] | None = []):

        exclude_modules = exclude_modules or []
        
        if hasattr(self, embedding_attr):
            embedding_layer = getattr(self, embedding_attr)
            if hasattr(embedding_layer, 'embed_dict'):
                for embed in embedding_layer.embed_dict.values():
                    self._embedding_params.append(embed.weight)

        for name, module in self.named_modules():
            # Skip self module
            if module is self:
                continue
            
            # Skip embedding layers
            if embedding_attr in name:
                continue
            
            # Skip BatchNorm and Dropout by checking module type
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                                   nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                continue
            
            # White-list: only include modules whose names contain specific keywords
            if include_modules is not None:
                should_include = any(inc_name in name for inc_name in include_modules)
                if not should_include:
                    continue
            
            # Black-list: exclude modules whose names contain specific keywords
            if any(exc_name in name for exc_name in exclude_modules):
                continue
            
            # Only add regularization for Linear layers
            if isinstance(module, nn.Linear):
                self._regularization_weights.append(module.weight)


    # compute regularization loss 
    def add_reg_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0, device=self.device)
        
        if self._embedding_l1_reg > 0 and len(self._embedding_params) > 0:
            for param in self._embedding_params:
                reg_loss += self._embedding_l1_reg * torch.sum(torch.abs(param))

        if self._embedding_l2_reg > 0 and len(self._embedding_params) > 0:
            for param in self._embedding_params:
                reg_loss += self._embedding_l2_reg * torch.sum(param ** 2)

        if self._dense_l1_reg > 0 and len(self._regularization_weights) > 0:
            for param in self._regularization_weights:
                reg_loss += self._dense_l1_reg * torch.sum(torch.abs(param))

        if self._dense_l2_reg > 0 and len(self._regularization_weights) > 0:
            for param in self._regularization_weights:
                reg_loss += self._dense_l2_reg * torch.sum(param ** 2)
        
        return reg_loss



    def _to_tensor(self, value, dtype: torch.dtype | None = None, device: str | torch.device | None = None) -> torch.Tensor:
        if value is None:
            raise ValueError("Cannot convert None to tensor.")
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        target_device = device if device is not None else self.device
        return tensor.to(target_device)

    def get_input(self, input_data: dict|pd.DataFrame):
        X_input = {}
        
        all_features = self.dense_features + self.sparse_features + self.sequence_features
        
        for feature in all_features:
            if feature.name not in input_data:
                continue
            feature_data = get_column_data(input_data, feature.name)
            if feature_data is None:
                continue
            if isinstance(feature, DenseFeature):
                dtype = torch.float32
            else:
                dtype = torch.long
            feature_tensor = self._to_tensor(feature_data, dtype=dtype)
            X_input[feature.name] = feature_tensor

        y = None
        if len(self.target) > 0:
            target_tensors = []
            for target_name in self.target:
                if target_name not in input_data:
                    continue
                target_data = get_column_data(input_data, target_name)
                if target_data is None:
                    continue
                target_tensor = self._to_tensor(target_data, dtype=torch.float32)
                if target_tensor.dim() > 1:
                    target_tensor = target_tensor.view(target_tensor.size(0), -1)
                    target_tensors.extend(torch.chunk(target_tensor, chunks=target_tensor.shape[1], dim=1))
                else:
                    target_tensors.append(target_tensor.view(-1, 1))
            if target_tensors:
                stacked = torch.cat(target_tensors, dim=1)
                if stacked.shape[1] == 1:
                    y = stacked.view(-1)
                else:
                    y = stacked

        return X_input, y

    def _set_metrics(self, metrics: list[str] | dict[str, list[str]] | None = None):
        """Configure metrics for model evaluation using the metrics module."""
        self.metrics, self.task_specific_metrics, self.best_metrics_mode = configure_metrics(
            task=self.task,
            metrics=metrics,
            target_names=self.target
        ) # ['auc', 'logloss'], {'target1': ['auc', 'logloss'], 'target2': ['mse']}, 'max'
        
        if not hasattr(self, 'early_stopper') or self.early_stopper is None:
            self.early_stopper = EarlyStopper(patience=self.early_stop_patience, mode=self.best_metrics_mode)


    def compile(self, 
                optimizer = "adam",
                optimizer_params: dict | None = None,
                scheduler: str | torch.optim.lr_scheduler._LRScheduler | type[torch.optim.lr_scheduler._LRScheduler] | None = None,
                scheduler_params: dict | None = None,
                loss: str | nn.Module | list[str | nn.Module] | None= "bce"):
        if optimizer_params is None:
            optimizer_params = {}
        
        self._optimizer_name = optimizer if isinstance(optimizer, str) else optimizer.__class__.__name__
        self._optimizer_params = optimizer_params
        if isinstance(scheduler, str):
            self._scheduler_name = scheduler
        elif scheduler is not None:
            self._scheduler_name = scheduler.__class__.__name__ if hasattr(scheduler, '__class__') else str(scheduler)
        else:
            self._scheduler_name = None
        self._scheduler_params = scheduler_params or {}
        self._loss_config = loss
        
        # set optimizer
        self.optimizer_fn = get_optimizer_fn(
            optimizer=optimizer, 
            params=self.parameters(), 
            **optimizer_params
        )
        
        # set loss functions
        if self.nums_task == 1:
            task_type = self.task if isinstance(self.task, str) else self.task[0]
            loss_value = loss[0] if isinstance(loss, list) else loss
            # For ranking and multitask, use pointwise training
            training_mode = 'pointwise' if self.task_type in ['ranking', 'multitask'] else None
            # Use task_type directly, not self.task_type for single task
            self.loss_fn = [get_loss_fn(task_type=task_type, training_mode=training_mode, loss=loss_value)]
        else:
            self.loss_fn = []
            for i in range(self.nums_task):
                task_type = self.task[i] if isinstance(self.task, list) else self.task
                
                if isinstance(loss, list):
                    loss_value = loss[i] if i < len(loss) else None
                else:
                    loss_value = loss
                
                # Multitask always uses pointwise training
                training_mode = 'pointwise'
                self.loss_fn.append(get_loss_fn(task_type=task_type, training_mode=training_mode, loss=loss_value))
        
        # set scheduler
        self.scheduler_fn = get_scheduler_fn(scheduler, self.optimizer_fn, **(scheduler_params or {})) if scheduler else None

    def compute_loss(self, y_pred, y_true):
        if y_true is None:
            return torch.tensor(0.0, device=self.device)
        
        if self.nums_task == 1:
            loss = self.loss_fn[0](y_pred, y_true)
            return loss
        
        else:
            task_losses = []
            for i in range(self.nums_task):
                task_loss = self.loss_fn[i](y_pred[:, i], y_true[:, i])
                task_losses.append(task_loss)
            return torch.stack(task_losses)


    def _prepare_data_loader(self, data: dict|pd.DataFrame|DataLoader, batch_size: int = 32, shuffle: bool = True):
        if isinstance(data, DataLoader):
            return data
        tensors = []
        all_features = self.dense_features + self.sparse_features + self.sequence_features
        
        for feature in all_features:
            column = get_column_data(data, feature.name)
            if column is None:
                raise KeyError(f"Feature {feature.name} not found in provided data.")
            
            if isinstance(feature, SequenceFeature):
                if isinstance(column, pd.Series):
                    column = column.values
                if isinstance(column, np.ndarray) and column.dtype == object:
                    column = np.array([np.array(seq, dtype=np.int64) if not isinstance(seq, np.ndarray) else seq for seq in column])
                if isinstance(column, np.ndarray) and column.ndim == 1 and column.dtype == object:
                    column = np.vstack([c if isinstance(c, np.ndarray) else np.array(c) for c in column])  # type: ignore
                tensor = torch.from_numpy(np.asarray(column, dtype=np.int64)).to('cpu')
            else:
                dtype = torch.float32 if isinstance(feature, DenseFeature) else torch.long
                tensor = self._to_tensor(column, dtype=dtype, device='cpu')
            
            tensors.append(tensor)
        
        label_tensors = []
        for target_name in self.target:
            column = get_column_data(data, target_name)
            if column is None:
                continue
            label_tensor = self._to_tensor(column, dtype=torch.float32, device='cpu')
            
            if label_tensor.dim() == 1:
                # 1D tensor: (N,) -> (N, 1)
                label_tensor = label_tensor.view(-1, 1)
            elif label_tensor.dim() == 2:
                if label_tensor.shape[0] == 1 and label_tensor.shape[1] > 1:
                    label_tensor = label_tensor.t()
            
            label_tensors.append(label_tensor)
        
        if label_tensors:
            if len(label_tensors) == 1 and label_tensors[0].shape[1] > 1:
                y_tensor = label_tensors[0]
            else:
                y_tensor = torch.cat(label_tensors, dim=1)

            if y_tensor.shape[1] == 1:
                y_tensor = y_tensor.squeeze(1)
            tensors.append(y_tensor)
        
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


    def _batch_to_dict(self, batch_data: tuple) -> dict:
        result = {}
        all_features = self.dense_features + self.sparse_features + self.sequence_features
        
        for i, feature in enumerate(all_features):
            if i < len(batch_data):
                result[feature.name] = batch_data[i]
        
        if len(batch_data) > len(all_features):
            labels = batch_data[-1]
            
            if self.nums_task == 1:
                result[self.target[0]] = labels
            else:
                if labels.dim() == 2 and labels.shape[1] == self.nums_task:
                    if len(self.target) == 1:
                        result[self.target[0]] = labels
                    else:
                        for i, target_name in enumerate(self.target):
                            if i < labels.shape[1]:
                                result[target_name] = labels[:, i]
                elif labels.dim() == 1:
                    result[self.target[0]] = labels
                else:
                    for i, target_name in enumerate(self.target):
                        if i < labels.shape[-1]:
                            result[target_name] = labels[..., i]
        
        return result


    def fit(self, 
            train_data: dict|pd.DataFrame|DataLoader, 
            valid_data: dict|pd.DataFrame|DataLoader|None=None, 
            metrics: list[str]|dict[str, list[str]]|None = None, # ['auc', 'logloss'] or {'target1': ['auc', 'logloss'], 'target2': ['mse']}
            epochs:int=1, verbose:int=1, shuffle:bool=True, batch_size:int=32,
            user_id_column: str = 'user_id'):

        self.to(self.device)
        if not self._logger_initialized:
            setup_logger()
            self._logger_initialized = True
        self._verbose = verbose

        self._set_metrics(metrics) # add self.metrics, self.task_specific_metrics, self.best_metrics_mode, self.early_stopper
        self.summary()

        if not isinstance(train_data, DataLoader):
            train_loader = self._prepare_data_loader(train_data, batch_size=batch_size, shuffle=shuffle)
        else:
            train_loader = train_data
        
        # Extract user_ids from validation data for GAUC computation
        valid_user_ids: np.ndarray | None = None
        if valid_data is not None and not isinstance(valid_data, DataLoader):
            valid_loader = self._prepare_data_loader(valid_data, batch_size=batch_size, shuffle=False)
            # Extract user_ids if available
            if isinstance(valid_data, pd.DataFrame) and user_id_column in valid_data.columns:
                valid_user_ids = np.asarray(valid_data[user_id_column].values)
            elif isinstance(valid_data, dict) and user_id_column in valid_data:
                valid_user_ids = np.asarray(valid_data[user_id_column])
        elif valid_data is not None:
            valid_loader = valid_data
        else:
            valid_loader = None
        
        try:
            self._steps_per_epoch = len(train_loader)
        except TypeError:
            self._steps_per_epoch = None 
        
        self._epoch_index = 0
        self._stop_training = False
        self._best_metric = float('-inf') if self.best_metrics_mode == 'max' else float('inf')
        
        if self._verbose:
            logging.info(colorize("=" * 80, color="bright_green", bold=True))
            logging.info(colorize(f"Start training", color="bright_green", bold=True))
            logging.info(colorize("=" * 80, color="bright_green", bold=True))
            logging.info("")
            logging.info(colorize(f"Model device: {self.device}", color="bright_green"))
        
        for epoch in range(epochs):
            self._epoch_index = epoch
            train_loss = self.train_epoch(train_loader)

            if self._verbose:
                if self.nums_task == 1:
                    logging.info(colorize(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f}", color="white"))
                else:
                    task_labels = []
                    for i in range(self.nums_task):
                        if i < len(self.target):
                            task_labels.append(self.target[i])
                        else:
                            task_labels.append(f"task_{i}")
                    
                    loss_str = ", ".join([f"{task_labels[i]}_loss: {train_loss[i]:.4f}" for i in range(self.nums_task)])  # type: ignore
                    total_loss_val = np.sum(train_loss) if isinstance(train_loss, np.ndarray) else train_loss  # type: ignore
                    logging.info(colorize(f"Epoch {epoch + 1}/{epochs} - {loss_str}, total_loss: {total_loss_val:.4f}", color="white"))
            
            if valid_loader is not None:
                val_metrics = self.evaluate(valid_loader, user_ids=valid_user_ids) # {'auc': 0.75, 'logloss': 0.45} or {'auc_target1': 0.75, 'logloss_target1': 0.45, 'mse_target2': 3.2}
            
                if self._verbose:
                    if self.nums_task == 1:
                        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                        logging.info(colorize(f"Validation - {metrics_str}", color="cyan", bold=True))
                    else:
                        # multi task metrics
                        task_metrics = {}
                        for metric_key, metric_value in val_metrics.items():
                            for target_name in self.target:
                                if metric_key.endswith(f"_{target_name}"):
                                    if target_name not in task_metrics:
                                        task_metrics[target_name] = {}
                                    metric_name = metric_key.rsplit(f"_{target_name}", 1)[0]
                                    task_metrics[target_name][metric_name] = metric_value
                                    break

                        logging.info(colorize(f"Validation:", color="cyan", bold=True))

                        for target_name in self.target:
                            if target_name in task_metrics:
                                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in task_metrics[target_name].items()])
                                logging.info(colorize(f"  [{target_name}] {metrics_str}", color="cyan", bold=True))
                
                if self.nums_task == 1:
                    primary_metric_key = self.metrics[0]
                else:
                    primary_metric_key = f"{self.metrics[0]}_{self.target[0]}"
                
                primary_metric = val_metrics.get(primary_metric_key, val_metrics[list(val_metrics.keys())[0]])
                improved = False
                
                if self.best_metrics_mode == 'max':
                    if primary_metric > self._best_metric:
                        self._best_metric = primary_metric
                        self.save_weights(self.best)
                        improved = True
                else:
                    if primary_metric < self._best_metric:
                        self._best_metric = primary_metric
                        improved = True
                
                if improved:
                    if self._verbose:
                        logging.info(colorize(f"Validation {primary_metric_key} improved to {self._best_metric:.4f}", color="yellow"))
                    self.save_weights(self.checkpoint)
                    self.early_stopper.trial_counter = 0
                else:
                    self.early_stopper.trial_counter += 1
                    if self._verbose:
                        logging.info(colorize(f"No improvement for {self.early_stopper.trial_counter} epoch(s)", color="yellow"))
                
                if self.early_stopper.trial_counter >= self.early_stopper.patience:
                    self._stop_training = True
                    if self._verbose:
                        logging.info(colorize(f"Early stopping triggered after {epoch + 1} epochs", color="bright_red", bold=True))
                    break
            else:
                self.save_weights(self.checkpoint)
            
            if self._stop_training:
                break
            
            if self.scheduler_fn is not None:
                if isinstance(self.scheduler_fn, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if valid_loader is not None:
                        self.scheduler_fn.step(primary_metric)
                else:
                    self.scheduler_fn.step()
                    
        if self._verbose:
            logging.info("\n")
            logging.info(colorize("Training finished.", color="bright_green", bold=True))
            logging.info("\n")
        
        if valid_loader is not None:
            if self._verbose:
                logging.info(colorize(f"Load best model from: {self.checkpoint}", color="bright_blue"))
            self.load_weights(self.checkpoint)
        
        return self

    def train_epoch(self, train_loader: DataLoader) -> Union[float, np.ndarray]:
        if self.nums_task == 1:
            accumulated_loss = 0.0
        else:
            accumulated_loss = np.zeros(self.nums_task, dtype=np.float64)
        
        self.train()
        num_batches = 0

        if self._verbose:
            # For streaming datasets without known length, set total=None to show progress without percentage
            if self._steps_per_epoch is not None:
                batch_iter = enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {self._epoch_index + 1}", total=self._steps_per_epoch))
            else:
                # Streaming mode: show batch count without total
                batch_iter = enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {self._epoch_index + 1}"))
        else:
            batch_iter = enumerate(train_loader)

        for batch_index, batch_data in batch_iter:
            batch_dict = self._batch_to_dict(batch_data)
            X_input, y_true = self.get_input(batch_dict)

            y_pred = self.forward(X_input)
            loss = self.compute_loss(y_pred, y_true)
            reg_loss = self.add_reg_loss()

            if self.nums_task == 1:
                total_loss = loss + reg_loss
            else:
                total_loss = loss.sum() + reg_loss
            
            self.optimizer_fn.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer_fn.step()
            
            if self.nums_task == 1:
                accumulated_loss += loss.item()
            else:
                accumulated_loss += loss.detach().cpu().numpy()
            
            num_batches += 1

        if self.nums_task == 1:
            return accumulated_loss / num_batches
        else:
            return accumulated_loss / num_batches


    def evaluate(self, data_loader: DataLoader, user_ids: np.ndarray | None = None) -> dict:
        """
        Evaluate the model on validation data.
        
        Args:
            data_loader: DataLoader for evaluation
            user_ids: Optional user IDs for computing GAUC metric
            
        Returns:
            Dictionary of metric values
        """
        self.eval()
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                batch_dict = self._batch_to_dict(batch_data)
                X_input, y_true = self.get_input(batch_dict)
                y_pred = self.forward(X_input)
                
                if y_true is not None:
                    y_true_list.append(y_true.cpu().numpy())
                if y_pred is not None:
                    y_pred_list.append(y_pred.cpu().numpy())

        if len(y_true_list) > 0:
            y_true_all = np.concatenate(y_true_list, axis=0)
        else:
            y_true_all = None
            
        if len(y_pred_list) > 0:
            y_pred_all = np.concatenate(y_pred_list, axis=0)
        else:
            y_pred_all = None
        
        metrics_dict = self.evaluate_metrics(y_true_all, y_pred_all, self.metrics, user_ids)
        
        return metrics_dict


    def evaluate_metrics(self, y_true: np.ndarray|None, y_pred: np.ndarray|None, metrics: list[str], user_ids: np.ndarray|None = None) -> dict:
        """Evaluate metrics using the metrics module."""
        task_specific_metrics = getattr(self, 'task_specific_metrics', None)
        
        return evaluate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
            task=self.task,
            target_names=self.target,
            task_specific_metrics=task_specific_metrics,
            user_ids=user_ids
        )


    def predict(self, data: dict|pd.DataFrame|DataLoader, batch_size: int = 32) -> np.ndarray:
        self.eval()

        if not isinstance(data, DataLoader):
            data_loader = self._prepare_data_loader(data, batch_size=batch_size, shuffle=False)
        else:
            data_loader = data
        
        y_pred_list = []
        
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Predicting", disable=self._verbose == 0):
                batch_dict = self._batch_to_dict(batch_data)
                X_input, _ = self.get_input(batch_dict)
                y_pred = self.forward(X_input)

                if y_pred is not None:
                    y_pred_list.append(y_pred.cpu().numpy())

        if len(y_pred_list) > 0:
            y_pred_all = np.concatenate(y_pred_list, axis=0)
            return y_pred_all
        else:
            return np.array([])
    
    def save_weights(self, model_path: str):
        torch.save(self.state_dict(), model_path)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)


    def summary(self):
        logger = logging.getLogger()
        
        logger.info(colorize("=" * 80, color="bright_blue", bold=True))
        logger.info(colorize(f"Model Summary: {self.model_name}", color="bright_blue", bold=True))
        logger.info(colorize("=" * 80, color="bright_blue", bold=True))
        
        logger.info("")
        logger.info(colorize("[1] Feature Configuration", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))
        
        if self.dense_features:
            logger.info(f"Dense Features ({len(self.dense_features)}):")
            for i, feat in enumerate(self.dense_features, 1):
                embed_dim = feat.embedding_dim if hasattr(feat, 'embedding_dim') else 1
                logger.info(f"  {i}. {feat.name:20s}")
        
        if self.sparse_features:
            logger.info(f"Sparse Features ({len(self.sparse_features)}):")

            max_name_len = max(len(feat.name) for feat in self.sparse_features)
            max_embed_name_len = max(len(feat.embedding_name) for feat in self.sparse_features)
            name_width = max(max_name_len, 10) + 2
            embed_name_width = max(max_embed_name_len, 15) + 2
            
            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Vocab Size':>12} {'Embed Name':>{embed_name_width}} {'Embed Dim':>10}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*embed_name_width} {'-'*10}")
            for i, feat in enumerate(self.sparse_features, 1):
                vocab_size = feat.vocab_size if hasattr(feat, 'vocab_size') else 'N/A'
                embed_dim = feat.embedding_dim if hasattr(feat, 'embedding_dim') else 'N/A'
                logger.info(f"  {i:<4} {feat.name:<{name_width}} {str(vocab_size):>12} {feat.embedding_name:>{embed_name_width}} {str(embed_dim):>10}")
        
        if self.sequence_features:
            logger.info(f"Sequence Features ({len(self.sequence_features)}):")

            max_name_len = max(len(feat.name) for feat in self.sequence_features)
            max_embed_name_len = max(len(feat.embedding_name) for feat in self.sequence_features)
            name_width = max(max_name_len, 10) + 2
            embed_name_width = max(max_embed_name_len, 15) + 2
            
            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Vocab Size':>12} {'Embed Name':>{embed_name_width}} {'Embed Dim':>10} {'Max Len':>10}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*embed_name_width} {'-'*10} {'-'*10}")
            for i, feat in enumerate(self.sequence_features, 1):
                vocab_size = feat.vocab_size if hasattr(feat, 'vocab_size') else 'N/A'
                embed_dim = feat.embedding_dim if hasattr(feat, 'embedding_dim') else 'N/A'
                max_len = feat.max_len if hasattr(feat, 'max_len') else 'N/A'
                logger.info(f"  {i:<4} {feat.name:<{name_width}} {str(vocab_size):>12} {feat.embedding_name:>{embed_name_width}} {str(embed_dim):>10} {str(max_len):>10}")
        

        logger.info("")
        logger.info(colorize("[2] Model Parameters", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        logger.info(f"Total Parameters:        {total_params:,}")
        logger.info(f"Trainable Parameters:    {trainable_params:,}")
        logger.info(f"Non-trainable Parameters: {non_trainable_params:,}")
        
        logger.info("Layer-wise Parameters:")
        for name, module in self.named_children():
            layer_params = sum(p.numel() for p in module.parameters())
            if layer_params > 0:
                logger.info(f"  {name:30s}: {layer_params:,}")
        
        logger.info("")
        logger.info(colorize("[3] Training Configuration", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))
        
        logger.info(f"Task Type:               {self.task}")
        logger.info(f"Number of Tasks:         {self.nums_task}")
        logger.info(f"Metrics:                 {self.metrics}")
        logger.info(f"Target Columns:          {self.target}")
        logger.info(f"Device:                  {self.device}")
        
        if hasattr(self, '_optimizer_name'):
            logger.info(f"Optimizer:               {self._optimizer_name}")
            if self._optimizer_params:
                for key, value in self._optimizer_params.items():
                    logger.info(f"  {key:25s}: {value}")
        
        if hasattr(self, '_scheduler_name') and self._scheduler_name:
            logger.info(f"Scheduler:               {self._scheduler_name}")
            if self._scheduler_params:
                for key, value in self._scheduler_params.items():
                    logger.info(f"  {key:25s}: {value}")
        
        if hasattr(self, '_loss_config'):
            logger.info(f"Loss Function:           {self._loss_config}")
        
        logger.info("Regularization:")
        logger.info(f"  Embedding L1:          {self._embedding_l1_reg}")
        logger.info(f"  Embedding L2:          {self._embedding_l2_reg}")
        logger.info(f"  Dense L1:              {self._dense_l1_reg}")
        logger.info(f"  Dense L2:              {self._dense_l2_reg}")
        
        logger.info("Other Settings:")
        logger.info(f"  Early Stop Patience:   {self.early_stop_patience}")
        logger.info(f"  Max Gradient Norm:     {self._max_gradient_norm}")
        logger.info(f"  Model ID:              {self.model_id}")
        logger.info(f"  Checkpoint Path:       {self.checkpoint}")
        
        logger.info("")
        logger.info("")


class BaseMatchModel(BaseModel):
    """
    Base class for match (retrieval/recall) models
    Supports pointwise, pairwise, and listwise training modes
    """
    
    @property
    def task_type(self) -> str:
        return 'match'
    
    @property
    def support_training_modes(self) -> list[str]:
        """
        Returns list of supported training modes for this model.
        Override in subclasses to restrict training modes.
        
        Returns:
            List of supported modes: ['pointwise', 'pairwise', 'listwise']
        """
        return ['pointwise', 'pairwise', 'listwise']
    
    def __init__(self,
                 user_dense_features: list[DenseFeature] | None = None,
                 user_sparse_features: list[SparseFeature] | None = None,
                 user_sequence_features: list[SequenceFeature] | None = None,
                 item_dense_features: list[DenseFeature] | None = None,
                 item_sparse_features: list[SparseFeature] | None = None,
                 item_sequence_features: list[SequenceFeature] | None = None,
                 training_mode: Literal['pointwise', 'pairwise', 'listwise'] = 'pointwise',
                 num_negative_samples: int = 4,
                 temperature: float = 1.0,
                 similarity_metric: Literal['dot', 'cosine', 'euclidean'] = 'dot',
                 device: str = 'cpu',
                 embedding_l1_reg: float = 0.0,
                 dense_l1_reg: float = 0.0,
                 embedding_l2_reg: float = 0.0,
                 dense_l2_reg: float = 0.0,
                 early_stop_patience: int = 20,
                 model_id: str = 'baseline'):
        
        all_dense_features = []
        all_sparse_features = []
        all_sequence_features = []
        
        if user_dense_features:
            all_dense_features.extend(user_dense_features)
        if item_dense_features:
            all_dense_features.extend(item_dense_features)
        if user_sparse_features:
            all_sparse_features.extend(user_sparse_features)
        if item_sparse_features:
            all_sparse_features.extend(item_sparse_features)
        if user_sequence_features:
            all_sequence_features.extend(user_sequence_features)
        if item_sequence_features:
            all_sequence_features.extend(item_sequence_features)
        
        super(BaseMatchModel, self).__init__(
            dense_features=all_dense_features,
            sparse_features=all_sparse_features,
            sequence_features=all_sequence_features,
            target=['label'],  
            task='binary',  
            device=device,
            embedding_l1_reg=embedding_l1_reg,
            dense_l1_reg=dense_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            dense_l2_reg=dense_l2_reg,
            early_stop_patience=early_stop_patience,
            model_id=model_id
        )
        
        self.user_dense_features = list(user_dense_features) if user_dense_features else []
        self.user_sparse_features = list(user_sparse_features) if user_sparse_features else []
        self.user_sequence_features = list(user_sequence_features) if user_sequence_features else []
        
        self.item_dense_features = list(item_dense_features) if item_dense_features else []
        self.item_sparse_features = list(item_sparse_features) if item_sparse_features else []
        self.item_sequence_features = list(item_sequence_features) if item_sequence_features else []
        
        self.training_mode = training_mode
        self.num_negative_samples = num_negative_samples
        self.temperature = temperature
        self.similarity_metric = similarity_metric
    
    def get_user_features(self, X_input: dict) -> dict:
        user_input = {}
        all_user_features = self.user_dense_features + self.user_sparse_features + self.user_sequence_features
        for feature in all_user_features:
            if feature.name in X_input:
                user_input[feature.name] = X_input[feature.name]
        return user_input
    
    def get_item_features(self, X_input: dict) -> dict:
        item_input = {}
        all_item_features = self.item_dense_features + self.item_sparse_features + self.item_sequence_features
        for feature in all_item_features:
            if feature.name in X_input:
                item_input[feature.name] = X_input[feature.name]
        return item_input
    
    def compile(self, 
                optimizer = "adam",
                optimizer_params: dict | None = None,
                scheduler: str | torch.optim.lr_scheduler._LRScheduler | type[torch.optim.lr_scheduler._LRScheduler] | None = None,
                scheduler_params: dict | None = None,
                loss: str | nn.Module | list[str | nn.Module] | None= None):
        """
        Compile match model with optimizer, scheduler, and loss function.
        Validates that training_mode is supported by the model.
        """
        from reclib.loss import validate_training_mode
        
        # Validate training mode is supported
        validate_training_mode(
            training_mode=self.training_mode,
            support_training_modes=self.support_training_modes,
            model_name=self.model_name
        )
        
        # Call parent compile with match-specific logic
        if optimizer_params is None:
            optimizer_params = {}
        
        self._optimizer_name = optimizer if isinstance(optimizer, str) else optimizer.__class__.__name__
        self._optimizer_params = optimizer_params
        if isinstance(scheduler, str):
            self._scheduler_name = scheduler
        elif scheduler is not None:
            self._scheduler_name = scheduler.__class__.__name__ if hasattr(scheduler, '__class__') else str(scheduler)
        else:
            self._scheduler_name = None
        self._scheduler_params = scheduler_params or {}
        self._loss_config = loss
        
        # set optimizer
        self.optimizer_fn = get_optimizer_fn(
            optimizer=optimizer, 
            params=self.parameters(), 
            **optimizer_params
        )
        
        # Set loss function based on training mode
        loss_value = loss[0] if isinstance(loss, list) else loss
        self.loss_fn = [get_loss_fn(
            task_type='match',
            training_mode=self.training_mode,
            loss=loss_value
        )]
        
        # set scheduler
        self.scheduler_fn = get_scheduler_fn(scheduler, self.optimizer_fn, **(scheduler_params or {})) if scheduler else None

    def compute_similarity(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        if self.similarity_metric == 'dot':
            if user_emb.dim() == 3 and item_emb.dim() == 3:
                # [batch_size, num_items, emb_dim] @ [batch_size, num_items, emb_dim]
                similarity = torch.sum(user_emb * item_emb, dim=-1)  # [batch_size, num_items]
            elif user_emb.dim() == 2 and item_emb.dim() == 3:
                # [batch_size, emb_dim] @ [batch_size, num_items, emb_dim]
                user_emb_expanded = user_emb.unsqueeze(1)  # [batch_size, 1, emb_dim]
                similarity = torch.sum(user_emb_expanded * item_emb, dim=-1)  # [batch_size, num_items]
            else:
                similarity = torch.sum(user_emb * item_emb, dim=-1)  # [batch_size]
        
        elif self.similarity_metric == 'cosine':
            if user_emb.dim() == 3 and item_emb.dim() == 3:
                similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
            elif user_emb.dim() == 2 and item_emb.dim() == 3:
                user_emb_expanded = user_emb.unsqueeze(1)
                similarity = F.cosine_similarity(user_emb_expanded, item_emb, dim=-1)
            else:
                similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
        
        elif self.similarity_metric == 'euclidean':
            if user_emb.dim() == 3 and item_emb.dim() == 3:
                distance = torch.sum((user_emb - item_emb) ** 2, dim=-1)
            elif user_emb.dim() == 2 and item_emb.dim() == 3:
                user_emb_expanded = user_emb.unsqueeze(1)
                distance = torch.sum((user_emb_expanded - item_emb) ** 2, dim=-1)
            else:
                distance = torch.sum((user_emb - item_emb) ** 2, dim=-1)
            similarity = -distance 
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        similarity = similarity / self.temperature
        
        return similarity
    
    def user_tower(self, user_input: dict) -> torch.Tensor:
        raise NotImplementedError
    
    def item_tower(self, item_input: dict) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self, X_input: dict) -> torch.Tensor:
        user_input = self.get_user_features(X_input)
        item_input = self.get_item_features(X_input)
        
        user_emb = self.user_tower(user_input)   # [B, D]
        item_emb = self.item_tower(item_input)   # [B, D]
        
        if self.training and self.training_mode in ['pairwise', 'listwise']:
            return user_emb, item_emb

        similarity = self.compute_similarity(user_emb, item_emb)  # [B]
        
        if self.training_mode == 'pointwise':
            return torch.sigmoid(similarity)
        else:
            return similarity
    
    def compute_loss(self, y_pred, y_true):
        if self.training_mode == 'pointwise':
            if y_true is None:
                return torch.tensor(0.0, device=self.device)
            return self.loss_fn[0](y_pred, y_true)
        
        # pairwise / listwise using inbatch neg
        elif self.training_mode in ['pairwise', 'listwise']:
            if not isinstance(y_pred, (tuple, list)) or len(y_pred) != 2:
                raise ValueError(
                    "For pairwise/listwise training, forward should return (user_emb, item_emb). "
                    "Please check BaseMatchModel.forward implementation."
                )
            
            user_emb, item_emb = y_pred  # [B, D], [B, D]
            
            logits = torch.matmul(user_emb, item_emb.t())  # [B, B]
            logits = logits / self.temperature            
            
            batch_size = logits.size(0)
            targets = torch.arange(batch_size, device=logits.device)  # [0, 1, 2, ..., B-1]
            
            # Cross-Entropy = InfoNCE
            loss = F.cross_entropy(logits, targets)
            return loss
        
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
    
    def _set_metrics(self, metrics: list[str] | None = None):
        if metrics is not None and len(metrics) > 0:
            self.metrics = [m.lower() for m in metrics]
        else:
            self.metrics = ['auc', 'logloss']
        
        self.best_metrics_mode = 'max'  
        
        if not hasattr(self, 'early_stopper') or self.early_stopper is None:
            self.early_stopper = EarlyStopper(patience=self.early_stop_patience, mode=self.best_metrics_mode)
    
    def encode_user(self, data: dict | pd.DataFrame | DataLoader, batch_size: int = 512) -> np.ndarray:
        self.eval()
        
        if not isinstance(data, DataLoader):
            user_data = {}
            all_user_features = self.user_dense_features + self.user_sparse_features + self.user_sequence_features
            for feature in all_user_features:
                if isinstance(data, dict):
                    if feature.name in data:
                        user_data[feature.name] = data[feature.name]
                elif isinstance(data, pd.DataFrame):
                    if feature.name in data.columns:
                        user_data[feature.name] = data[feature.name].values
            
            data_loader = self._prepare_data_loader(user_data, batch_size=batch_size, shuffle=False)
        else:
            data_loader = data
        
        embeddings_list = []
        
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Encoding users", disable=self._verbose == 0):
                batch_dict = self._batch_to_dict(batch_data)
                user_input = self.get_user_features(batch_dict)
                user_emb = self.user_tower(user_input)
                embeddings_list.append(user_emb.cpu().numpy())
        
        embeddings = np.concatenate(embeddings_list, axis=0)
        return embeddings
    
    def encode_item(self, data: dict | pd.DataFrame | DataLoader, batch_size: int = 512) -> np.ndarray:
        self.eval()
        
        if not isinstance(data, DataLoader):
            item_data = {}
            all_item_features = self.item_dense_features + self.item_sparse_features + self.item_sequence_features
            for feature in all_item_features:
                if isinstance(data, dict):
                    if feature.name in data:
                        item_data[feature.name] = data[feature.name]
                elif isinstance(data, pd.DataFrame):
                    if feature.name in data.columns:
                        item_data[feature.name] = data[feature.name].values

            data_loader = self._prepare_data_loader(item_data, batch_size=batch_size, shuffle=False)
        else:
            data_loader = data
        
        embeddings_list = []
        
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_loader, desc="Encoding items", disable=self._verbose == 0):
                batch_dict = self._batch_to_dict(batch_data)
                item_input = self.get_item_features(batch_dict)
                item_emb = self.item_tower(item_input)
                embeddings_list.append(item_emb.cpu().numpy())
        
        embeddings = np.concatenate(embeddings_list, axis=0)
        return embeddings
