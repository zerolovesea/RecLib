"""
Metrics computation and configuration for model evaluation.
"""
import logging
from typing import Literal
import numpy as np
from sklearn.metrics import (
    roc_auc_score, log_loss, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, r2_score,
)


TASK_DEFAULT_METRICS = {
    'binary': ['auc', 'gauc', 'ks', 'logloss', 'accuracy', 'precision', 'recall', 'f1'],
    'regression': ['mse', 'mae', 'rmse', 'r2', 'mape'],
    'multilabel': ['auc', 'hamming_loss', 'subset_accuracy', 'micro_f1', 'macro_f1'],
}



def compute_ks(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic."""
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)
    
    if n_pos > 0 and n_neg > 0:
        cum_pos_rate = np.cumsum(y_true_sorted == 1) / n_pos
        cum_neg_rate = np.cumsum(y_true_sorted == 0) / n_neg
        ks_value = np.max(np.abs(cum_pos_rate - cum_neg_rate))
        return float(ks_value)
    return 0.0


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error."""
    mask = y_true != 0
    if np.any(mask):
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return 0.0


def compute_msle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Log Error."""
    y_pred_pos = np.maximum(y_pred, 0)
    return float(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_pos)))


def compute_gauc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    user_ids: np.ndarray | None = None
) -> float:
    if user_ids is None:
        # If no user_ids provided, fall back to regular AUC
        try:
            return float(roc_auc_score(y_true, y_pred))
        except:
            return 0.0
    
    # Group by user_id and calculate AUC for each user
    user_aucs = []
    user_weights = []
    
    unique_users = np.unique(user_ids)
    
    for user_id in unique_users:
        mask = user_ids == user_id
        user_y_true = y_true[mask]
        user_y_pred = y_pred[mask]
        
        # Skip users with only one class (cannot compute AUC)
        if len(np.unique(user_y_true)) < 2:
            continue
        
        try:
            user_auc = roc_auc_score(user_y_true, user_y_pred)
            user_aucs.append(user_auc)
            user_weights.append(len(user_y_true))
        except:
            continue
    
    if len(user_aucs) == 0:
        return 0.0
    
    # Weighted average
    user_aucs = np.array(user_aucs)
    user_weights = np.array(user_weights)
    gauc = float(np.sum(user_aucs * user_weights) / np.sum(user_weights))
    
    return gauc


def configure_metrics(
    task: str | list[str],                            # 'binary' or ['binary', 'regression']
    metrics: list[str] | dict[str, list[str]] | None, # ['auc', 'logloss'] or {'task1': ['auc'], 'task2': ['mse']}
    target_names: list[str]                           # ['target1', 'target2'] 
) -> tuple[list[str], dict[str, list[str]] | None, str]:
    
    primary_task = task[0] if isinstance(task, list) else task
    nums_task = len(task) if isinstance(task, list) else 1
    
    metrics_list: list[str] = []
    task_specific_metrics: dict[str, list[str]] | None = None

    if isinstance(metrics, dict):
        metrics_list = []
        task_specific_metrics = {}
        for task_name, task_metrics in metrics.items():
            if task_name not in target_names:
                logging.warning(
                    "Task '%s' not found in targets %s, skipping its metrics",
                    task_name,
                    target_names,
                )
                continue

            lowered = [m.lower() for m in task_metrics]
            task_specific_metrics[task_name] = lowered
            for metric in lowered:
                if metric not in metrics_list:
                    metrics_list.append(metric)

    elif metrics:
        metrics_list = [m.lower() for m in metrics]

    else:
        # No user provided metrics, derive per task type
        if nums_task > 1 and isinstance(task, list):
            deduped: list[str] = []
            for task_type in task:
                # Inline get_default_metrics_for_task logic
                if task_type not in TASK_DEFAULT_METRICS:
                    raise ValueError(f"Unsupported task type: {task_type}")
                for metric in TASK_DEFAULT_METRICS[task_type]:
                    if metric not in deduped:
                        deduped.append(metric)
            metrics_list = deduped
        else:
            # Inline get_default_metrics_for_task logic
            if primary_task not in TASK_DEFAULT_METRICS:
                raise ValueError(f"Unsupported task type: {primary_task}")
            metrics_list = TASK_DEFAULT_METRICS[primary_task]
    
    if not metrics_list:
        # Inline get_default_metrics_for_task logic
        if primary_task not in TASK_DEFAULT_METRICS:
            raise ValueError(f"Unsupported task type: {primary_task}")
        metrics_list = TASK_DEFAULT_METRICS[primary_task]
    
    best_metrics_mode = get_best_metric_mode(metrics_list[0], primary_task)
    
    return metrics_list, task_specific_metrics, best_metrics_mode


def get_best_metric_mode(first_metric: str, primary_task: str) -> str:
    """Determine if metric should be maximized or minimized."""
    first_metric_lower = first_metric.lower()
    if first_metric_lower in {'auc', 'gauc', 'ks', 'accuracy', 'acc', 'precision', 'recall', 'f1', 'r2', 'micro_f1', 'macro_f1'}:
        return 'max'
    if first_metric_lower in {'logloss', 'mse', 'mae', 'rmse', 'mape', 'msle'}:
        return 'min'
    if primary_task == 'regression':
        return 'min'
    return 'max'


def compute_single_metric(
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    user_ids: np.ndarray | None = None
) -> float:
    y_p_binary = (y_pred > 0.5).astype(int)
    
    try:
        if metric == 'auc':
            value = float(roc_auc_score(y_true, y_pred, average='macro' if task_type == 'multilabel' else None))
        elif metric == 'gauc':
            value = float(compute_gauc(y_true, y_pred, user_ids))
        elif metric == 'ks':
            value = float(compute_ks(y_true, y_pred))
        elif metric == 'logloss':
            value = float(log_loss(y_true, y_pred))
        elif metric in ('accuracy', 'acc'):
            value = float(accuracy_score(y_true, y_p_binary))
        elif metric == 'precision':
            value = float(precision_score(y_true, y_p_binary, average='samples' if task_type == 'multilabel' else 'binary', zero_division=0))
        elif metric == 'recall':
            value = float(recall_score(y_true, y_p_binary, average='samples' if task_type == 'multilabel' else 'binary', zero_division=0))
        elif metric == 'f1':
            value = float(f1_score(y_true, y_p_binary, average='samples' if task_type == 'multilabel' else 'binary', zero_division=0))
        elif metric == 'micro_f1':
            value = float(f1_score(y_true, y_p_binary, average='micro', zero_division=0))
        elif metric == 'macro_f1':
            value = float(f1_score(y_true, y_p_binary, average='macro', zero_division=0))
        elif metric == 'mse':
            value = float(mean_squared_error(y_true, y_pred))
        elif metric == 'mae':
            value = float(mean_absolute_error(y_true, y_pred))
        elif metric == 'rmse':
            value = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == 'r2':
            value = float(r2_score(y_true, y_pred))
        elif metric == 'mape':
            value = float(compute_mape(y_true, y_pred))
        elif metric == 'msle':
            value = float(compute_msle(y_true, y_pred))
        else:
            logging.warning(f"Metric '{metric}' is not supported, returning 0.0")
            value = 0.0
    except Exception as exception:
        logging.warning(f"Failed to compute metric {metric}: {exception}")
        value = 0.0
    
    return value


def evaluate_metrics(
    y_true: np.ndarray | None,
    y_pred: np.ndarray | None,
    metrics: list[str],                                       # ['auc', 'logloss']
    task: str | list[str],                                    # 'binary' or ['binary', 'regression']
    target_names: list[str],                                  # ['target1', 'target2']
    task_specific_metrics: dict[str, list[str]] | None = None, # {'target1': ['auc', 'logloss'], 'target2': ['mse']}
    user_ids: np.ndarray | None = None                        # User IDs for GAUC computation
) -> dict: # {'auc': 0.75, 'logloss': 0.45, 'mse_target2': 3.2}
    
    result = {}
    
    if y_true is None or y_pred is None:
        return result
    
    # Main evaluation logic
    primary_task = task[0] if isinstance(task, list) else task
    nums_task = len(task) if isinstance(task, list) else 1
    
    # Single task evaluation
    if nums_task == 1:
        for metric in metrics:
            metric_lower = metric.lower()
            value = compute_single_metric(metric_lower, y_true, y_pred, primary_task, user_ids)
            result[metric_lower] = value
    
    # Multi-task evaluation
    else:
        for metric in metrics:
            metric_lower = metric.lower()
            for task_idx in range(nums_task):
                # Check if metric should be computed for given task
                should_compute = True
                if task_specific_metrics is not None and task_idx < len(target_names):
                    task_name = target_names[task_idx]
                    should_compute = metric_lower in task_specific_metrics.get(task_name, [])
                else:
                    # Get task type for specific index
                    if isinstance(task, list) and task_idx < len(task):
                        task_type = task[task_idx]
                    elif isinstance(task, str):
                        task_type = task
                    else:
                        task_type = 'binary'
                    
                    if task_type in ['binary', 'multilabel']:
                        should_compute = metric_lower in {'auc', 'ks', 'logloss', 'accuracy', 'acc', 'precision', 'recall', 'f1', 'micro_f1', 'macro_f1'}
                    elif task_type == 'regression':
                        should_compute = metric_lower in {'mse', 'mae', 'rmse', 'r2', 'mape', 'msle'}
                
                if not should_compute:
                    continue
                
                target_name = target_names[task_idx]
                
                # Get task type for specific index
                if isinstance(task, list) and task_idx < len(task):
                    task_type = task[task_idx]
                elif isinstance(task, str):
                    task_type = task
                else:
                    task_type = 'binary'
                
                y_true_task = y_true[:, task_idx]
                y_pred_task = y_pred[:, task_idx]
                
                # Compute metric
                value = compute_single_metric(metric_lower, y_true_task, y_pred_task, task_type, user_ids)
                result[f'{metric_lower}_{target_name}'] = value
    
    return result
