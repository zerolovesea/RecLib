"""
Common utilities for RecLib

Date: create on 13/11/2025
Author:
    Yang Zhou, zyaztec@gmail.com
"""


def get_task_type(model) -> str:
    """
    Get task type from model.
    
    Args:
        model: Model instance
        
    Returns:
        Task type string (e.g., 'binary', 'regression', 'match', 'ranking', 'multitask')
        
    Examples:
        >>> task_type = get_task_type(model)
        >>> print(task_type)  # 'binary'
    """
    return model.task_type
