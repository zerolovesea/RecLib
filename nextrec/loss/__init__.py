from nextrec.loss.match_losses import (
    BPRLoss,
    HingeLoss,
    TripletLoss,
    SampledSoftmaxLoss,
    CosineContrastiveLoss,
    InfoNCELoss,
    ListNetLoss,
    ListMLELoss,
    ApproxNDCGLoss,
)

from nextrec.loss.loss_utils import (
    get_loss_fn,
    validate_training_mode,
    VALID_TASK_TYPES,
)

__all__ = [
    # Match losses
    "BPRLoss",
    "HingeLoss",
    "TripletLoss",
    "SampledSoftmaxLoss",
    "CosineContrastiveLoss",
    "InfoNCELoss",
    # Listwise losses
    "ListNetLoss",
    "ListMLELoss",
    "ApproxNDCGLoss",
    # Utilities
    "get_loss_fn",
    "validate_training_mode",
    "VALID_TASK_TYPES",
]
