from typing import Optional
from reclib.utils.tools import get_auto_embedding_dim

class BaseFeature(object):
    def __repr__(self):
        params = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") 
        }
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"

class SequenceFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        vocab_size: int,
        max_len: int = 20,
        embedding_name: str = '',
        embedding_dim: Optional[int] = 4,
        combiner: str = "mean",
        padding_idx: Optional[int] = None,
        init_type: str='normal',
        init_params: dict|None = None,
        l1_reg: float = 0.0,
        l2_reg: float = 1e-5,
        trainable: bool = True,
    ):

        self.name = name
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_name = embedding_name or name
        self.embedding_dim = embedding_dim or get_auto_embedding_dim(vocab_size)

        self.init_type = init_type
        self.init_params = init_params or {}
        self.combiner = combiner
        self.padding_idx = padding_idx
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.trainable = trainable
    
class SparseFeature(BaseFeature):
    def __init__(self, 
                 name: str, 
                 vocab_size: int, 
                 embedding_name: str = '', 
                 embedding_dim: int = 4, 
                 padding_idx: int | None = None,
                 init_type: str='normal',
                 init_params: dict|None = None,
                 l1_reg: float = 0.0,                 
                 l2_reg: float = 1e-5,
                 trainable: bool = True):
        
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_name = embedding_name or name
        self.embedding_dim = embedding_dim or get_auto_embedding_dim(vocab_size)

        self.init_type = init_type
        self.init_params = init_params or {}
        self.padding_idx = padding_idx
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.trainable = trainable

class DenseFeature(BaseFeature):
    def __init__(self, 
                 name: str, 
                 embedding_dim: int = 1):

        self.name = name
        self.embedding_dim = embedding_dim



