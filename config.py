from transformers import PretrainedConfig
from typing import List

class LLMConfig(PretrainedConfig):
     model_type="whetstone"
     def __init__(self,
                  dim:int=512,
                  n_layers: int= 8,
                  n_heads: int=8,
                  n_kv_heads: int=8,
                  vocab_size: int=6400,
                  hidden_dim: int=None,
                  multiple_of: int=64,
                  norm_eps: float=1e-5,
                  max_seq_len: int=1024,
                  rope_theta: int=1e6,
                  dropout: float=0.0,
                  ):
          super().__init__()
          self.dim = dim
          self.n_layers=n_layers
          self.n_heads=n_heads
          self.n_kv_heads=n_kv_heads
          self.vocab_size=vocab_size
          self.hidden_dim=hidden_dim
          self.multiple_of=multiple_of
          self.norm_eps=norm_eps
          self.max_seq_len=max_seq_len
          self.rope_theta=rope_theta
          self.dropout=dropout
          self._attn_implementation_internal = None
          # 添加配置验证
          self._validate_config()
     
     def _validate_config(self):
         """验证配置参数的有效性"""
         assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
         assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
         assert self.max_seq_len > 0, "max_seq_len must be positive"