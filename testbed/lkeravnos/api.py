import os, sys
import numpy as np

# set up path to load .pyd
_module_path = os.path.dirname(__file__)
sys.path.insert(0, _module_path)

try:
  import keravnos  # this resolves to keravnos.pyd in the same directory
except ImportError as e:
  raise ImportError("Failed to import 'keravnos.pyd'. Is it compiled and placed correctly?") from e

# ----------------------- high-level front-end ----------------------- #

class Transformer:
  """Python interface for managing CUDA-based Transformer instances"""

  @staticmethod
  def construct(
    name: str,
    batch_size: int = 2,
    sequence_length: int = 2048,
    vocab_size: int = 48000,
    num_dims: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    ff_multiplier: int = 12,
    verbose: bool = False
  ):
    """Construct and register a new transformer"""
    keravnos.construct(
      name,
      batch_size,
      sequence_length,
      vocab_size,
      num_dims,
      num_heads,
      num_layers,
      ff_multiplier,
      verbose
    )

  @staticmethod
  def destruct(name: str, verbose: bool = False):
    """Destruct and unregister a transformer"""
    keravnos.destruct(name, verbose)

  @staticmethod
  def feed_token_ids(name: str, token_ids, verbose: bool = False):
    """Feed token ids into the transformer"""
    keravnos.feed_token_ids(name, token_ids, verbose)

  @staticmethod
  def edit_tensor(name: str, tensor_id: str, input_tensor, verbose: bool = False):
    """Edit tensor"""
    keravnos.edit_tensor(name, tensor_id, input_tensor, verbose)

  @staticmethod
  def causal_self_attention(name: str, use_bias: bool = True, dropout: float = 0.5, seed: int = 0, verbose: bool = False):
    """Causal self attention"""
    keravnos.causal_self_attention(name, use_bias, dropout, verbose)

  @staticmethod
  def get_info(name: str, verbose: bool = False) -> dict:
    """Retrieve transformer information"""
    return keravnos.get_info(name, verbose)

  @staticmethod
  def get_token_ids(name: str, verbose: bool = False):
    """Retrieve transformer information"""
    return keravnos.get_token_ids(name, verbose)

  @staticmethod
  def get_token_embed(name: str, verbose: bool = False):
    """Retrieve transformer token embedding weight"""
    return keravnos.get_token_embed(name, verbose)
  
  @staticmethod
  def get_pos_embed(name: str, verbose: bool = False):
    """Retrieve transformer positional embedding weight"""
    return keravnos.get_pos_embed(name, verbose)

  @staticmethod
  def get_tensor(name: str, tensor_id: str, verbose: bool = False):
    """Retrieve tensor"""
    return keravnos.get_tensor(name, tensor_id, verbose)
  