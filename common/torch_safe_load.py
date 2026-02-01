
""" torch_safe_load
    Centralized PyTorch >= 2.6 safe-unpickling policy
    Used by ALL scripts that load MMEngine / MMAction checkpoints.
"""

from torch.serialization import add_safe_globals
from mmengine.logging.history_buffer import HistoryBuffer
import numpy as np
from numpy.dtypes import Float64DType, Int64DType

def enable_checkpoint_loading():
    """  Allowlist all known-safe globals required to load
    MMEngine / MMAction checkpoints created by this project.
    """
add_safe_globals([HistoryBuffer,
                  np._core.multiarray._reconstruct,
                  Float64DType, Int64DType,
                  ])
