import math
from dataclasses import dataclass
from typing import Union 
import torch 

@dataclass
class MambaConfig:

    d_model: int 
    n_layers : int
    d_state=16
    d_conv=4
    expand=2
    dt_rank="auto"
    dt_min=0.001
    dt_max=0.1
    dt_init="random"
    dt_scale=1.0
    dt_init_floor=1e-4
    conv_bias=True
    bias=False
    use_fast_path=True  # Fused kernel options
    layer_idx=None
    device=None
    dtype=None


    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128  # width=d_model
