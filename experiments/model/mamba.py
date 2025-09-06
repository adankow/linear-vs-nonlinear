
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mamba_ssm.modules.mamba_simple import Mamba 
from config.mamba import MambaConfig

class ResidualMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mamba_layer = Mamba(
                                        config.d_model,
                                        d_state = config.d_state,
                                        d_conv = config.d_conv,
                                        expand = config.expand,
                                        dt_rank = config.dt_rank,
                                        dt_min = config.dt_min,
                                        dt_max = config.dt_max,
                                        dt_init = config.dt_init,
                                        dt_scale = config.dt_scale,
                                        dt_init_floor = config.dt_init_floor,
                                        conv_bias = config.conv_bias,
                                        bias = config.bias,
                                        use_fast_path = config.use_fast_path,  # Fused kernel options
                                        layer_idx = config.layer_idx,
                                        device = config.device,
                                        dtype = config.dtype
                                    )
        self.norm = torch.nn.RMSNorm(config.d_model, config.rms_norm_eps, config.mup)
    def forward(self, x):
        output = self.mamba_layer(self.norm(x)) + x
        return output
 
class MambaLM(nn.Module):
    def __init__(self, config: MambaConfig, vocab_size, embeding_dim):
        super().__init__()
        self.layers = nn.Sequential(*[ResidualMamba(config) for _ in range(config.n_layers)])
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeding_dim)

        self.lm_head = nn.Linear(
            in_features=embeding_dim,
            out_features=vocab_size,
            bias=False,
        )

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.layers(x)
        x = self.lm_head(x)
        return x