import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Positional encoding (NeRF section 5.1)
class PositionEncoder(nn.Module):

    def __init__(self, input_dim, N_freqs, max_freq, periodic_fns=[torch.sin, torch.cos],
                 log_sampling=True, include_input=True, trainable=False):
        super().__init__()

        self.periodic_fns = periodic_fns
        self.include_input = include_input or len(periodic_fns) == 0
        self.out_dim = len(periodic_fns) * input_dim * N_freqs

        # Identity map if no periodic_fns provided
        if self.include_input:
            self.out_dim += input_dim

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        if trainable:
            self.freq_bands = nn.Parameter(freq_bands, requires_grad=True)
        else:
            self.register_buffer('freq_bands', freq_bands, persistent=False)

    def forward(self, inputs):
        N_freqs = len(self.freq_bands)

        embeds = []
        for periodic_fn in self.periodic_fns:
            x_freq = inputs[..., None].expand(inputs.shape+(N_freqs,)) * self.freq_bands # [batch_shape, 3, N_freq]
            x_freq = periodic_fn(x_freq)
            embeds.append(x_freq.transpose(-1, -2))  # [batch_shape, N_freq, 3]
        embeds = torch.stack(embeds, -2) # [batch_shape, N_freq, N_fns, 3]
        embeds = embeds.reshape(inputs.shape[:-1]+(-1,)) # [batch_shape, N_freq x N_fns x 3]

        if self.include_input:
            embeds = torch.cat([inputs, embeds], -1)

        return embeds

# Integrated positional encoding (MipNeRF section 3.1)
# https://github.com/google/mipnerf/blob/main/internal/mip.py
class IntegratedPositionEncoder(nn.Module):

    def __init__(self, input_dim, N_freqs, max_freq, log_sampling=True, trainable=False):
        super().__init__()
        self.out_dim = 2 * input_dim * N_freqs

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        if trainable:
            self.freq_bands = nn.Parameter(freq_bands, requires_grad=True)
        else:
            self.register_buffer('freq_bands', freq_bands, persistent=False)

    def expected_sin(self, x, x_var):
        """Estimates mean and variance of sin(z), z ~ N(x, var)."""
        # When the variance is wide, shrink sin towards zero.
        y = torch.exp(-0.5 * x_var) * torch.sin(x)
        y_var = torch.maximum(
            torch.zeros_like(x),
            0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2
        )
        return y, y_var

    def forward(self, x, x_cov, diag=True):
        """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].
        Args:
            x, [N_pts, 3], variables to be encoded. Should be in [-pi, pi].
            x_cov, [N_pts, 3, 3], covariance matrices for `x`.
            diag: bool, if true, expects input covariances to be diagonal (full
            otherwise).
        Returns:
            encoded: [N_pts, 3], encoded variables.
        """
        if not diag:
            x_cov = torch.diagonal(x_cov, dim1=-2, dim2=-1)

        y = x[..., None, :] * self.freq_bands[:, None] # [N_pts, 1, 3] x [N_freqs, 1] -> [N_pts, N_freqs, 3]
        y = y.reshape(x.shape[:-1]+(-1,)) # [N_pts, N_freqs * 3]
        y_var = x_cov[..., None, :] * self.freq_bands[:, None]**2 # [N_pts, 1, 3] x [N_freqs, 1] -> [N_pts, N_freqs, 3]
        y_var = y_var.reshape(x.shape[:-1]+(-1,)) # [N_pts, N_freqs * 3]

        return self.expected_sin(
            torch.cat([y, y + 0.5 * math.pi], -1),
            torch.cat([y_var, y_var], -1)
        )[0]


# Positional encoding (NeRF section 5.1)
# https://github.com/yenchenlin/nerf-pytorch/blob/a15fd7cb363e93f933012fd1f1ad5395302f63a4/run_nerf_helpers.py#L15
class PositionEncoding_depr(nn.Module):

    def __init__(self, input_dim, N_freqs, max_freq, periodic_fns,
                 log_sampling=True, include_input=True):
        super(PositionEncoding, self).__init__()

        embed_fns = []
        d = input_dim
        out_dim = 0

        # Identity map if no periodic_fns provided
        if include_input or len(periodic_fns) == 0:
            embed_fns.append(lambda x : x)
            out_dim += d

        if len(periodic_fns) != 0:
            if log_sampling:
                freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            else:
                freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

            for freq in freq_bands:
                for p_fn in periodic_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                    out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
