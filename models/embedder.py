import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

# Positional encoding (section 5.1)
class Embedder(nn.Module):

    def __init__(self, input_dim, N_freqs, max_freq, periodic_fns,
                 log_sampling=True, include_input=True):
        super(Embedder, self).__init__()

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

# def get_embedder(input_dims, multires, i=0):
#     if i == -1:
#         return nn.Identity(), input_dims
    
#     embed_kwargs = {
#                 'include_input' : True,
#                 'input_dims' : input_dims,
#                 'max_freq_log2' : multires-1,
#                 'num_freqs' : multires,
#                 'log_sampling' : True,
#                 'periodic_fns' : [torch.sin, torch.cos],
#     }
    
#     embedder_obj = Embedder(**embed_kwargs)
#     embed = lambda x, eo=embedder_obj : eo.embed(x)

#     return embed, embedder_obj.out_dim