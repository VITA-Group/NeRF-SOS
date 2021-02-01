import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from layers.nerf_mlp import NerfMLP
from layers.position_embedding import Embedder

from utils.check_error import *

# Point query with embedding
class NerfVoxelNet(nn.Module):
    
    def __init__(self, args, input_dim, output_dim, net_depth, net_width, skips=[]):
        super(NerfVoxelNet, self).__init__()

        self.chunk = args.netchunk

        # Provide empty periodic_fns to specify identity embedder
        periodic_fns = []
        if args.i_embed >= 0:
            periodic_fns = [torch.sin, torch.cos]

        self.embedder = Embedder(input_dim, args.multires, args.multires-1,
                                 periodic_fns, log_sampling=True, include_input=True)
        input_ch = self.embedder.out_dim

        input_ch_views = 0
        self.embeddirs = None
        if args.use_viewdirs:
            self.embeddirs = Embedder(input_dim, args.multires_views, args.multires_views-1,
                                      periodic_fns, log_sampling=True, include_input=True)
            input_ch_views = self.embeddirs.out_dim

        output_ch = output_dim
        self.mlp = NerfMLP(net_depth, net_width, skips=skips,
                           input_ch=input_ch, output_ch=output_ch,
                           input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    
    def batchify(self, inputs):
        """Single forward feed that applies to smaller batches.
        """
        query_batches = []
        for i in range(0, inputs.shape[0], self.chunk):
            end = min(i+self.chunk, inputs.shape[0])
            h = self.mlp(inputs[i:end]) # [N_chunk, C]
            query_batches.append(h)
        outputs = torch.cat(query_batches, 0) # [N_pts, C]
        return outputs

    def forward(self, inputs, viewdirs=None, **kwargs):
        """Prepares inputs and applies network.
        """
        # Flatten
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) # [N_pts, C]
        if viewdirs is not None:
            input_dirs = viewdirs[:,None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        # Batchify
        output_chunks = []
        for i in range(0, inputs_flat.shape[0], self.chunk):
            end = min(i+self.chunk, inputs_flat.shape[0])

            embedded = self.embedder(inputs_flat[i:end])
            # append view direction embedding
            if self.embeddirs is not None:
                embedded_dirs = self.embeddirs(input_dirs_flat[i:end])
                embedded = torch.cat([embedded, embedded_dirs], -1)

            h = self.mlp(embedded) # [N_chunk, C]
            output_chunks.append(h)
        outputs_flat = torch.cat(output_chunks, 0) # [N_pts, C]

        # Unflatten
        sh = list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        return torch.reshape(outputs_flat, sh)
