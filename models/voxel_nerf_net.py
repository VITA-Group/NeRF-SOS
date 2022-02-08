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

import matplotlib.pyplot as plt

from models.sampler import StratifiedSampler, ImportanceSampler
from models.renderer import VolumetricRenderer
from models.nerf_mlp import VolumeInterpolater
from models.nerf_net import NeRFNet

class VoxelNeRFNet(NeRFNet):
    
    def __init__(self, volume_size, N_samples=64, N_importance=64, ray_chunk=1024*32, pts_chuck=1024*64,
        perturb=1., raw_noise_std=0., white_bkgd=False):
        
        super().__init__(netdepth=1, netwidth=1, netdepth_fine=1, netwidth_fine=1,
            viewdirs=False, use_embed=False, multires=0, multires_views=0, conv_embed=False,
            N_samples=N_samples, N_importance=N_importance, ray_chunk=ray_chunk, pts_chuck=pts_chuck,
            perturb=perturb, raw_noise_std=raw_noise_std, white_bkgd=white_bkgd)

        del self.nerf
        del self.nerf_fine

        # create nerf mlps
        self.nerf = VolumeInterpolater(volume_size)
        self.nerf_fine = self.nerf

    def load_from_numpy(self, np_vol):
        self.nerf.load_from_numpy(np_vol)
