import os, sys
import numpy as np
import imageio
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image import img2mse

import matplotlib.pyplot as plt

def polar_to_rotmat(azimuths, zeniths):
    view_dir = -torch.stack([torch.sin(zeniths) * torch.cos(azimuths),
                             torch.cos(zeniths), 
                             torch.sin(zeniths) * torch.sin(azimuths)], -1) # [batch_shape, 3]
    up_dir = torch.Tensor([0., 1., 0.]).expand(view_dir.shape) # [batch_shape, 3]

    # Grama-schmidta algorithm
    left_dir = torch.cross(up_dir, view_dir, dim=-1) # [batch_shape, 3]
    left_dir /= torch.norm(left_dir, 2, dim=-1, keepdim=True) # normalize
    up_dir = torch.cross(view_dir, left_dir, dim=-1) # [batch_shape, 3]
    return torch.stack([left_dir, up_dir, view_dir], -1) # [batch_shape, 3, 3]

def polar_to_xyz(azimuths, zeniths, rad=1.):
    return torch.stack([rad * torch.sin(zeniths) * torch.cos(azimuths),
                        rad * torch.cos(zeniths), 
                        rad * torch.sin(zeniths) * torch.sin(azimuths)], -1) # [batch_shape, 3]

class VoteNet(nn.Module):
    
    def __init__(self, args, nerf):
        super(VoteNet, self).__init__()

        self.nerf = nerf
        self.bound = (args.near, args.far)

    def mse_dist(self, rgb, gts):
        return torch.norm(rgb - gts, p=2, dim=-1, keepdim=True) # [batch_size, C] -> [batch_size, 1]

    def vote_rays(self, rays_o, rays_d, gts, rots, ts):
        """
        Compute voting of expected rotation and translation for each ray.
        Param:
        rays_o: input origins [N_imgs, N_rays, 3]
        rays_d: input directions [N_imgs, N_rays, 3]
        gts: input groundtruth [N_imgs, N_rays, C]
        """
        # Batchify
        ts = ts.expand(rays_o.shape[:-1] + ts.shape) # [N_imgs, N_rays, A_sample, Z_sample, 3]
        rots = rots.expand(rays_o.shape[:-1] + rots.shape) # [N_imgs, N_rays, A_sample, Z_sample, 3, 3]

        print(rays_o.shape, rays_d.shape)

        rays_o = rays_o[..., None, None, :, None] # [N_imgs, N_rays, 1, 1, 3, 1]
        rays_o = torch.matmul(rots, rays_o).squeeze(-1) # [N_imgs, N_rays, A_sample, Z_sample, 3]
        rays_o += ts # [N_imgs, N_uvs, A_sample, Z_sample, 3]

        rays_d = rays_d[..., None, None, :, None] # [N_imgs, N_rays, 1, 1, 3, 1]
        rays_d = torch.matmul(rots, rays_d).squeeze(-1) # [N_imgs, N_rays, A_sample, Z_sample, 3]

        ret_dict = self.nerf((rays_o, rays_d), self.bound) # [N_imgs, N_rays, A_sample, Z_sample, C]

        # Gaussian likelihood voting
        gts = gts[..., None, None, :]
        # votes = torch.exp(-self.mse_dist(ret_dict['rgb'], gts)) # [N_imgs, N_rays, A_sample, Z_sample, 1]
        votes = -self.mse_dist(ret_dict['rgb'], gts) # [N_imgs, N_rays, A_sample, Z_sample, 1]
        # votes = 1. / (self.mse_dist(ret_dict['rgb'], gts) + 1e-8) # [N_imgs, N_rays, A_sample, Z_sample, 1]
        # normalize per ray
        # votes = votes / torch.sum(torch.sum(votes, -2, keepdim=True), -3, keepdim=True)
        # softmax normalize per ray
        votes = votes.reshape(votes.shape[:2] + (-1, votes.shape[-1])) # [N_imgs, N_rays, A_sample * Z_sample, 1]
        votes = F.softmax(votes, 2) # [N_imgs, N_rays, A_sample * Z_sample, 1]
        # sum over rays per image
        votes = torch.sum(votes, 1) # [N_imgs, A_sample * Z_sample, 1]

        return votes

    def forward(self, rays_o, rays_d, gts, **kwargs):
        """
        Voting for the expected rotation and translation.
        Param:
        rays_o: input origins [N_imgs, N_rays, 3]
        rays_d: input directions [N_imgs, N_rays, 3]
        gts: input groundtruth [N_imgs, N_rays, C]
        Return:
        E[.]: Expected polar angles [N_imgs, 2]
        """
        A_sample, Z_sample = 64, 64
        azimuths, zeniths = torch.meshgrid(torch.linspace(-math.pi, math.pi, A_sample), # [A_sample, Z_sample]
                                           torch.linspace(-math.pi, math.pi, Z_sample)) # [A_sample, Z_sample]

        ts = polar_to_xyz(azimuths, zeniths) # [A_sample, Z_sample, 3]
        rots = polar_to_rotmat(azimuths, zeniths) # [A_sample, Z_sample, 3, 3]

        N_total = rays_o.shape[0] * rays_o.shape[1]
        chunk = 1024

        votes = torch.zeros((rays_o.shape[0], A_sample * Z_sample, 1))
        for i in range(0, rays_o.shape[1], chunk):
            end = min(i+chunk, rays_o.shape[1])
            votes += self.vote_rays(rays_o[:, i:end, ...], rays_d[:, i:end, ...], gts[:, i:end, ...], rots, ts)

        # Softmax normalization
        votes = F.softmax(votes, 1) # [N_imgs, A_sample * Z_sample, 1]

        heatmap = votes.reshape(votes.shape[0], A_sample, Z_sample, votes.shape[-1])[0]
        heatmap = heatmap.cpu().numpy().squeeze()
        img = plt.imshow(heatmap, origin='lower', cmap='Spectral_r')
        plt.colorbar(img)
        plt.savefig('net/heatmap.jpg')

        # Generate grids
        grids = torch.stack([azimuths, zeniths], -1) # [A_sample, Z_sample, 2]
        grids = grids.reshape(-1, grids.shape[-1]) # [A_sample * Z_sample, 2]
        
        # Expectation
        grids = grids.expand((votes.shape[0],) + grids.shape) # [N_imgs, A_sample * Z_sample, 2]
        E_grids = torch.sum(votes * grids, 1) # [N_imgs, 2]

        return E_grids
