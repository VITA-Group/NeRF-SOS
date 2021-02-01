
import os, sys
import math, random, time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Ray helpers
# def get_rays(H, W, focal, c2w):
#     i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
#     i = i.t()
#     j = j.t()
#     dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, torch.ones_like(i)], -1)
#     # Rotate ray directions from camera frame to the world frame
#     rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = c2w[:3,-1].expand(rays_d.shape)
#     return rays_o, rays_d

def get_ortho_rays(H, W, c2w, ps=1., us=1., z_dir=1.):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i, j = i.t(), j.t()

    # Rotate ray directions from camera frame to the world frame
    dirs = torch.stack([torch.zeros_like(i), torch.zeros_like(i), z_dir*torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    origins = torch.stack([(i-W*.5)*ps/us, -(j-H*.5)*ps/us, torch.zeros_like(i)], -1)
    origins = torch.sum(origins[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_o = origins - c2w[:3,-1].expand(origins.shape)    # translation

    return torch.stack([rays_o, rays_d], 0)

def get_persp_rays(H, W, focal, c2w, ps=1., us=1., z_dir=1.):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)*ps/focal/us, -(j-H*.5)*ps/focal/us, z_dir*np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return np.stack([rays_o, rays_d], 0)

def get_rays_with_id(i, H, W, focal, c2w):
    rays_o, rays_d = get_rays_np(H, W, focal, c2w)
    rays = np.stack([rays_o, rays_d], 0)
    ids = (i * np.ones((2, H, W, 1))).astype(np.float32)
    rays = np.concatenate([rays, ids], -1)
    return rays

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d
