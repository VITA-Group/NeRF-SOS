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

from layers.point_sampling import StratifiedSampler, ImportanceSampler
from layers.volumetric_rendering import VolumetricRenderer
from net.nerf_voxel_net import NerfVoxelNet
# from layers.camera_transform import *

from utils.check_error import *
from utils.image_utils import *

class NerfNet(nn.Module):
    
    def __init__(self, args):
        super(NerfNet, self).__init__()

        # Create sampler
        self.point_sampler = StratifiedSampler(args.N_samples, perturb=args.perturb, lindisp=False, pytest=False)
        self.importance_sampler = None
        if args.N_importance > 0:
            self.importance_sampler = ImportanceSampler(args.N_importance, perturb=args.perturb, lindisp=False, pytest=False)

        # Create transformer
        # self.cam_transformer = None
        # if args.num_cameras > 0:
        #     self.cam_transformer = CameraTransformer(args.num_cameras, trainable=args.trainable_cams)

        # Ray renderer
        self.renderer = VolumetricRenderer()

        # Maximum number of rays to process simultaneously. Used to control maximum memory usage. Does not affect final results.
        self.chunk = args.chunk
        # Save if use view directions (which cannot be changed after building networks)
        self.use_viewdirs = args.use_viewdirs

        # net params
        input_ch = 3
        output_ch = 4
        skips = [4]

        self.nerf = NerfVoxelNet(args, input_ch, output_ch, args.netdepth, args.netwidth, skips)
        self.nerf_fine = self.nerf

        # self.nerf_fine = None
        # if args.N_importance > 0:
        #     self.nerf_fine = NerfVoxelNet(args, input_ch, output_ch, args.netdepth, args.netwidth, skips)

        # render parameters
        self.render_kwargs_train = {
            'N_importance' : args.N_importance,
            'N_samples' : args.N_samples,
            'perturb' : args.perturb,
            'raw_noise_std' : args.raw_noise_std,
            'retraw' : True, 'retpts' : False
        }

        # copy from train rendering first
        self.render_kwargs_test = self.render_kwargs_train.copy()
        # no perturbation
        self.render_kwargs_test['perturb'] = 0.
        self.render_kwargs_test['raw_noise_std'] = 0.
    
    def load_net(self, ckpt):
        if isinstance(ckpt, str):
            ckpt = torch.load(ckpt)
        try:
            self.load_state_dict(ckpt['network_state_dict'])
        except KeyError:
            # For compatibility
            self.load_state_dict(ckpt['network_fn_state_dict'])

    def render_rays(self, rays_o, rays_d, near, far, viewdirs=None, raw_noise_std=0.,
        verbose=False, retraw = False, retpts=False, pytest=False, **kwargs):
        """Volumetric rendering.
        Args:
          ray_o: origins of rays. [N_rays, 3]
          ray_d: directions of rays. [N_rays, 3]
          near: the minimal distance. [N_rays, 1]
          far: the maximal distance. [N_rays, 1]
          raw_noise_std: If True, add noise on raw output from nn
          verbose: bool. If True, print more debugging info.
        Returns:
          rgb: [N_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          raw: [N_rays, N_samples, C]. Raw predictions from model.
          pts: [N_rays, N_samples, 3]. Sampled points.
          rgb0: See rgb_map. Output for coarse model.
          raw0: See raw. Output for coarse model.
          pts0: See acc_map. Output for coarse model.
          z_std: [N_rays]. Standard deviation of distances along ray for each sample.
        """
        bounds = torch.cat([near, far], -1) # [N_rays, 2]

        # Primary sampling
        pts, z_vals, _ = self.point_sampler(rays_o, rays_d, bounds, **kwargs)  # [N_rays, N_samples, 3]
        raw = self.nerf(pts, viewdirs)
        ret = self.renderer(raw, pts, raw_noise_std=raw_noise_std, pytest=pytest)

        # Buffer raw/pts
        if retraw:
            ret['raw'] = raw
        if retpts:
            ret['pts'] = pts
        
        # Secondary sampling
        no_importance = kwargs['N_importance'] <= 0 if 'N_importance' in kwargs else False # Support temporarily disabling resampling
        resample = (self.importance_sampler is not None) and (not no_importance)
        if resample:
            # backup coarse model output
            ret0 = ret

            # resample
            pts, z_vals, sampler_extras = self.importance_sampler(rays_o, rays_d, z_vals, **ret, **kwargs) # [N_rays, N_samples + N_importance, 3]
            # obtain raw data
            raw = self.nerf_fine(pts, viewdirs)
            # render raw data
            ret = self.renderer(raw, pts, raw_noise_std=raw_noise_std, pytest=pytest)
            
            # Buffer raw/pts
            if retraw:
                ret['raw'] = raw
            if retpts:
                ret['pts'] = pts

            # compute std of resampled point along rays
            ret['z_std'] = torch.std(sampler_extras['z_samples'], dim=-1, unbiased=False)  # [N_rays]

            # buffer coarse model output
            for k in ret0:
                ret[k+'0'] = ret0[k]

        return ret

    def forward(self, ray_batch, bound_batch, test=False, **kwargs):
        """Render rays
        Args:
          ray_batch: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
        Returns:
          ret_all includes the following returned values:
          rgb_map: [batch_size, 3]. Predicted RGB values for rays.
          raw: [batch_size, N_sample, C]. Raw data of each point.
          weight_map: [batch_size, N_sample, C]. Convert raw to weight scale (0-1).
          acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        """

        # Render settings
        if not test:
            render_kwargs = self.render_kwargs_train.copy()
            render_kwargs.update(kwargs)
        else:
            render_kwargs = self.render_kwargs_test.copy()
            render_kwargs.update(kwargs)

        # Disentangle ray batch
        rays_o, rays_d = ray_batch
        assert rays_o.shape == rays_d.shape

        # Flatten ray batch
        old_shape = rays_d.shape # [..., 3(+id)]
        rays_o = torch.reshape(rays_o, [-1,rays_o.shape[-1]]).float()
        rays_d = torch.reshape(rays_d, [-1,rays_d.shape[-1]]).float()

        # Provide ray directions as input
        if self.use_viewdirs: 
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, viewdirs.shape[-1]]).float()  

        # Disentangle bound batch
        near, far = bound_batch
        if isinstance(near, int) or isinstance(near, float):
            near = near * torch.ones_like(rays_d[...,:1], dtype=torch.float)
        if isinstance(far, int) or isinstance(far, float):
            far = far * torch.ones_like(rays_d[...,:1], dtype=torch.float)

        # Batchify rays
        all_ret = {}
        for i in range(0, rays_o.shape[0], self.chunk):
            end = min(i+self.chunk, rays_o.shape[0])
            chunk_o, chunk_d = rays_o[i:end], rays_d[i:end]
            chunk_n, chunk_f = near[i:end], far[i:end]
            chunk_v = viewdirs[i:end] if self.use_viewdirs else None
            # Render function
            ret = self.render_rays(chunk_o, chunk_d, chunk_n, chunk_f, viewdirs=chunk_v, **render_kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}

        # Unflatten
        for k in all_ret:
            k_sh = list(old_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh) # [input_rays_shape, per_ray_output_shape]

        return all_ret

    def loss(self, ret_dict, target_s):
        rgb = ret_dict['rgb']
        img_loss = img2mse(rgb, target_s)
        psnr = mse2psnr(img_loss)
        extras = dict(mse=img_loss, psnr=psnr)

        loss = img_loss
        if 'rgb0' in ret_dict:
            img_loss0 = img2mse(ret_dict['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

            extras['mse0'] = img_loss0
            extras['psnr0'] = psnr0

        return loss, extras

    # query raw data for points
    def forward_pts(self, pts_batch, test=False, **kwargs):
        raw = self.nerf(pts_batch)
#         raw = 1.0 - torch.exp(-F.relu(raw))
        return raw

    # l1 loss or l2 loss?
    def loss_raw(self, raw, target_s):
        return F.loss_mse(raw, target_s)
