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
from models.renderer import MipVolumetricRenderer
from models.nerf_mlp import MipNeRFMLP
from models.nerf_net import NeRFNet

class MipNeRFNet(NeRFNet):
    
    def __init__(self, netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, N_samples=64, N_importance=64,
        viewdirs=True, use_embed=True, multires=10, multires_views=4, ray_chunk=1024*32, pts_chuck=1024*64,
        perturb=1., raw_noise_std=0., white_bkgd=False):
        
        super().__init__(netdepth=netdepth, netwidth=256, netdepth_fine=netdepth_fine, netwidth_fine=netwidth_fine,
            viewdirs=viewdirs, use_embed=use_embed, multires=multires, multires_views=multires_views, conv_embed=False,
            N_samples=N_samples, N_importance=N_importance, ray_chunk=ray_chunk, pts_chuck=pts_chuck,
            perturb=perturb, raw_noise_std=raw_noise_std, white_bkgd=white_bkgd)

        del self.nerf
        del self.nerf_fine
        del self.renderer

        # Ray renderer
        self.renderer = MipVolumetricRenderer(raw_noise_std=raw_noise_std, white_bkgd=white_bkgd)

        # create nerf mlps
        self.nerf = MipNeRFMLP(input_dim=3, output_dim=4, net_depth=netdepth, net_width=netwidth, skips=[4],
                viewdirs=viewdirs, use_embed=use_embed, multires=multires, multires_views=multires_views, netchunk=pts_chuck)
        self.nerf_fine = self.nerf

    def lift_gaussian(self, rays_d, t_mean, t_var, r_var, diag):
        """Lift a Gaussian defined along a ray to 3D coordinates.
        rays_d: [N_rays, 3], mean on the ray
        t_mean: [N_rays, N_samples], mean on the ray
        t_var: [N_rays, N_samples], variance along ray
        r_var: [N_rays, N_samples], variance perpendicular to ray
        diag: boolean, whether compute the whole covariance matrix or only diagonal
        """
        mean = rays_d[..., None, :] * t_mean[..., None]

        d_mag_sq = torch.maximum(
            torch.full_like(rays_d[..., :1], 1e-10), torch.sum(rays_d**2, -1, keepdims=True)
        ) # [N_rays, 1]

        if diag:
            d_outer_diag = rays_d**2 # [N_rays, 3]
            null_outer_diag = 1. - d_outer_diag / d_mag_sq # [N_rays, 3]
            t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :] # [N_rays, N_samples, 3]
            xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :] # [N_rays, N_samples, 3]
            cov_diag = t_cov_diag + xy_cov_diag # [N_rays, N_samples, 3]
            return mean, cov_diag # [N_rays, N_samples, 3], # [N_rays, N_samples, 3]
        else:
            d_outer = rays_d[..., :, None] * rays_d[..., None, :] # d d^T: [N_rays, 3, 3]
            d_norm = rays_d / torch.sqrt(d_mag_sq) # [N_rays, 3]
            eye = torch.eye(d.shape[-1]).expand(d_outer.shape) # [N_rays, 3, 3]
            null_outer = eye - d_norm[..., :, None] * d_norm[..., None, :] # [N_rays, 3, 3]
            t_cov = t_var[..., None, None] * d_outer[..., None, :, :] # [N_rays, N_samples, 3, 3]
            xy_cov = r_var[..., None, None] * null_outer[..., None, :, :] # [N_rays, N_samples, 3, 3]
            cov = t_cov + xy_cov
            return mean, cov # [N_rays, N_samples, 3], # [N_rays, N_samples, 3, 3]


    def conical_frustum_to_gaussian(self, rasy_d, t0, t1, base_radius, diag, stable=True):
        """Approximate a conical frustum as a Gaussian distribution (mean+cov).
        Assumes the ray is originating from the origin, and base_radius is the
        radius at dist=1. Doesn't assume `d` is normalized.
        Args:
            rays_d: [N_rays, 3], the axis of the cone
            t0: [N_rays, N_samples], the starting distance of the frustum.
            t1: [N_rays, N_samples], the ending distance of the frustum.
            base_radius: [N_rays, N_samples], the scale of the radius as a function of distance.
            diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
            stable: boolean, whether or not to use the stable computation described in
                the paper (setting this to False will cause catastrophic failure).
        Returns:
            a Gaussian (mean and covariance).
        """
        if stable:
            mu = (t0 + t1) / 2
            hw = (t1 - t0) / 2
            t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
            t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                (3 * mu**2 + hw**2)**2)
            r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                (hw**4) / (3 * mu**2 + hw**2))
        else:
            t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
            r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
            t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
            t_var = t_mosq - t_mean**2
        return self.lift_gaussian(rasy_d, t_mean, t_var, r_var, diag)


    def cylinder_to_gaussian(rays_d, t0, t1, radius, diag):
        """Approximate a cylinder as a Gaussian distribution (mean+cov).
        Assumes the ray is originating from the origin, and radius is the
        radius. Does not renormalize `d`.
        Args:
            rays_d: [N_rays, 3], the axis of the cylinder
            t0: [N_rays, N_samples], the starting distance of the cylinder.
            t1: [N_rays, N_samples], the ending distance of the cylinder.
            radius: [N_rays, N_samples], the radius of the cylinder
            diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
        Returns:
            a Gaussian (mean and covariance).
        """
        t_mean = (t0 + t1) / 2
        r_var = radius**2 / 4
        t_var = (t1 - t0)**2 / 12
        return self.lift_gaussian(rays_d, t_mean, t_var, r_var, diag)


    def cast_rays(self, z_vals, rays_o, rays_d, radii, ray_shape='cone', diag=True):
        """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
        Args:
            z_vals: [N_rays, N_samples], the "fencepost" distances along the ray.
            origins: [N_rays, 3], the ray origin coordinates.
            directions: [N_rays, 3], the ray direction vectors.
            radii: [N_rays], the radii (base radii for cones) of the rays.
            ray_shape: str, the shape of the ray, must be 'cone' or 'cylinder'.
            diag: boolean, whether or not the covariance matrices should be diagonal.
        Returns:
            a tuple of arrays of means and covariances.
        """
        t0 = z_vals[..., :-1] # [N_rays, N_samples]
        t1 = z_vals[..., 1:] # [N_rays, N_samples]
        radii = radii.expand(t0.shape) # [N_rays, N_samples]
        if ray_shape == 'cone':
            gaussian_fn = self.conical_frustum_to_gaussian
        elif ray_shape == 'cylinder':
            gaussian_fn = self.cylinder_to_gaussian
        else:
            raise ValueError('Unknow ray shape:', gaussian_fn)
        means, covs = gaussian_fn(rays_d, t0, t1, radii, diag)
        means = means + rays_o[..., None, :]
        return means, covs
    
    def render_rays(self, rays_o, rays_d, near, far, radii, viewdirs=None, raw_noise_std=0.,
        verbose=False, retraw = False, retpts=False, pytest=False, **kwargs):
        """Volumetric rendering.
        Args:
          ray_o: origins of rays. [N_rays, 3]
          ray_d: directions of rays. [N_rays, 3]
          near: the minimal distance. [N_rays, 1]
          far: the maximal distance. [N_rays, 1]
          radii: the maximal distance. [N_rays, 1]
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
        z_vals = self.point_sampler(rays_o, rays_d, bounds, zvals_only=True, **kwargs)  # [N_rays, N_samples, 3]
        pts, pts_cov = self.cast_rays(z_vals, rays_o, rays_d, radii)
        viewdirs_c = viewdirs[..., None, :].expand(pts.shape) if self.use_viewdirs else None # [N_rays, 3] -> [N_rays, N_samples, 3]
        raw = self.nerf(pts, pts_cov, viewdirs_c)
        ret = self.renderer(raw, z_vals, rays_d, pad=False, raw_noise_std=raw_noise_std)

        # Buffer raw/pts
        if retraw:
            ret['raw'] = raw
        if retpts:
            ret['pts'] = pts

        # Secondary sampling
        N_importance = kwargs.get('N_importance', self.N_importance)
        if (self.importance_sampler is not None) and (N_importance > 0):
            # backup coarse model output
            ret0 = ret

            # Do a blurpool.
            weights = ret['weights']
            weights_pad = torch.cat([
                weights[..., :1],
                weights,
                weights[..., -1:],
            ], -1)
            weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
            weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])
            z_mids = (z_vals[...,1:] + z_vals[...,:-1]) / 2.

            # resample
            z_vals, sampler_extras = self.importance_sampler(rays_o, rays_d, z_mids, weights_blur, zvals_only=True, **kwargs) # [N_rays, N_samples + N_importance, 3]
            pts, pts_cov = self.cast_rays(z_vals, rays_o, rays_d, radii)
            viewdirs_f = viewdirs[..., None, :].expand(pts.shape) if self.use_viewdirs else None # [N_rays, 3] -> [N_rays, N_samples, 3]
            # obtain raw data
            raw = self.nerf_fine(pts, pts_cov, viewdirs_f)
            # render raw data
            ret = self.renderer(raw, z_vals, rays_d, pad=False, raw_noise_std=raw_noise_std)
            
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

    def forward(self, ray_batch, bound_batch, radii, **kwargs):
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
        if self.training:
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
        if isinstance(near, (int, float)):
            near = near * torch.ones_like(rays_d[...,:1], dtype=torch.float)
        if isinstance(far, (int, float)):
            far = far * torch.ones_like(rays_d[...,:1], dtype=torch.float)

        # Extract radius
        if isinstance(radii, (int, float)):
            radii = radii * torch.ones_like(rays_d[...,:1], dtype=torch.float)

        # Batchify rays
        all_ret = {}
        for i in range(0, rays_o.shape[0], self.chunk):
            end = min(i+self.chunk, rays_o.shape[0])
            chunk_o, chunk_d = rays_o[i:end], rays_d[i:end]
            chunk_n, chunk_f, chunk_r = near[i:end], far[i:end], radii[i:end]
            chunk_v = viewdirs[i:end] if self.use_viewdirs else None

            # Render function
            ret = self.render_rays(chunk_o, chunk_d, chunk_n, chunk_f, chunk_r, viewdirs=chunk_v, **render_kwargs)
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
