import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.error import *

# Volumetric render rays: \int T(t) C(v+td) \sigma(o+td) dt, where T(t) = exp(-\int^{t} \sigma(o+sd) ds
class VolumetricRenderer(nn.Module):
    def __init__(self, act_fn=F.relu, white_bkgd=False, raw_noise_std=0.):
        """
        NeRF Volumetric Renderer
        """
        super().__init__()
        self.raw_noise_std = raw_noise_std
        self.white_bkgd = white_bkgd
        self.act_fn = act_fn
        return

    def forward(self, raw, z_vals, rays_d, **kwargs):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples, C]. Prediction from model.
            z_vals: [num_rays, num_samples]. Point intervals sampled along the ray.
            rays_d: [num_rays, 3]. Ray directions.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        dists = z_vals[...,1:] - z_vals[...,:-1]
        # dists = torch.linalg.norm(pts[..., 1:, :] - pts[..., :-1, :], ord=2, dim=-1) # [N_rays, N_samples-1]
        dists = torch.cat([dists, 1e10 * torch.ones_like(dists[...,:1])], -1)  # Infinite padding: [N_rays, N_samples]
        dists = dists * torch.linalg.norm(rays_d[..., None, :], ord=2, dim=-1)

        # sigmoid normalizes color to 0-1
        rgb = torch.sigmoid(raw[..., :-1])  # [N_rays, N_samples, 3]

        # Generate random noises
        noise = 0.
        raw_noise_std = kwargs.get('raw_noise_std', self.raw_noise_std)
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., -1].shape, device=raw.device) * raw_noise_std

        # apply quadrature rule: a(t) = 1 - exp(-\sigma(o+td) dt)
        alpha = raw[..., -1] + noise # # [N_rays, N_samples]
        # alpha = (alpha[..., 1:] + alpha[..., :-1]) / 2.  # [N_rays, N_samples]
        alpha = 1.-torch.exp(-self.act_fn(alpha) * dists) # [N_rays, N_samples]
        # CHECK_ALL_ZERO(dist=dists, alpha=alpha, raw_rgb=rgb)

        # calculate transmittance: T(t) = exp(-\int^{t} \sigma(o+sd) ds) = \prod^{t} [1 - a(s)] ds
        # TF: weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        Ts = torch.cat([torch.ones_like(alpha[..., :1]), 1.-alpha + 1e-10], -1) # [N_rays, N_samples+1]
        Ts = torch.cumprod(Ts, -1)[..., :-1] # [N_rays, N_samples] # Exclude the last one, keep the first one

        # volumetric rendering: C = \int T(t) a(t) c(o+td) dt
        weights = alpha * Ts # [N_rays, N_samples]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        # depth = E[t] = \int T(t) a(t) t dt
        depth_map = torch.sum(weights * z_vals, -1, keepdim=True) # [N_rays, 1]
        # acc = \int T(t) a(t) dt
        acc_map = torch.sum(weights, -1, keepdim=True) # [N_rays, 1]
        depth_map[acc_map <= 1e-10] = 1e10 # set depth of vacancy to inf
        # disparity = 1 / depth
        disp_map = 1. / torch.max(torch.full_like(depth_map, 1e-10), depth_map / acc_map) # [N_rays, 1]

        # render white background
        white_bkgd = kwargs.get('white_bkgd', self.white_bkgd)
        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map)

        return dict(rgb=rgb_map, disp=disp_map, acc=acc_map, weights=weights, depth=depth_map)


# Volumetric render rays for mipnerf: \int T(t) C(v+td) \sigma(o+td) dt, where T(t) = exp(-\int^{t} \sigma(o+sd) ds
# refer to https://github.com/google/mipnerf/blob/84c969e0a623edd183b75693aed72a7e7c22902d/internal/mip.py#L187
class MipVolumetricRenderer(VolumetricRenderer):
    def __init__(self, act_fn=F.relu, white_bkgd=False, raw_noise_std=0.):
        """
        Mip-NeRF Volumetric Renderer
        """
        super().__init__(act_fn=act_fn, white_bkgd=white_bkgd, raw_noise_std=raw_noise_std)
        return

    def forward(self, raw, z_vals, rays_d, **kwargs):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples, C]. Prediction from model.
            z_vals: [num_rays, num_samples]. Point intervals sampled along the ray.
            rays_d: [num_rays, 3]. Ray directions.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        mids = (z_vals[...,1:] + z_vals[...,:-1]) / 2.
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = dists * torch.linalg.norm(rays_d[..., None, :], ord=2, dim=-1)

        # Generate random noises
        noise = 0.
        raw_noise_std = kwargs.get('raw_noise_std', self.raw_noise_std)
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., -1].shape, device=raw.device) * raw_noise_std

        # apply quadrature rule: a(t) = 1 - exp(-\sigma(o+td) dt)
        alpha = raw[..., -1] + noise # # [N_rays, N_samples]
        # alpha = (alpha[..., 1:] + alpha[..., :-1]) / 2.  # [N_rays, N_samples]
        alpha = 1.-torch.exp(-self.act_fn(alpha) * dists) # [N_rays, N_samples]
        # CHECK_ALL_ZERO(dist=dists, alpha=alpha, raw_rgb=rgb)

        # calculate transmittance: T(t) = exp(-\int^{t} \sigma(o+sd) ds) = \prod^{t} [1 - a(s)] ds
        # TF: weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        Ts = torch.cat([torch.ones_like(alpha[..., :1]), 1.-alpha + 1e-10], -1) # [N_rays, N_samples+1]
        Ts = torch.cumprod(Ts, -1)[..., :-1] # [N_rays, N_samples] # Exclude the last one, keep the first one

        # volumetric rendering: C = \int T(t) a(t) c(o+td) dt
        weights = alpha * Ts # [N_rays, N_samples]
        # sigmoid normalizes color to 0-1
        rgb = torch.sigmoid(raw[..., :-1])  # [N_rays, N_samples, 3]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        # depth = E[t] = \int T(t) a(t) t dt
        depth_map = torch.sum(weights * mids, -1, keepdim=True) # [N_rays, 1]
        # acc = \int T(t) a(t) dt
        acc_map = torch.sum(weights, -1, keepdim=True) # [N_rays, 1]
        depth_map[acc_map <= 1e-10] = 1e10 # set depth of vacancy to inf
        # disparity = 1 / depth
        disp_map = 1. / torch.max(torch.full_like(depth_map, 1e-10), depth_map / acc_map) # [N_rays, 1]

        # render white background
        white_bkgd = kwargs.get('white_bkgd', self.white_bkgd)
        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map)

        return dict(rgb=rgb_map, disp=disp_map, acc=acc_map, weights=weights, depth=depth_map)

# Integration along rays: \int V(o + td) dt
class ProjectionRenderer(nn.Module):
    def __init__(self, raw_noise_std=0.):
        """
        Nerf MLP backbone
        """
        super(ProjectionRenderer, self).__init__()
        self.raw_noise_std = raw_noise_std
        return

    def forward(self, raw, pts, raw_noise_std=0., **kwargs):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, C]. Prediction from model.
            pts: [num_rays, num_samples along ray, 3]. Sampled points.
        Returns:
            rgb_map: [num_rays, C]. Estimated RGB color of a ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        """

        # compute distance interval along rays to perform integration
        dists = torch.norm(pts[...,1:,:] - pts[...,:-1,:], dim=-1) # [N_rays, N_samples-1, 3]

        # \int V(o + td) dt
        values = (raw[..., :-1, :] + raw[..., 1:, :]) / 2.0 # [N_rays, N_samples-1, C]
        rgb_map = torch.sum(values * dists[..., None], dim=-2) # [N_rays, C]

        # Weight = 1 - exp{ max(density*vol, 0) }
        weights = torch.mean(raw, -1)  # [N_rays, N_sample-1]
        dists = torch.cat([dists, dists[..., -1, None]], -1) # Repeat padding. [N_rays, N_sample]
        weights = 1.-torch.exp(-F.relu(weights) * dists) # Compute weight

        return dict(rgb=rgb_map, weights=weights)
