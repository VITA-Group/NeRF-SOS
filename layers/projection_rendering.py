import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.check_error import *

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

        # raw noises
        # if 'raw_noise_std' in kwargs:
            # raw_noise_std = kwargs['raw_noise_std']
        # else:
            # raw_noise_std = self.raw_noise_std
        # if raw_noise_std > 0.0:
            # raw += torch.randn(raw.shape) * raw_noise_std

        # \int V(o + td) dt
        values = (raw[..., :-1, :] + raw[..., 1:, :]) / 2.0 # [N_rays, N_samples-1, C]
        rgb_map = torch.sum(values * dists[..., None], dim=-2) # [N_rays, C]

        # Weight = 1 - exp{ max(density*vol, 0) }
        weights = torch.mean(raw, -1)  # [N_rays, N_sample-1]
        dists = torch.cat([dists, dists[..., -1, None]], -1) # Repeat padding. [N_rays, N_sample]
        weights = 1.-torch.exp(-F.relu(weights) * dists) # Compute weight

        return dict(rgb=rgb_map, weights=weights)
