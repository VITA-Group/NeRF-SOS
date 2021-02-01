import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.check_error import *

# Volumetric render rays: \int T(t) C(v+td) \sigma(o+td) dt, where T(t) = exp(-\int^{t} \sigma(o+sd) ds
class VolumetricRenderer(nn.Module):
    def __init__(self, act_fn=F.relu, white_bkgd=False, raw_noise_std=0.):
        """
        Nerf MLP backbone
        """
        super(VolumetricRenderer, self).__init__()
        self.raw_noise_std = raw_noise_std
        self.white_bkgd = white_bkgd
        self.act_fn = act_fn
        return

    def forward(self, raw, pts, pytest=False, **kwargs):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, C]. Prediction from model.
            pts: [num_rays, num_samples along ray, 3]. Sampled points.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        dists = torch.norm(pts[..., 1:, :] - pts[..., :-1, :], dim=-1) # [N_rays, N_samples-1]
        # dists = torch.cat([dists, 1e10 * torch.ones_like(dists[...,-1:])], -1)  # Infinite padding: [N_rays, N_samples]
        dists = torch.cat([dists, dists[..., -1:]], -1) # Repeat padding: [N_rays, N_sample]

        # sigmoid normalizes color to 0-1
        rgb = torch.sigmoid(raw[..., :-1])  # [N_rays, N_samples, 3]
        # rgb = (rgb[..., 1:, :] + rgb[..., :-1, :]) / 2.  # [N_rays, N_samples, 3]

        # Generate noises
        if 'raw_noise_std' in kwargs:
            raw_noise_std = kwargs['raw_noise_std']
        else:
            raw_noise_std = self.raw_noise_std
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., -1].shape) * raw_noise_std

            # overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., -1].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

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

        ### Debugging
        # CHECK_ALL_ZERO(alpha=alpha, weights=weights, rgb=rgb_map)
        # CHECK(weights=weights, rgb=rgb_map)

        # depth = E[t] = \int T(t) a(t) t dt
        depth_map = torch.sum(weights * dists, -1, keepdim=True) # [N_rays, 1]
        # acc = \int T(t) a(t) dt
        acc_map = torch.sum(weights, -1, keepdim=True)
        depth_map[acc_map < 1e-10] = 1e10 # set depth of vacancy to inf
        # disp = 1 / depth
        # depth = depth_map / torch.max(1e-10*torch.ones_like(acc_map), acc_map)
        # disp_map = 1. / torch.max(1e-10*torch.ones_like(depth_map), depth_map/acc_map)
        disp_map = 1. / torch.max(1e-10*torch.ones_like(depth_map), depth_map)

        ### Debugging
        # CHECK(depth=depth_map, acc=acc_map, disp=disp_map)

        # render white background
        if 'white_bkgd' in kwargs:
            white_bkgd = kwargs['white_bkgd']
        else:
            white_bkgd = self.white_bkgd
        if white_bkgd:
            # mask_out = white_bg - mask_in
            rgb_map = rgb_map + (1. - acc_map)

        return dict(rgb=rgb_map, disp=disp_map, acc=acc_map, weights=weights, depth=depth_map)
