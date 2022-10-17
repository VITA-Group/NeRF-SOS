import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as st

# TODO: remove this dependency
# from torchsearchsorted import searchsorted

# Stratified Sampling Layer
class StratifiedSampler(nn.Module):

    def __init__(self, N_samples, perturb=0.0, lindisp=False, pytest=False):
        """ Init layered sampling
        init_planes: [N_planes, 4], Ax + By + Cz = D
        trainable: Whether planes can be trained by optimizer
        """
        super(StratifiedSampler, self).__init__()
        self.N_samples = N_samples
        self.perturb = perturb
        self.lindisp = lindisp
        self.pytest = pytest

    def forward(self, rays_o, rays_d, bounds, zvals_only=False, **render_kwargs):
        """ Generate sample points
        Args:
        rays_o: [N_rays, 3] origin points of rays
        rays_d: [N_rays, 3] directions of rays
        
        bounds: [N_rays, 2] near far boundary
        
        render_kwargs: other render parameters

        Return:
        pts: [N_rays, N_samples, 3] Point samples on every ray
        z_vals: [N_rays, N_samples] The z-values of rays
        """
        
        perturb = render_kwargs.get('perturb', self.perturb)
        N_samples = render_kwargs.get('N_samples', self.N_samples)
        
        # Sample uniformly from near to far
        N_rays = rays_o.shape[0]
        near, far = bounds[..., 0, None], bounds[..., 1, None] # [N_rays, 1] > fortress 1.2, 14.72
        t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
        if not self.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        # uniform noise
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)

            # Pytest, overwrite u with numpy's fixed random numbers
            if self.pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand
        if not zvals_only:
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
            return pts, z_vals
        else:
            return z_vals
    
# Importance Resampling Layer
class ImportanceSampler(nn.Module):

    def __init__(self, N_importance, perturb=0.0, lindisp=False, pytest=False):
        """ Init layered sampling
        init_planes: [N_planes, 4], Ax + By + Cz = D
        trainable: Whether planes can be trained by optimizer
        """
        super(ImportanceSampler, self).__init__()
        self.N_importance = N_importance
        self.perturb = perturb
        self.lindisp = lindisp
        self.pytest = pytest
        
    # Hierarchical sampling (section 5.2)
    def sample_pdf(self, bins, weights, det=False):
        # Get pdf
        weights = weights + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=self.N_importance, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [self.N_importance])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [self.N_importance], device=cdf.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if self.pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [self.N_importance]
            if det:
                u = np.linspace(0., 1., self.N_importance)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples

    def forward(self, rays_o, rays_d, z_vals, weights, zvals_only=False, **render_kwargs):
        """ Generate sample points
        Args:
        rays_o: [N_rays, 3] origin points of rays
        rays_d: [N_rays, 3] directions of rays
        
        z_vals: [N_rays, N_samples, 2] z-values obtained from previous near-far sampler
        weights: [N_rays, N_samples, 1] sample weights of rays from previous near-far sampler
        
        render_kwargs: other render parameters

        Return:
        pts: [N_rays, N_samples, 3] Point samples on every ray
        z_vals: [N_rays, N_samples] The z-values of rays
        """
        
        perturb = render_kwargs['perturb'] if 'perturb' in render_kwargs else self.perturb
        
        ret_extras = {}

        # Importance resampling
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = self.sample_pdf(z_vals_mid, weights[...,1:-1], det=(perturb==0.0))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        # Return new samples
        ret_extras['z_samples'] = z_samples

        if not zvals_only:
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
            return pts, z_vals, ret_extras
        else:
            return z_vals, ret_extras

# Layered Sampling Layer
class LayeredSampler(nn.Module):

    def __init__(self, init_planes, perturb=0.0, trainable=False, pytest=False):
        """ Init layered sampling
        init_planes: [N_planes, 4], Ax + By + Cz = D
        trainable: Whether planes can be trained by optimizer
        """
        super(LayeredSampler, self).__init__()
        
        self.trainable = trainable
        self.perturb = perturb
        self.pytest = pytest

        if self.trainable:
            self.Ds = nn.Parameter(torch.Tensor(init_planes[:, -1])) # [N_planes, 1]
#             self.ns = nn.Parameter(torch.Tensor(init_planes[:, :3])) # [N_planes, 3]
        else:
            self.register_buffer('Ds', torch.Tensor(init_planes[:, -1])) # [N_planes, 3]
        
        self.register_buffer('ns', torch.Tensor(init_planes[:, :3])) # [N_planes, 3]

    # Return if sampler supports resampling
    def has_resampling(self):
        return False

    def forward(self, rays_o, rays_d, zvals_only=False, **render_kwargs):
        """ Generate sample points
        Args:
        rays_o: [N_rays, 3] origin points of rays
        rays_d: [N_rays, 3] directions of rays

        render_kwargs: other render parameters

        Return:
        pts: [N_rays, N_samples, 3] Point samples on every ray
        z_vals: [N_rays, N_samples] The z-values of rays
        """
        perturb = render_kwargs['perturb'] if 'perturb' in render_kwargs else self.perturb
        
        ## Compute z_vals
        a = self.Ds[None, :] - torch.sum((rays_o[:, None, :]) * self.ns[None, :, :], dim=-1) # dot(o_i, n_j) -> [N_rays, N_planes]
        b = torch.sum(rays_d[:, None, :] * self.ns[None, :, :], dim=-1)                      # dot(d_i, n_j) -> [N_rays, N_planes]
#         assert not torch.isnan(a).any() and not torch.isinf(a).any()
#         assert not torch.isnan(b).any() and not torch.isinf(b).any() and not (torch.abs(b) < 1e-6).any()

        z_vals = a / b
#         assert (z_vals > 0).any()

        ## Add noise if perturn is greater than 2.0!!!
        ## Since we usually don't use perturb in layered rendering.
        if perturb >= 2.0:
            if perturb < 4.0:
                # stratified samples in those intervals
                # get intervals between samples
                mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                upper = torch.cat([mids, z_vals[...,-1:]], -1)
                lower = torch.cat([z_vals[...,:1], mids], -1)
                t_rand = torch.rand(z_vals.shape, device=z_vals.device)

                # Pytest, overwrite u with numpy's fixed random numbers
                if self.pytest:
                    np.random.seed(0)
                    t_rand = np.random.rand(*list(z_vals.shape))
                    t_rand = torch.Tensor(t_rand)

                z_vals = lower + (upper - lower) * t_rand
            else:
                # Gaussian sampling over layers
                t_rand = (perturb - 4.0) * torch.randn(z_vals.shape, device=z_vals.device)
                t_rand[t_rand > 1.0] = 1.0
                t_rand[t_rand < -1.0] = -1.0

                step = 0.5 * (z_vals[..., 1:] - z_vals[..., :-1])

                length = torch.cat([step, torch.zeros(z_vals.shape[:-1], device=z_vals.device)[..., None]], -1)
                z_vals[t_rand > 0.0] += t_rand[t_rand > 0.0] * length[t_rand > 0.0]

                length = torch.cat([torch.zeros(z_vals.shape[:-1], device=z_vals.device)[..., None], step], -1)
                z_vals[t_rand < 0.0] += t_rand[t_rand < 0.0] * length[t_rand < 0.0]

        if not zvals_only:
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
            if (torch.isnan(pts).any() or torch.isinf(pts).any()):
                print(f"! [Numerical Error] pts contains nan or inf.")

            return pts, z_vals
        else:
            return z_vals