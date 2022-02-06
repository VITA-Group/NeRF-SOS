import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Positional encoding (NeRF section 5.1)
class PositionEncoding(nn.Module):

    def __init__(self, input_dim, N_freqs, max_freq, periodic_fns=[torch.sin, torch.cos],
                 log_sampling=True, include_input=True, trainable=False):
        super(PositionEncoding, self).__init__()

        self.periodic_fns = periodic_fns
        self.include_input = include_input or len(periodic_fns) == 0
        self.out_dim = len(periodic_fns) * input_dim * N_freqs

        # Identity map if no periodic_fns provided
        if self.include_input:
            self.out_dim += input_dim

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        if trainable:
            self.freq_bands = nn.Parameter(freq_bands, requires_grad=True)
        else:
            self.register_buffer('freq_bands', freq_bands, persistent=False)

    def forward(self, inputs):
        N_freqs = len(self.freq_bands)

        embeds = []
        for periodic_fn in self.periodic_fns:
            x_freq = inputs[..., None].expand(inputs.shape+(N_freqs,)) * self.freq_bands # [batch_shape, 3, N_freq]
            x_freq = periodic_fn(x_freq)
            embeds.append(x_freq.transpose(-1, -2))  # [batch_shape, N_freq, 3]
        embeds = torch.stack(embeds, -2) # [batch_shape, N_freq, N_fns, 3]
        embeds = embeds.reshape(inputs.shape[:-1]+(-1,)) # [batch_shape, N_freq x N_fns x 3]

        if self.include_input:
            embeds = torch.cat([inputs, embeds], -1)

        return embeds


# def expected_sin(x, x_var):
#   """Estimates mean and variance of sin(z), z ~ N(x, var)."""
#   # When the variance is wide, shrink sin towards zero.
#   y = jnp.exp(-0.5 * x_var) * math.safe_sin(x)
#   y_var = jnp.maximum(
#       0, 0.5 * (1 - jnp.exp(-2 * x_var) * math.safe_cos(2 * x)) - y**2)
#   return y, y_var


# def lift_gaussian(d, t_mean, t_var, r_var, diag):
#   """Lift a Gaussian defined along a ray to 3D coordinates."""
#   mean = d[..., None, :] * t_mean[..., None]

#   d_mag_sq = jnp.maximum(1e-10, jnp.sum(d**2, axis=-1, keepdims=True))

#   if diag:
#     d_outer_diag = d**2
#     null_outer_diag = 1 - d_outer_diag / d_mag_sq
#     t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
#     xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
#     cov_diag = t_cov_diag + xy_cov_diag
#     return mean, cov_diag
#   else:
#     d_outer = d[..., :, None] * d[..., None, :]
#     eye = jnp.eye(d.shape[-1])
#     null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
#     t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
#     xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
#     cov = t_cov + xy_cov
#     return mean, cov


# def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
#   """Approximate a conical frustum as a Gaussian distribution (mean+cov).
#   Assumes the ray is originating from the origin, and base_radius is the
#   radius at dist=1. Doesn't assume `d` is normalized.
#   Args:
#     d: jnp.float32 3-vector, the axis of the cone
#     t0: float, the starting distance of the frustum.
#     t1: float, the ending distance of the frustum.
#     base_radius: float, the scale of the radius as a function of distance.
#     diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
#     stable: boolean, whether or not to use the stable computation described in
#       the paper (setting this to False will cause catastrophic failure).
#   Returns:
#     a Gaussian (mean and covariance).
#   """
#   if stable:
#     mu = (t0 + t1) / 2
#     hw = (t1 - t0) / 2
#     t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
#     t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
#                                       (3 * mu**2 + hw**2)**2)
#     r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
#                               (hw**4) / (3 * mu**2 + hw**2))
#   else:
#     t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
#     r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
#     t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
#     t_var = t_mosq - t_mean**2
#   return lift_gaussian(d, t_mean, t_var, r_var, diag)


# def cylinder_to_gaussian(d, t0, t1, radius, diag):
#   """Approximate a cylinder as a Gaussian distribution (mean+cov).
#   Assumes the ray is originating from the origin, and radius is the
#   radius. Does not renormalize `d`.
#   Args:
#     d: jnp.float32 3-vector, the axis of the cylinder
#     t0: float, the starting distance of the cylinder.
#     t1: float, the ending distance of the cylinder.
#     radius: float, the radius of the cylinder
#     diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
#   Returns:
#     a Gaussian (mean and covariance).
#   """
#   t_mean = (t0 + t1) / 2
#   r_var = radius**2 / 4
#   t_var = (t1 - t0)**2 / 12
#   return lift_gaussian(d, t_mean, t_var, r_var, diag)


# def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
#   """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
#   Args:
#     t_vals: float array, the "fencepost" distances along the ray.
#     origins: float array, the ray origin coordinates.
#     directions: float array, the ray direction vectors.
#     radii: float array, the radii (base radii for cones) of the rays.
#     ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
#     diag: boolean, whether or not the covariance matrices should be diagonal.
#   Returns:
#     a tuple of arrays of means and covariances.
#   """
#   t0 = t_vals[..., :-1]
#   t1 = t_vals[..., 1:]
#   if ray_shape == 'cone':
#     gaussian_fn = conical_frustum_to_gaussian
#   elif ray_shape == 'cylinder':
#     gaussian_fn = cylinder_to_gaussian
#   else:
#     assert False
#   means, covs = gaussian_fn(directions, t0, t1, radii, diag)
#   means = means + origins[..., None, :]
#   return means, covs


# def integrated_pos_enc(x_coord, min_deg, max_deg, diag=True):
#   """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].
#   Args:
#     x_coord: a tuple containing: x, jnp.ndarray, variables to be encoded. Should
#       be in [-pi, pi]. x_cov, jnp.ndarray, covariance matrices for `x`.
#     min_deg: int, the min degree of the encoding.
#     max_deg: int, the max degree of the encoding.
#     diag: bool, if true, expects input covariances to be diagonal (full
#       otherwise).
#   Returns:
#     encoded: jnp.ndarray, encoded variables.
#   """
#   if diag:
#     x, x_cov_diag = x_coord
#     scales = jnp.array([2**i for i in range(min_deg, max_deg)])
#     shape = list(x.shape[:-1]) + [-1]
#     y = jnp.reshape(x[..., None, :] * scales[:, None], shape)
#     y_var = jnp.reshape(x_cov_diag[..., None, :] * scales[:, None]**2, shape)
#   else:
#     x, x_cov = x_coord
#     num_dims = x.shape[-1]
#     basis = jnp.concatenate(
#         [2**i * jnp.eye(num_dims) for i in range(min_deg, max_deg)], 1)
#     y = math.matmul(x, basis)
#     # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
#     # to jax.vmap(jnp.diag)((basis.T @ covs) @ basis).
#     y_var = jnp.sum((math.matmul(x_cov, basis)) * basis, -2)

#   return expected_sin(
#       jnp.concatenate([y, y + 0.5 * jnp.pi], axis=-1),
#       jnp.concatenate([y_var] * 2, axis=-1))[0]


# Positional encoding (NeRF section 5.1)
# https://github.com/yenchenlin/nerf-pytorch/blob/a15fd7cb363e93f933012fd1f1ad5395302f63a4/run_nerf_helpers.py#L15
class PositionEncoding_depr(nn.Module):

    def __init__(self, input_dim, N_freqs, max_freq, periodic_fns,
                 log_sampling=True, include_input=True):
        super(PositionEncoding, self).__init__()

        embed_fns = []
        d = input_dim
        out_dim = 0

        # Identity map if no periodic_fns provided
        if include_input or len(periodic_fns) == 0:
            embed_fns.append(lambda x : x)
            out_dim += d

        if len(periodic_fns) != 0:
            if log_sampling:
                freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            else:
                freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

            for freq in freq_bands:
                for p_fn in periodic_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                    out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
