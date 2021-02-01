import sys, os, math
import numpy as np
import pandas as pd
import scipy, scipy.fft
import scipy.interpolate

import mrc
import pickle
import json

import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm, trange

from focustools.util import CorrectCTF, Project, BackProject

def to_pytorch_complex(x):
    return torch.stack([torch.Tensor(x.real), torch.Tensor(x.imag)], 0) # [2, ...]

def to_numpy_complex(x):
    x = x.cpu().numpy()
    return x[0] + 1j * x[1]

def pytorch_complex_mul(x1, x2):
    return torch.stack([x1[0]*x2[0]-x1[1]*x2[1], x1[0]*x2[1]+x1[1]*x2[0]], 0) # [2, ...]

def pytorch_complex_expi(x):
    return torch.stack([torch.cos(x), torch.sin(x)], 0) # [2, ...]

def pytorch_complex_conj(x):
    return torch.stack([x[0], -x[1]], 0) # [2, ...]

def pytorch_complex_abs(x):
    return torch.sqrt(x[0]**2 + x[1]**2) # [...]

def pytorch_fftfreq(n):
    if n % 2 == 0:
        raster = torch.linspace(-n//2, n//2-1, n)
    else:
        raster = torch.linspace(-(n-1)//2, (n-1)//2, n)
    return raster / n

def pytorch_phase_shift(Fx, shift):
    grids = torch.stack(torch.meshgrid([pytorch_fftfreq(n) for n in Fx.shape[1:]]), -1)

    # exp{-2j*pi*(k*x)/N}
    phase_s = torch.sum(grids * torch.Tensor(shift), -1)
    phase_s = -2 * np.pi * phase_s
    phase_s = pytorch_complex_expi(phase_s) # [2, ...]

    # complex multiplication
    return pytorch_complex_mul(Fx, phase_s) # [2, ...]

def pytorch_affine_transform(vol, theta, interp_mode='bilinear'):

    # transpose for align axis with [y, x, z] (the indexing order)
    vol = vol.permute([0, 2, 3, 1])

    if theta.shape[-1] == 3: # 3 x 3 rotation matrix
        theta = torch.cat([theta, torch.zeros_like(theta[..., :1])], -1) # (3, 4)

    input = vol[None, ...] # [N=1, C=2, D, H, W]
    grids = F.affine_grid(theta[None, ...], input.shape, align_corners=False)
    output = F.grid_sample(input, grids, mode=interp_mode, padding_mode='zeros', align_corners=False) # [N=1, C=2, D, H, W]
    output = output.squeeze(0) # [C=2, D, H, W]

    return output.permute([0, 3, 1, 2])

def volume_project(vol, pose, ffted=False, plotax=None):

    # fast Fourier transform
    if not ffted:
        # twice size to prevent aliasing
        pad_shape = [math.ceil(s * 2) for s in vol.shape]
        shift_shape = np.array([-math.ceil(s // 2) for s in vol.shape], dtype=np.int32)

        Fvol = scipy.fft.fftn(vol, s=pad_shape, workers=-1) # all cpu
        Fvol = scipy.fft.fftshift(Fvol)
    else:
        Fvol = vol
        shift_shape = None

    Fvol = to_pytorch_complex(Fvol)
    if shift_shape is not None:
        Fvol = pytorch_phase_shift(Fvol, shift_shape) # [2, D, H, W]

    # R = torch.Tensor(euler_to_rotmat(pose[0], pose[1], pose[2])[0])
    R = torch.Tensor(pose_to_rotmat(pose[0], pose[1], pose[2])[0])
    Fvol = pytorch_affine_transform(Fvol, R, interp_mode='nearest') # [2, D, H, W]

    # slice orthogonal to x-axis
    slice_pos = Fvol.shape[1]//2
    Fslice = Fvol[:, slice_pos, :, :]

    # To accelerate, we can first slice then perform shifting back
    # because the slice is aligned with the y-z plane
    if shift_shape is not None:
        Fslice = pytorch_phase_shift(Fslice, -shift_shape[1:]) # [2, D, H, W]

    # optional: translation using phase shift
    if pose[3] != 0 and pose[4] != 0:
        Fslice = pytorch_phase_shift(Fslice, [pose[3], pose[4]])
    
    if plotax is not None:
        pm = plotax.imshow(pytorch_complex_abs(Fslice).cpu().numpy())
        plt.colorbar(pm, ax=plotax)

    Fslice = to_numpy_complex(Fslice)
    Fslice = scipy.fft.ifftshift(Fslice)
    slice = scipy.fft.ifftn(Fslice)

    # remember to crop back
    return slice.real[:vol.shape[1], :vol.shape[2]]

def volume_backproject(img, pose, ffted=False, return_fft=False, return_weights=False, plotax=None):

    # fast Fourier transform
    if not ffted:
        # twice size to prevent aliasing
        pad_shape = [math.ceil(s * 2) for s in img.shape]
        shift_img = [math.ceil(s // 2) for s in img.shape]

        Fimg = scipy.fft.fftn(img, s=pad_shape, workers=-1) # all cpu
        Fimg = scipy.fft.fftshift(Fimg)
    else:
        Fimg = img
        # shift_img = None

    Fimg = to_pytorch_complex(Fimg)

    if plotax is not None:
        pm = plotax.imshow(pytorch_complex_abs(Fimg).cpu().numpy())
        plt.colorbar(pm, ax=plotax)

    # optional: translation using phase shift
    if pose[3] != 0 and pose[4] != 0:
        Fimg = pytorch_phase_shift(Fimg, [-pose[3], -pose[4]])

    # shift image to center
    if shift_img is not None:
        Fimg = pytorch_phase_shift(Fimg, shift_img)

    # construct empty volume (suppose square D = H)
    Fvol = torch.zeros(3, Fimg.shape[1], Fimg.shape[1], Fimg.shape[2]) # [real+imag+mask, D, H, W]

    # put back to the y-z plane
    slice_pos = Fvol.shape[1]//2
    Fvol[0, slice_pos, ...] = Fimg[0] # real
    Fvol[1, slice_pos, ...] = Fimg[1] # imag
    Fvol[2, slice_pos, ...] = 1.      # mask

    # R = torch.Tensor(euler_to_rotmat(pose[0], pose[1], pose[2])[0].T) # inverse
    R = torch.Tensor(pose_to_rotmat(pose[0], pose[1], pose[2])[0].T)
    Fvol = pytorch_affine_transform(Fvol, R, interp_mode='nearest') # [3, D, H, W]
    Fvol, W = Fvol[:2, ...], Fvol[2, ...] # [2, D, H, W], [D, H, W]

    if return_fft:
        Fvol = to_numpy_complex(Fvol)
        if return_weights:
            return Fvol, W.cpu().numpy()
        else:
            return Fvol
    else:
        # shift back
        vol_shape = (img.shape[0], img.shape[0], img.shape[1])
        if shift_img is not None:
            shift_vol = [0]+[math.ceil(s // 2) for s in vol_shape[1:]]
            Fvol = pytorch_phase_shift(Fvol, shift_vol) # [2, D, H, W]
            
        Fvol = to_numpy_complex(Fvol)
        Fvol = scipy.fft.ifftshift(Fvol)
        Fvol = scipy.fft.ifftn(Fvol)
        return Fvol.real[:vol_shape[0], :vol_shape[1], :vol.shape[2]]

# Sanity check
if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 3))
    ax0, ax1, ax2 = fig.subplots(1, 3)

    ax0.imshow(img, cmap='gray')
    ax0.axis('off')

    pad_shape = [s * 2 for s in img.shape]
    shift_shape = np.array([-math.ceil(s // 2) for s in img.shape], dtype=np.int32)
    Fimg = scipy.fft.fftn(img, s=pad_shape, workers=-1)
    Fimg = scipy.fft.fftshift(Fimg)
    # Fimg = to_pytorch_complex(Fimg)
    # Fimg = pytorch_phase_shift(Fimg, shift_shape)

    Fvol = np.zeros((Fimg.shape[0], Fimg.shape[0], Fimg.shape[1]), dtype=np.complex64)
    Fvol[Fvol.shape[0]//2, :, :] = Fimg

    Fslice = Fvol[Fvol.shape[0]//2, :, :]
    # Fslice = pytorch_phase_shift(Fimg, -shift_shape)
    Fslice = scipy.fft.ifftshift(Fslice)
    slice = scipy.fft.ifftn(Fslice).real
    ax1.imshow(slice[:img.shape[0], :img.shape[1]], cmap='gray')
    ax1.axis('off')

    R = torch.Tensor(euler_to_rotmat(75.676, 8.5, 230.5)[0])
    Fvol = to_pytorch_complex(Fvol)
    Fvol = pytorch_affine_transform(Fvol, R.T) # [2, D, H, W]
    Fvol = pytorch_affine_transform(Fvol, R) # [2, D, H, W]
    Fvol = to_numpy_complex(Fvol)

    Fslice = Fvol[Fvol.shape[0]//2, :, :]
    # Fslice = pytorch_phase_shift(Fimg, -shift_shape)
    Fslice = scipy.fft.ifftshift(Fslice)
    slice = scipy.fft.ifftn(Fslice).real
    ax2.imshow(slice[:img.shape[0], :img.shape[1]], cmap='gray')
    ax2.axis('off')