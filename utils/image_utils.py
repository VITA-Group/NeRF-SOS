
import os, sys
import math, random, time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import imageio

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def to8b(x):
    return (255*(x-x.min())/(x.max()-x.min())).astype(np.uint8)

def export_images(rgbs, save_dir, H=0, W=0):
    rgb8s = []
    for i, rgb in enumerate(rgbs):
        # Resize
        if H > 0 and W > 0:
            rgb = rgb.reshape([H, W])

        filename = os.path.join(save_dir, '{:03d}.npy'.format(i))
        np.save(filename, rgb)
        
        # Convert to image
        rgb8 = to8b(rgb)
        filename = os.path.join(save_dir, '{:03d}.png'.format(i))
        imageio.imwrite(filename, rgb8)
        rgb8s.append(rgb8)
    
    return np.stack(rgb8s, 0)

def export_video(rgbs, save_path, fps=30, quality=8):
    imageio.mimwrite(save_path, to8b(rgbs), fps=fps, quality=quality)