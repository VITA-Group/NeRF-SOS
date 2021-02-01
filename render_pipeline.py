import os, sys
import math, time, random

import numpy as np

import imageio
import json
import mrc

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from net.nerf_net import NerfNet
from net.vote_net import VoteNet, polar_to_rotmat, polar_to_xyz

from data.collater import Image_Batch_Collate

from utils.image_utils import *
from utils.ray_utils import get_ortho_rays
from utils.render_collector import SaveImageCollector, SaveVideoCollector

def render_pipeline(model, render_set, args, collector=None, idxs=None, test=False, **render_kwargs):
    H, W = args.H, args.W
    collate_fn = Image_Batch_Collate(H, W)

    ret = {} # Return data

    # generate indexing array
    if idxs is not None:
        if isinstance(idxs, int): idxs = [idxs]
        indices = np.concatenate([np.arange(H*W*idx, H*W*(idx+1)) for idx in idxs], -1)
        render_set = torch.utils.data.Subset(render_set, indices)

    # Large batch size is okay thanks to the internal batchifying
    render_loader = torch.utils.data.DataLoader(render_set, batch_size=args.H*args.W, shuffle=False, 
                                                collate_fn=collate_fn, pin_memory=args.pin_memory)
    for i, (batch_rays, _) in enumerate(tqdm(render_loader)):
        batch_rays = batch_rays.to(args.device)

        # Run nerf
        ret_dict = model(batch_rays, (args.near, args.far), test=test, **render_kwargs)
        for k, v in ret_dict.items(): ret_dict[k] = v.detach().cpu().numpy()
        ret_dict['batch_num'] = i
        
        # call collector if specified
        if collector is not None:
            collector(ret_dict)

def exhibit_pipeline(model, render_sets, args):

    with torch.no_grad():
        # Obtain the ID of used checkpoint
        _, ckpt = os.path.split(args.ckpt_file) # remove directory
        ckpt, _ = os.path.splitext(ckpt) # remove extension

        # Render test set or orbitary camera
        testsavedir = os.path.join(args.run_dir, f'exhibit_test_{ckpt}')
        os.makedirs(testsavedir, exist_ok=True)
        render_pipeline(model, render_sets['test'], args, collector=SaveImageCollector(testsavedir), test=True)
        print('Done saving', testsavedir)

        # Saving smoother path
        # if not args.render_test:
        #     export_video(rgbs, os.path.join(testsavedir, 'video.mp4'))
        # print('Done export video', testsavedir)

        # 2020.7.20 Also save all train set to check overfitting
        trainsavedir = os.path.join(args.run_dir, f'exhibit_train_{ckpt}')
        os.makedirs(trainsavedir, exist_ok=True)
        render_pipeline(model, render_sets['train'], args, collector=SaveImageCollector(trainsavedir), test=True)
        print('Done saving', trainsavedir)


def video_pipeline(model, render_set, args):

    with torch.no_grad():
        # Obtain the ID of used checkpoint
        _, ckpt = os.path.split(args.ckpt_file) # remove directory
        ckpt, _ = os.path.splitext(ckpt) # remove extension

        # Render test set or orbitary camera
        savedir = os.path.join(args.run_dir, f'exhibit_video_{ckpt}')
        os.makedirs(savedir, exist_ok=True)

        # TO-DO
        collector = SaveVideoCollector(savedir, fps=30, quality=8, save_frames=True)
        render_pipeline(model, render_set, args, collector=collector, test=True)
        collector.export_video('video.mp4')
        print('Done saving', savedir)

        # Saving smoother path
        # export_video(rgbs, os.path.join(savedir, 'video.mp4'))
        # print('Done export video', savedir)

def export_pipeline(model, args, suffix, c2w=None, render_set=None, idx=0):
    """
    Export volume by query MLP
    suffix: any information need to append into the directory name
    c2w: specify querying rays' viewing point. Default: Identity
    render_set & idx: querying rays at index idx from a render_set.
    """

    # Recalculate samples for extracting volume
    ps, us = args.pixel_size, args.unit_size
    H, W = args.H, args.W
    N_samples = round((args.far - args.near) * us / ps)
    print('Recalculate N_samples: %d' % N_samples)

    with torch.no_grad():
        if render_set is not None:
            subset = torch.utils.data.Subset(render_set, np.arange(H*W*idx, H*W*(idx+1)))
            collate_fn = Image_Batch_Collate(H, W)
            render_loader = torch.utils.data.DataLoader(subset, batch_size=H*W, shuffle=False,
                                                        collate_fn=collate_fn, pin_memory=args.pin_memory)
            batch_rays, _ = next(iter(render_loader))
        else:
            # Use identity by default
            if c2w is None:
                c2w = torch.cat([torch.eye(3), torch.Tensor([0, 0, max(H, W) / 2.0 * ps / us]).view(-1, 1)], -1)
            rays_exhibit = get_ortho_rays(H, W, c2w, ps, us)[None, ...]
            batch_rays = rays_exhibit[0]

        batch_rays = batch_rays.to(args.device)
        # Query data: no importance sampling. ONLY uniform samplings (N_importance=0).
        ret = model(batch_rays, (args.near, args.far), test=True, retraw=True, retpts=True, N_samples=N_samples, N_importance=0)
        rgb, raw, pts = ret['rgb'], ret['raw'], ret['pts']
        print('Rerender:', rgb.shape)
        print('Volume:', raw.shape)
        print('Sample points:', pts.shape)

        # Obtain the ID of used checkpoint
        _, ckpt = os.path.split(args.ckpt_file) # remove directory
        ckpt, _ = os.path.splitext(ckpt) # remove extension

        savedir = os.path.join(args.run_dir, f'export_{ckpt}_{suffix}')
        os.makedirs(savedir, exist_ok=True)
        
        # Saving raw
        export_images([rgb.cpu().numpy()], savedir)
        print('Done rerendering', os.path.join(savedir), raw.shape)
        np.save(os.path.join(savedir, 'raw.npy'), raw.cpu().numpy())
        print('Done exporting', os.path.join(savedir, 'raw.npy'))
        np.save(os.path.join(savedir, 'pts.npy'), pts.cpu().numpy())
        print('Done exporting', os.path.join(savedir, 'pts.npy'))

        # Export mrc
        vol = raw[..., -1]
        meta_data = {
            'm': np.array(vol.shape, dtype=np.float32),
            'd': np.array(vol.shape, dtype=np.float32) * args.pixel_size,
            'wave': np.array([0., 0., 0., 0., 0.]),

            # copy from relion_locres_filtered.mrc
            'zxy0': np.array([1.7639892e-19, 2.3412895e-41, 2.6452156e-02]),

            # title
            'NumTitles': 1,
            'title': f'{args.expname}_{ckpt}_{suffix}'
        }
        mrc.imwrite(os.path.join(savedir, 'raw.mrc'), vol.cpu().numpy(), metadata=meta_data)
        print('Done exporting', os.path.join(savedir, 'raw.mrc'))

def search_for_campose(model, args, a, z):

    # Recalculate samples for extracting volume
    ps, us = args.pixel_size, args.unit_size
    H, W = args.H, args.W

    with torch.no_grad():
        print(a, z)
        a, z = torch.Tensor([a]), torch.Tensor([z])
        c2w = polar_to_rotmat(a, z)
        batch_rays = get_ortho_rays(H, W, c2w, ps, us)        
        batch_rays = batch_rays.to(args.device)
        # Query data: no importance sampling. ONLY uniform samplings (N_importance=0).
        ret = model(batch_rays, (args.near, args.far), test=True, retraw=True, retpts=True)
        rgb, raw, pts = ret['rgb'], ret['raw'], ret['pts']
        rgb = rgb.reshape(1, -1, rgb.shape[-1])
        # print(rgb.shape)

        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
        i, j = i.t(), j.t()

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.stack([torch.zeros_like(i), torch.zeros_like(i), torch.ones_like(i)], -1)
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = torch.stack([(i-W*.5)*ps/us, -(j-H*.5)*ps/us, torch.zeros_like(i)], -1)

        # print(rays_o.shape, rays_d.shape)
        rays_o = rays_o.reshape(1, -1, rays_o.shape[-1])
        rays_d = rays_o.reshape(1, -1, rays_d.shape[-1])

        # print(rays_o.shape[1]//8)
        i_sample = np.random.choice(rays_o.shape[1], size=1024, replace=False)

        votenet = VoteNet(args, model)
        votes = votenet(rays_o[:, i_sample, ...], rays_d[:, i_sample, ...], rgb[:, i_sample, ...])

        print(votes)
