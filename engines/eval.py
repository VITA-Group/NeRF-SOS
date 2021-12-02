import os, sys
import math, time, random

import numpy as np

import imageio
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from utils.image import to8b, img2mse, mse2psnr, ssim, lpips
from utils.ray import get_ortho_rays

def eval_one_view(model, batch, near_far, device, **render_kwargs):

    model.eval()

    near, far = near_far
    with torch.no_grad():
        batch_rays = batch['rays'].to(device)

        # Run nerf
        ret_dict = model(batch_rays, (near, far), **render_kwargs)
        for k, v in ret_dict.items():
            ret_dict[k] = v.cpu()

        metric_dict = {}
        if 'target_s' in batch:
            target_s = batch['target_s']
            ret_dict['target_s'] = target_s

            mse = img2mse(ret_dict['rgb'], target_s)
            metric_dict['mse'] = mse
            metric_dict['psnr'] = mse2psnr(mse)
            metric_dict['ssim'] = ssim(ret_dict['rgb'], target_s, format='HWC')
            metric_dict['lpips'] = lpips(ret_dict['rgb'], target_s, format='HWC')

        return ret_dict, metric_dict

def evaluate(model, dataset, device, save_dir=None, **render_kwargs):

    near, far = dataset.near_far()

    all_metrics = {}
    for i, batch in enumerate(dataset):
        ret_dict, metric_dict = eval_one_view(model, batch, (near, far), device=device, **render_kwargs)

        for k in ['mse', 'psnr', 'ssim', 'lpips']:
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(metric_dict[k].item())

        img, disp, alpha = ret_dict['rgb'].numpy(), ret_dict['disp'].numpy(), ret_dict['acc'].numpy()
        print(f"[TEST] Iter {i+1}/{len(dataset)} MSE: {metric_dict['mse'].item()} PSNR: {metric_dict['psnr'].item()} "
            f"SSIM: {metric_dict['ssim'].item()} LPIPS: {metric_dict['lpips'].item()}")

        if save_dir is not None:
            np.save(os.path.join(save_dir, f'rgb_{i:03d}.npy'), img)
            imageio.imwrite(os.path.join(save_dir, f'rgb_{i:03d}.png'), to8b(img))

            np.save(os.path.join(save_dir, f'disp_{i:03d}.npy'), disp)
            imageio.imwrite(os.path.join(save_dir, f'disp_{i:03d}.png'), to8b(disp / np.max(disp)))

            np.save(os.path.join(save_dir, f'alpha_{i:03d}.npy'), alpha)
            imageio.imwrite(os.path.join(save_dir, f'alpha_{i:03d}.png'), to8b(alpha / np.max(alpha)))

    total_mse = np.array(all_metrics['mse']).mean()
    total_psnr = mse2psnr(total_mse)
    total_ssim = np.array(all_metrics['ssim']).mean()
    total_lpips = np.array(all_metrics['lpips']).mean()
    print(f'[TEST] MSE: {total_mse.item()} PSNR: {total_psnr.item()} SSIM: {total_ssim} LPIPS: {total_lpips}')

    all_metrics['total_mse'] = total_mse.item()
    all_metrics['total_psnr'] = total_psnr.item()
    all_metrics['total_ssim'] = total_ssim
    all_metrics['total_lpips'] = total_lpips

    if save_dir is not None:
        with open(os.path.join(save_dir, 'log.json'), 'w') as f:
            json.dump(all_metrics, f)
        with open(os.path.join(save_dir, 'log.txt'), 'w') as f:
            for i in range(len(all_metrics['mse'])):
                print(f"[TEST] Iter {i+1}/{len(dataset)} MSE: {all_metrics['mse'][i]} PSNR: {all_metrics['psnr'][i]} "
                    f"SSIM: {all_metrics['ssim'][i]} LPIPS: {all_metrics['lpips'][i]}", file=f)
            print(f"[TEST] MSE: {all_metrics['total_mse']} PSNR: {all_metrics['total_psnr']} "
                f"SSIM: {all_metrics['total_ssim']} LPIPS: {all_metrics['total_lpips']}", file=f)

    return {'mse': total_mse.item(), 'psnr': total_psnr.item(), 'ssim': total_ssim, 'lpips': total_lpips}

def render_video(model, dataset, device, save_dir, suffix='', fps=30, quality=8, **render_kwargs):

    near, far = dataset.near_far()

    rgbs, disps = [], []
    for i, batch in enumerate(tqdm(dataset, desc='Rendering')):
        ret_dict, metric_dict = eval_one_view(model, batch, (near, far), device=device, **render_kwargs)
        rgbs.append(ret_dict['rgb'])
        disps.append(ret_dict['disp'])

    rgb_video = torch.stack(rgbs, 0).numpy()
    np.save(os.path.join(save_dir, f"rgb{'_' + suffix if suffix else ''}.npy"), rgb_video)
    imageio.mimwrite(os.path.join(save_dir, f"rgb{'_' + suffix if suffix else ''}.mp4"),
        to8b(rgb_video), fps=fps, quality=quality)

    disp_video = torch.stack(disps, 0).numpy()
    np.save(os.path.join(save_dir, f"disp{'_' + suffix if suffix else ''}.npy"), disp_video)
    imageio.mimwrite(os.path.join(save_dir, f"disp{'_' + suffix if suffix else ''}.mp4"),
        to8b(disp_video / np.max(disp_video)), fps=fps, quality=quality)

def export_density(model, extents=(2.0, 2.0, 2.0), voxel_size=2./256., save_dir='', device=torch.device('cpu')):
    model.eval()

    with torch.no_grad():
        h, w, d = extents
        pts = torch.stack(torch.meshgrid(
            torch.linspace(-w/2, w/2, int(w/voxel_size)),
            torch.linspace(-h/2, h/2, int(h/voxel_size)),
            torch.linspace(-d/2, d/2, int(d/voxel_size))
        ), dim=-1).to(device).float() # [W, H, D, 3]
        viewdirs = torch.zeros_like(pts).to(device)
        raw = model.nerf_fine(pts, viewdirs=viewdirs)
        sigma = torch.maximum(raw[..., -1], torch.zeros_like(raw[..., -1]))
        sigma = sigma.cpu().numpy()
        if save_dir:
            np.save(os.path.join(save_dir, 'density.npy'), sigma)

            import mrc
            mrc.imsave(os.path.join(save_dir, 'density.mrc'), sigma)

        return sigma
