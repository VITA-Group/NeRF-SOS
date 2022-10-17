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

from utils.image import to8b, img2mse, mse2psnr, ssim, lpips, color_pallete, colorize_np
from utils.ray import get_ortho_rays
from pdb import set_trace as st
from sklearn.metrics import adjusted_rand_score
from utils.misc import segmap_cluster
from models.extractor import VitExtractor
import cv2

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    # batch = batch.div_(255.0)
    return (batch - mean) / std

def eval_one_view(model, batch, near_far, radii, device, clus_no_sfm=False, N_cluster=2, **render_kwargs):

    model.eval()

    near, far = near_far
    with torch.no_grad():
        batch_rays = batch['rays'].to(device)

        # Run nerf
        ret_dict = model(batch_rays, (near, far), radii=radii, **render_kwargs)
        for k, v in ret_dict.items():
            ret_dict[k] = v.cpu()

        if 'semantics' in ret_dict.keys():
            if 'masks' in batch.keys():
                sem_gt = batch['masks'].cpu().numpy()
            else:
                print("[Warning!] masks gt not in batch.")
                sem_gt = np.zeros_like(ret_dict['disp'])
            
            if clus_no_sfm:
                sem_prob = ret_dict['semantics'].detach().cpu().float()
                sem_pred_sft = torch.argmax(sem_prob.softmax(dim=-1), -1).unsqueeze(-1).numpy()
            else:
                sem_prob = ret_dict['semantics'].detach().cpu().float().softmax(dim=-1)
                sem_pred_sft = torch.argmax(sem_prob, -1).unsqueeze(-1).numpy()
            sem_pred_clus = segmap_cluster(sem_prob, n_clusters=N_cluster)

            sem_gt, sem_pred_clus, sem_pred_sft = \
                sem_gt.astype(np.int32), sem_pred_clus.astype(np.int32), sem_pred_sft.astype(np.int32)
            
            print(sem_pred_sft.shape, sem_pred_clus.shape)
            ret_dict["sem"] = sem_pred_sft
            ret_dict["clustering"] = sem_pred_clus
            fg_idx = sem_gt==1
            clus_ari = adjusted_rand_score(sem_gt.reshape(-1), sem_pred_clus.reshape(-1))
            clus_ari_fg = adjusted_rand_score(sem_gt[fg_idx].reshape(-1), sem_pred_clus[fg_idx].reshape(-1))
            sem_ari= adjusted_rand_score(sem_gt.reshape(-1), sem_pred_sft.reshape(-1))
            sem_ari_fg = adjusted_rand_score(sem_gt[fg_idx].reshape(-1), sem_pred_sft[fg_idx].reshape(-1))

            clus_ari = torch.Tensor([clus_ari]).to(batch_rays.device)
            clus_ari_fg = torch.Tensor([clus_ari_fg]).to(batch_rays.device)
            sem_ari = torch.Tensor([sem_ari]).to(batch_rays.device)
            sem_ari_fg = torch.Tensor([sem_ari_fg]).to(batch_rays.device)
        else:
            clus_ari = clus_ari_fg = sem_ari = sem_ari_fg = torch.Tensor([0]).to(batch_rays.device)
        
        metric_dict = {}
        if 'target_s' in batch:
            target_s = batch['target_s']
            ret_dict['target_s'] = target_s

            mse = img2mse(ret_dict['rgb'], target_s)
            metric_dict['mse'] = mse
            metric_dict['psnr'] = mse2psnr(mse)
            metric_dict['ssim'] = ssim(ret_dict['rgb'], target_s, format='HWC')
            metric_dict['lpips'] = lpips(ret_dict['rgb'], target_s, format='HWC')
            metric_dict['clus_ari'] = clus_ari
            metric_dict['clus_ari_fg'] = clus_ari_fg
            metric_dict['sem_ari'] = sem_ari
            metric_dict['sem_ari_fg'] = sem_ari_fg

        return ret_dict, metric_dict


def evaluate(model, dataset, device, save_dir=None, fast_mode=False, ret_cluster=False, clus_no_sfm=False, N_cluster=2, find_fg=True,  **render_kwargs):

    near, far = dataset.near_far()
    radii = dataset.radii()

    if find_fg:
        dino = VitExtractor(model_name='dino_vits16', device=device)

    all_metrics = {}
    for i, batch in enumerate(dataset):
        if fast_mode:
            if i >= 1:
                continue
        ret_dict, metric_dict = eval_one_view(model, batch, (near, far), radii=radii, device=device, clus_no_sfm=clus_no_sfm, N_cluster=N_cluster, **render_kwargs)

        for k in ['mse', 'psnr', 'ssim', 'lpips', 'clus_ari', 'clus_ari_fg', 'sem_ari', 'sem_ari_fg']:
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(metric_dict[k].item())

        img, disp, alpha, depth = ret_dict['rgb'].numpy(), ret_dict['disp'].numpy(), ret_dict['acc'].numpy(), ret_dict['depth'].numpy()
        # if 'semantics' in ret_dict.keys():
        #     sem = ret_dict['semantics'].softmax(dim=-1)
        #     sem = torch.argmax(sem, -1)
        #     sem = sem.unsqueeze(-1).cpu().numpy()
        #     if ret_cluster:
        #         if clus_no_sfm:
        #             clustering = ret_dict['semantics']
        #         else:
        #             clustering = ret_dict['semantics'].softmax(dim=-1)
        #         clustering = clustering.cpu().numpy()
        #         clustering = segmap_cluster(clustering, n_clusters=N_cluster)
                # print("> clustering.shape:", clustering.shape)
                # clustering = seg2color(clustering.squeeze(-1), color_pallete)
        
        clustering = ret_dict["clustering"]
        sem = ret_dict["sem"]
        if find_fg:
            dino_in = ret_dict['rgb'].to(device)
            dino_in = dino_in.unsqueeze(0).permute(0, 3, 1, 2)
            dino_in = normalize_batch(dino_in)
            dino_ret = dino.get_vit_attn_feat_noresize(dino_in)
            attn = dino_ret['attn']
            attn = attn.reshape([1, 1, dino_in.shape[2] // 16, dino_in.shape[3] // 16 ])
            attn = F.interpolate(attn, (dino_in.shape[2], dino_in.shape[3]))
            attn = attn.permute(0, 2, 3, 1).squeeze(0)
            attn = attn.detach().cpu().numpy()
            if np.mean(attn[clustering==1]) < np.mean(attn[clustering==0]):
                clustering = np.ones_like(clustering) - clustering
        print()
        print(np.unique(clustering), np.sum(clustering==0), np.sum(clustering==1))
        print(np.unique(sem), np.sum(sem==0), np.sum(sem==1))
        clustering = (clustering * 255).astype(np.uint8)
        sem = (sem * 255).astype(np.uint8)
        # clustering = seg2color(clustering.squeeze(-1), color_pallete)
        # sem = seg2color(sem.squeeze(-1), color_pallete)
        print(np.unique(clustering), np.sum(clustering==0), np.sum(clustering==1))
        print(np.unique(sem), np.sum(sem==0), np.sum(sem==1))


        print(f"[TEST] Iter {i+1}/{len(dataset)} MSE: {metric_dict['mse'].item()} PSNR: {metric_dict['psnr'].item()} "
            f"SSIM: {metric_dict['ssim'].item()} LPIPS: {metric_dict['lpips'].item()}"
             f"clus_ari: {metric_dict['clus_ari'].item()} clus_ari_fg: {metric_dict['clus_ari_fg'].item()}"
             f"sem_ari: {metric_dict['sem_ari'].item()} sem_ari_fg: {metric_dict['sem_ari_fg'].item()}")

        if save_dir is not None:
            # np.save(os.path.join(save_dir, f'rgb_{i:03d}.npy'), img)
            imageio.imwrite(os.path.join(save_dir, f'rgb_{i:03d}.png'), to8b(img))

            # np.save(os.path.join(save_dir, f'depth_{i:03d}.npy'), depth)
            imageio.imwrite(os.path.join(save_dir, f'depth_{i:03d}.png'), to8b(depth / np.max(depth)))
            depth = depth.squeeze(-1)
            depth = colorize_np(depth, cmap_name='jet', append_cbar=True)
            imageio.imwrite(os.path.join(save_dir, f'depth_{i:03d}_.png'), to8b(depth / np.max(depth)))

            # np.save(os.path.join(save_dir, f'alpha_{i:03d}.npy'), alpha)
            imageio.imwrite(os.path.join(save_dir, f'alpha_{i:03d}.png'), to8b(alpha / np.max(alpha)))

            if 'semantics' in ret_dict.keys():
                imageio.imwrite(os.path.join(save_dir, f'sem_{i:03d}.png'), sem)
            if ret_cluster:
                imageio.imwrite(os.path.join(save_dir, f'clus_{i:03d}.png'), clustering)
                cv2.imwrite(os.path.join(save_dir, f'clus_{i:03d}_cv2.png'), clustering)

    total_mse = np.array(all_metrics['mse']).mean()
    total_psnr = mse2psnr(total_mse)
    total_ssim = np.array(all_metrics['ssim']).mean()
    total_lpips = np.array(all_metrics['lpips']).mean()
    total_lpips = np.array(all_metrics['lpips']).mean()
    total_clus_ari = np.array(all_metrics['clus_ari']).mean()
    total_clus_ari_fg = np.array(all_metrics['clus_ari_fg']).mean()
    total_sem_ari = np.array(all_metrics['sem_ari']).mean()
    total_sem_ari_fg = np.array(all_metrics['sem_ari_fg']).mean()

    print(f'[TEST] MSE: {total_mse.item()} PSNR: {total_psnr.item()} SSIM: {total_ssim} LPIPS: {total_lpips}'
    f'total_clus_ari: {total_clus_ari.item()}, total_clus_ari_fg: {total_clus_ari_fg.item()}, total_sem_ari: {total_sem_ari.item()}, total_sem_ari_fg: {total_sem_ari_fg.item()}')

    all_metrics['total_mse'] = total_mse.item()
    all_metrics['total_psnr'] = total_psnr.item()
    all_metrics['total_ssim'] = total_ssim
    all_metrics['total_lpips'] = total_lpips
    all_metrics['total_clus_ari'] = total_clus_ari.item()
    all_metrics['total_clus_ari_fg'] = total_clus_ari_fg.item()
    all_metrics['total_sem_ari'] = total_sem_ari.item()
    all_metrics['total_sem_ari_fg'] = total_sem_ari_fg.item()

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


def render_video(model, dataset, device, save_dir, suffix='', fps=30, quality=8, 
    ret_cluster=True, fast_mode=False, clus_no_sfm=False, N_cluster=2, find_fg=True, **render_kwargs):

    near, far = dataset.near_far()
    radii = dataset.radii()

    if find_fg:
        dino = VitExtractor(model_name='dino_vits16', device=device)

    rgbs, disps, sems = [], [], []
    if ret_cluster:
        clusters = []
    print("> fast_mode:", fast_mode)
    for i, batch in enumerate(tqdm(dataset, desc='Rendering')):
        # if fast_mode:
            # if i >= 2:
            #     continue
        ret_dict, metric_dict = eval_one_view(model, batch, (near, far), radii=radii, device=device, clus_no_sfm=clus_no_sfm, N_cluster=N_cluster, **render_kwargs)
        if 'sem' in ret_dict.keys():
            sems.append(ret_dict['sem'])
        if ret_cluster:
            clustering = ret_dict['clustering']
            if find_fg:
                dino_in = ret_dict['rgb'].to(device)
                dino_in = dino_in.unsqueeze(0).permute(0, 3, 1, 2)
                dino_in = normalize_batch(dino_in)
                dino_ret = dino.get_vit_attn_feat_noresize(dino_in)
                attn = dino_ret['attn']
                attn = attn.reshape([1, 1, dino_in.shape[2] // 16, dino_in.shape[3] // 16 ])
                attn = F.interpolate(attn, (dino_in.shape[2], dino_in.shape[3]))
                attn = attn.permute(0, 2, 3, 1).squeeze(0)
                attn = attn.detach().cpu().numpy()
                if np.mean(attn[clustering==1]) < np.mean(attn[clustering==0]):
                    clustering = np.ones_like(clustering) - clustering
                
            clusters.append(clustering)

        rgbs.append(ret_dict['rgb'])
        disps.append(ret_dict['disp'])

    rgb_video = torch.stack(rgbs, 0).numpy()
    # np.save(os.path.join(save_dir, f"rgb{'_' + suffix if suffix else ''}.npy"), rgb_video)
    imageio.mimwrite(os.path.join(save_dir, f"rgb{'_' + suffix if suffix else ''}.mp4"),
        to8b(rgb_video), fps=fps, quality=quality)

    disp_video = torch.stack(disps, 0).numpy()
    # np.save(os.path.join(save_dir, f"disp{'_' + suffix if suffix else ''}.npy"), disp_video)
    imageio.mimwrite(os.path.join(save_dir, f"disp{'_' + suffix if suffix else ''}.mp4"),
        to8b(disp_video / np.max(disp_video)), fps=fps, quality=quality)

    if 'semantics' in ret_dict.keys():
        # sem_video = torch.stack(sems, 0)
        # sem_video = torch.argmax(sem_video.softmax(dim=-1), -1).float().numpy()
        sem_video = np.stack(sems, 0)
        imageio.mimwrite(os.path.join(save_dir, f"sem{'_' + suffix if suffix else ''}.mp4"),
            to8b(sem_video), fps=fps, quality=quality)
    if ret_cluster:
        clust_video = (np.stack(clusters, 0) * 255).astype(np.uint8)
        imageio.mimwrite(os.path.join(save_dir, f"clus{'_' + suffix if suffix else ''}.mp4"),
            clust_video, fps=fps, quality=quality)


def seg2color(seg, color_pallete=color_pallete):
    '''seg has to be HxW dimension
    '''
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in color_pallete.items():
        color_seg[seg == label, :] = color
    return color_seg

def export_density(model, extents=(2.0, 2.0, 2.0), voxel_size=2./256., save_dir='', device=torch.device('cpu')):
    model.eval()

    with torch.no_grad():
        h, w, d = extents
        pts = torch.stack(torch.meshgrid(
            torch.linspace(-w/2, w/2, int(w/voxel_size)),
            torch.linspace(-h/2, h/2, int(h/voxel_size)),
            torch.linspace(-d/2, d/2, int(d/voxel_size))
        ), dim=-1).to(device).float() # [W, H, D, 3]
        pts = pts * 14
        viewdirs = torch.zeros_like(pts).to(device)
        raw = model.nerf_fine(pts, viewdirs=viewdirs)
        sigma = torch.maximum(raw[..., -1], torch.zeros_like(raw[..., -1]))
        sigma = sigma.cpu().numpy()
        if save_dir:
            # np.save(os.path.join(save_dir, 'density.npy'), sigma)

            import mrc
            mrc.imsave(os.path.join(save_dir, 'density.mrc'), sigma)
            vis_percentage(sigma, thres_percent=None, path=os.path.join(save_dir, 'density.ply'))

        return sigma


def vis_percentage(alpha, thres_percent=99, path="logs/1.ply"):
    import open3d as o3d
    if alpha.shape[0] < alpha.shape[-1]:
        alpha = np.transpose(alpha, (1,2,0))
    if thres_percent is not None:
        thres = np.percentile(alpha, thres_percent)
    else:
        thres = 1e-6
    print(f"[Volume Activate Rate:] {(alpha > thres).mean()}")
    xyz_min = np.array([0,0,0])
    xyz_max = np.array(alpha.shape)
    cam_frustrm_lst = []
    aabb_01 = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 0]])
    out_bbox = o3d.geometry.LineSet()
    out_bbox.points = o3d.utility.Vector3dVector(xyz_min + aabb_01 * (xyz_max - xyz_min))
    out_bbox.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(12)])
    out_bbox.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])

    xyz = np.stack((alpha > thres).nonzero(), -1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz / alpha.shape * (xyz_max - xyz_min) + xyz_min)
    # pcd.colors = o3d.utility.Vector3dVector(color[:, :3])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=max((xyz_max - xyz_min) / alpha.shape))

    # Save as ply
    print(f"[PLY Saving:], saving ply file in {path}")
    o3d.io.write_voxel_grid(path, voxel_grid)

