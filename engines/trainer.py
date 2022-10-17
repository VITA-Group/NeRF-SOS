import os, sys, copy
import math, time, random

import numpy as np

import imageio
import json
from sacrebleu import dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.image import to8b, img2mse, mse2psnr
from engines.eval import eval_one_view, evaluate, render_video
from pdb import set_trace as st
from torchvision import transforms as pth_transforms
from utils.image import get_similarity_matrix
from sklearn.metrics import adjusted_rand_score
from utils.misc import segmap_cluster
from engines.eval import seg2color

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    # batch = batch.div_(255.0)
    return (batch - mean) / std


def train_one_step(batch, model, optimizer, scheduler, train_loader, global_step, losses, device, args):
    
    model, dino = model
    seg_loss, contrast_loss, correlation_loss, geoCorrelation_loss = losses

    model.train()

    near, far = train_loader.dataset.near_far()
    radii = train_loader.dataset.radii()
    img_h, img_w = train_loader.dataset.height, train_loader.dataset.width

    # make sure on cuda
    # if patch_tune is True: batch_rays.shape, gt.shape, masks.shape torch.Size([1, 1024, 2, 3]), torch.Size([1, 1024, 3]), torch.Size([1, 1024, 1])
    # if patch_tune is False: batch_rays.shape, gt.shape, masks.shape torch.Size([2, 1024, 3]), torch.Size([1024, 3]), torch.Size([1024, 1])
    if len(batch) == 3:
        batch_rays, gt, masks = batch
        batch_rays, gt, masks = batch_rays.to(device), gt.to(device), masks.to(device)
    elif len(batch) == 4:
        batch_rays, gt, masks, poses = batch
        batch_rays, gt, masks, poses = batch_rays.to(device), gt.to(device), masks.to(device), poses.to(device)
    elif len(batch) == 2:
        batch_rays, gt = batch
        batch_rays, gt = batch_rays.to(device), gt.to(device)
    elif len(batch) == 5:
        batch_rays, gt, masks, poses, start_idx = batch
        batch_rays, gt, masks, poses, start_idx = batch_rays.to(device), gt.to(device), masks.to(device), poses.to(device), start_idx.to(device)
    else:
        raise RuntimeError

    if args.patch_tune:
        # if batch_size=1, gt.shape: torch.Size([1, 1024, 3])
        batch_rays, gt, masks = batch_rays.reshape(-1, *batch_rays.shape[2:]), gt.reshape(-1, *gt.shape[2:]), masks.reshape(-1, *masks.shape[2:])
        batch_rays = batch_rays.permute(1, 0, 2)
    

    #####  Core optimization loop  #####
    ret_dict = model(batch_rays, (near, far), radii=radii) # no extraction
    if args.patch_tune:
        # gt: [2048, 3] > [2, 1024, 3]
        gt = torch.stack(torch.chunk(gt, args.batch_size, dim=0)).squeeze(1)
        gt = gt.reshape([gt.shape[0], args.patch_size, args.patch_size, *gt.shape[2:]])
        masks = torch.stack(torch.chunk(masks, args.batch_size, dim=0)).squeeze(1)
        masks = masks.reshape([masks.shape[0], args.patch_size, args.patch_size, *masks.shape[2:]])
        ret_dict['rgb'] = torch.stack(torch.chunk(ret_dict['rgb'], args.batch_size, dim=0)).squeeze(1)
        ret_dict['rgb'] = ret_dict['rgb'].reshape([ret_dict['rgb'].shape[0], args.patch_size, args.patch_size, *ret_dict['rgb'].shape[2:]])

        ret_dict['rgb0'] = torch.stack(torch.chunk(ret_dict['rgb0'], args.batch_size, dim=0)).squeeze(1)
        ret_dict['rgb0'] = ret_dict['rgb0'].reshape([ret_dict['rgb0'].shape[0], args.patch_size, args.patch_size, *ret_dict['rgb0'].shape[2:]])

        ret_dict['depth0'] = torch.stack(torch.chunk(ret_dict['depth0'], args.batch_size, dim=0)).squeeze(1)
        ret_dict['depth0'] = ret_dict['depth0'].reshape([ret_dict['depth0'].shape[0], args.patch_size, args.patch_size, *ret_dict['depth0'].shape[2:]])

        ret_dict['depth'] = torch.stack(torch.chunk(ret_dict['depth'], args.batch_size, dim=0)).squeeze(1)
        ret_dict['depth'] = ret_dict['depth'].reshape([ret_dict['depth'].shape[0], args.patch_size, args.patch_size, *ret_dict['depth'].shape[2:]])

        ray_o, ray_d = batch_rays
        ray_o = torch.stack(torch.chunk(ray_o, args.batch_size, dim=0)).squeeze(1)
        ray_o = ray_o.reshape([ray_o.shape[0], args.patch_size, args.patch_size, *ray_o.shape[2:]])
        ray_d = torch.stack(torch.chunk(ray_d, args.batch_size, dim=0)).squeeze(1)
        ray_d = ray_d.reshape([ray_d.shape[0], args.patch_size, args.patch_size, *ray_d.shape[2:]])

        if 'semantics' in ret_dict:
            #(B*N, C) > (B, Patch, Patch, C)
            ret_dict['semantics'] = torch.stack(torch.chunk(ret_dict['semantics'], args.batch_size, dim=0)).squeeze(1)
            ret_dict['semantics'] = ret_dict['semantics'].reshape([ret_dict['semantics'].shape[0], args.patch_size, args.patch_size, *ret_dict['semantics'].shape[2:]])
            ret_dict['semantics0'] = torch.stack(torch.chunk(ret_dict['semantics0'], args.batch_size, dim=0)).squeeze(1)
            ret_dict['semantics0'] = ret_dict['semantics0'].reshape([ret_dict['semantics0'].shape[0], args.patch_size, args.patch_size, *ret_dict['semantics0'].shape[2:]])

    # dino forward pass
    if args.use_dino:
        import cv2
        dino_in = ret_dict['rgb'].permute(0, 3, 1, 2)
        dino_in = F.interpolate(dino_in, (args.patch_size*args.patch_stride, args.patch_size*args.patch_stride))
        dino_in = normalize_batch(dino_in)

        dino_ret = dino.get_vit_attn_feat(dino_in)
        attn, cls_, feat = dino_ret['attn'], dino_ret['cls_'], dino_ret['feat']
        attn = attn.reshape([attn.shape[0], attn.shape[1], int(math.sqrt(attn.shape[-1])), int(math.sqrt(attn.shape[-1]))])

    optimizer.zero_grad()
    rgb = ret_dict['rgb']
    img_loss = img2mse(rgb, gt)
    psnr = mse2psnr(img_loss)

    loss = args.rgb_w * img_loss

    if 'rgb0' in ret_dict:
        img_loss0 = img2mse(ret_dict['rgb0'], gt)
        psnr0 = mse2psnr(img_loss0)
        loss = loss + (args.rgb_w * img_loss0)

    sem_loss0, sem_loss1 = torch.Tensor([0]), torch.Tensor([0])
    
    similarity_matrix = get_similarity_matrix(cls_)

    if args.use_correlation:
        # semantics0 = ret_dict['semantics0'].softmax(dim=-1)
        # semantics = ret_dict['semantics'].softmax(dim=-1)
        semantics0 = ret_dict['semantics0']
        semantics = ret_dict['semantics']
        semantics0 = semantics0.reshape(args.batch_size, args.patch_size, args.patch_size, semantics0.shape[-1])
        semantics0 = semantics0.permute(0, 3, 1, 2)
        semantics = semantics.reshape(args.batch_size, args.patch_size, args.patch_size, semantics.shape[-1])
        semantics = semantics.permute(0, 3, 1, 2)
        feat = feat.reshape(args.batch_size,  int(math.sqrt(feat.shape[-2])), int(math.sqrt(feat.shape[-2])), feat.shape[-1])
        feat = feat.permute(0, 3, 1, 2)
        correlation_c_l = args.correlation_w * correlation_loss(feat, semantics0, similarity_matrix)
        correlation_f_l = args.correlation_w * correlation_loss(feat, semantics, similarity_matrix)
        loss += correlation_c_l
        loss += correlation_f_l
    else:
        correlation_c_l, correlation_f_l = torch.Tensor([0]), torch.Tensor([0])
    
    if args.use_geoCorr:
        K = train_loader.dataset.K
        img_h = train_loader.dataset.height
        img_w = train_loader.dataset.width
        semantics0 = ret_dict['semantics0']
        semantics = ret_dict['semantics']
        semantics0 = semantics0.permute(0, 3, 1, 2) #(B, P, P, C) > (B, C, P, P)
        semantics = semantics.permute(0, 3, 1, 2)
        depth0 = ret_dict['depth0']
        depth = ret_dict['depth']
        depth0 = depth0.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)#(B, P, P, C) > (B, C, P, P)

        ray_o, ray_d = ray_o.permute(0, 3, 1, 2), ray_d.permute(0, 3, 1, 2)
        geo_correlation_c_l = args.Gcorrelation_w * geoCorrelation_loss(depth, semantics0, [ray_o, ray_d, gt], similarity_matrix)
        geo_correlation_f_l = args.Gcorrelation_w * geoCorrelation_loss(depth, semantics, [ray_o, ray_d, gt], similarity_matrix)
        # geo_correlation_c_l = args.correlation_w * correlation_loss(depth, semantics)
        # geo_correlation_f_l = args.correlation_w * correlation_loss(depth, semantics0)
        loss += geo_correlation_c_l
        loss += geo_correlation_f_l
    else:
        geo_correlation_c_l, geo_correlation_f_l = torch.Tensor([0]), torch.Tensor([0])

    if args.use_contrast:
        contrast_l = args.contrast_w * contrast_loss(cls_)
        loss += contrast_l
    else:
        contrast_l = torch.Tensor([0]).to(rgb.device)

    if (global_step % args.i_print == 0 and global_step > 0) or global_step==1:
        if "semantics" in ret_dict:
            if args.clus_no_sfm:
                sem_prob = ret_dict['semantics'].detach().cpu().float()
                sem_pred_sft = torch.argmax(sem_prob.softmax(dim=-1), -1).unsqueeze(-1).numpy()
            else:
                sem_prob = ret_dict['semantics'].detach().cpu().float().softmax(dim=-1)
                sem_pred_sft = torch.argmax(sem_prob, -1).unsqueeze(-1).numpy()

            sem_pred_clus = np.zeros([args.batch_size, args.patch_size, args.patch_size, 1])
            for i in range(args.batch_size):
                res = segmap_cluster(sem_prob[i, ...], n_clusters=args.N_cluster)
                sem_pred_clus[i, ...] = res
            sem_gt = masks.detach().cpu().float().numpy()
            fg_idx = sem_gt==1

            clus_ari = adjusted_rand_score(sem_gt.reshape(-1), sem_pred_clus.reshape(-1))
            clus_ari_fg = adjusted_rand_score(sem_gt[fg_idx].reshape(-1), sem_pred_clus[fg_idx].reshape(-1))

            sem_ari = adjusted_rand_score(sem_gt.reshape(-1), sem_pred_sft.reshape(-1))
            sem_ari_fg = adjusted_rand_score(sem_gt[fg_idx].reshape(-1), sem_pred_sft[fg_idx].reshape(-1))
        else:
            clus_ari = clus_ari_fg = sem_ari = sem_ari_fg = 0
    else:
        clus_ari = clus_ari_fg = sem_ari = sem_ari_fg = 0

    # Optimize
    loss.backward()
    optimizer.step()
    scheduler.step(global_step)

    return dict(loss=loss, psnr=psnr, sem0=sem_loss0, sem1=sem_loss1, img0=img_loss0, img1=img_loss, 
    contrast=contrast_l, corr0=correlation_c_l, corr1=correlation_f_l,
    geo_corr0=geo_correlation_c_l,
    geo_corr1=geo_correlation_f_l,
    clus_ari=clus_ari,
    clus_ari_fg = clus_ari_fg,
    sem_ari = sem_ari,
    sem_ari_fg = sem_ari_fg
    )


def save_checkpoint(path, global_step, model, optimizer):
    save_dict = {
        'global_step': global_step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(save_dict, path)
