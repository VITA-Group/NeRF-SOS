import os, sys, copy
import math, time, random

import numpy as np

import imageio
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.image import to8b, img2mse, mse2psnr
from engines.eval import eval_one_view, evaluate, render_video


def train_one_step(batch, model, optimizer, scheduler, train_loader, global_step, device):
    
    model.train()

    near, far = train_loader.dataset.near_far()

    # make sure on cuda
    batch_rays, gt = batch
    batch_rays, gt = batch_rays.to(device), gt.to(device)

    #####  Core optimization loop  #####
    ret_dict = model(batch_rays, (near, far)) # no extraction

    optimizer.zero_grad()

    rgb = ret_dict['rgb']
    img_loss = img2mse(rgb, gt)
    psnr = mse2psnr(img_loss)

    loss = img_loss

    if 'rgb0' in ret_dict:
        img_loss0 = img2mse(ret_dict['rgb0'], gt)
        psnr0 = mse2psnr(img_loss0)

        loss = loss + img_loss0

    # Optimize
    loss.backward()
    optimizer.step()
    scheduler.step(global_step)

    return dict(loss=loss, psnr=psnr)

def train_one_epoch(model, optimizer, scheduler, train_loader, test_set, exhibit_set, summary_writer, global_step, max_steps,
    run_dir, device, i_print=100, i_img=500, log_img_idx=0, i_weights=10000, i_testset=50000, i_video=50000):

    near, far = train_loader.dataset.near_far()

    start_step = global_step
    epoch = global_step // len(train_loader) + 1
    time0 = time.time()

    for (batch_rays, gt) in train_loader:
        model.train()

        # counter accumulate
        global_step += 1

        # make sure on cuda
        batch_rays, gt = batch_rays.to(device), gt.to(device)

        #####  Core optimization loop  #####
        ret_dict = model(batch_rays, (near, far)) # no extraction

        optimizer.zero_grad()

        rgb = ret_dict['rgb']
        img_loss = img2mse(rgb, gt)
        psnr = mse2psnr(img_loss)

        loss = img_loss

        if 'rgb0' in ret_dict:
            img_loss0 = img2mse(ret_dict['rgb0'], gt)
            psnr0 = mse2psnr(img_loss0)

            loss = loss + img_loss0

        # Optimize
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)

        ############################
        ##### Rest is logging ######
        ############################

        # logging errors
        if global_step % i_print == 0 and global_step > 0:
            dt = time.time() - time0
            time0 = time.time()
            avg_time = dt / min(global_step - start_step, i_print)
            print(f"[TRAIN] Iter: {global_step}/{max_steps} Loss: {loss.item()} PSNR: {psnr.item()} Average Time: {avg_time}")

            # log training metric
            summary_writer.add_scalar('train/loss', loss, global_step)
            summary_writer.add_scalar('train/psnr', psnr, global_step)

            # log learning rate
            lr_groups = {}
            for i, param in enumerate(optimizer.param_groups):
                lr_groups['group_'+str(i)] = param['lr']
            summary_writer.add_scalars('l_rate', lr_groups, global_step)

        # logging images
        if global_step % i_img == 0 and global_step > 0:
            # Output test images to tensorboard
            ret_dict, metric_dict = eval_one_view(model, test_set[log_img_idx], (near, far), device=device)
            summary_writer.add_image('test/rgb', to8b(ret_dict['rgb'].numpy()), global_step, dataformats='HWC')
            summary_writer.add_image('test/disp', to8b(ret_dict['disp'].numpy() / np.max(ret_dict['disp'].numpy())), global_step, dataformats='HWC')

            # Render test set to tensorboard looply
            ret_dict, metric_dict = eval_one_view(model, test_set[(global_step//i_img-1) % len(test_set)], (near, far), device=device)
            summary_writer.add_image('loop/rgb', to8b(ret_dict['rgb'].numpy()), global_step, dataformats='HWC')
            summary_writer.add_image('loop/disp', to8b(ret_dict['disp'].numpy() / np.max(ret_dict['disp'].numpy())), global_step, dataformats='HWC')

        # save checkpoint
        if global_step % i_weights == 0 and global_step > 0:
            path = os.path.join(run_dir, 'checkpoints', '{:08d}.ckpt'.format(global_step))
            print('Checkpointing at', path)
            save_checkpoint(path, global_step, model, optimizer)

        # test images
        if global_step % i_testset == 0 and global_step > 0:
            print("Evaluating test images ...")
            save_dir = os.path.join(run_dir, 'testset_{:08d}'.format(global_step))
            os.makedirs(save_dir, exist_ok=True)
            metric_dict = evaluate(model, test_set, device=device, save_dir=save_dir)

            # log testing metric
            summary_writer.add_scalar('test/mse', metric_dict['mse'], global_step)
            summary_writer.add_scalar('test/psnr', metric_dict['psnr'], global_step)

        # exhibition video
        if global_step % i_video==0 and global_step > 0 and exhibit_set is not None:
            render_video(model, exhibit_set, device=device, save_dir=run_dir, suffix=str(global_step))

        # End training if finished
        if global_step >= max_steps:
            print(f'Train ends at global_step={global_step}')
            break

    return global_step

def save_checkpoint(path, global_step, model, optimizer):
    save_dict = {
        'global_step': global_step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(save_dict, path)
