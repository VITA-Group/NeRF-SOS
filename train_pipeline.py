import os, sys
import math, time, random

import numpy as np

import imageio
import json
import mrc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from net.nerf_net import NerfNet
from data.collater import Ray_Batch_Collate
from render_pipeline import render_pipeline

from utils.render_collector import SaveImageCollector, TensorboardCollector

def train_pipeline(model, train_set, test_set, render_sets, args):

    print('Begin training')
    print('TRAIN views size:', len(train_set))
    print('TEST views size:', len(test_set))
    print('RENDER views are:', render_sets.keys())

    # schedule learning rate: scheme I
    def scheduler_1(iter):
        decay_rate = 0.1
        decay_steps = args.lr1_decay_iter
        return args.lr1_decay_rate ** (iter / args.lr1_decay_iter)
    
    # schedule learning rate: scheme II
    def scheduler_2(iter):
        iter = iter + 1
        if iter < args.lr2_warmup_iter:
            return iter / args.lr2_warmup_iter
        elif iter > args.lr2_decay_iter:
            r = (iter - args.lr2_decay_iter) / (args.N_iters - args.lr2_decay_iter)
            return (1. - args.lr2_scale) * math.exp(-r) + args.lr2_scale
        else:
            return 1.

    # Create optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=[scheduler_1] * len(optimizer.param_groups))
    # Start iters
    start_iter = 0

    # Load from checkpoint if specified
    if args.ckpt_file is not None and not args.no_reload:  
        print('Reloading optimizer from', args.ckpt_file)
        ckpt = torch.load(args.ckpt_file)
        # Load current step
        start_iter = ckpt['global_step']
        # Load optimizer
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # Summary writers
    writer = SummaryWriter(log_dir=args.log_dir)
    print("Tensorboard directory:", args.log_dir)

    # write images to summary
    def tb_add_image(ret_dict):
        rgb = ret_dict['rgb']
        # Write images
        test_images = test_images.astype(np.float32) / 255.0
        test_images = np.tile(test_images, (3,))
        writer.add_images('test/images', test_images, global_step, dataformats='NHWC')

    # compute current epoch and epoch to go
    iters_per_epoch = math.ceil(len(train_set) / args.N_rand)
    epoch = start_iter // iters_per_epoch
    max_epoch = math.ceil(args.N_iters / iters_per_epoch)

    # counter and timer
    global_step = start_iter
    time0 = time.time()

    # use to specify which test image to be rendered
    loop_image_id = 0
    
    print('Current epoch:', epoch)
    print('Total epochs:', max_epoch)
    print('Current iteration:', global_step)
    print('Iterations per epoch:', iters_per_epoch)
    print('Total iterations:', args.N_iters)
    
    collate_fn = Ray_Batch_Collate() # Create collater
    if args.no_batching:
        # No shuffle nor parallel data
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.N_rand, shuffle=False, 
                                                   collate_fn=collate_fn, pin_memory=args.pin_memory)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.N_rand, shuffle=True, 
                                                   collate_fn=collate_fn, pin_memory=args.pin_memory)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.N_rand, shuffle=False, 
                                              collate_fn=collate_fn, pin_memory=args.pin_memory)

    for ep in range(epoch, max_epoch):
        print('New epoch started: epoch=%d, iters=%d' % (ep, global_step))
        for (batch_rays, target_s) in tqdm(train_loader):
            # counter accumulate
            global_step += 1
            
            # make sure on cuda
            batch_rays, target_s = batch_rays.to(args.device), target_s.to(args.device)

            #####  Core optimization loop  #####
            ret_dict = model(batch_rays, (args.near, args.far), test=False) # no extraction

            optimizer.zero_grad()
            loss, extra_loss = model.loss(ret_dict, target_s)
            mse = extra_loss['mse']
            psnr = extra_loss['psnr']

            # Optimize
            loss.backward()
            optimizer.step()
            # Tune learning rate
            scheduler.step(global_step)

            ############################
            ##### Rest is logging ######
            ############################

            # logging errors
            if global_step%args.i_print == 0 and global_step > 0:
                dt = time.time() - time0
                time0 = time.time()
                avg_time = dt / min(global_step-start_iter, args.i_print)
                tqdm.write(f"[TRAIN] Iter: {global_step} Loss: {loss.item()} PSNR: {psnr.item()} Average Time: {avg_time}")

                # log training metric
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/psnr', psnr, global_step)

                # log learning rate
                lr_groups = {}
                for i, param in enumerate(optimizer.param_groups):
                    lr_groups['group_'+str(i)] = param['lr']
                writer.add_scalars('l_rate', lr_groups, global_step)

            # logging images
            if global_step%args.i_img == 0 and global_step > 0:
                # Output test images to tensorboard
                with torch.no_grad():
                    render_pipeline(model, render_sets['test'], args, test=True, idxs=[args.log_img_test],
                                    collector=TensorboardCollector(writer, 'test/image', global_step, combine=False))

                # Output train images to tensorboard
                with torch.no_grad():
                    render_pipeline(model, render_sets['train'], args, test=True, idxs=[args.log_img_train],
                                    collector=TensorboardCollector(writer, 'train/image', global_step, combine=False))

                # Render test set to tensorboard looply
                with torch.no_grad():
                    render_pipeline(model, render_sets['test'], args, test=True, idxs=[loop_image_id],
                                    collector=TensorboardCollector(writer, 'loop/image', global_step, combine=True))
                loop_image_id = (loop_image_id + 1) % (len(render_sets['test']) // args.H // args.W)

            # save checkpoint
            if global_step%args.i_weights == 0:
                path = os.path.join(args.ckpt_dir, '{:08d}.tar'.format(global_step))
                to_save = {
                    'global_step': global_step,
                    'network_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                torch.save(to_save, path)
                print('Saved checkpoints at', path)

            # test image
            if global_step%args.i_testset == 0 and global_step > 0:
                testsavedir = os.path.join(args.run_dir, 'testset_{:08d}'.format(global_step))
                os.makedirs(testsavedir, exist_ok=True)
                with torch.no_grad():
                    render_pipeline(model, render_sets['test'], args, collector=SaveImageCollector(testsavedir), test=True)

                # 2020.7.15 Also save train set
                trainsavedir = os.path.join(args.run_dir, 'trainset_{:08d}'.format(global_step))
                os.makedirs(trainsavedir, exist_ok=True)
                with torch.no_grad():
                    render_pipeline(model, render_sets['test'], args, collector=SaveImageCollector(trainsavedir), test=True)

            # if global_step%args.i_video==0 and global_step > 0:
            #     # Turn on exhibiting mode
            #     with torch.no_grad():
            #         rgbs = render_path(model, render_sets['exhibit'], k_extract=['rgb'], test=True)

            #         moviebase = os.path.join(args.run_dir, '{}_spiral_{:08d}_'.format(expname, global_step))
            #         export_video(rgbs, moviebase + 'rgb.mp4')
            
            # End training if finished
            if global_step >= args.N_iters:
                print('Interrupt training: global_step=%d' % global_step)
                return
