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
from data.collater import Point_Batch_Collate
from utils.render_collector import SaveImageCollector, TensorboardCollector

from render_pipeline import render_pipeline

def pc2nerf_pipeline(model, train_set, test_set, render_sets, args):

    print('Begin training')
    print('TRAIN point count:', len(train_set))
    print('TEST ray count:', len(test_set))
    print('RENDER keys are:', render_sets.keys())

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
    
    if args.no_batching:
        # No shuffle nor parallel data
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.N_rand, shuffle=False, 
                                                   collate_fn=Point_Batch_Collate(), pin_memory=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.N_rand, shuffle=True, 
                                                   collate_fn=Point_Batch_Collate(), pin_memory=False)

    for ep in range(epoch, max_epoch):
        print('New epoch started: epoch=%d, iters=%d' % (ep, global_step))
        for (batch_pts, target_s) in tqdm(train_loader):
            # counter accumulate
            global_step += 1
            
            # make sure on cuda
            batch_pts, target_s = batch_pts.to(args.device), target_s.to(args.device)

            #####  Core optimization loop  #####
            raw = model.forward_pts(batch_pts, test=False) 

            optimizer.zero_grad()
            loss = model.loss_mrc(raw, target_s)
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
                tqdm.write(f"[TRAIN] Iter: {global_step} Loss: {loss.item()} Average Time: {avg_time}")

                # log training loss
                writer.add_scalar('train/loss', loss, global_step)

                # log learning rate
                lr_groups = {}
                for i, param in enumerate(optimizer.param_groups):
                    lr_groups['group_'+str(i)] = param['lr']
                writer.add_scalars('l_rate', lr_groups, global_step)

            # logging images
            if global_step%args.i_img == 0 and global_step > 0:
                # Output test images to tensorboard
                with torch.no_grad():
                    render_pipeline(model, render_sets['test'], args, test=True, idxs=[0],
                                    collector=TensorboardCollector(writer, 'test/image', global_step, combine=False))

                # Render test set to tensorboard looply
                with torch.no_grad():
                    render_pipeline(model, render_sets['test'], args, test=True, idxs=[loop_image_id],
                                    collector=TensorboardCollector(writer, 'loop/image', global_step, combine=True))
                loop_image_id = (loop_image_id + 1) % (len(render_sets['test']) // args.H // args.W)

            # save checkpoint
            if global_step%args.i_weights == 0:
                path = os.path.join(args.ckpt_dir, '{:06d}.tar'.format(global_step))
                to_save = {
                    'global_step': global_step,
                    'network_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                torch.save(to_save, path)
                print('Saved checkpoints at', path)

            # test image
            if global_step%args.i_testset == 0 and global_step > 0:
                testsavedir = os.path.join(args.run_dir, 'testset_{:06d}'.format(global_step))
                os.makedirs(testsavedir, exist_ok=True)
                with torch.no_grad():
                    render_pipeline(model, render_sets['test'], args, collector=SaveImageCollector(testsavedir), test=True)

            # if global_step%args.i_video==0 and global_step > 0:
            #     # Turn on exhibiting mode
            #     with torch.no_grad():
            #         rgbs = render_path(model, render_sets['exhibit'], k_extract=['rgb'], test=True)

            #         moviebase = os.path.join(args.run_dir, '{}_spiral_{:06d}_'.format(expname, global_step))
            #         export_video(rgbs, moviebase + 'rgb.mp4')
            
            # End training if finished
            if global_step >= args.N_iters:
                print('Interrupt training: global_step=%d' % global_step)
                return
