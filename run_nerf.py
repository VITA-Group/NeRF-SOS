import os, sys, copy
import math, time, random, shutil

import numpy as np

import imageio
import json
import configargparse
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from utils.config import compare_args, update_args
from utils.image import to8b

from data.datasets import RayNeRFDataset, ViewNeRFDataset, ExhibitNeRFDataset
from data.collater import RayBatchCollater, ViewBatchCollater
from models.nerf_net import NeRFNet
from engines.lr import LRScheduler
from engines.trainer import train_one_step, save_checkpoint
from engines.eval import eval_one_view, evaluate, render_video, export_density

def create_arg_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--data_path", "--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')
    parser.add_argument("--gpuid", type=int, default=0, 
                        help='gpu id for cuda')
    parser.add_argument("--eval", action='store_true', 
                        help='only evaluate without training')
    parser.add_argument("--eval_video", action='store_true', 
                        help='render video during evaluation')
    parser.add_argument("--eval_vol", action='store_true', 
                        help='export density volume during evaluation')

    parser.add_argument("--save_rays", action='store_true', 
                        help='save rays, near, far for visualization')
    parser.add_argument("--save_pts", action='store_true', 
                        help='save point samples for visualization')

    # Training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--max_steps", "--N_iters", type=int, default=500000, 
                        help='max iteration number (number of iteration to finish training)')
    parser.add_argument("--batch_size", "--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--ray_chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--pts_chunk", type=int, default=1024*256, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--verbose", action='store_true', 
                        help='print more when training')

    # hyper-parameter for learning scheduler
    parser.add_argument("--decay_step", type=int, default=250, 
                        help='exponential learning rate decay iteration (in 1000 steps)')
    parser.add_argument("--decay_rate", type=float, default=0.1, 
                        help='exponential learning rate decay scale')

    # reload option
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_path", type=str, default='', 
                        help='specific weights npy file to reload for coarse network')

    parser.add_argument("--pin_mem", action='store_true', default=True,
                        help='turn on pin memory for data loading')
    parser.add_argument("--no_pin_mem", action='store_false', dest='pin_memory',
                        help='turn off pin memory for data loading')
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--num_workers", type=int, default=8,
                        help='number of workers used for data loading')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', default=True,
                        help='enable full 5D input, using 3D without view dependency')
    parser.add_argument("--no_viewdirs", action='store_false', dest='use_viewdirs',
                        help='disable full 5D input, using 3D without view dependency')
    parser.set_defaults(use_viewdirs=True)
    parser.add_argument("--use_embed", action='store_true', default=True, 
                        help='turn on positional encoding')
    parser.add_argument("--no_embed", action='store_false', dest='use_embed', 
                        help='turn on positional encoding')
    parser.set_defaults(use_embed=True)
    parser.add_argument("--conv_embed", action='store_true', default=False, 
                        help='turn on 1D convolutional positional encoding')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument('--white_bkgd', action='store_true', default=False,
                        help='Render synthetic data on white background. Only for blender/LINEMOD dataset')

    # additional training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
 
    # dataset options
    parser.add_argument("--dataset_type", type=str, default='nerf', 
                        help='options: nerf / point cloud')
    parser.add_argument("--subsample", type=int, default=0, 
                    help='subsampling rate if applicable')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=200, 
                        help='frequency of console/tensorboard printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--log_img_idx", type=int, default=0, 
                    help='the view idx used for logging while testing')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser

def main(args):

    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    # Create log dir and copy the config file
    run_dir = os.path.join(args.basedir, args.expname)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    log_dir = os.path.join(run_dir, 'tensorboard')

    # Save/reload config
    if not os.path.exists(run_dir):
        if not args.eval:
            os.makedirs(run_dir)
            os.makedirs(ckpt_dir)
            os.makedirs(log_dir)

            # Dump training configuration
            config_path = os.path.join(run_dir, 'args.txt')
            parser.write_config_file(args, [config_path])
            # Backup the default config file for checking
            shutil.copy(args.config, os.path.join(run_dir, 'config.txt'))
        else:
            print("Error: The specified working directory does not exists!")
            return
    else:
        config_path = os.path.join(run_dir, 'args.txt')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_file, _ = parser.parse_known_args(args=[], config_file_contents=f.read())
                # Hyper-parameters to reload
                keys = ['netdepth', 'netwidth', 'netdepth_fine', 'netwidth_fine', 'use_embed',
                        'multires', 'multires_views', 'use_viewdirs']
                if not compare_args(args, config_file, keys):
                    print('Reloading network parameters from', config_path)
                    update_args(args, config_file, keys)

    # Create model and optimizer
    model = NeRFNet(netdepth=args.netdepth, netwidth=args.netwidth, netwidth_fine=args.netwidth_fine, netdepth_fine=args.netdepth_fine,
        N_samples=args.N_samples, N_importance=args.N_importance, viewdirs=args.use_viewdirs, use_embed=args.use_embed, multires=args.multires,
        multires_views=args.multires_views, conv_embed=args.conv_embed, ray_chunk=args.ray_chunk, pts_chuck=args.pts_chunk, perturb=args.perturb,
        raw_noise_std=args.raw_noise_std, white_bkgd=args.white_bkgd).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    scheduler = LRScheduler(optimizer=optimizer, init_lr=args.lrate, decay_rate=args.decay_rate, decay_steps=args.decay_step*1000)
    global_step = 0

    # find and load checkpoint
    ckpt_path = args.ckpt_path
    if not ckpt_path and not args.no_reload:
        # chronological order
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if len(ckpt_files) > 0:
            sort_fn = lambda x: os.path.splitext(x)[0]
            ckpt_files = sorted(ckpt_files, key=sort_fn)
            ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
    ckpt_dict = None
    if os.path.exists(ckpt_path):
        ckpt_dict = torch.load(ckpt_path)

    # reload from checkpoint
    if ckpt_dict is not None:
        print("Reloading from checkpoint:", ckpt_path)
        global_step = ckpt_dict['global_step']
        model.load_state_dict(ckpt_dict['model'])
        optimizer.load_state_dict(ckpt_dict['optimizer'])

    # Create dataset
    print("Loading nerf data:", args.data_path)
    test_set = RayNeRFDataset(args.data_path, subsample=args.subsample, split='test', cam_id=False)
    try:
        exhibit_set = ExhibitNeRFDataset(args.data_path, subsample=args.subsample)
    except FileNotFoundError:
        exhibit_set = None
        print("Warning: No exhibit set!")

    ####### Training stage #######
    if not args.eval:
        if not args.no_batching:
            train_set = RayNeRFDataset(args.data_path, subsample=args.subsample, split='train', cam_id=False)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                collate_fn=RayBatchCollater(), num_workers=args.num_workers, pin_memory=args.pin_mem)
        else:
            train_set = ViewNeRFDataset(args.data_path, args.batch_size, subsample=args.subsample, split='train', cam_id=False,
                precrop_iters=args.precrop_iters, precrop_frac=args.precrop_frac, start_iters=global_step)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, 
                collate_fn=ViewBatchCollater(), num_workers=0, pin_memory=args.pin_mem)

        near, far = train_loader.dataset.near_far()

        # Summary writers
        summary_writer = SummaryWriter(log_dir=log_dir)
        while global_step < args.max_steps:
            time0 = time.time()
            epoch = global_step // len(train_loader) + 1
            for batch in train_loader:

                # global_step = train_one_epoch(model, optimizer, scheduler,
                #     train_loader, test_set, exhibit_set, summary_writer,
                #     global_step, args.max_steps, run_dir, device,
                #     i_print=args.i_print, i_img=args.i_img, log_img_idx=args.log_img_idx,
                #     i_weights=args.i_weights, i_testset=args.i_testset, i_video=args.i_video)

                # counter accumulate
                global_step += 1
                # train for one batch
                ret_dict = train_one_step(batch, model, optimizer, scheduler, train_loader, global_step, device)
                loss, psnr = ret_dict['loss'], ret_dict['psnr']

                ############################
                ##### Rest is logging ######
                ############################

                # logging errors
                if global_step % args.i_print == 0 and global_step > 0:
                    avg_time = (time.time() - time0) / args.i_print
                    print(f"[TRAIN] Iter: {global_step}/{args.max_steps} Loss: {loss.item()} PSNR: {psnr.item()} Average Time: {avg_time}")
                    time0 = time.time()

                    # log training metric
                    summary_writer.add_scalar('train/loss', loss, global_step)
                    summary_writer.add_scalar('train/psnr', psnr, global_step)

                    # log learning rate
                    lr_groups = {}
                    for i, param in enumerate(optimizer.param_groups):
                        lr_groups['group_'+str(i)] = param['lr']
                    summary_writer.add_scalars('l_rate', lr_groups, global_step)

                # logging images
                if global_step % args.i_img == 0 and global_step > 0:
                    # Output test images to tensorboard
                    ret_dict, metric_dict = eval_one_view(model, test_set[args.log_img_idx], (near, far), device=device)
                    summary_writer.add_image('test/rgb', to8b(ret_dict['rgb'].numpy()), global_step, dataformats='HWC')
                    summary_writer.add_image('test/disp', to8b(ret_dict['disp'].numpy() / np.max(ret_dict['disp'].numpy())), global_step, dataformats='HWC')

                    # Render test set to tensorboard looply
                    ret_dict, metric_dict = eval_one_view(model, test_set[(global_step//args.i_img-1) % len(test_set)], (near, far), device=device)
                    summary_writer.add_image('loop/rgb', to8b(ret_dict['rgb'].numpy()), global_step, dataformats='HWC')
                    summary_writer.add_image('loop/disp', to8b(ret_dict['disp'].numpy() / np.max(ret_dict['disp'].numpy())), global_step, dataformats='HWC')

                # save checkpoint
                if global_step % args.i_weights == 0 and global_step > 0:
                    path = os.path.join(run_dir, 'checkpoints', '{:08d}.ckpt'.format(global_step))
                    print('Checkpointing at', path)
                    save_checkpoint(path, global_step, model, optimizer)

                # test images
                if global_step % args.i_testset == 0 and global_step > 0:
                    print("Evaluating test images ...")
                    save_dir = os.path.join(run_dir, 'testset_{:08d}'.format(global_step))
                    os.makedirs(save_dir, exist_ok=True)
                    metric_dict = evaluate(model, test_set, device=device, save_dir=save_dir)

                    # log testing metric
                    summary_writer.add_scalar('test/mse', metric_dict['mse'], global_step)
                    summary_writer.add_scalar('test/psnr', metric_dict['psnr'], global_step)

                # exhibition video
                if global_step % args.i_video==0 and global_step > 0 and exhibit_set is not None:
                    render_video(model, exhibit_set, device=device, save_dir=run_dir, suffix=str(global_step))

                # End training if finished
                if global_step >= args.max_steps:
                    print(f'Train ends at global_step={global_step}')
                    break   

        save_checkpoint(os.path.join(ckpt_dir, 'last.ckpt'), global_step, model, optimizer)

    ####### Testing stage #######
    save_dir = os.path.join(run_dir, 'eval')
    os.makedirs(save_dir, exist_ok=True)
    evaluate(model, test_set, device=device, save_dir=save_dir)
    if args.eval_video and exhibit_set is not None:
        render_video(model, exhibit_set, device=device, save_dir=save_dir)
    if args.eval_vol:
        export_density(model, extents=(2., 2., 2.), voxel_size=2./256, device=device, save_dir=save_dir)

if __name__=='__main__':
    # Random seed
    np.random.seed(0)

    # Read arguments and configs
    parser = create_arg_parser()
    args, _ = parser.parse_known_args()

    # enable error detection
    torch.autograd.set_detect_anomaly(True)

    main(args)





