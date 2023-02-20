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

from data.datasets import RayNeRFDataset, ViewNeRFDataset, ExhibitNeRFDataset, PatchNeRFDataset
from data.collater import RayBatchCollater, ViewBatchCollater, PatchBatchCollater
from models.nerf_net import NeRFNet
from models.mip_nerf_net import MipNeRFNet
from engines.lr import LRScheduler
from engines.trainer import train_one_step, save_checkpoint
from engines.eval import eval_one_view, evaluate, render_video, export_density
from pdb import set_trace as st
from models.dino import Dino
from models.extractor import VitExtractor
from utils.image import NeRFContrastive, CorrelationLoss, GeoCorrelationLoss

def create_arg_parser():
    parser = configargparse.ArgumentParser()

    # basic options
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--gpuid", type=int, default=0,
                        help='gpu id for cuda')
    parser.add_argument("--eval", action='store_true',
                        help='only evaluate without training')
    parser.add_argument("--eval_video", action='store_true',
                        help='render video during evaluation')
    parser.add_argument("--eval_vol", action='store_true',
                        help='export density volume during evaluation')
    parser.add_argument("--vol_extents", nargs='+', type=float, default=2.,
                        help='extent of exported density volume')
    parser.add_argument("--vol_size", type=float, default=2./256,
                        help='voxel size for exported density volume')

    # dataset options
    parser.add_argument("--data_path", "--datadir", type=str, required=True,
                        help='input data directory')
    parser.add_argument('--data_type', '--dataset_type', type=str, required=True,
                        help='dataset type', choices=['llff', 'blender', 'LINEMOD', 'deepvoxels', 'toydesk',
                        'toydesk_custom', 'tankstemple_custom'])
    parser.add_argument("--subsample", type=int, default=0,
                    help='subsampling rate if applicable')

    # flags for llff
    parser.add_argument('--ndc', action='store_true', default=False,
        help='Turn on NDC device. Only for llff dataset')
    parser.add_argument('--spherify', action='store_true', default=False,
        help='Turn on spherical 360 scenes. Only for llff dataset')
    parser.add_argument('--factor', type=int, default=8,
        help='Downsample factor for LLFF images. Only for llff dataset')
    parser.add_argument('--llffhold', type=int, default=8,
        help='Hold out every 1/N images as test set. Only for llff dataset')

    # flags for blend
    parser.add_argument('--half_res', action='store_true', default=False,
        help='Load half-resolution (400x400) images instead of full resolution (800x800). Only for blender dataset.')
    parser.add_argument('--white_bkgd', action='store_true', default=False,
        help='Render synthetic data on white background. Only for blender/LINEMOD dataset')
    parser.add_argument('--test_skip', type=int, default=8,
        help='will load 1/N images from test/val sets. Only for large datasets like blender/LINEMOD/deepvoxels.')

    ## flags for deepvoxels
    parser.add_argument('--dv_scene', type=str, default='greek',
        help='Shape of deepvoxels scene. Only for deepvoxels dataset', choices=['armchair', 'cube', 'greek', 'vase'])

    # Training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--max_steps", "--N_iters", type=int, default=200000,
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
    parser.add_argument("--mipnerf", action='store_true', default=False,
                        help='use mipnerf model')
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

    # additional training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console/tensorboard printout and metric loggin')
    parser.add_argument("--i_verbose",   type=int, default=500,
                        help='frequency of printing')
    parser.add_argument("--i_img",     type=int, default=900000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--log_img_idx", type=int, default=0,
                    help='the view idx used for logging while testing')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    # new added
    parser.add_argument("--use_semantics", action='store_true', default=True,
                        help='add another semantic branch')
    parser.add_argument("--no_semantics", action='store_true', default=False,
                        help='add another semantic branch')
    parser.add_argument("--sem_w", type=float, default=0,
                        help='semantic loss weight')
    parser.add_argument("--rgb_w", type=float, default=1,
                        help='rgb loss weight')
    parser.add_argument("--load_nostrict", action='store_true', default=False,
                        help='strict when loading ckpt')
    parser.add_argument("--patch_tune", action='store_true', default=False,
                        help='finetuning using patch wise')
    parser.add_argument("--patch_size", type=int, default=32,
                        help='patch size for dataloader')
    parser.add_argument("--patch_stride", type=int, default=1,
                        help='patch stride for dataloader')
    parser.add_argument("--bin_thres", type=float, default=0.3,
                        help='binary threshold for dino segmentation')
    parser.add_argument("--use_dino", action='store_true', default=False,
                        help='add dino as output')
    parser.add_argument("--use_contrast", action='store_true', default=False,
                        help='use contrastive loss')
    parser.add_argument("--fast_mode", action='store_true', default=False,
                        help='only eval first image')
    parser.add_argument("--contrast_w", type=float, default=0,
                        help='contrast loss weight')
    parser.add_argument("--verbose", action='store_true', default=False,
                        help='verbose print')
    parser.add_argument("--sem_layer", type=int, default=2,
                        help='semantic branch layer number')
    parser.add_argument("--fix_backbone", action='store_true', default=False,
                        help='fix nerf backbone')
    parser.add_argument("--ret_cluster", action='store_true', default=False,
                        help='return and save clustering results')
    parser.add_argument("--correlation_w", type=float, default=0.001,
                        help='correlation loss weight')
    parser.add_argument("--Gcorrelation_w", type=float, default=0.001,
                        help='correlation loss weight')
    parser.add_argument("--use_correlation", action='store_true', default=False,
                        help='use correlation to compute loss')
    parser.add_argument("--clus_no_sfm", action='store_true', default=False,
                        help='no soft max before clustering')
    parser.add_argument("--sem_dim", type=int, default=2,
                        help='semantic branch layer dimension')
    parser.add_argument("--N_cluster", type=int, default=2,
                        help='cluster number')
    parser.add_argument("--self_corr_w", type=float, default=0,
                        help='self correlation loss weight')
    parser.add_argument("--sem_with_coord", action='store_true', default=False,
                        help='input coordinate for semantic branch')
    parser.add_argument("--sem_with_geo", action='store_true', default=False,
                        help='input coordinate for semantic branch')
    parser.add_argument("--use_geoCorr", action='store_true', default=False,
                        help='use geometry correlation loss')
    parser.add_argument("--pos_corr_w", type=float, default=0,
                        help='positive correlation loss weight')
    parser.add_argument("--use_sim_matrix", action='store_true', default=False,
                        help='use similar matrix to find negative pair')
    parser.add_argument('--app_corr_params', nargs='*', default=[None, None, None, None],
                        help="self_shift, self_weight, neg_shift, neg_weight")
    parser.add_argument('--geo_corr_params', nargs='*', default=[None, None, None, None],
                        help="self_shift, self_weight, neg_shift, neg_weight")
    parser.add_argument("--use_masks", action='store_true', default=False,
                        help='use_masks')
    parser.add_argument("--rand_neg", action='store_true', default=False,
                        help='rand_neg')
    return parser

def main(args):

    if args.no_semantics:
        args.use_semantics = False
    print(f'> Semantic branch is {args.use_semantics}, semantic weight is {args.sem_w}')
    print(f"> spherify: {args.spherify}")

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
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        # config_path = os.path.join(run_dir, 'args.txt')
        # if os.path.exists(config_path):
        #     with open(config_path, 'r') as f:
        #         config_file, _ = parser.parse_known_args(args=[], config_file_contents=f.read())
        #         # Hyper-parameters to reload
        #         keys = ['netdepth', 'netwidth', 'netdepth_fine', 'netwidth_fine', 'use_embed',
        #                 'conv_embed', 'multires', 'multires_views', 'use_viewdirs']
        #         if not compare_args(args, config_file, keys):
        #             print('Reloading network parameters from', config_path)
        #             update_args(args, config_file, keys)

    # Create model and optimizer
    if args.mipnerf:
        model = MipNeRFNet(netdepth=args.netdepth, netwidth=args.netwidth, netwidth_fine=args.netwidth_fine, netdepth_fine=args.netdepth_fine,
        N_samples=args.N_samples, N_importance=args.N_importance, viewdirs=args.use_viewdirs, use_embed=args.use_embed, multires=args.multires,
        multires_views=args.multires_views, ray_chunk=args.ray_chunk, pts_chuck=args.pts_chunk, perturb=args.perturb,
        raw_noise_std=args.raw_noise_std, white_bkgd=args.white_bkgd).to(device)

    else:
        model = NeRFNet(netdepth=args.netdepth, netwidth=args.netwidth, netwidth_fine=args.netwidth_fine, netdepth_fine=args.netdepth_fine,
            N_samples=args.N_samples, N_importance=args.N_importance, viewdirs=args.use_viewdirs, use_embed=args.use_embed, multires=args.multires,
            multires_views=args.multires_views, conv_embed=args.conv_embed, ray_chunk=args.ray_chunk, pts_chuck=args.pts_chunk, perturb=args.perturb,
            raw_noise_std=args.raw_noise_std, white_bkgd=args.white_bkgd, use_semantics=args.use_semantics, sem_layer=args.sem_layer, sem_dim=args.sem_dim,
            sem_with_coord=args.sem_with_coord).to(device)

    if args.fix_backbone:
        from utils.misc import find_params
        my_list = 'semantic_linear'
        # params_specify, params_base = find_params(model, my_list)
        # optimizer = torch.optim.Adam([{'params': params_specify}, {'params': params_base, 'lr': 0}], lr=args.lrate, betas=(0.9, 0.999))
        # scheduler = LRScheduler(optimizer=optimizer, init_lr=(args.lrate, 0), decay_rate=args.decay_rate, decay_steps=args.decay_step*1000)
        for p in model.nerf.mlp.named_parameters():
            if my_list not in p[0]:
                p[1].requires_grad=False
        for p in model.nerf_fine.mlp.named_parameters():
            if my_list not in p[0]:
                p[1].requires_grad=False

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    scheduler = LRScheduler(optimizer=optimizer, init_lr=args.lrate, decay_rate=args.decay_rate, decay_steps=args.decay_step*1000)
    print(f'[Check require grad or not]: {list(model.nerf_fine.mlp.pts_linears[0].parameters())[0].requires_grad}')

    if args.use_dino:
        dino = VitExtractor(model_name='dino_vits16', device=device)
        # dino = Dino(arch='vit_small', patch_size=8, image_size=(args.patch_size, args.patch_size), device=device, fix=True, \
        #     ckpt_path='', checkpoint_key='teacher')
    else:
        dino = None
    print("Num of Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    global_step = 0

    # find and load checkpoint
    ckpt_path = args.ckpt_path
    print(f"[ckpt_path param:] {ckpt_path}")
    if not ckpt_path and not args.no_reload:
        # chronological order
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if len(ckpt_files) > 0:
            sort_fn = lambda x: os.path.splitext(x)[0]
            ckpt_files = sorted(ckpt_files, key=sort_fn)
            ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
    if ckpt_path not in [None, 'None', '']:
        if not os.path.exists(ckpt_path):
            print(f"[Error:] ckpt path {ckpt_path} not exist!")
            exit(0)
    ckpt_dict = None
    if os.path.exists(ckpt_path):
        print(f'>>> Load strict: {not args.load_nostrict}')
        ckpt_dict = torch.load(ckpt_path, map_location='cpu')

    # reload from checkpoint
    if ckpt_dict is not None:
        print("Reloading from checkpoint:", ckpt_path)
        global_step = ckpt_dict['global_step']
        model.load_state_dict(ckpt_dict['model'], strict=(not args.load_nostrict))
        try:
            optimizer.load_state_dict(ckpt_dict['optimizer'])
        except:
             print(f"[Error]: optimizer initialization failed!")

    # Create eval dataset
    print("Loading nerf data:", args.data_path)
    test_set = RayNeRFDataset(args.data_path, args, subsample=args.subsample, split='test', cam_id=False, use_masks=args.use_masks)
    try:
        exhibit_set = ExhibitNeRFDataset(args.data_path, args, subsample=args.subsample, use_semantics=False)
    except FileNotFoundError:
        exhibit_set = None
        print("Warning: No exhibit set!")

    ####### Evaluate #######
    if args.eval:
        print('> Start to evaluate')
        save_dir = f'{run_dir}/eval'
        os.makedirs(save_dir, exist_ok=True)
        metric_dict = evaluate(model, test_set, device=device, save_dir=save_dir, fast_mode=args.fast_mode, ret_cluster=args.ret_cluster, N_cluster=args.N_cluster, clus_no_sfm=args.clus_no_sfm)
        exit(0)
    ####### Eval video #######
    if args.eval_video and exhibit_set is not None:
        render_video(model, exhibit_set, device=device, save_dir=run_dir, suffix=args.expname,
            ret_cluster=args.ret_cluster, clus_no_sfm=args.clus_no_sfm, N_cluster=args.N_cluster, fast_mode=args.fast_mode)
        exit(0)
    ####### Eval Density ########
    if args.eval_vol:
        print('> Start to export density')
        extents = args.vol_extents
        save_dir = f'{run_dir}/eval'
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(args.vol_extents, (float, int)):
            extents = (args.vol_extents,)
        if len(extents) == 1:
            extents = extents * 3
        if len(extents) != 3:
            print('Unsupported length of extents:', extents)
            return
        print('Exporting volume ...')
        export_density(model, extents=extents, voxel_size=args.vol_size, device=device, save_dir=save_dir)
        exit(0)

    ####### Training stage #######
    if not args.eval:
        # Create train dataset
        if not args.no_batching:
            if not args.patch_tune:
                train_set = RayNeRFDataset(args.data_path, args, subsample=args.subsample, split='train', cam_id=False)
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
            collate_fn=RayBatchCollater(), num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
            else:
                train_set = PatchNeRFDataset(args.data_path, args, subsample=args.subsample, split='train', cam_id=False,
                crop_size=args.patch_size*args.patch_stride, patch_stride=args.patch_stride, bin_thres=args.bin_thres, ret_k=args.use_geoCorr)
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
            collate_fn=PatchBatchCollater(), num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
        else:
            train_set = ViewNeRFDataset(args.data_path, args.batch_size, args, subsample=args.subsample, split='train', cam_id=False,
                precrop_iters=args.precrop_iters, precrop_frac=args.precrop_frac, start_iters=global_step, bin_thres=args.bin_thres)
            # number of workers must be zero, because there is an iteration counter inside.
            # multi-threading will duplicate accumulation to that counter.
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True,
                collate_fn=ViewBatchCollater(), num_workers=0, pin_memory=args.pin_mem, drop_last=True)

        near, far = train_loader.dataset.near_far()
        seg_loss = nn.CrossEntropyLoss() if args.use_semantics else None
        contrast_loss = NeRFContrastive(device=device) if args.use_contrast else None
        correlation_loss = CorrelationLoss(args) if args.use_correlation else None
        geoCorrelation_loss = GeoCorrelationLoss(args) if args.use_geoCorr else None

        # Summary writers
        summary_writer = SummaryWriter(log_dir=log_dir)
        print(f"> Start Iteration from {global_step}, semantics is {args.use_semantics}")
        while global_step < args.max_steps:
            time0 = time.time()
            epoch = global_step // len(train_loader) + 1

            for batch in train_loader:
                # counter accumulate
                global_step += 1
                # train for one batch
                loss_list = [seg_loss, contrast_loss, correlation_loss, geoCorrelation_loss]
                ret_dict = train_one_step(batch, [model, dino], optimizer, scheduler, train_loader, global_step, loss_list, device, args)
                loss, psnr = ret_dict['loss'], ret_dict['psnr']
                sem0, sem1 = ret_dict['sem0'], ret_dict['sem1']
                img0, img1 = ret_dict['img0'], ret_dict['img1']
                contrast = ret_dict['contrast']
                corr0, corr1 = ret_dict['corr0'], ret_dict['corr1']
                geo_corr0, geo_corr1 = ret_dict['geo_corr0'], ret_dict['geo_corr1']
                clus_ari, clus_ari_fg = ret_dict['clus_ari'], ret_dict['clus_ari_fg']
                sem_ari, sem_ari_fg = ret_dict['sem_ari'], ret_dict['sem_ari_fg']

                ############################
                ##### Rest is logging ######
                ############################
                args.verbose = False
                if contrast_loss is not None:
                    contrast_loss.verbose = False

                if global_step % args.i_verbose == 0 or global_step==1:
                    args.verbose = True
                    if contrast_loss is not None:
                        contrast_loss.verbose = True

                if (global_step % args.i_print == 0 and global_step > 0) or global_step==1:
                    geoCorrelation_loss.verbose = True
                    correlation_loss.verbose = True

                    avg_time = (time.time() - time0) / args.i_print
                    print(f"[Logging infor]: expname: {args.expname}")
                    print(f"[TRAIN] Iter: {global_step}/{args.max_steps} Loss: {round(loss.item(),4)} L_sem0:{round(sem0.item(),4)} L_sem1:{round(sem1.item(),4)} L_img0:{round(img0.item(),4)} L_img1:{round(img1.item(),4)} L_contrast:{round(contrast.item(),4)}" )
                    print(f"L_corr0:{round(corr0.item(),4)}  L_corr1:{round(corr1.item(),4)} L_geo_corr0:{round(geo_corr0.item(),4)}  L_geo_corr1:{round(geo_corr1.item(),4)} PSNR: {round(psnr.item(), 4)} Average Time: {round(avg_time,4)}")
                    print(f"clus_ari: {round(clus_ari,4)}  clus_ari_fg: {round(clus_ari_fg,4)}  sem_ari: {round(sem_ari,4)}  sem_ari_fg: {round(sem_ari_fg,4)}  ")
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
                    ret_dict, metric_dict = eval_one_view(model, test_set[args.log_img_idx], (near, far), radii=test_set.radii(), device=device)
                    summary_writer.add_image('test/rgb', to8b(ret_dict['rgb'].numpy()), global_step, dataformats='HWC')
                    summary_writer.add_image('test/disp', to8b(ret_dict['disp'].numpy() / np.max(ret_dict['disp'].numpy())), global_step, dataformats='HWC')

                    # Render test set to tensorboard looply
                    ret_dict, metric_dict = eval_one_view(model, test_set[(global_step//args.i_img-1) % len(test_set)], (near, far), radii=test_set.radii(), device=device)
                    summary_writer.add_image('loop/rgb', to8b(ret_dict['rgb'].numpy()), global_step, dataformats='HWC')
                    summary_writer.add_image('loop/disp', to8b(ret_dict['disp'].numpy() / np.max(ret_dict['disp'].numpy())), global_step, dataformats='HWC')

                # save checkpoint
                if global_step % args.i_weights == 0 and global_step > 0:
                    path = os.path.join(run_dir, 'checkpoints', '{:08d}.ckpt'.format(global_step))
                    print('Checkpointing at', path)
                    save_checkpoint(path, global_step, model, optimizer)
                    path = os.path.join(run_dir, 'checkpoints', 'latest.ckpt')
                    save_checkpoint(path, global_step, model, optimizer)

                # test images
                if global_step % args.i_testset == 0 and global_step > 0:
                    print("Evaluating test images ...")
                    save_dir = os.path.join(run_dir, 'testset_{:08d}'.format(global_step))
                    os.makedirs(save_dir, exist_ok=True)
                    metric_dict = evaluate(model, test_set, device=device, save_dir=save_dir,
                        fast_mode=args.fast_mode, ret_cluster=args.ret_cluster, clus_no_sfm=args.clus_no_sfm)

                    # log testing metric
                    summary_writer.add_scalar('test/mse', metric_dict['mse'], global_step)
                    summary_writer.add_scalar('test/psnr', metric_dict['psnr'], global_step)

                # exhibition video
                if global_step % args.i_video==0 and global_step > 0 and exhibit_set is not None:
                    render_video(model, exhibit_set, device=device, save_dir=run_dir, suffix=str(global_step),
                    fast_mode=args.fast_mode, ret_cluster=args.ret_cluster, clus_no_sfm=args.clus_no_sfm, N_cluster=args.N_cluster)

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
        render_video(model, exhibit_set, device=device, save_dir=save_dir, fast_mode=False,
        ret_cluster=args.ret_cluster, clus_no_sfm=args.clus_no_sfm, N_cluster=args.N_cluster)


if __name__=='__main__':
    # Random seed
    np.random.seed(0)

    # enable error detection
    torch.autograd.set_detect_anomaly(True)

    # Read arguments and configs
    parser = create_arg_parser()
    args, _ = parser.parse_known_args()

    main(args)
