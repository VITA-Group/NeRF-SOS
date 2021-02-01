
import os, sys
import math, random, time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import imageio
import json

import configargparse

def create_arg_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')
    parser.add_argument("--gpuid", type=int, default=0, 
                        help='gpu id for cuda')
    parser.add_argument("--action", type=str, default='train', 
                        help='choose to perform which action (train/exhibit/export)')

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
    parser.add_argument("--N_iters", type=int, default=2000000, 
                        help='max iteration number (number of iteration to finish training)')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*256, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--verbose", action='store_true', 
                        help='print more when training')

    # hyper-parameter for learning scheduler 1
    parser.add_argument("--lr1_decay_iter", type=int, default=250000, 
                        help='scheduler 1: exponential learning rate decay iteration (in steps)')
    parser.add_argument("--lr1_decay_rate", type=float, default=0.1, 
                        help='scheduler 2: exponential learning rate decay scale')

    # hyper-parameter for learning scheduler 2
    parser.add_argument("--lr2_warmup_iter", type=int, default=1000, 
                        help='scheduler 2: warmup decay stage iteration (in steps)')
    parser.add_argument("--lr2_decay_iter", type=int, default=3000, 
                        help='scheduler 2: start decay stage iteration (in steps)')
    parser.add_argument("--lr2_scale", type=float, default=0.009, 
                        help='scheduler 2: decay scale')

    # reload option
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_file", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    # parser.add_argument("--no_viewdirs", action='store_true',
    #                     help='disable full 5D input, using 3D without view dependency')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    # parser.add_argument("--multires_views", type=int, default=4, 
                        # help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # additional training options
    # parser.add_argument("--no_camera_id", action='store_true', 
    #                     help='do not concat camera id with each ray')   
    # parser.add_argument("--trainable_cams", action='store_true', 
    #                     help='optimize camera pose jointly')
    parser.add_argument("--prior_loss", type=str, default='none',
                        help='priors on the volumetric reconstruction')
    parser.add_argument("--prior_coeff", type=float, default=0.001,
                        help='coefficient for the prior loss')
 
    # dataset options
    parser.add_argument("--dataset_type", type=str, default='nerf', 
                        help='options: nerf / point cloud')
    parser.add_argument("--dataset_subsample", type=int, default=8, 
                    help='subsampling rate if applicable')

    # corruptions
    parser.add_argument("--corrupt_cams", action='store_true', 
                        help='whether corrupt camera extrinsics using a perturbation')
    parser.add_argument("--corrupt_cams_t", type=float, default=0.1,  
                        help='how large are perturbation in rotation degree')
    parser.add_argument("--corrupt_cams_r", type=float, default=5.0,  
                        help='how large are perturbation in rotation degree')
    parser.add_argument("--noise_level", type=float, default=0.1,  
                        help='how strong are the gaussian noises added to corrupt images')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console/tensorboard printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    parser.add_argument("--log_img_train", type=int, default=0, 
                    help='the image idx used for logging while training')
    parser.add_argument("--log_img_test", type=int, default=0, 
                    help='the image idx used for logging while testing')

    return parser

def read_config_file(file_path):
    config_file = configargparse.DefaultConfigFileParser()
    with open(file_path, 'r') as f:
        args = config_file.parse(f)
        print(args)
    return args

def update_arg_config(args, config_file, keys=[]):
    if len(keys) > 0:
        keys = set(vars(args)).intersection(set(keys))
    else:
        keys = set(vars(args))

    for name in keys:
        if name in config_file:
            setattr(args, name, config_file[name])

# Compare args1 with args2
def compare_args(args1, args2, keys=[]):

    if len(keys) == 0:
        keys = vars(args2).keys()

    same = True
    for k in keys:
        if not hasattr(args1, k) or getattr(args1, k) != getattr(args2, k):
            print(k)
            same = False
            break
            # setattr(args, name, config_file[name])
    return same

# Update args1 with args2
def update_args(args1, args2, keys=[]):

    if len(keys) == 0:
        keys = vars(args2).keys()

    for k in keys:
        if hasattr(args1, k):
            setattr(args1, k, getattr(args2, k))
    return args1