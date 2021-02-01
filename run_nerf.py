import os, sys
import math, time, random, shutil

import numpy as np

import imageio
import json
import mrc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from utils.config_utils import *

from data.load_nerf import load_nerf

from net.nerf_net import NerfNet

from train_pipeline import train_pipeline
from render_pipeline import exhibit_pipeline, export_pipeline, video_pipeline
from render_pipeline import search_for_campose
from pc2nerf_pipeline import pc2nerf_pipeline

def find_checkpoint(args):
    sort_fn = lambda x: int(os.path.splitext(x)[0])
    if args.ckpt_file is None:
        # chronological order
        os.makedirs(args.ckpt_dir, exist_ok=True)
        ckpts = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.tar')]
        ckpts = sorted(ckpts, key=sort_fn)
        args.ckpt_file = os.path.join(args.ckpt_dir, ckpts[-1]) if len(ckpts) > 0 else None
    else:
        # add suffix if necessary
        _, suffix = os.path.splitext(args.ckpt_file)
        if suffix != '.tar':
            args.ckpt_file += '.tar'

        args.ckpt_file = os.path.join(args.ckpt_dir, args.ckpt_file)

def load_dataset(args):
    if args.dataset_type == 'nerf':
        data_device = torch.device('cpu')
        train_set, test_set, render_sets, bds, hw, extras = load_nerf(args.datadir, args.no_batching, args.N_rand, no_cam_id=True,
                                                                      subsample=args.dataset_subsample, device=data_device)
        print('Loaded nerf', len(train_set), len(test_set), args.datadir, extras['data_device'])
        
        args.pin_memory = (extras['data_device'] == torch.device('cpu'))
        
        H, W = hw
        args.H, args.W = int(H), int(W)
        print('H W', args.H, args.W)

        args.near, args.far = bds
        print('NEAR FAR', args.near, args.far)
        
        # Disable camera transformation
        args.num_cameras = 0
        # Disable view direction
        args.use_viewdirs = False

        # pixel size in angst, unit size in angst
        args.pixel_size, args.unit_size = 1.0, 1.0

    elif args.dataset_type == 'pointcloud':
        print('No implementation')
        
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        exit(-1)
    
    return train_set, test_set, render_sets, bds, hw, extras

if __name__=='__main__':
    # Random seed
    np.random.seed(0)

    # Read arguments and configs
    parser = create_arg_parser()
    args = parser.parse_args()

    # set default cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(args.gpuid)
    args.device = device
    print('Running on %s:%d' % (device, args.gpuid))

    # enable error detection
    torch.autograd.set_detect_anomaly(True)

    # Create log dir and copy the config file
    args.run_dir = os.path.join(args.basedir, args.expname)
    args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')
    args.log_dir = os.path.join(args.run_dir, 'tensorboard')

    # Save and load config
    if not os.path.exists(args.run_dir):
        if args.action == 'train':
            os.makedirs(args.run_dir)

            # Dump training configuration
            config_path = os.path.join(args.run_dir, 'args.txt')
            parser.write_config_file(args, [config_path])
            # Backup the default config file for checking
            shutil.copy(args.config, os.path.join(args.run_dir, 'config.txt'))
        else:
            print("Error: The specified working directory does not exists!")
            exit(-1)
    else:
        config_path = os.path.join(args.run_dir, 'args.txt')
        if os.path.exists(config_path):
            print('Reloading network parameters from', config_path)
            with open(config_path, 'r') as f:
                config_file, _ = parser.parse_known_args(args=[], config_file_contents=f.read())

                # Check hyper-parameter
                keys = ['netdepth', 'netwidth', 'netdepth_fine', 'netwidth_fine', 'i_embed',
                        'multires', 'use_viewdirs']
                if not compare_args(args, config_file, keys):
                    ans = input('The network parameters are not compatible, reload hyper-parameters? (Y/N)')
                    if ans.lower() == 'y':
                        update_args(args, config_file, keys)

    # Detect checkpoints
    find_checkpoint(args)
    if args.ckpt_file is not None:
        if not os.path.exists(args.ckpt_file):
            print("Error: The specified checkpoint does not exists!")
            exit(-1)

    train_set, test_set, render_sets, bds, hw, extras = load_dataset(args)

    # Create nerf model
    model = NerfNet(args)

    # reloading
    if args.ckpt_file is not None and not args.no_reload:
        print('Reloading model from', args.ckpt_file)
        model.load_net(args.ckpt_file)

    # Run rerendering pipeline
    if args.action == 'exhibit':
        exhibit_pipeline(model, render_sets, args)

    # Exporting volume pipeline
    elif args.action == 'export':
        export_pipeline(model, args, 'frontview')

        """
        # Export from a specific view in rendering set
        key, idx = 'test', 3
        export_pipeline(model, args, f"{start_iter}_{key}_{idx}", render_set=render_sets[key], idx=idx)
        """

    # Run training pipeline
    elif args.action == 'train':
        # Start training
        if args.dataset_type == 'pointcloud':
            pc2nerf_pipeline(model, train_set, test_set, render_sets, args)
        else:
            train_pipeline(model, train_set, test_set, render_sets, args)

    elif args.action == 'video':
        if 'exhibit' not in render_sets:
            print("No exhibition dataset available!")
            exit(-1)
        video_pipeline(model, render_sets['exhibit'], args)

    elif args.action == 'campose':
        # search_for_campose(model, args, math.pi*20./32., math.pi*5./32.)
        # search_for_campose(model, args, math.pi*20./32., math.pi*20./32.)
        print("No implementation!")
        exit(-1)

    else:
        print("Unknown action:", args.action)
        exit(-1)




