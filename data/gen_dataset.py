import os, sys
import numpy as np
import json
import random
import time

from tqdm import tqdm, trange
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    sys.path.append('..')

from utils.ray import get_persp_rays, get_persp_intrinsic

from data.load_llff import load_llff_data
from data.load_deepvoxels import load_dv_data
from data.load_LINEMOD import load_LINEMOD_data
from data.load_blender import load_blender_data

import configargparse


def create_arg_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Path to config file')
    parser.add_argument('--data_type', '--dataset_type', type=str, required=True, help='Dataset type',
        choices=['llff', 'blender', 'LINEMOD', 'deepvoxels'])
    parser.add_argument('--data_path', '--datadir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_path', type=str, default='', help='Path to save processed dataset directory')

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

    return parser

def generate_dataset(args, output_path):

    if not os.path.exists(args.data_path):
        print('Dataset path not exists:', args.data_path)
        exit(-1)

    K = None # intrinsic matrix
    if args.data_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.data_path, factor=args.factor,
            recenter=True, bd_factor=.75, spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.data_path)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])

        if args.ndc:
            near = 0.
            far = 1.
        else:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        print('NEAR FAR', near, far)

    elif args.data_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.data_path, args.half_res, args.test_skip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.data_path)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.data_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.data_path, args.half_res, args.test_skip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.data_type == 'deepvoxels':
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.dv_scene, basedir=args.data_path, testskip=args.test_skip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.data_path)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type:', args.data_type)
        exit(-1)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    if K is None:
        K = get_persp_intrinsic(H, W, focal)
    print('Intrinsic matrix:', K)
    print('Train/valid/test split', i_train, i_val, i_test)

    print('Calculating train/valid/test rays ...')
    rays = torch.stack([get_persp_rays(H, W, K, torch.tensor(p)) for p in tqdm(poses[:,:3,:4])], 0) # [N, ro+rd, H, W, 3]
    rays = rays.permute([0, 2, 3, 1, 4]).numpy().astype(np.float32) # [N, H, W, ro+rd, 3]
    print('Done.', rays.shape)

    print('Splitting train/valid/test rays ...')
    rays_train, rgbs_train = rays[i_train], images[i_train]
    rays_val, rgbs_val = rays[i_val], images[i_val]
    rays_test, rgbs_test = rays[i_test], images[i_test]

    print('Calculating exhibition rays ...')
    rays_exhibit = torch.stack([get_persp_rays(H, W, K, torch.tensor(p)) for p in tqdm(render_poses[:,:3,:4])], 0) # [N, ro+rd, H, W, 3]
    rays_exhibit = rays_exhibit.permute([0, 2, 3, 1, 4]).numpy().astype(np.float32) # [N, H, W, ro+rd, 3]
    print('Done.', rays_exhibit.shape)

    print('Training set:', rays_train.shape, rgbs_train.shape)
    print('Validation set:', rays_val.shape, rgbs_val.shape)
    print('Testing set:', rays_test.shape, rgbs_test.shape)
    print('Exhibition set:', rays_exhibit.shape)

    print('Saving to: ', output_path)
    np.save(os.path.join(output_path, 'rays_train.npy'), rays_train)
    np.save(os.path.join(output_path, 'rgbs_train.npy'), rgbs_train)

    np.save(os.path.join(output_path, 'rays_val.npy'), rays_val)
    np.save(os.path.join(output_path, 'rgbs_val.npy'), rgbs_val)

    np.save(os.path.join(output_path, 'rays_test.npy'), rays_test)
    np.save(os.path.join(output_path, 'rgbs_test.npy'), rgbs_test)

    np.save(os.path.join(output_path, 'rays_exhibit.npy'), rays_exhibit)

    # Save meta data
    meta_dict = {
        'H': H, 'W': W, 'focal': float(focal),
        'near': float(near), 'far': float(far),

        'i_train': i_train.tolist(), 'i_val': i_val.tolist(), 'i_test': i_test.tolist(),

        'ndc': args.ndc, 'factor': args.factor,
        'spherify': args.spherify, 'llffhold': args.llffhold,

        'half_res': args.half_res, 'white_bkgd': args.white_bkgd,
        'test_skip': args.test_skip, 'dv_scene': args.dv_scene
    }
    print("Meta data:", meta_dict)
    with open(os.path.join(output_path, 'meta.json'), 'w') as f:
        json.dump(meta_dict, f)

if __name__ == '__main__':

    parser = create_arg_parser()
    args, _ = parser.parse_known_args()

    output_path = args.output_path
    if not args.output_path:
        output_path = args.data_path
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    generate_dataset(args, output_path)
