import os, sys
import math, random
import numpy as np
import torch
import json
import cv2
from pdb import set_trace as st

from data.gen_dataset import generate_dataset
from utils.misc import *

class BaseNeRFDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, args, split='train', subsample=0, cam_id=False, rgb=True, use_masks=True, bin_thres=0.3, ret_k=False):

        super().__init__()

        self.split = split

        # if dataset not generated yet
        if not os.path.exists(os.path.join(root_dir, 'meta.json')):
            print('Dataset not prepared, generating rays for dataset ...')
            generate_dataset(args, root_dir)

        # Read metadata
        with open(os.path.join(root_dir, 'meta.json'), 'r') as f:
            self.meta_dict = json.load(f)
            
            required_keys = ['near', 'far']
            if not np.all([(k in self.meta_dict) for k in required_keys]):
                raise IOError('Missing required meta data')
        
        # Construct loaded filename
        rgbs_name, rays_name = 'rgbs_' + split, 'rays_' + split
        if use_masks:
            masks_name = 'masks_' + split
        if ret_k:
            poses_name = 'poses_' + split
        # add subsample suffix
        if subsample != 0:
            rgbs_name, rays_name = rgbs_name + f'_x{subsample}', rays_name + f'_x{subsample}'
        # add extension name
        rgbs_name, rays_name = rgbs_name + '.npy', rays_name + '.npy'
        if use_masks:
            masks_name = masks_name + '.npy'
        if ret_k:
            poses_name = poses_name + '.npy'

        self.rays = np.load(os.path.join(root_dir, rays_name)) # [N, H, W, ro+rd, 3]

        # RGB files may not exist considering exhibit set
        if rgb:
            rgb_path = os.path.join(root_dir, rgbs_name)
            self.rgbs = np.load(rgb_path) # [N, H, W, C]
        if use_masks:
            mask_path = os.path.join(root_dir, masks_name)
            # self.masks_1 = np.load(mask_path)
            # self.masks_0 = 1 - self.masks_1
            # self.masks = np.concatenate([self.masks_0, self.masks_1], -1)
            try:
                self.masks = np.load(mask_path)
            except:
                print(f" Warning! Masks path is wrong, use all one masks")
                self.masks = np.ones([self.rays.shape[0], self.rays.shape[1], self.rays.shape[2], 1])
            if bin_thres != -1:
                self.masks = (self.masks > bin_thres).astype(np.long)
            else:
                self.masks = self.masks.astype(np.float32)
            print(f'> {split} binary threshold: {bin_thres}')

        if ret_k:
            K = np.eye(3).astype(np.float32)
            K[0, 0] = K[1, 1] = self.meta_dict["focal"]
            K[0, -1] = self.meta_dict["W"] / 2.
            K[1, -1] = self.meta_dict["H"] / 2.
            self.K = torch.from_numpy(K).cuda()
            poses_path = os.path.join(root_dir, poses_name)
            try:
                self.poses = np.load(poses_path)
            except:
                print(f"[Warning!] pose path {poses_path} is wrong.")
                self.poses = np.zeros([self.rays.shape[0], 3, 4])
        else:
            self.poses = np.zeros([self.rays.shape[0], 3, 4])
        print(f'> {split} poses shape: {self.poses.shape}')

        # add camera ids
        self.has_cam_id = cam_id
        if cam_id:
            # ids = np.arange(self.rays.shape[0], dtype=np.int64) # [N,]
            # ids = np.reshape(ids, [-1, 1, 1, 1]) # [N, 1, 1, 1]
            # ids = np.tile(ids, (1,)+self.rays.shape[1:2]+(1,)) # [N, H, W, ro+rd, 1]

            # # Sanity check
            # for i in range(self.rays.shape[0]):
            #     assert np.all(ids[i] == i)

            # self.cam_ids = ids # [N, H, W, id]

            self.cam_ids = np.arange(self.rays.shape[0], dtype=np.int64)

        # Basic attributes
        self.height = self.rays.shape[1]
        self.width = self.rays.shape[2]

        self.image_count = self.rays.shape[0]
        self.image_step = self.height * self.width

    def num_images(self):
        return self.image_count
        
    def height_width(self):
        return self.height, self.width

    def near_far(self):
        return self.meta_dict['near'], self.meta_dict['far']

    def radii(self):
        return 2. / max(self.height, self.width) * 2 / math.sqrt(12)

class RayNeRFDataset(BaseNeRFDataset):

    def __init__(self, root_dir, args, split='train', subsample=0, cam_id=False, use_masks=True, bin_thres=0.3):

        super().__init__(root_dir, args, split=split, subsample=subsample, cam_id=cam_id, rgb=True, use_masks=use_masks, bin_thres=bin_thres)

        self.use_masks = use_masks
        # Cast to tensors
        self.rays = torch.from_numpy(self.rays).float()
        self.rgbs = torch.from_numpy(self.rgbs).float()
        if use_masks:
            self.masks = torch.from_numpy(self.masks).long()
        else:
            self.masks = torch.zeros_like(self.rgbs)[..., 0]
        
        self.class_w = weights_log(self.masks)

        if self.has_cam_id:
            self.cam_ids = torch.from_numpy(self.cam_ids).long()
        
        '''
        '''
        rgb = self.rgbs[0, ...].numpy()
        cv2.imwrite("logs/rgb.png", (rgb*255).astype(np.uint8))
        msk = self.masks[0, ...].numpy()
        cv2.imwrite("logs/msk.png", (msk*255).astype(np.uint8))

        if split == 'train':
            self.rays = self.rays.reshape([-1, 2, self.rays.shape[-1]]) # [N * H * W, ro+rd, 3]
            self.rgbs = self.rgbs.reshape([-1, self.rgbs.shape[-1]]) # [N * H * W, 3]
            self.masks = self.masks.reshape([-1, self.masks.shape[-1]]) # [N * H * W, 3]
        else:
            self.rays = self.rays.permute([0, 3, 1, 2, 4]) # [N, ro+rd, H, W, 3]

    def __len__(self):
        return self.rays.shape[0]

    def __getitem__(self, i):
        # Prohibit multiple workers
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:
        #     raise ValueError("Error BatchNerfDataset does not support multi-processing")

        if self.split == 'train' and self.has_cam_id:
            if self.has_cam_id:
                return dict(rays = self.rays[i], target_s = self.rgbs[i], cam_id = self.cam_ids[i // self.image_step]) # rays=[3,], cam_id=[1,]
            else:
                return dict(rays = self.rays[i], target_s = self.rgbs[i], masks = self.masks[i]) # [3,]
        else:
            return dict(rays = self.rays[i], target_s = self.rgbs[i], masks = self.masks[i]) # [3,]


class PatchNeRFDataset(BaseNeRFDataset):

    def __init__(self, root_dir, args, split='train', subsample=0, cam_id=False, use_masks=True, crop_size=32, patch_stride=1, bin_thres=0.3, ret_k=False):

        super().__init__(root_dir, args, split=split, subsample=subsample, cam_id=cam_id, rgb=True, use_masks=use_masks, bin_thres=bin_thres, ret_k=ret_k)

        self.use_masks = use_masks
        self.crop_size = crop_size
        self.patch_stride = patch_stride
        self.ret_k = ret_k

        if ret_k:
            pass

        # Cast to tensors
        self.rays = torch.from_numpy(self.rays).float()
        self.rgbs = torch.from_numpy(self.rgbs).float()
        if use_masks:
            if bin_thres != -1:
                self.masks = torch.from_numpy(self.masks).long()
            else:
                self.masks = torch.from_numpy(self.masks).float()

        else:
            self.masks = torch.zeros_like(self.rgbs)[..., 0]

        self.poses = torch.from_numpy(self.poses).float()

        self.class_w = weights_log(self.masks)

        if self.has_cam_id:
            self.cam_ids = torch.from_numpy(self.cam_ids).long()
        
        '''
        rgb = self.rgbs[0, ...].numpy()
        cv2.imwrite("logs/rgb.png", (rgb*255).astype(np.uint8))
        msk = self.masks[0, ...].numpy()
        cv2.imwrite("logs/msk.png", (msk*255).astype(np.uint8))
        '''

        if split == 'train':
            pass
            # self.rays = self.rays.reshape([-1, 2, self.rays.shape[-1]]) # [N * H * W, ro+rd, 3]
            # self.rgbs = self.rgbs.reshape([-1, self.rgbs.shape[-1]]) # [N * H * W, 3]
            # self.masks = self.masks.reshape([-1, self.masks.shape[-1]]) # [N * H * W, 3]
        else:
            self.rays = self.rays.permute([0, 3, 1, 2, 4]) # [N, ro+rd, H, W, 3]
        
        print(f'> PatchNeRFDataset, rgbs/rays/masks shapes: {self.rgbs.shape} {self.rays.shape} {self.masks.shape}, use masks: {self.use_masks}')
        print(f'> PatchNeRFDataset, crop_size: {self.crop_size}, patch_stride: {self.patch_stride}')
        print(f'> mean of masks:{torch.mean(self.masks.float())}, bin thres: {bin_thres}')

    def __len__(self):
        return self.rays.shape[0]

    def __getitem__(self, i):
        # Prohibit multiple workers
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:
        #     raise ValueError("Error BatchNerfDataset does not support multi-processing")

        if self.split == 'train' and self.has_cam_id:
            if self.has_cam_id:
                return dict(rays = self.rays[i], target_s = self.rgbs[i], cam_id = self.cam_ids[i // self.image_step]) # rays=[3,], cam_id=[1,]
            else:
                return dict(rays = self.rays[i], target_s = self.rgbs[i], masks = self.masks[i]) # [3,]
        else:
            h_idx = random.randint(0, self.height-self.crop_size)
            w_idx = random.randint(0, self.width-self.crop_size)
            ray_sample = self.rays[i]
            rgb_sample = self.rgbs[i]
            msk_sample = self.masks[i]
            rays = ray_sample[h_idx:h_idx+self.crop_size:self.patch_stride, w_idx:w_idx+self.crop_size:self.patch_stride, :]
            rgbs = rgb_sample[h_idx:h_idx+self.crop_size:self.patch_stride, w_idx:w_idx+self.crop_size:self.patch_stride, :]
            masks = msk_sample[h_idx:h_idx+self.crop_size:self.patch_stride, w_idx:w_idx+self.crop_size:self.patch_stride, :]
            rays = rays.reshape([-1, 2, rays.shape[-1]])
            rgbs = rgbs.reshape([-1, rgbs.shape[-1]]) 
            masks = masks.reshape([-1, masks.shape[-1]])
            pose = self.poses[i]
            start_idx = torch.Tensor([h_idx, w_idx])
            # print(f"> iter  height {self.height} width {self.width} crop_size {self.crop_size} h_idx {h_idx} w_idx {w_idx}  h_idx+crop_size {h_idx+self.crop_size} w_idx+crop_size {w_idx+self.crop_size}")
            return dict(rays = rays, target_s = rgbs, masks = masks, poses = pose, start_idx=start_idx) # [3,]


class ViewNeRFDataset(BaseNeRFDataset):

    def __init__(self, root_dir, batch_size, args, split='train', subsample=0, cam_id=False, precrop_iters=0, precrop_frac=0.5, start_iters=0):

        super().__init__(root_dir, args, split=split, subsample=subsample, cam_id=cam_id, rgb=True)

        self.batch_size = batch_size
        self.precrop_iters = precrop_iters
        self.precrop_frac = precrop_frac
        self.counter = start_iters
        self.start_iters = start_iters

        # Cast to tensors
        self.rays = torch.from_numpy(self.rays).float()
        self.rgbs = torch.from_numpy(self.rgbs).float()
        if self.has_cam_id:
            self.cam_ids = torch.from_numpy(self.cam_ids).long()

        self.rays = self.rays.permute([0, 3, 1, 2, 4]) # [N, ro+rd, H, W, 3]

    def __len__(self):
        return self.rays.shape[0]

    def __getitem__(self, i):

        self.counter += 1

        rays_o, rays_d = self.rays[i]
        target = self.rgbs[i]

        N_rand = self.batch_size
        H, W = self.height_width()
        if self.counter < self.precrop_iters:
            dH = int(H//2 * self.precrop_frac)
            dW = int(W//2 * self.precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                    torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                ), -1)
            if self.counter == self.start_iters + 1:
                print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {self.precrop_iters}", self.counter, self.start_iters)                
        else:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 1)              # (N_rand, 2, 3)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        if self.split == 'train':
            if self.has_cam_id:
                return dict(rays = batch_rays, target_s = target_s, cam_id = self.cam_ids[i]) # rays=[N, 3], cam_id=[1,]
            else:
                return dict(rays = batch_rays, target_s = target_s) # [N, ro+rd, 3], [N, 3]
        else:
            return dict(rays = batch_rays, target_s = target_s) # [N, ro+rd, 3]

# Containing only rays for rendering, no rgb groundtruth
class ExhibitNeRFDataset(BaseNeRFDataset):

    def __init__(self, root_dir, args, subsample=0, use_semantics=False):
        super().__init__(root_dir, args, split='exhibit', subsample=subsample, cam_id=False, rgb=False, use_masks=use_semantics)

        self.rays = torch.from_numpy(self.rays).float()
        self.rays = self.rays.permute([0, 3, 1, 2, 4]) # [N, ro+rd, H, W, 3(+id)]

    def __len__(self):
        # return self.image_count * self.height * self.width
        return self.rays.shape[0]

    def __getitem__(self, i):
        return dict(rays = self.rays[i]) # [H, W, 3]
