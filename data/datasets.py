import os, sys
import math, random
import numpy as np
import torch
import json

class BaseNeRFDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, split='train', subsample=0, cam_id=False, rgb=True):

        super().__init__()

        self.split = split

        # Read metadata
        with open(os.path.join(root_dir, 'meta.json'), 'r') as f:
            self.meta_dict = json.load(f)
            
            required_keys = ['near', 'far']
            if not np.all([(k in self.meta_dict) for k in required_keys]):
                raise IOError('Missing required meta data')
        
        # Construct loaded filename
        rgbs_name, rays_name = 'rgbs_' + split, 'rays_' + split
        # add subsample suffix
        if subsample != 0:
            rgbs_name, rays_name = rgbs_name + f'_x{subsample}', rays_name + f'_x{subsample}'
        # add extension name
        rgbs_name, rays_name = rgbs_name + '.npy', rays_name + '.npy'

        self.rays = np.load(os.path.join(root_dir, rays_name)) # [N, H, W, ro+rd, 3]

        # RGB files may not exist considering exhibit set
        if rgb:
            rgb_path = os.path.join(root_dir, rgbs_name)
            self.rgbs = np.load(rgb_path) # [N, H, W, C]
        
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

class RayNeRFDataset(BaseNeRFDataset):

    def __init__(self, root_dir, split='train', subsample=0, cam_id=False):

        super().__init__(root_dir, split=split, subsample=subsample, cam_id=cam_id, rgb=True)

        # Cast to tensors
        self.rays = torch.from_numpy(self.rays).float()
        self.rgbs = torch.from_numpy(self.rgbs).float()
        if self.has_cam_id:
            self.cam_ids = torch.from_numpy(self.cam_ids).long()

        if split == 'train':
            self.rays = self.rays.reshape([-1, 2, self.rays.shape[-1]]) # [N * H * W, ro+rd, 3]
            self.rgbs = self.rgbs.reshape([-1, self.rgbs.shape[-1]]) # [N * H * W, 3]
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
                return dict(rays = self.rays[i], target_s = self.rgbs[i]) # [3,]
        else:
            return dict(rays = self.rays[i], target_s = self.rgbs[i]) # [3,]

class ViewNeRFDataset(BaseNeRFDataset):

    def __init__(self, root_dir, batch_size, split='train', subsample=0, cam_id=False, precrop_iters=0, precrop_frac=0.5, start_iters=0):

        super().__init__(root_dir, split=split, subsample=subsample, cam_id=cam_id, rgb=True)

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

        # print(self.counter, self.precrop_iters, self.precrop_frac, self.start_iters, batch_rays.shape, target_s.shape)

        if self.split == 'train':
            if self.has_cam_id:
                return dict(rays = batch_rays, target_s = target_s, cam_id = self.cam_ids[i]) # rays=[N, 3], cam_id=[1,]
            else:
                return dict(rays = batch_rays, target_s = target_s) # [N, ro+rd, 3], [N, 3]
        else:
            return dict(rays = batch_rays, target_s = target_s) # [N, ro+rd, 3]

# Containing only rays for rendering, no rgb groundtruth
class ExhibitNeRFDataset(BaseNeRFDataset):

    def __init__(self, root_dir, subsample=0):
        super().__init__(root_dir, split='exhibit', subsample=subsample, cam_id=False, rgb=False)

        self.rays = torch.from_numpy(self.rays).float()
        self.rays = self.rays.permute([0, 3, 1, 2, 4]) # [N, ro+rd, H, W, 3(+id)]

    def __len__(self):
        # return self.image_count * self.height * self.width
        return self.rays.shape[0]

    def __getitem__(self, i):
        return dict(rays = self.rays[i]) # [H, W, 3]

# def load_dataset(dataset_path, subsample=0, cam_id=False, device=torch.device("cpu")):

#     if not os.path.isdir(dataset_path):
#         raise ValueError("No such directory containing dataset:", dataset_path)

#     train_set = BatchNeRFDataset(dataset_path, subsample=subsample, split='train', cam_id=cam_id, device=device)
#     test_set = BatchNeRFDataset(dataset_path, subsample=subsample, split='test', cam_id=True, device=device)

#     exhibit_set = None
#     try:
#         exhibit_set = ExhibitNerfDataset(dataset_path, subsample=subsample, device=device)
#     except FileNotFoundError:
#         print("Warning: No exhibit set!")

#     return train_set, test_set, exhibit_set
