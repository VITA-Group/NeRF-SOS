import os, sys
import math, random
import numpy as np
import torch
import json

class NerfDataset(object):
    
    def __init__(self, root_dir, subsample=0, suffix=None, no_cam_id=False,
                 device=torch.device("cpu")):
        # Read metadata
        with open(os.path.join(root_dir, 'meta.json'), 'r') as f:
            self.meta_dict = json.load(f)
            
            required_keys = ['near', 'far']
            if not np.all([(k in self.meta_dict) for k in required_keys]):
                raise IOError('Missing required meta data')
        
        # Construct loaded filename
        rgbs_name, rays_name, mask_name = 'rgbs' + suffix, 'rays' + suffix, 'masks' + suffix
            
        if subsample != 0:
            rgbs_name += f'_x{subsample}'
            rays_name += f'_x{subsample}'
            mask_name += f'_x{subsample}'

        # add suffix
        rgbs_name += '.npy'
        rays_name += '.npy'
        mask_name += '.npy'

        print("Loading nerf data:", root_dir)
        self.rays = np.load(os.path.join(root_dir, rays_name)) # [N, H, W, ro+rd, 3]
        
        # RGB files may not exist considering exhibit set
        rgb_path = os.path.join(root_dir, rgbs_name)
        if os.path.exists(rgb_path):
            self.rgbs = np.load(rgb_path) # [N, H, W, C]
            # self.rgbs = self.rgbs[..., None]
        else:
            self.rgbs = np.zeros((1,), dtype=np.float32) # fake rgbs

        # if mask not exists, generate a white mask
        mask_path = os.path.join(root_dir, mask_name)
        if os.path.exists(mask_path):
            self.mask = np.load(mask_path) # [N, H, W]
        else:
            self.mask = np.ones(self.rgbs.shape[:-1], dtype=np.float32) # all-pass mask [N, H, W]

        print("Dataset loaded:", self.rays.shape, self.rgbs.shape, self.mask.shape)
        
        if not no_cam_id:
            ids = np.arange(self.rays.shape[0], dtype=np.float32) # [N,]
            ids = np.reshape(ids, [-1, 1, 1, 1, 1]) # [N, 1, 1, 1, 1]
            ids = np.tile(ids, (1,)+self.rays.shape[1:-1]+(1,)) # [N, H, W, ro+rd, 3]

            # Necessary check
            for i in range(self.rays.shape[0]):
                assert np.all(ids[i] == i)

            self.rays = np.concatenate([self.rays, ids], -1) # [N, H, W, ro+rd, 3+id]
            print('Done, add ids', self.rays.shape)
        
        # Cast to tensors
        self.rays = torch.from_numpy(self.rays).float().to(device)
        self.rgbs = torch.from_numpy(self.rgbs).float().to(device)
        self.mask = torch.from_numpy(self.mask).float().to(device)

        # Basic attributes
        self.height = self.rays.shape[1]
        self.width = self.rays.shape[2]

        self.image_count = self.rays.shape[0]
        self.image_step = self.height * self.width
        
        self.rays = self.rays.reshape([-1, 2, self.rays.shape[-1]]) # [N * H * W, ro+rd, 3(+id)]
        self.rgbs = self.rgbs.reshape([-1, self.rgbs.shape[-1]]) # [N * H * W, 3(+id)]
        self.mask = self.mask.reshape([-1, 1]) # [N * H * W, 1]
        print("Dataset reshaped:", self.rays.shape, self.rgbs.shape, self.mask.shape)

        """
        # replace background through mask
        if bg_color is not None:
            # bg_idx, _ = torch.where(self.mask < 0) # [N * H * W]
            # self.rgbs[bg_idx, :] = torch.Tensor(bg_color).to(self.rgbs.device)
            # print("Background replaced:", self.rgbs.shape)
            mask_idx, _ = torch.where(self.mask > 0) # [N * H * W]
            self.rays = self.rays[mask_idx, ...]
            self.rgbs = self.rgbs[mask_idx, ...]
            print("Dataset masked:", self.rays.shape, self.rgbs.shape)
        """

    def invert_mask(self):
        """
        mask_idx, _ = torch.where(self.mask > 0) # [N * H * W]
        self.rays = self.rays[mask_idx, ...]
        self.rgbs = self.rgbs[mask_idx, ...]
        print("Dataset masked:", self.rays.shape, self.rgbs.shape)
        """

        bg_idx, _ = torch.where(self.mask < 0) # [N * H * W]
        self.rgbs[mask_idx, :] = torch.Tensor([0., 1., 0.]).to(self.rgbs.device)

    def num_images(self):
        return self.image_count
        
    def height_width(self):
        return self.height, self.width

    def near_far(self):
        return self.meta_dict['near'], self.meta_dict['far']

class BatchNerfDataset(NerfDataset, torch.utils.data.Dataset):

    def __init__(self, root_dir, subsample=0, testset=False, no_cam_id=False,
                 device=torch.device("cpu")):
        torch.utils.data.Dataset.__init__(self)
        NerfDataset.__init__(self, root_dir, subsample, '_test' if testset else '_train',
                             no_cam_id=no_cam_id, device=device)

    def __len__(self):
        # return self.image_count * self.height * self.width
        return self.rays.shape[0]

    def __getitem__(self, i):
        # Prohibit multiple workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise ValueError("Error BatchNerfDataset does not support multi-processing")

        # Mask out the background (set to zeros)
        return dict(rays = self.rays[i], target_s = self.rgbs[i])
        # return dict(rays = self.rays[i], target_s = self.rgbs[i], mask = self.mask[i])

class MaskedNerfDataset(torch.utils.data.Dataset):

    def __init__(self, nerf_dataset):
        super(MaskedNerfDataset, self).__init__()
        self.dataset = nerf_dataset
        self.fg_idx, _ = torch.where(self.dataset.mask > 0)

    def __len__(self):
        return self.fg_idx.shape[0]

    def __getitem__(self, i):
        # Prohibit multiple workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise ValueError("Error MaskedNerfDataset does not support multi-processing")

        return dict(rays = self.dataset.rays[self.fg_idx[i]],
                    target_s = self.dataset.rgbs[self.fg_idx[i]])

    def num_images(self):
        return self.dataset.num_images()
        
    def height_width(self):
        return self.dataset.height_width()

    def near_far(self):
        return self.dataset.near_far()

class ScreenNerfDataset(torch.utils.data.Dataset):

    def __init__(self, nerf_dataset, bg_color=[0., 1., 0.]):
        super(ScreenNerfDataset, self).__init__()
        self.dataset = nerf_dataset
        self.bg_color = torch.Tensor(bg_color).to(self.dataset.rgbs.device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        # Prohibit multiple workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise ValueError("Error ScreenNerfDataset does not support multi-processing")

        target_s = self.dataset.rgbs[i] * self.dataset.mask[i] + self.bg_color * (1. - self.dataset.mask[i])
        # print(target_s, self.dataset.rgbs[i], self.dataset.mask[i])
        return dict(rays = self.dataset.rays[i], target_s = target_s)

    def num_images(self):
        return self.dataset.num_images()
        
    def height_width(self):
        return self.dataset.height_width()

    def near_far(self):
        return self.dataset.near_far()

# Containing only rays for rendering, no rgb groundtruth
class ExhibitNerfDataset(NerfDataset, torch.utils.data.Dataset):

    def __init__(self, root_dir, subsample=0, device=torch.device("cpu")):
        torch.utils.data.Dataset.__init__(self)
        NerfDataset.__init__(self, root_dir, subsample, '_exhibit', True, device)

    def __len__(self):
        return self.image_count * self.height * self.width

    def __getitem__(self, i):
        return dict(rays = self.rays[i])

def load_nerf(basedir, no_batch, batch_size, subsample=0, no_cam_id=False,
                max_train_vis=10, max_test_vis=10, device=torch.device("cpu")):

    if not os.path.isdir(basedir):
        raise ValueError("No such directory containing dataset:", basedir)
    
    train_set = BatchNerfDataset(basedir, subsample=subsample, testset=False, 
                                 no_cam_id=no_cam_id, device=device)
    test_set = BatchNerfDataset(basedir, subsample=subsample, testset=True, 
                                no_cam_id=True, device=device)

    H, W = train_set.height_width()
    near, far = train_set.near_far()
    extras = {}

    def pick_rays(dataset, max_vis):
        frame_indx = np.linspace(0, dataset.num_images() - 1, max_vis, dtype=np.int32)
        print("Rendered image index:", frame_indx)

        pick_indx = np.array([np.arange(i*H*W, (i+1)*H*W) for i in frame_indx])
        pick_indx = pick_indx.reshape(-1)
        
        return torch.utils.data.Subset(dataset, pick_indx)

    print("Picking rendering set ...")
    
    train_render = pick_rays(train_set, max_train_vis)
    test_render = pick_rays(test_set, max_test_vis)    
    render_sets = {'train': train_render, 'test': test_render}

    # train_set = MaskedNerfDataset(train_set)
    # train_set = ScreenNerfDataset(train_set, bg_color=[0.5, 0.5, 0.5])

    try:
        exhibit_render = ExhibitNerfDataset(basedir, subsample=subsample, device=device)
        render_sets['exhibit'] = exhibit_render
    except FileNotFoundError:
        print("No exhibit set!")

    extras['data_device'] = device
    extras['num_train_images'] = train_set.num_images()
    extras['num_test_images'] = test_set.num_images()
    extras['num_per_image_pixels'] = H * W
    
    print("Done, loading nerf data.")

    return train_set, test_set, render_sets, (near, far), (H, W), extras
