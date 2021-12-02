import os, sys
import numpy as np
import torch

# Pre-defined collaters

class RayBatchCollater:
    def __init__(self):
        pass

    def __call__(self, xs):
        batch_rays = torch.stack([torch.as_tensor(x['rays']) for x in xs], 0)
        batch_rays = torch.transpose(batch_rays, 0, 1)
        
        batch_rgbs = None
        if 'target_s' in xs[0]:
            batch_rgbs = torch.stack([torch.as_tensor(x['target_s']) for x in xs], 0)

        batch_cam_ids = None
        if 'cam_id' in xs[0]:
            batch_cam_ids = torch.stack([torch.as_tensor(x['cam_id']) for x in xs], 0)
            return batch_rays, batch_rgbs, batch_cam_ids

        return batch_rays, batch_rgbs

class ViewBatchCollater:
    def __init__(self):
        pass

    def __call__(self, xs):
        batch_rays = torch.cat([torch.as_tensor(x['rays']) for x in xs], 0)
        batch_rays = torch.transpose(batch_rays, 0, 1)

        # When under exhibit mode, no groundtruth will be given
        batch_rgbs = None
        if 'target_s' in xs[0]:
            batch_rgbs = torch.cat([torch.as_tensor(x['target_s']) for x in xs], 0)

        batch_cam_ids = None
        if 'cam_id' in xs[0]:
            batch_cam_ids = torch.cat([
                torch.full((batch_rays.shape[0],), x['cam_id'], dtype=torch.int64) for x in xs
            ], 0)
            return batch_rays, batch_rgbs, batch_cam_ids

        return batch_rays, batch_rgbs

class ExhibitCollater:
    def __init__(self, H, W):
        self.H, self.W = H, W

    def __call__(self, xs):
        batch_rays = torch.stack([torch.as_tensor(x['rays']) for x in xs], 0)
        batch_rays = torch.transpose(batch_rays, 0, 1)
        batch_rays = batch_rays.reshape((batch_rays.shape[0], self.H, self.W, batch_rays.shape[-1]))

        # When under exhibit mode, no groundtruth will be given
        batch_rgbs = None
        if 'target_s' in xs[0]:
            batch_rgbs = torch.stack([torch.as_tensor(x['target_s']) for x in xs], 0)
            batch_rgbs = batch_rgbs.reshape((self.H, self.W, batch_rgbs.shape[-1]))

        return batch_rays, batch_rgbs

class PointBatchCollater:
    def __init__(self):
        pass

    def __call__(self, xs):
        batch_pts = torch.stack([torch.as_tensor(x['pts']) for x in xs], 0)
        batch_rgbs = torch.stack([torch.as_tensor(x['target_s']) for x in xs], 0)
#         batch_rgbs = F.relu(batch_rgbs)
        return batch_pts, batch_rgbs