
import os, sys
import math, random, time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import imageio
import lpips
from pdb import set_trace as st
try:
    import utils.ssim as ssim_utils
except:
    import ssim as ssim_utils

try:
    import pytorch3d
except:
    print(f"[Import Error]! import pytorch3d failed")

try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import matplotlib as mpl
    from matplotlib import cm
    import cv2
except:
    print(f"[Import Error]! import matplotlib failed")


lpips_alex = lpips.LPIPS(net='alex') # best forward scores
lpips_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization


color_pallete = {
    0:(0, 0, 0), 
    1:(255, 255, 255), 2:(214, 168, 168), 3:(50, 115, 53), #door
    4:(177, 78, 78), 5:(87, 41, 41), 6:(29, 14, 14),       #door
    7:(222, 210, 179), 8:(168, 186, 214), 9:(111, 140, 187), 10:(68, 97, 144), #window
    28:(244, 247, 50), #stairs
    18:(247, 214, 253), 20:(231, 132, 250), 24: (103, 5, 123),#home appliance
    11:(245, 250, 245), 12:(224, 241, 225), 13:(203, 232, 204), 14:(182, 223, 184), #furniture
    15:(161, 214, 164), 16:(140, 205, 143), 17:(119, 196, 123), 19:(98, 187, 103), #furniture
    21:(78, 177, 83), 22:(68, 157, 73), 23:(59, 136, 63), 25:(50, 115, 53), 
    26:(41, 94, 44), 27:(32, 73, 34),       #furniture
    29:(239, 220, 220), 30:(177, 78, 78),   #equipment
    31:(222, 210, 179), 32:(200, 180, 128), 33:(177, 150, 78), 34:(127, 107, 55) #stuff
    }


def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None):
    fig = Figure(figsize=(1.2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = ['{:3.2f}'.format(x) for x in tick_loc]
    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(x, cmap_name='jet', mask=None, append_cbar=False):
    if mask is not None:
        # vmin, vmax = np.percentile(x[mask], (1, 99))
        vmin = np.min(x[mask])
        vmax = np.max(x[mask])
        vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        x = np.clip(x, vmin, vmax)
        # print(vmin, vmax)
    else:
        vmin = x.min()
        vmax = x.max() + 1e-5

    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.zeros_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name)

    if append_cbar:
        x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new, cbar

# Misc
def img2mse(x, y, reduction='mean'):
    diff = torch.mean((x - y) ** 2, -1)
    if reduction == 'mean':
        return torch.mean(diff)
    elif reduction == 'sum':
        return torch.sum(diff)
    elif reduction == 'none':
        return diff

def mse2psnr(x):
    if isinstance(x, float):
        x = torch.tensor([x])
    return -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

def ssim(img1, img2, window_size = 11, size_average = True, format='NCHW'):
    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    return ssim_utils.ssim(img1, img2, window_size, size_average)

def lpips(img1, img2, net='alex', format='NCHW'):
    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    if net == 'alex':
        return lpips_alex(img1, img2)
    elif net == 'vgg':
        return lpips_vgg(img1, img2)

def to8b(x):
    return (255*(x-x.min())/(x.max()-x.min())).astype(np.uint8)

def export_images(rgbs, save_dir, H=0, W=0):
    rgb8s = []
    for i, rgb in enumerate(rgbs):
        # Resize
        if H > 0 and W > 0:
            rgb = rgb.reshape([H, W])

        filename = os.path.join(save_dir, '{:03d}.npy'.format(i))
        np.save(filename, rgb)
        
        # Convert to image
        rgb8 = to8b(rgb)
        filename = os.path.join(save_dir, '{:03d}.png'.format(i))
        imageio.imwrite(filename, rgb8)
        rgb8s.append(rgb8)
    
    return np.stack(rgb8s, 0)

def export_video(rgbs, save_path, fps=30, quality=8):
    imageio.mimwrite(save_path, to8b(rgbs), fps=fps, quality=quality)


def get_similarity_matrix(x):
    batch_size = x.shape[0]
    similarity_matrix = F.cosine_similarity(x.unsqueeze(0), x.unsqueeze(1), dim=2)
    return similarity_matrix

class NeRFContrastive(nn.Module):
    def __init__(self, temperature=1, device=None, verbose=False, min_max_contrast=True):
        super().__init__()
        self.device = device
        self.verbose = verbose
        self.min_max_contrast = min_max_contrast
        self.register_buffer("temperature", torch.tensor(temperature).to(device))

    def forward(self, embeddings):
        self.batch_size = embeddings.shape[0]
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        
        if self.min_max_contrast:
            mask = torch.eye(embeddings.shape[0], dtype=torch.bool).to(self.device)
            similarity_matrix = similarity_matrix[~mask]
            min = similarity_matrix[torch.argmin(similarity_matrix)]
            max = similarity_matrix[torch.argmax(similarity_matrix)]
            loss =  -torch.log(max / (max + min))
            if self.verbose:
                pass
                # print()
                # print(f"similarity_matrix: {similarity_matrix}")
                # print(f"max sim: {max}, min sim: {min}")
        else:
            raise NotImplementedError 
        
        return loss


class ContrastiveLossELI5(nn.Module):
    def __init__(self, batch_size, temperature=0.5, verbose=True):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
 
        representations = torch.cat([z_i, z_j], dim=0)
        st()
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
            
        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]), 0.0)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
                
            loss_ij = -torch.log(numerator / denominator)
                
            return loss_ij.squeeze(0)
 
        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss



class CorrelationLoss(nn.Module):

    def __init__(self, args=None):
        super(CorrelationLoss, self).__init__()
        self.zero_clamp = True
        self.stabalize = False
        self.pointwise = True
        self.feature_samples = 11
        self.self_shift = 0.18
        self.self_weight = 0.67
        self.neg_shift = 0.46
        self.neg_weight = 0.63
        self.verbose = False
        self.rand_neg = args.rand_neg

        if args is not None:
            self.self_corr_w = args.self_corr_w
            self.use_sim_matrix = args.use_sim_matrix
            app_corr_params = args.app_corr_params
            app_corr_params = [float(x) for x in app_corr_params]
            self.self_shift, self.self_weight, self.neg_shift, self.neg_weight = app_corr_params
        else:
            print("[Warning!] args is None")
            self.self_corr_w = 1
            self.use_sim_matrix = True
        print(f"> self_corr_w in CorrelationLoss is: {self.self_corr_w}, app_corr_params: {app_corr_params}")
        if self.rand_neg:
            print(f"Warning: rand_neg is set {self.rand_neg}")

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2
    
    def tensor_correlation(self, a, b):
        return torch.einsum("nchw,ncij->nhwij", a, b)
    
    def norm(self, t):
        return F.normalize(t, dim=1, eps=1e-10)

    def sample(self, t: torch.Tensor, coords: torch.Tensor):
        return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)
    
    def super_perm(self, size: int, device: torch.device):
        perm = torch.randperm(size, device=device, dtype=torch.long)
        perm[perm == torch.arange(size, device=device)] += 1
        return perm % size

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = self.tensor_correlation(self.norm(f1), self.norm(f2))

            if self.pointwise: # True #
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = self.tensor_correlation(self.norm(c1), self.norm(c2))

        if self.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor,
                orig_code: torch.Tensor,
                sim_matrix: torch.Tensor
                ):

        coord_shape = [orig_feats.shape[0], self.feature_samples, self.feature_samples, 2] # what's feature_samples?

        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1 # coord_shape: [16, 11, 11, 2]  * 2 - 1  let range [-1, 1]
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1 # coord_shape: [16, 11, 11, 2]

        feats = self.sample(orig_feats, coords1)
        code = self.sample(orig_code, coords1)

        # find negtive pair
        if sim_matrix is None:
            neg_indx = self.super_perm(orig_feats.shape[0], orig_feats.device)
        else:
            assert len(sim_matrix.shape) == 2
            neg_indx = torch.min(sim_matrix, dim=0)[1]
        
        if self.rand_neg:
            neg_indx = torch.randperm(sim_matrix.shape[0], device=orig_feats.device, dtype=torch.long)

        neg_feats = orig_feats[neg_indx]
        neg_code = orig_code[neg_indx]
        neg_feats = self.sample(neg_feats, coords2)
        neg_code = self.sample(neg_code, coords2)

        neg_corr_loss, neg_corr_cd = self.helper(
            feats, neg_feats, code, neg_code, self.neg_shift)
        
        self_corr_loss, self_corr_cd = self.helper(
            feats, feats, code, code, self.self_shift)

        return self.neg_weight * neg_corr_loss.mean() + self.self_weight * self_corr_loss.mean()


class GeoCorrelationLoss(CorrelationLoss):
    def __init__(self, args=None):
        super(GeoCorrelationLoss, self).__init__(args)
        self.zero_clamp = True
        self.stabalize = False
        self.pointwise = True
        self.self_shift = 3
        self.self_weight = 0.67
        self.neg_shift = 10
        self.neg_weight = 0.63
        self.verbose = False
        self.max_depth = 15
        self.rand_neg = args.rand_neg
        
        if args is not None:
            self.self_corr_w = args.self_corr_w
            self.ps = args.patch_stride
            self.use_sim_matrix = args.use_sim_matrix
            geo_corr_params = args.geo_corr_params
            geo_corr_params = [float(x) for x in geo_corr_params]
            self.self_shift, self.self_weight, self.neg_shift, self.neg_weight = geo_corr_params
        else:
            print("[Warning!] args is None")
            self.self_corr_w = 1
            self.ps = 8
            self.use_sim_matrix = True
        print(f"> self_corr_w in GeoCorrelationLoss is: {self.self_corr_w}, app_corr_params: {geo_corr_params}")

        if self.rand_neg:
            print(f"Warning: rand_neg is set {self.rand_neg}")

    def tensor_correlation(self, a, b, is_f=False):
        x = a.unsqueeze(-1).unsqueeze(-1)
        y = b.unsqueeze(2).unsqueeze(3)
        ret = torch.sum(torch.abs(x-y), dim=1)
        ret = torch.abs(ret)
        ret = ret + (torch.ones_like(ret)*5e-2).to(ret.device)
        ret = 1 / ret
        ret[ret > self.max_depth] = torch.Tensor([self.max_depth]).to(ret.device)
        # ret = ret / torch.max(ret)
        return ret
    
    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = self.tensor_correlation(f1, f2, is_f=True)
            # fd = self.tensor_correlation(self.norm(f1), self.norm(f2))

            if self.pointwise: # True #
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = self.tensor_correlation(self.norm(c1), self.norm(c2))

        if self.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def depth2pts(self, depth, batch_rays):
        batch, channel, patch_h, patch_w = depth.shape
        ray_o, ray_d, rgbs = batch_rays
        XYZ = ray_o + ray_d * depth
        XYZ = XYZ.reshape(batch, -1, patch_h*patch_w)
        XYZ = XYZ.view(batch, 3, patch_h, patch_w)
        return XYZ
    
    def forward(self,
        orig_feats: torch.Tensor,
        orig_code: torch.Tensor,
        batch_rays: torch.Tensor,
        sim_matrix: torch.Tensor
        ):
        # depth filter
        orig_feats[orig_feats > self.max_depth] = orig_feats[orig_feats < self.max_depth].max()

        # orig_feats = 1 / torch.max(orig_feats, (torch.ones_like(orig_feats) * 1e-5).to(orig_feats.device)) 
        orig_feats = self.depth2pts(orig_feats, batch_rays)

        feats = orig_feats
        code = orig_code

        # find negtive pair
        if sim_matrix is None:
            neg_indx = self.super_perm(orig_feats.shape[0], orig_feats.device)
        else:
            assert len(sim_matrix.shape) == 2
            neg_indx = torch.min(sim_matrix, dim=0)[1]
        
        if self.rand_neg:
            neg_indx = torch.randperm(sim_matrix.shape[0], device=orig_feats.device, dtype=torch.long)

        neg_feats = orig_feats[neg_indx]
        neg_code = orig_code[neg_indx]

        neg_corr_loss, neg_corr_cd = self.helper(
            feats, neg_feats, code, neg_code, self.neg_shift)
        
        self_corr_loss, self_corr_cd = self.helper(
            feats, feats, code, code, self.self_shift)
        
        return self.neg_weight * neg_corr_loss.mean() + self.self_weight * self_corr_loss.mean()

    
if __name__ == "__main__":
    orig_feats  = torch.ones([3, 1, 32, 32]).float()
    orig_code  = torch.ones([3, 2, 32, 32]).float()
    K = torch.eye(3).float()
    poses = torch.ones([3, 3, 4]).float()
    loss = GeoCorrelationLoss()
    res = loss(orig_feats=orig_feats, orig_code=orig_code, K=K, poses=poses)
    print(res)