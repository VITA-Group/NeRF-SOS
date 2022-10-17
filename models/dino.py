# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)
import vision_transformer as vits
from pdb import set_trace as st


class Dino():
    def __init__(self, arch=None, patch_size=None, image_size=None, device=None, fix=True, ckpt_path=None, \
        checkpoint_key='teacher'):
        self.arch = arch
        self.patch_size = patch_size
        self.device = device
        self.fix = fix
        self.ckpt_path = ckpt_path
        self.image_size = image_size

        self.model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        if fix:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        
        self.model.to(device)
        # self.transform = pth_transforms.Compose([
        #     pth_transforms.Resize(self.image_size),
        #     pth_transforms.ToTensor(),
        #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        if os.path.isfile(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]

            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(ckpt_path, msg))
        
        else:
            print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
            url = None
            if arch == "vit_small" and patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif arch == "vit_small" and patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            elif arch == "vit_base" and patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif arch == "vit_base" and patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
                self.model.load_state_dict(state_dict, strict=True)
            else:
                print("There is no reference weights available for this model => We use random weights.")
        
    def forward_pass(self, img):
        # img_orisize = img.size[::-1]
        # img = self.transform(img)

        # make the image divisible by the patch size
        w, h = img.shape[3] - img.shape[3] % self.patch_size, img.shape[2] - img.shape[2] % self.patch_size
        img = img[:, :, :h, :w]

        w_featmap = img.shape[-1] // self.patch_size
        h_featmap = img.shape[-2] // self.patch_size
        attentions = self.model.get_last_selfattention(img)
        attentions = torch.mean(attentions, dim=1).unsqueeze(1)
        nh = attentions.shape[1] # number of head
        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        attentions = attentions.reshape(nh, h_featmap, w_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=self.patch_size, mode="nearest")[0]
        return attentions
        # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=self.patch_size, mode="nearest")[0].cpu().numpy()
        st()


if __name__ == "__main__":
    device = 'cuda:1'
    dino = Dino(arch='vit_small', patch_size=8, image_size=(756, 1008), device=device, fix=True, ckpt_path='', checkpoint_key='teacher')

    def normalize_batch(batch):
        batch = batch.permute(0, 3, 1, 2)
        # normalize using imagenet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        batch = batch.div_(255.0)
        return (batch - mean) / std

    img_path = '/ssd1/xx/datasets/nerf_llff_data_fullres/fortress/images_4/image000.png'
    # img = Image.open(img_path)
    # img = img.convert('RGB')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device).float()
    img = img.unsqueeze(0)
    attn = dino.forward_pass(normalize_batch(img))
    attn = attn.cpu().numpy()
    attn -= np.min(attn)
    attn /= np.max(attn)
    attn *= 255.
    attn = attn.astype(np.uint8)
    cv2.imwrite("logs/1.png", attn.transpose(1,2,0))
    st()    
