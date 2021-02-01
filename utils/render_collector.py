import os, sys
import math, random, time
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.image_utils import *

class SaveImageCollector(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        return

    def __call__(self, ret_dict):
        i, rgb = ret_dict['batch_num'], ret_dict['rgb']

        filename = os.path.join(self.save_dir, f'{i:03d}.npy')
        np.save(filename, rgb)

        # Convert to image
        filename = os.path.join(self.save_dir, f'{i:03d}.png')
        imageio.imwrite(filename, to8b(rgb))

class TensorboardCollector(object):
    def __init__(self, writer, tag, step, combine=False):
        """
        writer: summary writer object
        tag: tag for logging
        step: current step
        combine: output all images into one image or separate images
        """
        self.writer = writer
        self.tag = tag
        self.step = step
        self.combine = False

    def __call__(self, ret_dict):
        i, rgb = ret_dict['batch_num'], to8b(ret_dict['rgb'])
        tag = self.tag if self.combine else self.tag + str(i)
        self.writer.add_image(tag, rgb, self.step, dataformats='HWC')

class SaveVideoCollector(object):
    def __init__(self, save_dir, fps=30, quality=8, save_frames=False):
        self.save_dir = save_dir
        self.fps = fps
        self.quality = quality
        self.save_frames = save_frames

        self.rgbs = []

    def __call__(self, ret_dict):
        i, rgb = ret_dict['batch_num'], ret_dict['rgb']

        if self.save_frames:
            filename = os.path.join(self.save_dir, f'{i:03d}.npy')
            np.save(filename, rgb)

            # Convert to image
            filename = os.path.join(self.save_dir, f'{i:03d}.png')
            imageio.imwrite(filename, to8b(rgb))

        self.rgbs.append(rgb)

    def export_video(self, filename):
        rgbs = np.stack(self.rgbs, 0)
        imageio.mimwrite(os.path.join(self.save_dir, filename), to8b(rgbs),
                         fps=self.fps, quality=self.quality)