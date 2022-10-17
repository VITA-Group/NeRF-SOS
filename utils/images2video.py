import os
import sys
import cv2
import imageio
import numpy as np
from glob import glob

rgb_dir = "/ssd1/zhiwen/projects/nerf-coseg/logs/nerf-coseg-compare/truck/gt_rgb/"
seg_dir = "/ssd1/zhiwen/projects/nerf-coseg/logs/nerf-coseg-compare/truck/ours_seg_randneg_clus/"

rgb_path_list = sorted(glob(f"{rgb_dir}/*.png"))
seg_path_list = sorted(glob(f"{seg_dir}/*.png"))

rgb = cv2.imread(rgb_path_list[0])


rgbs, segs = [], []

for rgb_path, seg_path in zip(rgb_path_list, seg_path_list):
    rgb = cv2.imread(rgb_path)
    seg = cv2.imread(seg_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgbs.append(rgb)
    segs.append(seg)

rgbs = np.stack(rgbs)
segs = np.stack(segs)
imageio.mimwrite(f"{rgb_dir}rgb.mp4", rgbs, fps=2, quality=8)
imageio.mimwrite(f"{seg_dir}seg.mp4", segs, fps=2, quality=8)
