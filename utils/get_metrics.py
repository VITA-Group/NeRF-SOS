import os
import numpy as np
import sys
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))
import cv2
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as st
from sklearn.metrics import adjusted_rand_score, rand_score
from sklearn.metrics import confusion_matrix 
from utils.image import to8b, img2mse, mse2psnr, ssim, lpips, color_pallete
def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return IoU

def cal_seg_metrics():
    ars_list = []
    ars_fg_list = []
    iou_list = []
    gt_path = "/ssd1/xx/projects/nerf-coseg/logs/nerf-coseg-compare/flower/gt_seg/"
    gt_path_list = sorted(glob(f"{gt_path}/*.png"))
    res_path = "/ssd1/xx/projects/nerf-coseg/logs/nerf-coseg-compare/flower/docs_seg/"
    res_path_list = sorted(glob(f"{res_path}/*.png"))

    for x, y in zip(gt_path_list, res_path_list ):
        gt = cv2.imread(x)
        gt = gt[..., 0] / 255.
        img_shape = gt.shape[::-1]

        pred = cv2.imread(y)
        pred = pred[..., 0] / 255.
        pred = cv2.resize(pred, tuple(img_shape))
        pred = (pred >=0.5).astype(np.float32)
        
        # print(f"gt.shape: {gt.shape}, pred.shape: {pred.shape}")

        ars = adjusted_rand_score(gt.reshape(-1), pred.reshape(-1))
        # print(ars)

        ars_list.append(ars)

        fg_idx = (gt==1)
        gt_fg = gt[fg_idx]
        pred_fg = pred[fg_idx]
        ars_fg = adjusted_rand_score(gt_fg.reshape(-1), pred_fg.reshape(-1))
        # print(ars_fg)

        ars_fg_list.append(ars_fg)

        iou = compute_iou(pred, gt)
        iou_list.append(iou)
    print(np.array(iou_list).shape)
    print("mean:", np.mean(ars_list))
    # print("mean:", np.mean(ars_fg_list))
    print("iou:", np.mean(np.array(iou_list)[:, 0]), np.mean(np.array(iou_list)[:, 1]), np.mean(np.array(iou_list)))



def cal_render_metics():
    import torch
    psnr_list = []
    ssim_list = []
    lp_list = []
    gt_path = "/ssd1/xx/projects/nerf-coseg/logs/nerf-coseg-compare/flower/gt_rgb/"
    gt_path_list = sorted(glob(f"{gt_path}/*.png"))
    res_path = "/ssd1/xx/projects/nerf-coseg/logs/nerf-coseg-compare/flower/ours_rgb/"
    res_path_list = sorted(glob(f"{res_path}/*.png"))

    for x, y in zip(gt_path_list, res_path_list ):
        gt = cv2.imread(x)
        gt = gt / 255.
        img_shape = gt.shape[::-1]

        pred = cv2.imread(y)
        pred = pred/ 255.

        pred = torch.from_numpy(pred).float()
        gt = torch.from_numpy(gt).float()

        mse = img2mse(pred, gt)
        psnr = mse2psnr(mse)
        ss = ssim(pred, gt, format='HWC')
        lp = lpips(pred, gt, format='HWC')
        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ss.cpu().numpy())
        lp_list.append(lp.detach().cpu().numpy())

    print("psnr:", np.mean(psnr_list))
    print("ssim:", np.mean(ssim_list))
    print("lp:", np.mean(lp_list))


cal_seg_metrics()
# cal_render_metics()


