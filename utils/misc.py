import torch
from pdb import set_trace as st

def to8b(x):
    return (255*(x-x.min())/(x.max()-x.min())).astype(np.uint8)

def weights_log(masks):
    class_freq = [torch.sum(masks==0), torch.sum(masks==1)]
    class_freq = torch.Tensor(class_freq)
    weights = 1 / torch.log1p(class_freq)
    weights = len(class_freq) * weights / torch.sum(weights)
    print(f'> balanced class weight: {weights}')
    print(f'> class count (0): {torch.sum(masks==0)},  (1): {torch.sum(masks==1)}')
    return weights


def params(model):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, param in model.named_parameters():
        yield name, param


def find_params(model, my_list):
    params_list = list(params(model))
    params_base = []
    params_specify = []
    for name, param in params_list:
        specify = False
        for x in my_list:
            if x in name:
                specify = True
        if specify:
            params_specify.append(param)
        else:
            params_base.append(param)
    print(f"[Params]: len of params_specify:{len(params_specify)}, len of params_base:{len(params_base)}")
    return params_specify, params_base

def segmap_cluster(x, n_clusters=2, method='kmeans'):
    '''x in shape [H, W, class], type is numpy array
    '''
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        assert len(x.shape) == 3
        H, W, C = x.shape
        x_viewed = x.reshape(-1, C)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x_viewed)
        kmeans_orisize = kmeans.labels_.reshape(H, W, 1)
        return kmeans_orisize

    else:
        raise NotImplementedError


if __name__ == '__main__':
    import numpy as np
    import os
    # x = np.random.rand(1, 2, 40, 50)
    # label = segmap_cluster(x, n_clusters=2)
    # st()
    logits_dir = '/ssd1/xx/datasets/nerf_llff_data_fullres/fortress/masks'
    save_dir = '/ssd1/xx/datasets/nerf_llff_data_fullres/fortress/masks_kmeans'
    os.makedirs(save_dir, exist_ok=True)
    from glob import glob
    import cv2
    img_path_list = glob(f'{logits_dir}/*.png')
    for img_path in img_path_list:
        img = cv2.imread(img_path)[:, :, 0:1]
        img_inverse = np.ones_like(img) - img
        img = np.concatenate([img, img_inverse], -1)
        print(img.shape)
        label = segmap_cluster(img)
        print(label.shape)
        save_path = f'{save_dir}/{os.path.basename(img_path)}'
        cv2.imwrite(save_path, to8b(label)) 