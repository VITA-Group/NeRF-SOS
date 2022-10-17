import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as st
import sys
import os
# cwd = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, f"{cwd}/..")
# from utils.misc import segmap_cluster


def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor(nn.Module):
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature
    
    def get_feat_attn_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        attn = self.outputs_dict[VitExtractor.ATTN_KEY]
        cls_ = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return  {'attn': attn, 'cls_': cls_}

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map

    def get_vit_feature(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, 3, 1, 1)
        # x = F.interpolate(x, size=(224, 224))
        x = (x - mean) / std
        # print(x.shape, self.get_feature_from_input(x)[-1].shape)
        # return self.get_feature_from_input(x)[-1][0, 0, :]
        return self.get_feature_from_input(x)[-1][:, 1:, :]

    
    def get_vit_feature_attn(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224))
        x = (x - mean) / std
        # print(x.shape, self.get_feature_from_input(x)[-1].shape)
        # st()
        # return self.get_feature_from_input(x)[-1][0, 0, :]
        return self.get_feature_from_input(x)[-1][:, 0, :]

    
    def get_vit_attn_feat(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224))
        x = (x - mean) / std
        ret = self.get_feat_attn_from_input(x)
        attn = ret['attn'][-1].mean(1).unsqueeze(1)[:, :, 0, 1:]
        cls_ = ret['cls_'][-1][:, 0, :]
        feat = ret['cls_'][-1][:, 1:, :]
        return {'attn': attn, 'cls_': cls_, 'feat':feat}

    def get_vit_attn_feat_noresize(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, 3, 1, 1)
        # x = F.interpolate(x, size=(224, 224))
        x = (x - mean) / std
        ret = self.get_feat_attn_from_input(x)
        attn = ret['attn'][-1].mean(1).unsqueeze(1)[:, :, 0, 1:]
        cls_ = ret['cls_'][-1][:, 0, :]
        feat = ret['cls_'][-1][:, 1:, :]
        return {'attn': attn, 'cls_': cls_, 'feat':feat}

if __name__ == "__main__":
    import cv2
    import numpy as np
    from glob import glob
    device = 'cuda:0'
    dino = VitExtractor(model_name='dino_vits16', device=device)

    def segmap_cluster(x, n_clusters=2, method='kmeans'):
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

    def normalize_batch(batch):
        batch = batch.permute(0, 3, 1, 2)
        # normalize using imagenet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        batch = batch.div_(255.0)
        return (batch - mean) / std

    def run_attn_single_image(img_path='/ssd1/xx/datasets/nerf_llff_data_fullres/fortress/images_4/image000.png', \
        save_path="logs/1.png"):

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device).float()
        img = img.unsqueeze(0).permute(0, 3, 1, 2) / 255.
        aa = dino.get_vit_attn_feat_noresize(img)
        st()
        attn = aa['attn'].reshape([1, 1, 47, 63])
        attn = attn.detach().cpu().numpy()
        # attn -= np.min(attn)
        attn /= np.max(attn)
        attn *= 255.
        attn = attn.astype(np.uint8)
        cv2.imwrite(save_path, attn.squeeze())
    
    def run_vitFeat_single_image(img_path='/ssd1/xx/datasets/nerf_llff_data_fullres/fortress/images_4/image000.png', \
        save_path="logs/1.png"):
        if type(img_path) != str:
            img = img_path
        else:
            img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device).float()
        img = img.unsqueeze(0).permute(0, 3, 1, 2) / 255.
        B, C, H, W = img.shape
        dino_ret = dino.get_vit_feature(img)
        dino_ret = dino_ret.reshape([1, H//16, W//16, 384])
        dino_ret = dino_ret.squeeze(0).detach().cpu().numpy()
        clustering = segmap_cluster(dino_ret).astype(np.float32)
        clustering *= 255.
        clustering = clustering.astype(np.uint8)
        if save_path is not None:
            cv2.imwrite(save_path, clustering.squeeze())
        else:
            return clustering

    # img_path = '/ssd1/xx/datasets/nerf_llff_data_fullres/fortress/images_4/image000.png'
    # run_vitFeat_single_image(img_path=img_path, save_path="logs/1.png")
    # run_attn_single_image(img_path=img_path)

    def video2imgs(video_path):
        times = 0
        img_list = []
        camera = cv2.VideoCapture(video_path)
        while True:
            times+=1
            res, image = camera.read()
            if not res:
                print('not res , not image')
                break
            img_list.append(image)
        return img_list
    
    def imgs2video(img_list, video_path, fps=30):
        import imageio
        quality=8
        # rgb_video = np.stack(img_list, 0)
        # imageio.mimwrite(video_path, rgb_video, fps=fps, quality=quality)

        # img_list_rsz = []
        # for img in img_list:
        #     H, W, C = img.shape
        #     img_list_rsz.append(cv2.resize(img, (W*8, H*8)))
        #     print(np.max(img), np.min(img), np.mean(img))

        rgb_video = np.stack(img_list, 0).astype(np.uint8)
        imageio.mimwrite(video_path, rgb_video, fps=fps, quality=quality)
        return 
    
    def to8b(x):
        return (255*(x-x.min())/(x.max()-x.min())).astype(np.uint8)
    
    # video_path = "/ssd1/xx/datasets/toydesk_data/processed/our_desk_1/full_rgb/"
    # img_list = sorted(glob(f"{video_path}/*.png"))
    # # img_list = video2imgs(video_path)
    # # img_list = img_list[:2]
    # clus_list = []
    # from tqdm import tqdm 
    # for img in tqdm(img_list):
    #     clustering = run_vitFeat_single_image(img, None)
    #     clus_list.append(clustering)
    # print("> save video")
    # imgs2video(clus_list, "logs/toydesk.mp4", fps=2)
    video_path = "/ssd1/xx/datasets/nerf_synthetic/lego/test/"
    img_path_list = sorted(glob(f"{video_path}/*.png"))
    img_path_list = [x for x in img_path_list if "depth" not in x]
    img_path_list = [x for x in img_path_list if "normal" not in x]
    
    img_list = []
    for x in sorted(img_path_list, key= lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0])):
        img = cv2.imread(x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 400))
        img_list.append(img)
    imgs2video(img_list, "logs/lego.mp4")