from posixpath import basename
import numpy as np
import os, imageio
import struct
import collections
from pdb import set_trace as st

########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
def _load_data(basedir):
    pose_json = f'{basedir}/transforms_full.json'
    import json
    with open(pose_json) as f:
        pose_dict = json.load(f)
    
    max_list = []
    for item in pose_dict['frames']:
        idx = item['idx']
        max_list.append(idx)
    _max = max(max_list)

    img_path_list = []
    pose_list = []
    idx_list = []
    for item in pose_dict['frames']:
        basename = item['file_path']
        img_path = f'{basedir}/{basename}.png'
        img_path_list.append(img_path)
        pose = item['transform_matrix']
        pose = np.array(pose)
        pose_list.append(pose)
        idx = item['idx']
        idx_list.append(idx)

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    
    _img = imread(img_path_list[0])
    height, width, _ = _img.shape
    imgs = np.zeros([_max+1, height, width, 3]).astype(np.float32)
    poses = np.zeros([_max+1, 4, 4]).astype(np.float32)

    for i, p in zip(idx_list, pose_list):
        poses[i] = p
    
    for i, img_p in zip(idx_list, img_path_list):
        imgs[i] = imread(img_p)[...,:3]/255

    # poses = np.array(pose_list).astype(np.float32)
    # imgs = [imread(f)[...,:3]/255. for f in img_path_list]
    # imgs = np.stack(imgs, 0)
    masks = np.zeros_like(imgs)[..., 0]
    masks = np.expand_dims(masks, -1)
    # masks = [np.expand_dims(imread(f)/255., -1) for f in maskfiles]
    # masks = np.stack(masks, -1)
    # assert imgs.shape[:2] == masks.shape[:2]
    print('Loaded image data', imgs.shape, masks.shape)
    return poses, imgs, masks, idx_list


def normalize(x):
    return x / np.linalg.norm(x)


def load_toydesk_data(basedir):

    poses, imgs, masks, idx_list = _load_data(basedir) # factor=8 downsamples original imgs by 8x

    # fix_rot = np.array([1, 0, 0,
    #                 0, -1, 0,
    #                 0, 0, -1]).reshape(3, 3)
    fix_rot = np.array([1, 0, 0,
                    0, -1, 0,
                    0, 0, -1]).reshape(3, 3)
    poses_ = poses + 0
    for idx in range(poses.shape[0]):
        # poses_[idx, :3, :3] = poses[idx, :3, :3] @ fix_rot # equal to np.matmul(poses[0], fix_rot)
        poses_[idx, :3, :3] = poses[idx, :3, :3]  @ fix_rot

    del poses

    render_poses = None

    data_home, slice = basedir.split("/processed/")
    slice = slice.split('/')[0]
    split_path_train = f'{data_home}/split/{slice}_train_0.8/train.txt'
    split_path_test = f'{data_home}/split/{slice}_train_0.8/test.txt'
    with open(split_path_train) as f:
        i_train = f.readlines()
    i_train = [x.strip() for x in i_train]
    i_train = [x for x in i_train if x is not '']
    i_train = [int(x) for x in i_train]
    i_train = [x for x in i_train if x in idx_list]

    with open(split_path_test) as f:
        i_test = f.readlines()
    i_test = [x.strip() for x in i_test]
    i_test = [x for x in i_test if x is not '']
    i_test = [int(x) for x in i_test]
    i_test = [x for x in i_test if x in idx_list]

    images = imgs.astype(np.float32)
    poses = poses_.astype(np.float32)
    masks = masks.astype(np.float32)

    i_split = [np.array(i_train), np.array(i_test), np.array(i_test)] # validation = test
    hwf = None

    return images, poses, render_poses, masks, i_split, hwf


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)