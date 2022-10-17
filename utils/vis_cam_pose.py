import numpy as np
try:
    from utils.camera_pose_visualizer import CameraPoseVisualizer
except:
    from camera_pose_visualizer import CameraPoseVisualizer

import torch
from pdb import set_trace as st

# from kornia import create_meshgrid
def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    nH: int = -1,
    nW: int = -1,
    device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generates a coordinate grid for an image.
    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.
    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.
    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.
    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])
        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    if nH != -1 and nW != -1:
        ys = torch.linspace(0, height - 1, nH, device=device, dtype=dtype)
        xs = torch.linspace(0, width - 1, nW, device=device, dtype=dtype)
    else:
        xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
        ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2
def get_ray_directions(H, W, focal, nH=-1, nW=-1):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False, nH=nH, nW=nW)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)
    return directions
def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)
    
    # print(rays_d.shape, rays_o.shape)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d
def get_ray_directions(H, W, focal, nH=-1, nW=-1):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False, nH=nH, nW=nW)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)
    return directions
def revert_axis(extrinsics):
    R, T = extrinsics[:3, :3], extrinsics[:3, 3:]
    ww = np.array([[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]])
        #  [0, 0, 0, 1]])
    # R_ = R.T
    # T_ = -1 * R_ @ T
    # R_ = ww @ R_
    # R_ = ww @ R
    R_ = -ww @ R
    # T_ = ww @ T
    T_ = T
    # T_ = ww @ T
    # print(R_.shape, T_.shape)
    new = np.concatenate((R_, T_), axis=1)
    new = np.concatenate((new, np.array([[0, 0, 0, 1]])), axis=0)
    return new
def ww():
    data = np.load('/tmp/c2ws_render.npy')
    # print(data.shape)
    data = [data[0], data[10]]
    focal = 1446.165 / 200 / 4
    directions = get_ray_directions(160, 128, focal, nH=10, nW=10)
    pcl = o3d.geometry.PointCloud()
    vec = []
    for cur in data:
        print(np.linalg.inv(cur))
        rays_o, rays_d = get_rays(directions, torch.FloatTensor(np.linalg.inv(cur)[:3, :4]))
        print(rays_o.shape, rays_d.shape)
        rays_o = rays_o.numpy()
        rays_d = rays_d.numpy()
        rays_d = np.mean(rays_d, axis=0, keepdims=True)
        rays_o = np.mean(rays_o, axis=0, keepdims=True)
        for i in range(1000):
            vec.append(rays_o + i / 10.0 * rays_d)
import open3d as o3d
# import numpy as np
import torch
if __name__ == '__main__':
    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([-15, 15], [-15, 15], [-15, 15])
    poses = np.load("tmp/poses.npy")
    print(poses.shape)
    for p in range(0, poses.shape[0], 20):
        tmp = np.eye(4)
        pose = poses[p]
        tmp[:3, :4] = pose[:3, :4]
        visualizer.extrinsic2pyramid(tmp, 'c', 10)
    
    poses = np.load("tmp/render_poses.npy")
    print(poses.shape)
    for p in range(0, poses.shape[0], 20):
        tmp = np.eye(4)
        pose = poses[p]
        tmp[:3, :4] = pose[:3, :4]
        visualizer.extrinsic2pyramid(tmp, 'r', 10)

    # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
    # c1 = np.array([[ 0.42349423 , 0.74917624 ,-0.50930109 , 1.76508594],
    #                 [-0.76085978,  0.59929175 , 0.24888109, -0.8519389 ],
    #                 [ 0.49167574,  0.28210701 , 0.82381466,  0.56356287],
    #                 [ 0.        ,  0.         , 0.          ,1.        ]])
#     c1 = np.array([[ 0.42349423 ,-0.76085978 , 0.49167574 ,-1.67279995],
#                     [ 0.74917624,  0.59929175,  0.28210701 ,-0.97078552],
#                     [-0.50930109,  0.24888109,  0.82381466 , 0.64672032],
#                     [ 0.        ,  0.        ,  0.         , 1.        ]]
# )
#     # print(c1)
#     c2 = np.array([[ 0.06040315, -0.87269021  ,0.48452374 ,-1.57980436],
#                     [ 0.7407434 ,  0.36455593  ,0.56426783 ,-1.85998086],
#                     [-0.66906702,  0.32482421  ,0.66845984 , 1.16545529],
#                     [ 0.        ,  0.          ,0.         , 1.        ]])
    
        # visualizer.extrinsic2pyramid(cur, 'c', 10)
    # for cur in data:
    #     # visualizer.extrinsic2pyramid(np.linalg.inv(cur), 'r', 10)
    #     visualizer.extrinsic2pyramid(revert_axis(cur), 'r', 10)
    # data = np.load('/ssd1/xx/datasets/toydesk_data/processed/our_desk_1/rays_train.npy')
    # print(data.shape)
    # b, h, w, c, r = data.shape
    # data = data.reshape(b, -1, c, r)
    # pcl = o3d.geometry.PointCloud()
    # vec = []
    # for i in range(0, b, 30):
    #     tmp_i = data[i]
    #     for j in range(0, data[i].shape[0], 64):
    #         rays_o = tmp_i[j, 0, :]
    #         rays_d = tmp_i[j, 1, :]
    #         for i in range(0, 100, 10):
    #             vec.append(rays_o + i / 10.0 * rays_d)
    # vec = np.stack(vec)
    # print(vec.shape)
    # pcl.points = o3d.utility.Vector3dVector(vec)
    # o3d.io.write_point_cloud("logs/desk_train.ply", pcl)
    visualizer.show()