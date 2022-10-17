from cv2 import cvtColor
import open3d as o3d
import numpy as np
import cv2
from pdb import set_trace as st

def save_ply(xyz, rgbs=None, file_name='logs/1.ply'):
    '''xyz in a N x 3 order
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgbs is not None:
        rgbs = np.array(rgbs)
        colors = rgbs.reshape(-1, 3) / 255.
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file_name, pcd)


def depth2pts(depth, K, pose, rgb=None, scale=1):
    u = range(0, depth.shape[1])
    v = range(0, depth.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    depth = depth / scale
    # Z = depth.astype(float) / scale
    # X = (u - K[0, 2]) * Z / K[0, 0]
    # Y = (v - K[1, 2]) * Z / K[1, 1]
    X = u
    Y = v
    Z = np.ones_like(X)

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    # depth filter
    depth = depth.reshape(1, -1)
    depth[depth > 100] = np.max(depth[depth < 100])
    # depth = depth[(depth>np.quantile(depth,0.01)) & (depth<np.quantile(depth,0.09))]

    XYZ = np.vstack((X, Y, Z)) * depth
    XYZ = np.linalg.inv(K) @ XYZ
    
    XYZ = np.vstack((XYZ, np.ones(len(X))))

    XYZ = pose @ XYZ

    # position = np.vstack((X, Y, Z, np.ones(len(X))))
    # position = np.dot(pose, position)

    if rgb is not None:
        R = np.ravel(rgb[:, :, 0])[valid]
        G = np.ravel(rgb[:, :, 1])[valid]
        B = np.ravel(rgb[:, :, 2])[valid]
        R = np.expand_dims(R, 0)
        G = np.expand_dims(G, 0)
        B = np.expand_dims(B, 0)
        points = np.transpose(XYZ[0:3, :]).tolist()
        # points = np.transpose(np.vstack(position[0:3, :])).tolist()
        rgbs = np.transpose(np.concatenate([R, G, B], 0)).tolist()
    else:
        R = G = B = None
        points = np.transpose(XYZ[0:3, :]).tolist()

    return [points, rgbs]


def depth2pts_torch(depth, K, pose, rgb=None, scale=1):
    import torch
    # u = range(0, depth.shape[1])
    # v = range(0, depth.shape[0])

    # u, v = np.meshgrid(u, v)
    depth = torch.from_numpy(depth).float()
    K = torch.from_numpy(K).float()
    pose = torch.from_numpy(pose).float()
    v, u = torch.meshgrid([torch.arange(0, depth.shape[0], dtype=torch.float32, device=depth.device),
                                    torch.arange(0, depth.shape[1], dtype=torch.float32, device=depth.device)])

    # u = u.astype(float)
    # v = v.astype(float)

    depth = depth / scale
    # Z = depth.astype(float) / scale
    # X = (u - K[0, 2]) * Z / K[0, 0]
    # Y = (v - K[1, 2]) * Z / K[1, 1]
    X = u
    Y = v
    Z = torch.ones_like(X).to(X.device)

    # X = np.ravel(X)
    # Y = np.ravel(Y)
    # Z = np.ravel(Z)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    # depth filter
    depth = depth.reshape(-1)
    depth[depth > 100] = torch.max(depth[depth < 100])
    # depth = depth[(depth>np.quantile(depth,0.01)) & (depth<np.quantile(depth,0.09))]

    XYZ = torch.stack((X, Y, Z), dim=0)
    # XYZ = np.linalg.inv(K) @ XYZ
    XYZ = torch.matmul(torch.inverse(K), XYZ * depth)
    
    # XYZ = np.vstack((XYZ, np.ones(len(X))))
    xyz_homo = torch.cat((XYZ, torch.ones_like(depth).unsqueeze(0)), dim=0).to(XYZ.device)
    XYZ = torch.matmul(pose, xyz_homo).cpu().numpy()

    # XYZ = pose @ XYZ

    if rgb is not None:
        R = np.ravel(rgb[:, :, 0])[valid]
        G = np.ravel(rgb[:, :, 1])[valid]
        B = np.ravel(rgb[:, :, 2])[valid]
        R = np.expand_dims(R, 0)
        G = np.expand_dims(G, 0)
        B = np.expand_dims(B, 0)
        points = np.transpose(XYZ[0:3, :]).tolist()
        # points = np.transpose(np.vstack(position[0:3, :])).tolist()
        rgbs = np.transpose(np.concatenate([R, G, B], 0)).tolist()
    else:
        R = G = B = None
        points = np.transpose(XYZ[0:3, :]).tolist()

    return [points, rgbs]


def depthmap2pts(depth_path="/ssd1/xx/projects/nerf-coseg/logs/fortress_base_eval/eval/depth_000.npy", \
                img_path="/ssd1/xx/projects/nerf-coseg/logs/fortress_base_eval/eval/rgb_000.png", \
                poses_path= "/ssd1/xx/datasets/nerf_llff_data_fullres/fortress/poses_test.npy", \
                idx="0"):
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 842.8297729492188
    K[0, 2] = 1008/ 2.
    K[1, 2] = 756 / 2.
    print(K)
    poses = np.load(poses_path)
    pose = poses[int(idx), ...] # TODO inverse?
    # bottom = np.reshape([0,0,0,1.], [1,4])
    # pose = np.concatenate([pose[:3,:4], bottom], -2)
    # pose = np.linalg.inv(pose)[:3,:4]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    depth = np.load(depth_path)
    depth = np.squeeze(depth, -1)
    import copy
    pts_ = depth2pts(copy.deepcopy(depth), rgb=copy.deepcopy(img), K=copy.deepcopy(K), pose=copy.deepcopy(pose))
    pts = depth2pts_torch(copy.deepcopy(depth), rgb=copy.deepcopy(img), K=copy.deepcopy(K), pose=copy.deepcopy(pose))
    # print(np.mean(np.array(pts_)) == np.mean(np.array(pts)))
    print(np.max(np.array(pts_)), np.min(np.array(pts_)))
    # print(np.mean(np.array(pts)))

    if len(pts) == 2:
        points, rgbs = pts
        points_, rgbs_ = pts_
    else:
        points, rgbs = pts, None
        points_, rgbs_ = pts_, None
    save_ply(xyz=points_, rgbs=rgbs_, file_name=f"logs/{str(idx)}_np.ply")
    save_ply(xyz=points, rgbs=rgbs, file_name=f"logs/{str(idx)}_torch.ply")

if __name__ == "__main__":
    for i in range(0, 5, 1):
        print(i)
        depth_path = f"/ssd1/xx/projects/nerf-coseg/logs/fortress_base_eval/eval/depth_00{i}.npy"
        img_path = f"/ssd1/xx/projects/nerf-coseg/logs/fortress_base_eval/eval/rgb_00{i}.png"
        poses_path = "/ssd1/xx/datasets/nerf_llff_data_fullres/fortress/poses_test.npy"
        idx = int(i)
        depthmap2pts(depth_path=depth_path, img_path=img_path, idx=idx)
