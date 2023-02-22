# NeRF-SOS - PyTorch Implementation
Implementation of paper "NeRF-SOS: Any-View Self-supervised Object Segmentation from Complex Real-World Scenes " with PyTorch library.

[Project Page](https://zhiwenfan.github.io/NeRF-SOS/) | [Paper](https://arxiv.org/abs/2209.08776)

<div>
<img src="https://github.com/VITA-Group/NeRF-SOS/blob/main/datasets/imgs/flow_rgb.gif?raw=true" height="120"/>
<img src="https://github.com/VITA-Group/NeRF-SOS/blob/main/datasets/imgs/flower_seg.gif?raw=true" height="120"/>
<img src="https://github.com/VITA-Group/NeRF-SOS/blob/main/datasets/imgs/truck_rgb.gif?raw=true" height="120"/>
<img src="https://github.com/VITA-Group/NeRF-SOS/blob/main/datasets/imgs/truck_seg.gif?raw=true" height="120"/>
</div>

## Installation

We recommend users to use `conda` to install the running environment. The following dependencies are required:
```
pytorch=1.7.0
torchvision=0.8.0
cudatoolkit=11.0
tensorboard=2.7.0
opencv
imageio
imageio-ffmpeg
configargparse
scipy
matplotlib
tqdm
mrc
lpips
```

## Data Preparation

To run our code on NeRF dataset, users can either download our combined (RGB+poses+masks) datasets through [cloud drive](https://drive.google.com/file/d/1i0wN_cLgllMPJFG0w-si7oCoZjVT94Gr/view?usp=share_link) and then generate .npy training data following the subsequent suggestions, or download original NeRF data from official [cloud drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and official CO3Dv2 datasets. 
Then extract package files according to the following directory structure:

```
├── configs
│   ├── ...
│
├── datasets
│   ├── nerf_llff_data
│   │   └── flower 
│   │   └── fortress 
|   |   └── ...
│   ├── co3d_colmap
│   │   └── co3d_apple_110_13051_23361 
│   │   └── co3d_backpack_537_78249_152122  
|   |   └── ...
```

Then, please put the ``segments'' folder [LINK](https://drive.google.com/file/d/1gD5paJ8HBOFVyMRgweTc0jNTL41SmBFz/view?usp=sharing) under DATASET/SCENE/ (e.g., nerf_llff_data/flower/segments)


run the following command to generate training and testing data
```
cd data
python gen_dataset.py --config ../configs/flower_full.txt  --data_path /PATH_TO_DATA/nerf_llff_data_fullres/flower/  --data_type llff
```
### Download Prepared Data
#### Scene Flower
We provide a data sample for scene ``Flower'' (with multi-view images, camera poses, masks and the processed .npy training files) can be found in the [LINK](https://drive.google.com/file/d/1glu5KcPpXsLh9Im1b0X1M19Sja-HVkdL/view?usp=sharing), you can direct download it without any modification.

## Evaluation using our pre-trained ckpt

If you wanna see the rendered masks using the self-supervised trained model, run:
```
bash scrits/eval.sh
```
If you wanna see the rendered masks (video format) using the self-supervised trained model, run:
```
bash scrits/eval_video.sh
```

## Training

After preparing datasets, users can train a NeRF-SOS by the following command:
```
bash scripts/train_flower_node0.sh
bash scripts/train_fortress_node0.sh
bash scripts/train_co3d_apple_node0.sh
bash scripts/train_co3d_backpack_node0.sh
```

For more options, please check via the following instruction:
```
python run_nerf.py -h
```

While training, users can view logging information through `tensorboard`:
```
tensorboard --logdir='./logs' --port <your_port> --host 0.0.0.0
```

## Evaluation

When training is done, users can synthesize exhibition video by running:
```
bash scrits/eval.sh
```

## TODO
support COLMAP poses

## Citation

If you find this repo is helpful, please cite:

```

@article{fan2022nerf,
  title={NeRF-SOS: Any-View Self-supervised Object Segmentation from Complex Real-World Scenes},
  author={Fan, Zhiwen and Wang, Peihao and Gong, Xinyu and Jiang, Yifan and Xu, Dejia and Wang, Zhangyang},
  journal={arXiv e-prints},
  pages={arXiv--2209},
  year={2022}
}

```
