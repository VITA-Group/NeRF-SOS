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

To run our code on NeRF dataset, users need first download data from official [cloud drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Then extract package files according to the following directory structure:

```
├── configs
│   ├── ...
│
├── datasets
│   ├── nerf_llff_data
│   │   └── flower  # downloaded llff dataset
│   │   └── fortress   # downloaded llff dataset
|   |   └── ...
```

Then, please put the segments folder [LINK](https://drive.google.com/file/d/1gD5paJ8HBOFVyMRgweTc0jNTL41SmBFz/view?usp=sharing) under DATASET/SCENE/ (e.g., nerf_llff_data/flower/segments)

We provide a data sample for scene Flower in the [LINK](https://drive.google.com/file/d/1glu5KcPpXsLh9Im1b0X1M19Sja-HVkdL/view?usp=sharing), you can direct download it without any modification.

run the following command to generate training and testing data
```
cd data
python gen_dataset.py --config ../configs/flower_full.txt  --data_path /PATH_TO_DATA/nerf_llff_data_fullres/flower/  --data_type llff
```

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
bash scripts/flower_node0.sh
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
