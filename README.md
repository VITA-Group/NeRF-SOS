# Neural Radiance Field - PyTorch Implementation
Reimplementation of ECCV paper "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" with PyTorch library.
Our code's key features include simplicity, reusability, high-level encapsulation, and extensive tunable hyper-parameters.

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
│   │   └── fern
│   │   └── flower  # downloaded llff dataset
│   │   └── horns   # downloaded llff dataset
|   |   └── ...
|   ├── nerf_synthetic
|   |   └── lego
|   |   └── ship    # downloaded synthetic dataset
|   |   └── ...
```

## Training

After preparing datasets, users can train a vanilla NeRF by the following command:
```
python run_nerf.py --gpuid <gpu_id> --expname <output_folder> --config <default_config_file>
```
`<default_config_file>` is the path to the configuration file of your experiment instance. Examples and pre-defined configuration files are provided in `configs` folder. `<output_folder>` is the output folder name under `logs/` directory.

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
python run_nerf.py --action video --gpuid <your_device> --expname <output_folder> --config <default_config_file>
```

To visualize geometry, users can first generate density field with flag `--eval_vol`:
```
python run_nerf.py --eval --eval_vol --gpuid <gpu_id> --config <default_config_file>
```
The exported volume will be saved into `<expname>/eval/` directory (with both `.npy` and `.mrc` suffices). Then users can use Python library [PyMCubes](https://github.com/pmneila/PyMCubes) or [USCF Chimera](https://www.cgl.ucsf.edu/chimera/) to threshold and extract meshes from the density field.

## Acknowledgement

Our code is implemented based on the following repository.

```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
