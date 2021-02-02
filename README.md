# Nerf-PyTorch
Reimplementation of ECCV paper "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" with PyTorch library.
Our code's key features include simplicity, reusability, high-level encapsulation, and extensive tunable hyper-parameters.

## Installation

We recommend users to use `conda` to install the running environment. Please follow the command as below:
```
conda env create -f environment.yml
```

Our code has been tested on Ubuntu 18.04.2 LTS, PyTorch 1.3.1, and Cuda 10.0.

## Usage

For training a NeRF demo, users need to convert raw datasets to our format using `data/export_nerf.ipynb` notebook.
It only supports LLFF datasets for now.

After generating datasets, users can train a NeRF by the following command:
```
python run_nerf.py --action train --gpuid <your_device> --expname <output_folder> --config <default_config_file>
```

While training, users can view logging information through `tensorboard`:
```
tensorboard --logdir='./logs' --port <your_port> --host 0.0.0.0
```

When training is done, users can synthesize exhibition video by running:
```
python run_nerf.py --action video --gpuid <your_device> --expname <output_folder> --config <default_config_file>
```

For more options, please check via the following instruction:
```
python run_nerf.py -h
```