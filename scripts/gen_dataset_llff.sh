#! /bin/bash

cd data/
python gen_dataset.py --config  ../configs/fortress_full.txt  --data_path /ssd1/zhiwen/datasets/nerf_llff_data_fullres/fortress/  --data_type llff