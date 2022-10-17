#! /bin/bash

SCENE=flower
BATCH_SIZE=8
PATCH_SIZE=64
PATCH_STRIDE=8
EXPNAME=flower_B8_P64_PS6_1corr0.18,1,0.46,1_0.01geoCorr0.5,1,3,1_noTrainSFM
mkdir -p logs/$EXPNAME

python -u run_nerf.py --gpuid 0 \
--expname ${EXPNAME}  \
--config configs/${SCENE}_full.txt  \
--i_print 200  \
--i_verbose 200  \
--i_testset 500 \
--i_video 5000  \
--i_weights 5000 \
--max_steps 200000 \
--patch_tune  \
--batch_size $BATCH_SIZE \
--patch_size $PATCH_SIZE  \
--patch_stride  $PATCH_STRIDE   \
--load_nostrict  \
--sem_w 0 \
--use_dino \
--contrast_w 0 \
--use_correlation \
--use_geoCorr \
--correlation_w 1 \
--Gcorrelation_w 1 \
--fix_backbone \
--ret_cluster \
--clus_no_sfm \
--sem_with_coord \
--sem_dim 2 \
--use_masks \
--use_sim_matrix \
--app_corr_params 0.18 1 0.46 1 \
--geo_corr_params 1 1 5 1 \
--ckpt_path logs/${EXPNAME}/checkpoints/latest.ckpt \
--eval
# --eval_video