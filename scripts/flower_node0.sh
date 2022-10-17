#! /bin/bash

SCENE=flower
BATCH_SIZE=8
PATCH_SIZE=64
PATCH_STRIDE=6
EXPNAME=${SCENE}_B${BATCH_SIZE}_P${PATCH_SIZE}_PS${PATCH_STRIDE}_1corr0.18,1,0.46,1_0.01geoCorr0.5,1,3,1_rerun
# EXPNAME=${SCENE}_tmp
mkdir -p logs/$EXPNAME

python -u run_nerf.py --gpuid 2 \
--expname ${EXPNAME}  \
--config configs/${SCENE}_full.txt  \
--i_print 200  \
--i_verbose 200  \
--i_testset 500 \
--i_video 5000  \
--i_weights 1000 \
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
--fix_backbone \
--ret_cluster \
--clus_no_sfm \
--sem_with_coord \
--sem_dim 2 \
--use_sim_matrix \
--correlation_w 1 \
--Gcorrelation_w 0.01 \
--app_corr_params 0.18 1 0.46 1 \
--geo_corr_params 0.5 1 3 1 \
--fast_mode \
--ckpt_path ./pretrained_ckpt/flower_00150000.ckpt  2>&1 | tee -a logs/${EXPNAME}/${EXPNAME}.txt
