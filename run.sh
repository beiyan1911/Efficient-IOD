CUDA_VISIBLE_DEVICES=0 PORT=29057 ./tools/dist_train.sh configs/gfl_increment_voc/gfl_r50_fpn_1x_coco_fd_stage_0_tal.py 1  --work-dir=./tmp/er_results/gfl_increment_voc/gfl_r50_fpn_1x_coco_fd_stage_0_tal --cfg-options randomness.seed=2024


# 40+20X stage 0
CUDA_VISIBLE_DEVICES=0 PORT=29057 ./tools/dist_train.sh configs/gfl_increment/gfl_r50_fpn_1x_coco_fd_stage_0_tal.py 1  --work-dir=./work_dir/gfl_increment/gfl_r50_fpn_1x_coco_fd_stage_0_tal
# 40+20X stage 1
CUDA_VISIBLE_DEVICES=0 PORT=29057 ./tools/dist_train.sh configs/gfl_increment/gfl_r50_fpn_1x_coco_fd_40_20X_stage_1_adg_tal.py 1  --work-dir=./work_dir/gfl_increment/gfl_r50_fpn_1x_coco_fd_40_20X_stage_1_adg_tal
# 40+20X stage 2
CUDA_VISIBLE_DEVICES=0 PORT=29057 ./tools/dist_train.sh configs/gfl_increment/gfl_r50_fpn_1x_coco_fd_40_20X_stage_2_adg_tal.py 1  --work-dir=./work_dir/gfl_increment/gfl_r50_fpn_1x_coco_fd_40_20X_stage_2_adg_tal

