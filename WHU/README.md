> V4/GAC code: see `README_GAC.md`. The commands below are the V2/HPD base.

# DDP
## WHU-MARS-1000

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/HPD_UAD.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR logs/HPD_1000_DDP

# 1000 AS
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/HPD_UAD.yml \
  MODEL.DIST_TRAIN False \
  DATASETS.ROOT_DIR /project/code/data \
  TEST.WEIGHT logs/HPD_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/HPD_1000_test

# 1000GD（同一 1000 权重，不另训）
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/HPD_UAD.yml \
  MODEL.DIST_TRAIN False \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  DATASETS.FILTER_GD True \
  TEST.IMS_PER_BATCH 256 \
  TEST.WEIGHT logs/HPD_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/HPD_1000GD_test

# 36 条 modality×platform 诊断
CUDA_VISIBLE_DEVICES=1 python test.py \
  --config_file configs/HPD_UAD.yml \
  MODEL.DIST_TRAIN False \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  TEST.WEIGHT logs/HPD_1000_DDP/transformer_120.pth \
  TEST.DETAILED_DOMAIN_EVAL True \
  OUTPUT_DIR logs/HPD_1000_domain

# 2337
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/HPD_UAD.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  DATASETS.DATASET_DIR WHU-MARS-2337 \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR logs/HPD_2337_DDP