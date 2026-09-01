
DATASET=v2xsim
SETTING=lidar_only_with_noise


# 1. Prepare early fusion and no fusion model
########## Model prepare ##########

# CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
# -y opencood/hypes_yaml/${DATASET}/${SETTING}/pointpillar_early.yaml \
# --fusion_method early

# CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
# -y opencood/hypes_yaml/${DATASET}/${SETTING}/pointpillar_single.yaml \
# --fusion_method no

########## Model prepare ##########



# 2. Pretrain point selector
########## point selector ##########

CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
-y opencood/hypes_yaml/${DATASET}/${SETTING}/pointnet_single.yaml \
--run_test False \
--log early

########## point selector ##########


# 3. Pretrain vq-based completion module
########## lidar completion ##########

# CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_vqvae_ga_new.py \
# -y opencood/hypes_yaml/${DATASET}/${SETTING}/pointpillar_colc_baseline_vqvae.yaml \
# --pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_early  # early fusion model

########## lidar completion ##########


# 4. Train CoLC model
########## CoLC ##########
# CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_colc_ga.py \
# -y opencood/hypes_yaml/${DATASET}/${SETTING}/pointpillar_colc_kd.yaml \
# --fusion_method cecooper \
# --nei_model opencood/logs/${DATASET}_point_pillar_lidar_late \ 
# --pretrained_model opencood/logs/${DATASET}_point_pillar_lidar_early \
# --vqvae_model opencood/logs/$completion_model_path$ 
########## CoLC ##########

















