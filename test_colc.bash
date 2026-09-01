
DATASET=v2xsim

MODEL_DIR=opencood/logs/xx # Your trained CoLC model

FUSION_METHOD=cecooper # 所有协作车辆的中融合检测结果，使用场景所有车的gt box。[intermediate fusion dataset支持]

fg_ratio=0.2 # Foreground point selection ratio 
bg_ratio=0.1 # Background point selection ratio 


CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_colc.py \
--model_dir $MODEL_DIR \
--fusion_method $FUSION_METHOD \
--save_vis_interval 40 \
--nei_model opencood/logs/${DATASET}_pointnet_lidar_seg_early \ 
--fg_ratio ${fg_ratio} \
--bg_ratio ${bg_ratio} \
--note '_color_point' \
--vqvae_model opencood/logs/${Your_pretrained_vq_completion_model}

