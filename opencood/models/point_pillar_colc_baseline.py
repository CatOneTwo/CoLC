# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn> Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

# 在cecooper的基础上添加pilalr的VQ-VAE模块用于lidar completion

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv

from opencood.models.sub_modules.colc_vqvae import Vqvae

import numpy as np
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.tools import train_utils
from opencood.utils.roiaware_pool3d import roiaware_pool3d_utils
from opencood.utils.pcd_utils import mask_points_by_range

from opencood.utils.common_utils import torch_tensor_to_numpy, merge_features_to_dict

from pdb import set_trace as pause

import os
import cv2

def draw_bev_lidar(voxels, pth):
    cv2.imwrite(
        pth,
        voxels.max(dim=0)[0][:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255,
    )
    

class PointPillarColcBaseline(nn.Module):
    def __init__(self, args):
        super(PointPillarColcBaseline, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])

        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], # 384
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'], # 384
                                  kernel_size=1)
        
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2， # 384
        else:
            self.use_dir = False
        
        self.model_type = args['model_type'] # train plugin or vq-vae?

        self.lidar_completion = args['lidar_completion'] if 'lidar_completion' in args else False
        if self.lidar_completion:
            self.fusion = 'mean'
            # self.fusion = 'max'
            if 'fusion' in args:
                self.fusion = args['fusion']
            if self.fusion == 'weight':
                self.weight_layer = FusionUnit(64)
            if self.fusion == 'disco':
                self.disco_layer = DiscoFusion(64)

        # ===neighbor===
        self.nei_model = None
        # Foreground-Aware Point Cloud Sampling (FAPS)
        if 'neighbor_points_selection' in args:
            self.nei_supply =args['neighbor_points_selection']['nei_supply']
            self.score_threshold =args['neighbor_points_selection']['score_threshold']
            # 前景用最远点采样
            if 'fg_fps' in args['neighbor_points_selection']:
                self.fg_fps = args['neighbor_points_selection']['fg_fps'] # 最远点采样
            else:
                self.fg_fps = False
            # 背景用最远点采样
            if 'bg_rs' in args['neighbor_points_selection']:
                self.bg_rs = args['neighbor_points_selection']['bg_rs'] # 随机采样
            else:
                self.bg_rs = False
        else:
            self.nei_supply = False

        # ===ego===
        # Sparse-to-Dense LiDAR Completion (SDLC)
        self.vqvae = Vqvae(args['vqvae'])
        self.perceptual_loss = args['vqvae']['perceptual_loss'] if 'perceptual_loss' in args['vqvae'] else False


        if self.model_type == "codebook_training":
            self.backbone_fix()
        
        elif self.model_type != "codebook_training":
            for p in self.vqvae.parameters():
                p.requires_grad = False
            
    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
        if self.use_dir == True:
            for p in self.dir_head.parameters():
                p.requires_grad = False
        
    
    def forward(self, data_dict):
        if self.model_type == "codebook_training":
            output_dict = self.train_codebook(data_dict)
        else:
            output_dict = self.train_detector(data_dict)
        return output_dict

    def train_detector(self, data_dict):

        voxel_features = data_dict['processed_lidar_colc']['voxel_features'] # [M,32,4]
        voxel_coords = data_dict['processed_lidar_colc']['voxel_coords'] # [M,4]
        voxel_num_points = data_dict['processed_lidar_colc']['voxel_num_points'] # [M]

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict) # 
        batch_dict = self.scatter(batch_dict) # batch_dict['spatial_features'] [B, 64, 200, 504]
        
        if self.lidar_completion and 'processed_lidar_downsample' in data_dict: 

            # vis spatial features
            # vis_save_path_root = os.path.join('debug_spatial_features_v2xsim_')
            # if not os.path.exists(vis_save_path_root):
            #     os.makedirs(vis_save_path_root)

            spatial_features = batch_dict['spatial_features']
            
            # all agents pillar feature from downsample lidar
            voxel_features = data_dict['processed_lidar_downsample']['voxel_features'] # [M,32,4]
            voxel_coords = data_dict['processed_lidar_downsample']['voxel_coords'] # [M,4]
            voxel_num_points = data_dict['processed_lidar_downsample']['voxel_num_points'] # [M]

            batch_dict_sparse = {'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points}

            batch_dict_sparse = self.pillar_vfe(batch_dict_sparse) # 
            batch_dict_sparse = self.scatter(batch_dict_sparse) 
            sparse_spatial_features = batch_dict_sparse['spatial_features'] # 
            
            batch_dict = {
            'spatial_features_sparse':sparse_spatial_features
            }
            output_dict = self.vqvae.s2d_completion(batch_dict)
            rec_spatial_features = output_dict['generated_voxel']
            occ_mask = output_dict['occ_mask']
            record_len_nei = data_dict['record_len_nei']

            assert rec_spatial_features.size(0) == record_len_nei.size(0), f"Size mismatch: {rec_spatial_features} != {record_len_nei}"
            
            B, C, H, W = spatial_features.shape
            out = []
            for i in range(B):
                spatial_features_i = spatial_features[i].unsqueeze(0)
                # draw_bev_lidar(spatial_features_i[0], "{}/No{}_ori.png".format(vis_save_path_root, i))
                ind = (record_len_nei == i)
                if sum(ind) > 0:
                    rec_spatial_features_i = rec_spatial_features[ind]
                    occ_mask_i = occ_mask[ind]
                    ego_mask = (spatial_features_i.abs().sum(dim=1, keepdim=True) > 1e-6).float()
                    empty_mask = 1.0 - ego_mask
                    if self.fusion == 'mean':                        
                        weights_all = occ_mask_i * empty_mask   # 只补充空白区域 [N, 1, H, W]
                        filled_feat = (rec_spatial_features_i * weights_all).sum(dim=0)  # [C, H, W]
                        total_weight = weights_all.sum(dim=0)  # [C, H, W]

                        total_weight = total_weight.clamp(min=1e-6)
                        filled_feat = filled_feat / total_weight
                        spatial_features_i = spatial_features_i * ego_mask + filled_feat.unsqueeze(0) * empty_mask
                    elif self.fusion == 'max':
                        # 获取最大置信度的 neighbor index per voxel
                        occ_mask_i = occ_mask_i.squeeze(1)  # [N, H, W]
                        max_idx = occ_mask_i.argmax(dim=0)  # [H, W]

                        # 构建索引以选择最大 occ 的重建特征
                        N, C, H, W = rec_spatial_features_i.shape
                        rec_spatial_features_i = rec_spatial_features_i.permute(1, 0, 2, 3)  # [C, N, H, W]
                        # Gather 按照 max_idx 选取对应特征
                        gather_idx = max_idx.unsqueeze(0).expand(C, -1, -1).unsqueeze(1)  # [C, 1, H, W]
                        filled_feat = torch.gather(rec_spatial_features_i, dim=1, index=gather_idx).squeeze(1)  # [C, H, W]
                        # 融合 ego 特征
                        spatial_features_i = spatial_features_i * ego_mask + filled_feat.unsqueeze(0) * empty_mask
                    elif self.fusion == 'weight':
                        N = rec_spatial_features_i.size()[0]
                        empty_mask = empty_mask.expand(N, 1, H, W) 
                        fused_feats, weights = self.weight_layer(spatial_features_i, rec_spatial_features_i, occ_mask_i)  # [B,C,H,W], [B,1,H,W]    
                        # 3. 聚合融合特征，仅填补 ego 空区域
                        filled_feat = torch.sum(fused_feats * empty_mask, dim=0, keepdim=True)  # [1,C,H,W]
                        total_weight = torch.sum(weights * empty_mask, dim=0, keepdim=True).clamp(min=1e-6)  # [1,1,H,W]               
                        total_weight = total_weight.clamp(min=1e-6)
                        filled_feat = filled_feat / total_weight  # [1,C,H,W]
                        spatial_features_i = spatial_features_i * ego_mask + filled_feat * empty_mask[:1]
                    elif self.fusion == 'disco':
                        filled_feat = self.disco_layer(spatial_features_i, rec_spatial_features_i)
                        spatial_features_i = spatial_features_i * ego_mask + filled_feat * empty_mask
                    
                # draw_bev_lidar(spatial_features_i[0], "{}/No{}_rec.png".format(vis_save_path_root, i))
                out.append(spatial_features_i)
            
            fused_feat = torch.stack(out).squeeze(1)
            batch_dict['spatial_features'] = fused_feat

        output_dict = dict()

        batch_dict = self.backbone(batch_dict) 
        
        spatial_features_2d = batch_dict['spatial_features_2d']

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d) # [B, 256, 100, 252]


        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict.update( {
            'feature': spatial_features_2d,
            'pillar_feature': batch_dict['spatial_features'],
            'cls_preds': psm,
            'reg_preds': rm})
                       
        if self.use_dir:
            dm = self.dir_head(spatial_features_2d)
            output_dict.update({'dir_preds': dm})

        return output_dict
    
    def train_codebook(self, data_dict):
        # 单独训练vqvae

        # vis_save_path_root = os.path.join('debug_spatial_features_6')
        # if not os.path.exists(vis_save_path_root):
        #     os.makedirs(vis_save_path_root)
        
        # all agents pillar feature from original lidar
        voxel_features = data_dict['processed_lidar']['voxel_features'] # [M,32,4]
        voxel_coords = data_dict['processed_lidar']['voxel_coords'] # [M,4]
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points'] # [M]

        batch_dict_dense = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}
        
        batch_dict_dense = self.pillar_vfe(batch_dict_dense) # 
        batch_dict_dense = self.scatter(batch_dict_dense) # batch_dict['spatial_features'] [B, 64, 200, 504]
        dense_spatial_features = batch_dict_dense['spatial_features'] # 

        batch_dict_vqvae = {
            'spatial_features':dense_spatial_features
            }

        # for i in range(dense_spatial_features.size(0)):
        #     draw_bev_lidar(dense_spatial_features[i], "{}/No{}_ori.png".format(vis_save_path_root, i))

        # all agents pillar feature from downsample lidar
        if self.vqvae.rec_mode =='s2d':
            voxel_features = data_dict['processed_lidar_downsample']['voxel_features'] # [M,32,4]
            voxel_coords = data_dict['processed_lidar_downsample']['voxel_coords'] # [M,4]
            voxel_num_points = data_dict['processed_lidar_downsample']['voxel_num_points'] # [M]

            batch_dict = {'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points}

            batch_dict = self.pillar_vfe(batch_dict) # 
            batch_dict = self.scatter(batch_dict) # batch_dict['spatial_features'] [B, 64, 200, 504]
            sparse_spatial_features = batch_dict['spatial_features'] # 

            # for i in range(sparse_spatial_features.size(0)):
            #     draw_bev_lidar(sparse_spatial_features[i], "{}/No{}_ds.png".format(vis_save_path_root, i))

            batch_dict_vqvae.update({
                    'spatial_features_sparse':sparse_spatial_features
                })
        
        # if self.training:
        output_dict = self.vqvae(batch_dict_vqvae)
        # else:

        # output_dict = self.vqvae.s2d_completion(batch_dict_vqvae)

        # rec_spatial_features = output_dict['generated_voxel']
        # for i in range(rec_spatial_features.size(0)):
        #     draw_bev_lidar(rec_spatial_features[i], "{}/No{}_rec.png".format(vis_save_path_root, i))

        # perceptual loss
        if self.perceptual_loss and output_dict['occ_mask'].sum()>0:
            
            rec_feature = output_dict['generated_voxel']
            batch_dict_rec = {
                'spatial_features': rec_feature
            }
            batch_dict_rec = self.backbone(batch_dict_rec) 

            occ_mask = output_dict['occ_mask']
            dense_spatial_features = dense_spatial_features*occ_mask
            batch_dict_dense['spatial_features'] = dense_spatial_features
            batch_dict_dense = self.backbone(batch_dict_dense) 

            gt_feat = batch_dict_dense['spatial_features_2d']
            rec_feat = batch_dict_rec['spatial_features_2d']

            loss_perc = F.mse_loss(rec_feat, gt_feat)*10

            output_dict.update({
                'loss_lidar_per': loss_perc
            })
            

        return output_dict

    
    def neighbor_points_selection(self, batch_data, opencood_dataset, device):
        # 近邻端根据自身预测采样前景部分点云并传输，ego端融合近邻端点云    

        # 1. 近邻端做预测
        # ouput_dict = self.forward(batch_data['ego']) # nei model 近邻模型不更新
        ouput_dict = self.nei_model.forward(batch_data['ego']) # nei model 近邻模型不更新

        # 2. 基于近邻端预测提取前景的点云 (在model内部实现，将nei_model的输出和opencood_dataset传入model)
        transformation_matrix_torch = torch.from_numpy(np.identity(4)).float().to(device)
        transformation_matrix_clean_torch = torch.from_numpy(np.identity(4)).float().to(device)
        batch_data['ego'].update({'transformation_matrix': transformation_matrix_torch,
                                'transformation_matrix_clean': transformation_matrix_clean_torch})
    
        record_len = batch_data['ego']['label_dict']['record_len'] # [n1,n2,n3,...,n4]
        B = record_len.size(0) # batch size

        cls_preds = ouput_dict['cls_preds'] # [n1+n2+n3+n4, 2, 80, 80]
        reg_preds = ouput_dict['reg_preds'] # [n1+n2+n3+n4, 14, 80, 80]
        split_cls_preds = regroup(cls_preds, record_len)
        split_reg_preds = regroup(reg_preds, record_len)
        
        if self.use_dir:
            dir_preds = ouput_dict['dir_preds'] # [n1+n2+n3+n4, 4, 80, 80]
            split_dir_preds = regroup(dir_preds, record_len)

        processed_lidar_list = [] # 所有batch的早期融合点云
        processed_lidar_downsample_list = [] # 所有近邻的降采样voxel feature

        # nei_lidar_original_list = []
        # nei_lidar_downsample_list = []

        record_len_nei = []
        for b in range(B):
            batch_points = batch_data['ego']['original_lidar'][b] # all original projected lidar in curent batch 
            
            transformation_matrix_list = batch_data['ego']['transformation_matrix_list'][b]
            transformation_matrix_clean_list = batch_data['ego']['transformation_matrix_clean_list'][b]

            # 保存nei点云
            # nei_lidar_original = [] # 当前batch的近邻点云
            # nei_lidar_downsample = [] # 当前batch的近邻点云降采样
            
            processed_features_list = [] # 所有batch的早期融合点云

            projected_lidar_stack = [] # 当前batch的早期融合点云, ego+sparse nei
            ego_points = batch_points[0]
            projected_lidar_stack.append(ego_points) # ego lidar
                        

            batch_cls_preds = split_cls_preds[b] # [N, 2, 100, 252]
            batch_reg_preds = split_reg_preds[b]
            if self.use_dir:
                batch_dir_preds = split_dir_preds[b]

            if batch_cls_preds.size(0)!=record_len[b]:
                print('--------------------!!!!!!!-----------------------')
                print(batch_cls_preds.size(), record_len[b])
            
            # 近邻根据预测提取前景点云
            for item in range(1, batch_cls_preds.size(0)):
            # for item in range(batch_cls_preds.size(0)):
                item_ouput_dict = {'ego': {}}
                item_ouput_dict['ego'].update({
                    'cls_preds': batch_cls_preds[item].unsqueeze(0), # [1, 2, 80, 80]
                    'reg_preds': batch_reg_preds[item].unsqueeze(0), # [1, 14, 80, 80]
                    'pred_center' : True, # 直接预测7维坐标,
                    'score_threshold': self.score_threshold }) 
                
                if self.use_dir:
                    item_ouput_dict['ego'].update({'dir_preds': batch_dir_preds[item].unsqueeze(0)}) # [1, 4, 80, 80]

                # 直接用七维的预测值
                nei_pred_boxes, scores = opencood_dataset.post_processor.post_process(batch_data, item_ouput_dict)
                
                # 除ego之外的智能体
                points = batch_points[item] # [N,4]
                # points = batch_points[item+1] # [N,4]
                points = train_utils.to_device(torch.tensor(points), device)

                # nei选择传递的点云
                if self.nei_supply:
                    if nei_pred_boxes is None:
                        # 全是背景点云
                        if self.bg_rs:
                            points_not_in_box = points # 背景
                            bg_points_count = points_not_in_box.size(0)
                            target_point_count = int(bg_points_count * self.bg_rs['ratio']) # 随机采样20%
                            random_indices = np.random.choice(bg_points_count, target_point_count, replace=False)
                            points_not_in_box = points_not_in_box[random_indices]
                            points_in_box = points_not_in_box
                        else:
                            continue
                    else:
                        points_mask = roiaware_pool3d_utils.points_in_boxes_gpu(points[:, :3].unsqueeze(0), nei_pred_boxes.unsqueeze(0))
                        points_in_box = points[points_mask.squeeze() != -1] # 包含在预测框内的前景

                        # FPS: Farthest Point Sampling for predicted foreground points # 2025/3/28
                        if self.fg_fps and points_in_box.size(0) > self.fg_fps['min_points']: # 大于一定量级才需要采样
                            points_count = points_in_box.size(0)
                            target_point_count = int(points_count * self.fg_fps['ratio']) # 随机采样50%
                            points_in_box = farthest_point_sampling(points_in_box, target_point_count)

                        if self.bg_rs:
                            points_not_in_box = points[points_mask.squeeze() == -1] # 背景
                            bg_points_count = points_not_in_box.size(0)
                            target_point_count = int(bg_points_count * self.bg_rs['ratio']) # 随机采样20%
                            random_indices = np.random.choice(bg_points_count, target_point_count, replace=False)
                            points_not_in_box = points_not_in_box[random_indices]
                            points_in_box = torch.cat((points_in_box,points_not_in_box),dim=0) # 与原始点云聚合
                # else:
                    # early_fusion
                    # points_in_box = points
                
                # nei_lidar_original.append(points) # nei 原始点云
                # nei_lidar_downsample.append(points_in_box) # nei 降采样点云

                record_len_nei.append(b)
                processed_features = opencood_dataset.pre_processor.preprocess(points_in_box.cpu().numpy())
                processed_features_list.append(processed_features)

                projected_lidar_stack.append(points_in_box.cpu().numpy()) # ego点云 + 降采样点云    


            # 3. 近邻将采样的前景点云传递给ego进行融合
            merged_feature_dict = merge_features_to_dict(processed_features_list)
            processed_lidar_downsample_list.append(merged_feature_dict)
        
            # nei_lidar_original_list.append(nei_lidar_original)
            # nei_lidar_downsample_list.append(nei_lidar_downsample)
            
            batch_data['ego']['original_lidar'][b]=projected_lidar_stack # ego + sparse/rec nei

            # 当前batch融合的点云
            stack_lidar_np = np.vstack(projected_lidar_stack) # [[n1,4],[n2,4],[n3,4]] -> [n1+n2+n3,4]
            stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                        opencood_dataset.params['preprocess']['cav_lidar_range'])
            stack_feature_processed = opencood_dataset.pre_processor.preprocess(stack_lidar_np)
            processed_lidar_list.append(stack_feature_processed)

        # 如果有neighbor，就储存每个neighbor降采样后的pillar
        merged_feature_dict = merge_features_to_dict(processed_lidar_downsample_list)
        if 'voxel_features' in merged_feature_dict:
            processed_downsampled_lidar_torch_dict = \
                    opencood_dataset.pre_processor.collate_batch(merged_feature_dict)

            batch_data['ego'].update({'processed_lidar_downsample': train_utils.to_device(processed_downsampled_lidar_torch_dict, device)})
            batch_data['ego'].update({'record_len_nei': torch.tensor(record_len_nei).to(device)})

        # batch_data['ego'].update({
        #     'nei_lidar_original_list': nei_lidar_original_list, 
        #     'nei_lidar_downsample_list': nei_lidar_downsample_list
        # })

        # 将输入更新为新的早期融合数据
        processed_lidar_torch_dict = \
                opencood_dataset.pre_processor.collate_batch(processed_lidar_list)
        batch_data['ego'].update({'processed_lidar_colc': train_utils.to_device(processed_lidar_torch_dict, device)}) # ego点云和降采样点云融合
        
        return batch_data
    
        
def farthest_point_sampling(points:torch.Tensor, num_samples:int):
    """
    使用Farthest Point Sampling (FPS)算法从点云中选择一组关键点。

    参数:
    - points (torch.Tensor): 输入点云，每一行是一个点的坐标。
    - num_samples (int): 要选择的采样点的数量。

    返回:
    - selected_points (torch.Tensor): 选择的关键点的坐标。
    """
    num_points = points.size(0)
    selected_indices = points.new_empty(num_samples, dtype=torch.long)
    distances = points.new_ones(num_points) * float('inf')

    # 从输入点云中随机选择一个起始点
    start_index = 0
    selected_indices[0] = start_index
    selected_point = points[start_index]

    # 迭代选择最远点
    for i in range(1, num_samples):
        # 计算每个点与已选点之间的欧几里得距离
        dist_to_selected = torch.norm(points - selected_point, dim=1)

        # 更新距离，选择最远点
        distances = torch.min(distances, dist_to_selected)
        farthest_index = torch.argmax(distances)
        selected_indices[i] = farthest_index
        selected_point = points[farthest_index]

    selected_points = points[selected_indices]
    return selected_points


class FusionUnit(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.att_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat_ego, feat_neigh, occ=None):
        # feat_ego: [1, C, H, W] -> expand to [B, C, H, W]
        B, C, H, W = feat_neigh.shape
        feat_ego_exp = feat_ego.expand(B, -1, -1, -1)
        fusion_input = torch.cat([feat_ego_exp, feat_neigh], dim=1)  # [B, 2C, H, W]
        if occ is not None:
            weight = self.att_conv(fusion_input) * occ  # [B, 1, H, W]
        else:
            weight = self.att_conv(fusion_input)
        fused = weight * feat_neigh + (1 - weight) * feat_ego_exp  # [B, C, H, W]
        return fused, weight

class DiscoFusion(nn.Module):
    def __init__(self, feature_dims=64):
        super().__init__()
        from opencood.models.fuse_modules.disco_fuse import PixelWeightLayer
        self.pixel_weight_layer = PixelWeightLayer(feature_dims)

    def forward(self, feat_ego, feat_neigh, occ=None):
        # feat_ego: [1, C, H, W] -> expand to [B, C, H, W]
        B, C, H, W = feat_neigh.shape
        feat_ego_exp = feat_ego.expand(B, -1, -1, -1)
        fusion_input = torch.cat([feat_ego_exp, feat_neigh], dim=1)  # [B, 2C, H, W]
        # (N, 1, H, W)
        agent_weight = self.pixel_weight_layer(fusion_input)
        # (N, 1, H, W)
        agent_weight = F.softmax(agent_weight, dim=0)
        agent_weight = agent_weight.expand(-1, C, -1, -1)
        # (N, C, H, W)
        feature_fused = torch.sum(agent_weight * feat_neigh, dim=0)

        return feature_fused