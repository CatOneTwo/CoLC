# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn> Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


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
from opencood.utils import box_utils as box_utils
from opencood.utils.common_utils import torch_tensor_to_numpy, merge_features_to_dict

from pdb import set_trace as pause

from torch_cluster import fps as fps_cluster

import time

import cupoch as cph

import cv2

import os

def draw_bev_lidar(voxels, pth):
    cv2.imwrite(
        pth,
        voxels.max(dim=0)[0][:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255,
    )


class PointPillarColc(nn.Module):
    def __init__(self, args):
        super(PointPillarColc, self).__init__()

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
                                  kernel_size=1) # BIN_NUM = 2, output channels = 384
        else:
            self.use_dir = False
                    
        # ===neighbor===
        self.nei_model = None
        # Foreground-Aware Point Cloud Sampling (FAPS)
        if 'neighbor_points_selection' in args:
            self.nei_supply =args['neighbor_points_selection']['nei_supply']
            self.score_threshold =args['neighbor_points_selection']['score_threshold']
            # Apply farthest point sampling to foreground points.
            if 'fg_fps' in args['neighbor_points_selection']:
                self.fg_fps = args['neighbor_points_selection']['fg_fps'] # Farthest point sampling
            else:
                self.fg_fps = False
            # Apply random sampling to background points.
            if 'bg_rs' in args['neighbor_points_selection']:
                self.bg_rs = args['neighbor_points_selection']['bg_rs'] # Random sampling
            else:
                self.bg_rs = False
        else:
            self.nei_supply = False

        # ===ego===
        # Sparse-to-Dense LiDAR Completion (SDLC)
        self.vqvae = None
        self.fusion = args['fusion'] if 'fusion' in args else 'disco1'
        self.fuse_layer = DiscoFusion(64)
            

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar_colc']['voxel_features'] # [M,32,4]
        voxel_coords = data_dict['processed_lidar_colc']['voxel_coords'] # [M,4]
        voxel_num_points = data_dict['processed_lidar_colc']['voxel_num_points'] # [M]

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict) # 
        batch_dict = self.scatter(batch_dict) # batch_dict['spatial_features'] [B, 64, 200, 504]
        
        completion_time = 0
        fusion_time = 0
        enc_time = 0
        vq_time = 0
        dec_time = 0

        # if self.vqvae and 'processed_lidar_downsample' in data_dict: 
        if 'processed_lidar_downsample' in data_dict: 
            # vis spatial features
            # vis_save_path_root = os.path.join('debug_pillar_v2xsim')
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

            if batch_dict_sparse['pillar_features'].dim() == 2:

                batch_dict_sparse = self.scatter(batch_dict_sparse) 
                sparse_spatial_features = batch_dict_sparse['spatial_features'] # 

                # for j in range(sparse_spatial_features.size()[0]):
                #     draw_bev_lidar(sparse_spatial_features[j], "{}/No{}_sparse.png".format(vis_save_path_root, j))
                
                record_len_nei = data_dict['record_len_nei']

                assert sparse_spatial_features.size(0) == record_len_nei.size(0), f"Size mismatch: {sparse_spatial_features.size(0)} != {record_len_nei.size(0)}"

                if self.vqvae: # Completion
                    t1 = time.time()
                    batch_dict = {
                    'spatial_features_sparse':sparse_spatial_features
                    }
                    output_dict = self.vqvae.s2d_completion(batch_dict)
                    rec_spatial_features = output_dict['generated_voxel']
                    t2 = time.time()
                    completion_time += (t2 - t1)

                    enc_time += output_dict['enc_time']
                    vq_time += output_dict['quant_time']
                    dec_time += output_dict['dec_time']
                else:
                    # Use the transmitted full point cloud without completion.
                    rec_spatial_features = sparse_spatial_features

                # for j in range(rec_spatial_features.size()[0]):
                #     draw_bev_lidar(rec_spatial_features[j], "{}/No{}_dense.png".format(vis_save_path_root, j))

                # rec_spatial_features = sparse_spatial_features
                # occ_mask = output_dict['occ_mask']
                

                assert rec_spatial_features.size(0) == record_len_nei.size(0), f"Size mismatch: {rec_spatial_features.size(0)} != {record_len_nei.size(0)}"
                
                B, C, H, W = spatial_features.shape
                out = []
                
                t3 = time.time()
                for i in range(B):
                    spatial_features_i = spatial_features[i].unsqueeze(0)
                    # draw_bev_lidar(spatial_features_i[0], "{}/No{}_ori.png".format(vis_save_path_root, i))
                    ind = (record_len_nei == i)
                    if sum(ind) > 0:
                        # ego + nei_ds_lidar -> nei_rec_lidar
                        if self.fusion == 'disco1':
                            rec_spatial_features_i = rec_spatial_features[ind]
                            # occ_mask_i = occ_mask[ind]
                            ego_mask = (spatial_features_i.abs().sum(dim=1, keepdim=True) > 1e-6).float()
                            empty_mask = 1.0 - ego_mask
                            
                            filled_feat = self.fuse_layer(spatial_features_i, rec_spatial_features_i)
                            # draw_bev_lidar(filled_feat, "{}/No{}_nei_fused.png".format(vis_save_path_root, i))
                            
                            spatial_features_i = spatial_features_i * ego_mask + filled_feat * empty_mask
                        # ego -> ego + nei_rec_lidar
                        elif self.fusion == 'disco2':
                            rec_spatial_features_i = rec_spatial_features[ind]
                            # TODO: Add spatial_features_i to rec_spatial_features_i.
                            all_spatial_features_i = torch.cat([spatial_features_i, rec_spatial_features_i], dim=0)
                            spatial_features_i = self.fuse_layer(spatial_features_i, all_spatial_features_i).unsqueeze(0)

                    # draw_bev_lidar(spatial_features_i[0], "{}/No{}_rec.png".format(vis_save_path_root, i))
                    out.append(spatial_features_i)
                
                fused_feat = torch.stack(out).squeeze(1)
                batch_dict['spatial_features'] = fused_feat
                t4 = time.time()
                fusion_time += (t4 - t3)


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
        
        output_dict.update({
            'enc_time': enc_time,
            'vq_time': vq_time,
            'dec_time': dec_time,
            'completion_time': completion_time,
            'fusion_time': fusion_time
        })

        return output_dict
    
    def neighbor_points_selection(self, batch_data, opencood_dataset, device, icp=False):
        # The neighbor samples and transmits the foreground point cloud based on its own predictions,
        # while the ego merges the point clouds from neighbors.

        record_len = batch_data['ego']['label_dict']['record_len'] # [n1,n2,n3,...,n4]
        B = record_len.size(0) # batch size

        processed_lidar_list = [] # Early-fused point clouds for all batches
        processed_lidar_downsample_list = [] # Downsampled voxel features from all neighbors

        # nei_lidar_original_list = []
        # nei_lidar_downsample_list = []
        # nei_lidar_box_list = []

        record_len_nei = []
        for b in range(B):
            batch_points = batch_data['ego']['original_lidar'][b] # All original projected LiDAR points in the current batch

            # transformation_matrix_list = batch_data['ego']['transformation_matrix_list'][b] # Transformation matrices

            # Save neighboring point clouds.
            # nei_lidar_original = [] # Neighboring point clouds in the current batch
            # nei_lidar_downsample = [] # Downsampled neighboring point clouds in the current batch
            # nei_lidar_box = [] # Downsampled neighboring point clouds in the current batch
            
            processed_features_list = [] # Early-fused point clouds for all batches

            projected_lidar_stack = [] # Early-fused point clouds in the current batch: ego + sparse neighbors
            ego_points = batch_points[0]
            projected_lidar_stack.append(ego_points) # ego lidar
            
            

            faps_time = 0
            icp_time = 0

            # Extract foreground point clouds from neighbors based on predictions.
            for item in range(1, record_len[b]):

                # Agents other than ego.
                points = batch_points[item] # [N,4]
                points = train_utils.to_device(torch.tensor(points), device)

                if icp:
                    t3 = time.time() 
                    points = self.register_icp_fast(points, ego_points)
                    points = train_utils.to_device(torch.tensor(points), device)
                    t4 = time.time()
                    icp_time += (t4-t3)

                t1 = time.time()

                data_dict = {'origin_lidar': points.unsqueeze(0)}
                logits = self.nei_model.forward(data_dict)

                probs = torch.sigmoid(logits[0])          # Convert logits to probabilities
                preds = (probs > 0.5).long()            # Binary prediction mask

                # Select the point cloud to transmit from the neighbor.
                if self.nei_supply:
                    # rps for all points
                    # points_count = points.size(0)
                    # target_point_count = int(points_count * self.bg_rs['ratio']) # Randomly sample 20%
                    # random_indices = np.random.choice(points_count, target_point_count, replace=False)
                    # points_in_box = points[random_indices]
                    
                    points_in_box = points[preds.squeeze()==1]
                    # FPS: Farthest Point Sampling for predicted foreground points # 2025/3/28
                    if self.fg_fps and points_in_box.size(0) > self.fg_fps['min_points']: # Sample only above the minimum point count
                        idx = fps_cluster(points_in_box, ratio=self.fg_fps['ratio'])
                        points_in_box = points_in_box[idx]

                        # fg-rps
                        # fg_points_count = points_in_box.size(0)
                        # target_point_count = int(fg_points_count * self.fg_fps['ratio']) # Randomly sample 20%
                        # random_indices = np.random.choice(fg_points_count, target_point_count, replace=False)
                        # points_in_box = points_in_box[random_indices]

                    if self.bg_rs:
                        points_not_in_box = points[preds.squeeze() == 0] # Background
                        bg_points_count = points_not_in_box.size(0)
                        target_point_count = int(bg_points_count * self.bg_rs['ratio']) # Randomly sample 20%
                        random_indices = np.random.choice(bg_points_count, target_point_count, replace=False)
                        points_not_in_box = points_not_in_box[random_indices]
                        points_in_box = torch.cat((points_in_box,points_not_in_box),dim=0) # Merge with the original point cloud

                    # nei_lidar_original.append(points) # Original neighbor point cloud
                    # nei_lidar_downsample.append(points_in_box) # Downsampled neighbor point cloud

                    record_len_nei.append(b)

                    t2 = time.time() 
                    faps_time += (t2 - t1)
                    
                    # Used for pillar fusion.
                    processed_features = opencood_dataset.pre_processor.preprocess(points_in_box.cpu().numpy())
                    processed_features_list.append(processed_features)
                
                else:
                    points_in_box = points

                    # During testing, apply pillar fusion to the full point cloud as well.
                    # record_len_nei.append(b)
                    # processed_features = opencood_dataset.pre_processor.preprocess(points_in_box.cpu().numpy())
                    # processed_features_list.append(processed_features)
                
                # Used for point cloud fusion.
                # if icp: 
                #     t3 = time.time() 
                #     projected_lidar_stack.append(self.register_icp_fast(points_in_box, ego_points))
                #     t4 = time.time()
                #     icp_time += (t4-t3)
                # else:
                #     projected_lidar_stack.append(points_in_box.cpu().numpy()) # Ego point cloud + downsampled point clouds
                                
                projected_lidar_stack.append(points_in_box.cpu().numpy()) # Ego point cloud + downsampled point clouds
            
            if record_len[b] > 1:
                faps_time = faps_time/(record_len[b].cpu().numpy()-1)
                icp_time = icp_time/(record_len[b].cpu().numpy()-1)
            batch_data['ego'].update({'faps_time': faps_time,
                                      'icp_time': icp_time})

            # 3. Neighbors transmit sampled foreground points to ego for fusion.
            merged_feature_dict = merge_features_to_dict(processed_features_list)
            processed_lidar_downsample_list.append(merged_feature_dict)
        
            # nei_lidar_original_list.append(nei_lidar_original)
            # nei_lidar_downsample_list.append(nei_lidar_downsample)
            # nei_lidar_box_list.append(nei_lidar_box)
            
            batch_data['ego']['original_lidar'][b]=projected_lidar_stack # ego + sparse/rec nei

            # Fused point cloud for the current batch.
            stack_lidar_np = np.vstack(projected_lidar_stack) # [[n1,4],[n2,4],[n3,4]] -> [n1+n2+n3,4]
            stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                        opencood_dataset.params['preprocess']['cav_lidar_range'])
            stack_feature_processed = opencood_dataset.pre_processor.preprocess(stack_lidar_np)
            processed_lidar_list.append(stack_feature_processed)

        # If neighbors exist, store the downsampled pillars for each neighbor.
        merged_feature_dict = merge_features_to_dict(processed_lidar_downsample_list)
        if 'voxel_features' in merged_feature_dict:
            processed_downsampled_lidar_torch_dict = \
                    opencood_dataset.pre_processor.collate_batch(merged_feature_dict)

            batch_data['ego'].update({'processed_lidar_downsample': train_utils.to_device(processed_downsampled_lidar_torch_dict, device)})
            batch_data['ego'].update({'record_len_nei': torch.tensor(record_len_nei).to(device)})

        # batch_data['ego'].update({
        #     'nei_lidar_original_list': nei_lidar_original_list, 
        #     'nei_lidar_downsample_list': nei_lidar_downsample_list,
        #     'nei_lidar_box_list': nei_lidar_box_list
        # })

        # Update the input with the new early-fusion data.
        processed_lidar_torch_dict = \
                opencood_dataset.pre_processor.collate_batch(processed_lidar_list)
        batch_data['ego'].update({'processed_lidar_colc': train_utils.to_device(processed_lidar_torch_dict, device)}) # Fuse the ego and downsampled point clouds
        
        return batch_data

    def register_icp_fast(self, points_in_box, ego_points):
        if not cph.utility.is_cuda_available():
            print("⚠️ CUDA not available, falling back to CPU (performance will drop).")
        
        # ---------- Build CPU point clouds ----------
        source = cph.geometry.PointCloud()
        source.points = cph.utility.Vector3fVector(points_in_box[:, :3].cpu().numpy())

        target = cph.geometry.PointCloud()
        target.points = cph.utility.Vector3fVector(ego_points[:, :3])

        # ---------- Voxel downsampling ----------
        voxel_size = 0.4 # Larger values are faster
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        # ---------- Estimate normals (point-to-plane is more robust) ----------
        source_down.estimate_normals()
        target_down.estimate_normals()

        # ---------- GPU ICP parameters ----------
        criteria = cph.registration.ICPConvergenceCriteria(
            relative_fitness=1e-3,
            relative_rmse=1e-3,
            max_iteration=30
        )

        reg_p2p = cph.registration.registration_icp(
            source_down,
            target_down,
            max_correspondence_distance=1.0,
            init=np.eye(4),
            estimation_method=cph.registration.TransformationEstimationPointToPlane(),  # More robust
            criteria=criteria
        )

        if reg_p2p.fitness < 0.2 or reg_p2p.inlier_rmse > 1.0:
            print("ICP failed")

        # ---------- Apply the transformation ----------
        T_icp = reg_p2p.transformation
        points_in_box = (T_icp @ points_in_box.cpu().numpy().T).T
            
        return points_in_box

    

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
        
        # weight heatmap visualization
        # import matplotlib.pyplot as plt
        # for i in range(agent_weight.size(0)):
        #     heatmap_data = agent_weight[i].cpu().numpy()[0]
        #     plt.figure()
        #     plt.imshow(heatmap_data, cmap='inferno')
        #     plt.axis("off")

        #     plt.colorbar()
        #     plt.tight_layout()
        #     plt.savefig("weight{}_v2xsim_dsse.png".format(i))

        # pause()
        agent_weight = F.softmax(agent_weight, dim=0)
        agent_weight = agent_weight.expand(-1, C, -1, -1)
        # (N, C, H, W)
        feature_fused = torch.sum(agent_weight * feat_neigh, dim=0)

        return feature_fused
