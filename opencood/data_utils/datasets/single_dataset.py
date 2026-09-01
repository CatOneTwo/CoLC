# late fusion dataset
import random
import math
from collections import OrderedDict
import cv2
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.heter_utils import AgentSelector
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.roiaware_pool3d import roiaware_pool3d_utils

from pdb import set_trace as pause

def getSingleFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class SingleDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            
            self.cav_lidar_sample = False if 'cav_lidar_sample' not in params['fusion']['args'] else params['fusion']['args']['cav_lidar_sample']

            self.fg_aug = False if 'fg_aug' not in params['fusion']['args'] else params['fusion']['args']['fg_aug']
            self.bg_lc = False if 'bg_lc' not in params['fusion']['args'] else params['fusion']['args']['bg_lc']

            self.isv2vreal=False
            if 'v2vreal' in self.root_dir:
                self.isv2vreal=True

            self.isdairv2x=False
            if 'my_dair_v2x' in self.root_dir:
                self.isdairv2x=True
          
        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            
            reformat_data_dict = self.get_item(base_data_dict)

            return reformat_data_dict

        def get_item(self, base_data_dict):
            processed_data_dict = OrderedDict()

            # during training, we return a random cav's data
            # only one vehicle is in processed_data_dict
            
            # TODO 1 dair-v2x只传递路端，2. v2xset 和 v2xreal都传递

            if self.isdairv2x:
                
                if len(base_data_dict.items()) == 1:
                    return None
                selected_cav_id, selected_cav_base = list(base_data_dict.items())[1]
            else:
                if self.train:
                    selected_cav_id, selected_cav_base = random.choice(
                        list(base_data_dict.items())
                    )
                else:
                    selected_cav_id, selected_cav_base = list(base_data_dict.items())[0]
            
            # if self.fg_aug:
            selected_cav_base = self.foreground_augmentaion(selected_cav_base, selected_cav_id, base_data_dict)

            selected_cav_processed = self.get_item_single_car(selected_cav_base)
            processed_data_dict.update({"ego": selected_cav_processed})

            return processed_data_dict

        def get_item_single_car(self, selected_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}

            # lidar
            if self.load_lidar_file:
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                lidar_np = mask_points_by_range(lidar_np,
                                                self.params['preprocess'][
                                                    'cav_lidar_range']) 
                # lidar_np_ = lidar_np.copy()
                if self.cav_lidar_sample:

                    if self.cav_lidar_sample['method'] == 'RS': # random sampling, 随机采样
                        points_count = lidar_np.shape[0]
                        tmp = int(1/self.cav_lidar_sample['ratio'])
                        target_point_count = int(points_count / tmp) # 随机采样10%
                        random_indices = np.random.choice(points_count, target_point_count, replace=False)
                        lidar_np_ds = lidar_np[random_indices]

                        if self.bg_lc:
                            all_indices = np.arange(points_count)
                            remaining_indices = np.setdiff1d(all_indices, random_indices)
                            lidar_np = lidar_np[remaining_indices]

                    elif self.cav_lidar_sample['method'] == 'RRS': # random sampling, 随机生成的阈值
                        points_count = lidar_np.shape[0]
                        if self.train:
                            ratio = np.random.uniform(0.01, 0.1)
                        else:
                            ratio = 0.05
                        target_point_count = int(points_count * ratio) # 随机采样10%
                        random_indices = np.random.choice(points_count, target_point_count, replace=False)
                        lidar_np_ds = lidar_np[random_indices]
                    elif self.cav_lidar_sample['method'] == 'BGRS': # FG + BG random sampling, 背景随机采样
                        # calculate the transformation matrix
                        current_pose = selected_cav_base['params']['lidar_pose']
                        transformation_matrix = x1_to_x2(current_pose, current_pose)

                        if self.isdairv2x:
                            # dairv2x 需要单独提取路端的标注
                            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single(
                                [selected_cav_base], current_pose)
                        else:
                            # retrieve objects under ego coordinates
                            object_bbx_center, object_bbx_mask, object_ids = \
                                self.generate_object_center([selected_cav_base], transformation_matrix if self.isv2vreal else current_pose)           
                        # cav只传递包含前景的点云
                        gt_boxes = object_bbx_center[object_bbx_mask == 1]
                        points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(lidar_np[:, :3], gt_boxes) # (box_num, lidar_np_num)
                        lidar_np_fg = lidar_np[points_mask.sum(0) != 0] # 包含在预测框内的前景, 0为背景，1为前景 
                        if 'fg_ratio' in self.cav_lidar_sample and lidar_np_fg.shape[0]>50:
                            fg_points_count = lidar_np_fg.shape[0]
                            target_point_count = int(fg_points_count * self.cav_lidar_sample['fg_ratio']) # FPS 20%
                            selected_points = farthest_point_sampling(torch.tensor(lidar_np_fg), target_point_count)
                            lidar_np_fg = np.array(selected_points)

                        lidar_np_bg = lidar_np[points_mask.sum(0) == 0]
                        bg_points_count = lidar_np_bg.shape[0]
                        
                        target_point_count = int(bg_points_count * self.cav_lidar_sample['bg_ratio']) # RS 10%
                        random_indices = np.random.choice(bg_points_count, target_point_count, replace=False)
                        lidar_np_bg = lidar_np_bg[random_indices]

                        lidar_np_ds = np.vstack([lidar_np_fg,lidar_np_bg])
                

                if self.fg_aug and 'nei_lidar_np_fg' in selected_cav_base: # 前景点云增强
                    
                    nei_lidar_np = selected_cav_base['nei_lidar_np']
                    nei_lidar_np_fg = selected_cav_base['nei_lidar_np_fg']
                    
                    # vis debug
                    # from opencood.visualization import simple_vis
                    # import os
                    # vis_save_path_root = os.path.join('debug_fg_aug_v2xsim_4')
                    # if not os.path.exists(vis_save_path_root):
                    #     os.makedirs(vis_save_path_root)
                    # gt_range = self.params['preprocess']['cav_lidar_range']

                    # lidar_original_dict = ({0: lidar_np})
                    # # lidar_original_dict = ({0: lidar_np_})
                    # lidar_early_dict =({'ego': lidar_np, 
                    #                     '0': nei_lidar_np})
                    # lidar_aug_dict = ({'ego': lidar_np, 
                    #                     '0': nei_lidar_np_fg})

                    # vis_save_path = os.path.join(vis_save_path_root, 'ori.png')
                
                    # simple_vis.visualize_colorful({},
                    #                                 lidar_original_dict,
                    #                                 gt_range,
                    #                                 vis_save_path,
                    #                                 method='bev',
                    #                                 left_hand=False)
                    # vis_save_path = os.path.join(vis_save_path_root, 'early.png')
                
                    # simple_vis.visualize_colorful({},
                    #                                 lidar_early_dict,
                    #                                 gt_range,
                    #                                 vis_save_path,
                    #                                 method='bev',
                    #                                 left_hand=False)
                    # vis_save_path = os.path.join(vis_save_path_root, 'aug.png')
                
                    # simple_vis.visualize_colorful({},
                    #                                 lidar_aug_dict,
                    #                                 gt_range,
                    #                                 vis_save_path,
                    #                                 method='bev',
                    #                                 left_hand=False)
                    # pause()

                    lidar_np = np.vstack([lidar_np, nei_lidar_np_fg])
                    

            selected_cav_processed.update({'origin_lidar': lidar_np})

            if self.cav_lidar_sample:
                selected_cav_processed.update({'origin_lidar_sparse': lidar_np_ds})
                    
            return selected_cav_processed

        def foreground_augmentaion(self, selected_cav_base, selected_cav_id, base_data_dict):
            
            if not self.fg_aug:
                return selected_cav_base

            ego_pose = selected_cav_base['params']['lidar_pose']
            # 聚集近邻的早期融合点云
            projected_lidar_stack = []
            object_stack = []
            object_id_stack = []
            for cav_id, selected_nei_base in base_data_dict.items():

                if not self.isv2vreal:
                    distance = \
                    math.sqrt((selected_nei_base['params']['lidar_pose'][0] - ego_pose[0]) ** 2 
                    + (selected_nei_base['params']['lidar_pose'][1] - ego_pose[1]) ** 2)
                else: # v2v4real
                    distance = \
                    math.sqrt((selected_nei_base['params']['lidar_pose'][0,-1] - ego_pose[0,-1]) ** 2 
                    + (selected_nei_base['params']['lidar_pose'][1,-1] - ego_pose[1,-1]) ** 2)
                
                if distance > self.params['comm_range']:
                    continue
                
                transformation_matrix = x1_to_x2(selected_nei_base['params']['lidar_pose'], ego_pose)

                object_bbx_center, object_bbx_mask, object_ids = \
                        self.generate_object_center([selected_nei_base], transformation_matrix if self.isv2vreal else ego_pose)   

                object_stack.append(object_bbx_center[object_bbx_mask == 1])
                object_id_stack += object_ids

                if cav_id == selected_cav_id: # only need nei lidar
                    continue

                lidar_np = selected_nei_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # project the lidar to ego space
                lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                projected_lidar_stack.append(lidar_np)
            
            if len(projected_lidar_stack) > 0:
                unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]

                projected_lidar_stack =np.vstack(projected_lidar_stack)
                nei_lidar_np = mask_points_by_range(projected_lidar_stack, self.params['preprocess']['cav_lidar_range'])

                points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(nei_lidar_np[:, :3], object_stack) # (box_num, lidar_np_num)
                nei_lidar_np_fg = nei_lidar_np[points_mask.sum(0) != 0] # 包含在预测框内的前景, 0为背景，1为前景 
                
                selected_cav_base['nei_lidar_np'] = nei_lidar_np
                selected_cav_base['nei_lidar_np_fg'] = nei_lidar_np_fg

            return selected_cav_base


        def collate_batch(self, batch):
            """
            Customized collate function for pytorch dataloader during training
            for early and late fusion dataset.

            Parameters
            ----------
            batch : dict

            Returns
            -------
            batch : dict
                Reformatted batch.
            """
            # during training, we only care about ego.
            output_dict = {'ego': {}}

            origin_lidar = []
            origin_lidar_sparse = []

            for i in range(len(batch)):
                if self.isdairv2x and batch[i] ==None:
                    continue
                ego_dict = batch[i]['ego']                
                origin_lidar.append(ego_dict['origin_lidar'])  
                if self.cav_lidar_sample:
                    origin_lidar_sparse.append(ego_dict['origin_lidar_sparse'])

            # for centerpoint
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.cav_lidar_sample:
                origin_lidar_sparse = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar_sparse))
                origin_lidar_sparse = torch.from_numpy(origin_lidar_sparse)
                output_dict['ego'].update({'origin_lidar_sparse': origin_lidar_sparse})

            return output_dict

    return SingleDataset


