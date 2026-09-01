# singel seg dataset
# 每个场景选多个车辆，其他车辆转换到ego坐标系
import random
import math
from collections import OrderedDict
import cv2
import numpy as np
import torch
import copy
from icecream import ic

from opencood.utils import box_utils as box_utils

from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum_with_label,
)

from opencood.utils.roiaware_pool3d import roiaware_pool3d_utils

from pdb import set_trace as pause

def getSegearlyFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class EarlySegDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            
            self.isv2vreal=False
            if 'v2vreal' in self.root_dir:
                self.isv2vreal=True

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}
            ego_id = -1
            ego_lidar_pose = []

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    break
                
            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            projected_lidar_stack = []
            object_stack = []
            object_id_stack = []

            for cav_id, selected_cav_base in base_data_dict.items():
                 # check if the cav is within the communication range with ego
                # v2v4real is not list
                # if isinstance(selected_cav_base['params']['lidar_pose'], list):
                if not self.isv2vreal:
                    distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 
                    + (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
                else: # v2v4real
                    distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0,-1] - ego_lidar_pose[0,-1]) ** 2 
                    + (selected_cav_base['params']['lidar_pose'][1,-1] - ego_lidar_pose[1,-1]) ** 2)

                if distance > self.params['comm_range']:
                    continue
                
                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)

                projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
            
            # exclude all repetitive objects
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            # convert list to numpy array, (N, 4)
            projected_lidar_stack = np.vstack(projected_lidar_stack)

            # data augmentation
            projected_lidar_stack, object_bbx_center, mask = \
                self.augment(projected_lidar_stack, object_bbx_center, mask)
            
             # we do lidar filtering in the stacked lidar
            projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                        self.params['preprocess'][
                                                            'cav_lidar_range'])
            # augmentation may remove some of the bbx out of range
            object_bbx_center_valid = object_bbx_center[mask == 1]
            object_bbx_center_valid, range_mask = \
                box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                        self.params['preprocess'][
                                                            'cav_lidar_range'],
                                                        self.params['postprocess'][
                                                            'order'],
                                                        return_mask=True
                                                        )
            mask[object_bbx_center_valid.shape[0]:] = 0
            object_bbx_center[:object_bbx_center_valid.shape[0]] = \
                object_bbx_center_valid
            object_bbx_center[object_bbx_center_valid.shape[0]:] = 0
            unique_indices = list(np.array(unique_indices)[range_mask])

            # generate seg labels
            gt_box_center = object_bbx_center[mask == 1] # [M, 7]
            points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(projected_lidar_stack[:, :3], gt_box_center) 
            # points_mask: [gt_box_center.shape[0], lidar_np.shape[0]]
            fg_mask = np.any(points_mask, axis=0) # lidar_np.shape[0], FG:1 BG:0
            seg_labels = fg_mask.astype(np.int64)  

            processed_data_dict['ego'].update({'origin_lidar': projected_lidar_stack,
                                               'object_bbx_mask': mask,
                                               'seg_labels': seg_labels
                                               })
                                                           
            return processed_data_dict

        def get_item_single_car(self, selected_cav_base, ego_pose):
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

            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose)

            # retrieve objects under ego coordinates
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([selected_cav_base], transformation_matrix if self.isv2vreal else ego_pose)


            # lidar
            lidar_np = selected_cav_base['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            # remove points that hit ego vehicle
            lidar_np = mask_ego_points(lidar_np)

            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                        transformation_matrix)
            
            lidar_np = mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])

            
            # gt_box_center = object_bbx_center[object_bbx_mask == 1] # [M, 7]

            # points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(lidar_np[:, :3], gt_box_center) 
            # # points_mask: [gt_box_center.shape[0], lidar_np.shape[0]]
            # fg_mask = np.any(points_mask, axis=0) # lidar_np.shape[0], FG:1 BG:0
            # seg_labels = fg_mask.astype(np.int64)               

            selected_cav_processed.update(
                    {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
                    'object_ids': object_ids,
                    'projected_lidar': lidar_np})

            return selected_cav_processed


        def collate_batch_train(self, batch):
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
            label_list = []
            object_bbx_mask = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                origin_lidar.append(ego_dict['origin_lidar'])
                label_list.append(ego_dict['seg_labels'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])

            origin_lidar, label_list = downsample_lidar_minimum_with_label(pcd_np_list=origin_lidar, label_list=label_list)
            origin_lidar = torch.from_numpy(np.array(origin_lidar))
            label_torch = torch.from_numpy(np.array(label_list))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            output_dict['ego'].update({'origin_lidar': origin_lidar,
                                       'label_dict': label_torch,
                                       'object_bbx_mask': object_bbx_mask})

            return output_dict
    

        def collate_batch_test(self, batch):
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
            
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            batch = batch[0]
            output_dict = {}


            for cav_id, cav_content in batch.items():
                output_dict.update({cav_id: {}})
                output_dict[cav_id].update({
                    'origin_lidar': torch.from_numpy(np.array([cav_content['origin_lidar']])),
                    'label_dict': torch.from_numpy(np.array([cav_content['seg_labels']]))
                })

            return output_dict

        
    return EarlySegDataset


