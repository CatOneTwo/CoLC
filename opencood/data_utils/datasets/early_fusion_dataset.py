# early fusion dataset
import torch
import numpy as np
from opencood.utils.pcd_utils import downsample_lidar_minimum
import math
from collections import OrderedDict

from opencood.utils import box_utils
from opencood.utils.common_utils import merge_features_to_dict
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.heter_utils import AgentSelector
from opencood.utils.pose_utils import add_noise_data_dict

from opencood.utils.roiaware_pool3d import roiaware_pool3d_utils

from pdb import set_trace as pause

def getEarlyFusionDataset(cls):
    class EarlyFusionDataset(cls):
        """
        This dataset is used for early fusion, where each CAV transmit the raw
        point cloud to the ego vehicle.
        """
        def __init__(self, params, visualize, train=True):
            super(EarlyFusionDataset, self).__init__(params, visualize, train)
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            assert self.supervise_single is False
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']
            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.isv2vreal=False
            if 'v2vreal' in self.root_dir:
                self.isv2vreal=True

            self.isdairv2x=False
            if 'my_dair_v2x' in self.root_dir:
                print(self.root_dir)
                self.isdairv2x=True
            

            self.heterogeneous = False
            if 'heter' in params:
                self.heterogeneous = True
                self.selector = AgentSelector(params['heter'], self.max_cav)
            
            # 近邻只传递GT框内前景点云 cav only select foreground points
            self.cav_fg_select = False if 'cav_fg_select' not in params['fusion']['args']\
                                         else params['fusion']['args']['cav_fg_select']
            # 近邻的点云下采样
            self.cav_lidar_sample = False if 'cav_lidar_sample' not in params['fusion']['args'] else params['fusion']['args']['cav_lidar_sample']


        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

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

            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            projected_lidar_stack = []
            projected_lidar_dict = {}
            object_stack = []
            object_id_stack = []

            # loop over all CAVs to process information
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
                
                # distance = \
                #     math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                #             ego_lidar_pose[0]) ** 2 + (
                #                     selected_cav_base['params'][
                #                         'lidar_pose'][1] - ego_lidar_pose[
                #                         1]) ** 2)
                if distance > self.params['comm_range']:
                    continue

                # 只可视化近邻
                # if len(object_id_stack)!=0 and cav_id!=ego_id:
                #     projected_lidar_stack = []
                #     projected_lidar_dict = {}
                #     object_stack = []
                #     object_id_stack = []
                
                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_lidar_pose)
                # all these lidar and object coordinates are projected to ego
                # already.
                
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])
                

                projected_lidar_dict.update({'ego' if cav_id==ego_id else cav_id: selected_cav_processed['projected_lidar']})

                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
            
            # if len(set(object_id_stack))!=len(object_id_stack):
                # print(object_id_stack)
            
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

            # pre-process the lidar to voxel/bev/downsampled lidar
            lidar_dict = self.pre_processor.preprocess(projected_lidar_stack)

            # generate the anchor boxes
            anchor_box = self.post_processor.generate_anchor_box()

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': anchor_box,
                'processed_lidar': lidar_dict,
                'label_dict': label_dict})

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar': projected_lidar_stack,
                                                    'origin_lidar_dict': projected_lidar_dict})

            return processed_data_dict

        def get_item_single_car(self, selected_cav_base, ego_pose):
            """
            Project the lidar and bbx to ego space first, and then do clipping.

            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
            ego_pose : list
                The ego vehicle lidar pose under world coordinate.

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

            # filter lidar
            lidar_np = selected_cav_base['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            # remove points that hit itself
            lidar_np = mask_ego_points(lidar_np)

            # point clouds downsample 2024/9/8
            if self.cav_lidar_sample and selected_cav_base['ego']==False:
                points_count = lidar_np.shape[0]
                tmp = int(1/self.cav_lidar_sample['ratio'])
                if self.cav_lidar_sample['method'] == 'RS': # random sampling, 均匀随机采样
                    target_point_count = int(points_count / tmp) # 随机采样20%
                    random_indices = np.random.choice(points_count, target_point_count, replace=False)
                    lidar_np = lidar_np[random_indices]
                elif self.cav_lidar_sample['method'] == 'FPS': # Farthest Point Sampling
                    target_point_count = int(points_count / tmp) # 随机采样20%
                    selected_points = farthest_point_sampling(torch.tensor(lidar_np), target_point_count)
                    lidar_np = np.array(selected_points)
                elif self.cav_lidar_sample['method'] == 'HINTED': # 只对近邻区域随机采样
                    limit_range = self.params['preprocess']['cav_lidar_range']
                    distance_mask = (lidar_np[:, 0] > int(limit_range[0]/3)) & (lidar_np[:, 0] < int(limit_range[3]/3))
                    near_points = lidar_np[distance_mask] # 近的点
                    far_points = lidar_np[~distance_mask] # 远的点
                    near_points_count = near_points.shape[0]
                    target_near_point_count = int(near_points_count / tmp) # 近邻采样20%
                    random_indices = np.random.choice(near_points_count, target_near_point_count, replace=False)
                    
                    downsampled_near_point_cloud = near_points[random_indices] # 近的点云随机采样
                    lidar_np = np.concatenate((downsampled_near_point_cloud, far_points), 0)
                else:
                    print('cav lidar sample method is not valid!!')


            # project the lidar to ego space
            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                        transformation_matrix)

            # foreground point selection 2024/12/19
            if selected_cav_base['ego']==False and self.cav_fg_select:
                if self.isdairv2x:
                    # dairv2x 需要单独提取路端的标注
                    # generate targets label single GT, note the reference pose is itself.
                    object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single(
                        [selected_cav_base], ego_pose)
                
                # cav只传递包含前景的点云
                gt_boxes = object_bbx_center[object_bbx_mask == 1]
                points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(lidar_np[:, :3], gt_boxes)
                lidar_np = lidar_np[points_mask.sum(0) != 0] # 包含在预测框内的前景, 0为背景，1为前景
                
            
            selected_cav_processed.update(
                {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
                'object_ids': object_ids,
                'projected_lidar': lidar_np})

            return selected_cav_processed

        def collate_batch_test(self, batch):
            """
            Customized collate function for pytorch dataloader during testing
            for late fusion dataset.

            Parameters
            ----------
            batch : dict

            Returns
            -------
            batch : dict
                Reformatted batch.
            """
            # currently, we only support batch size of 1 during testing
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            batch = batch[0] # only ego

            output_dict = {}

            for cav_id, cav_content in batch.items():
                output_dict.update({cav_id: {}})
                # shape: (1, max_num, 7)
                object_bbx_center = \
                    torch.from_numpy(np.array([cav_content['object_bbx_center']]))
                object_bbx_mask = \
                    torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
                object_ids = cav_content['object_ids']

                # the anchor box is the same for all bounding boxes usually, thus
                # we don't need the batch dimension.
                if cav_content['anchor_box'] is not None:
                    output_dict[cav_id].update({'anchor_box':
                        torch.from_numpy(np.array(
                            cav_content[
                                'anchor_box']))})
                if self.visualize:
                    origin_lidar = [cav_content['origin_lidar']]

                # processed lidar dictionary
                processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(
                        [cav_content['processed_lidar']])
                # label dictionary
                label_torch_dict = \
                    self.post_processor.collate_batch([cav_content['label_dict']])

                # save the transformation matrix (4, 4) to ego vehicle
                transformation_matrix_torch = \
                    torch.from_numpy(np.identity(4)).float()
                transformation_matrix_clean_torch = \
                    torch.from_numpy(np.identity(4)).float()

                output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                            'object_bbx_mask': object_bbx_mask,
                                            'processed_lidar': processed_lidar_torch_dict,
                                            'label_dict': label_torch_dict,
                                            'object_ids': object_ids,
                                            'transformation_matrix': transformation_matrix_torch,
                                            'transformation_matrix_clean': transformation_matrix_clean_torch})

                if self.visualize:
                    origin_lidar = \
                        np.array(
                            downsample_lidar_minimum(pcd_np_list=origin_lidar))
                    origin_lidar = torch.from_numpy(origin_lidar)
                    output_dict[cav_id].update({'origin_lidar': origin_lidar,
                                                'origin_lidar_dict': cav_content['origin_lidar_dict']})

            return output_dict
        
        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            image_inputs_list = []
            # used to record different scenario
            label_dict_list = []
            origin_lidar = []
            
            # heterogeneous
            lidar_agent_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []
            
            ### 2022.10.10 single gt ####
            if self.supervise_single:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
                
                label_dict_list.append(ego_dict['label_dict'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                ### 2022.10.10 single gt ####
                if self.supervise_single:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])

                # heterogeneous
                if self.heterogeneous:
                    lidar_agent_list.append(ego_dict['lidar_agent'])

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)

                if self.heterogeneous:
                    lidar_agent = np.concatenate(lidar_agent_list)
                    lidar_agent_idx = lidar_agent.nonzero()[0].tolist()
                    for k, v in merged_feature_dict.items(): # 'voxel_features' 'voxel_num_points' 'voxel_coords'
                        merged_feature_dict[k] = [v[index] for index in lidar_agent_idx]

                if not self.heterogeneous or (self.heterogeneous and sum(lidar_agent) != 0):
                    processed_lidar_torch_dict = \
                        self.pre_processor.collate_batch(merged_feature_dict)
                    output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

                if self.heterogeneous:
                    camera_agent = 1 - lidar_agent
                    camera_agent_idx = camera_agent.nonzero()[0].tolist()
                    if sum(camera_agent) != 0:
                        for k, v in merged_image_inputs_dict.items(): # 'imgs' 'rots' 'trans' ...
                            merged_image_inputs_dict[k] = torch.stack([v[index] for index in camera_agent_idx])
                            
                if not self.heterogeneous or (self.heterogeneous and sum(camera_agent) != 0):
                    output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0]})


            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.supervise_single:
                output_dict['ego'].update({
                    "label_dict_single" : 
                        {"pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                        "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                        "targets": torch.cat(targets_single, dim=0)}
                })

            if self.heterogeneous:
                output_dict['ego'].update({
                    "lidar_agent_record": torch.from_numpy(np.concatenate(lidar_agent_list)) # [0,1,1,0,1...]
                })

            return output_dict

        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict, output_dict)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor
        
        # def post_process(self, data_dict, output_dict):
        #     """
        #     Process the outputs of the model to 2D/3D bounding box.

        #     Parameters
        #     ----------
        #     data_dict : dict
        #         The dictionary containing the origin input data of model.

        #     output_dict :dict
        #         The dictionary containing the output of the model.

        #     Returns
        #     -------
        #     pred_box_tensor : torch.Tensor
        #         The tensor of prediction bounding box after NMS.
        #     gt_box_tensor : torch.Tensor
        #         The tensor of gt bounding box.
        #     """
        #     pred_box_tensor, pred_score = \
        #         self.post_processor.post_process(data_dict, output_dict)
        #     gt_box_tensor, object_id_list = self.post_processor.generate_gt_bbx(data_dict)

        #     return pred_box_tensor, pred_score, gt_box_tensor, object_id_list

    return EarlyFusionDataset

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


