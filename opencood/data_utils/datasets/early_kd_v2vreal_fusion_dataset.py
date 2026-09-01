# intermediate fusion dataset
import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.heter_utils import AgentSelector
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json
from opencood.utils.roiaware_pool3d import roiaware_pool3d_utils


from pdb import set_trace as pause

def getEarlykdv2vrealFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class EarlyKDV2VRealFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            # intermediate and supervise single
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            # for ERMVP: Communication-Efﬁcient and Collaboration-Robust Multi-Vehicle Perception in Challenging Environments
            self.supervise_ego = True if ('supervise_ego' in params['model']['args'] and params['model']['args']['supervise_ego']) \
                                        else False
            # self.proj_first = True if 'proj_first' not in params['fusion']['args']\
            #                              else params['fusion']['args']['proj_first']
            
            self.proj_first = True

            # self.nei_only = True if 'nei_only' not in params['fusion']['args']\
                                        #  else params['fusion']['args']['nei_only']
            
            # 近邻的点云下采样
            self.cav_lidar_sample = False if 'cav_lidar_sample' not in params['fusion']['args'] else params['fusion']['args']['cav_lidar_sample']
            
            # ego + neighbor downsample lidar
            if self.cav_lidar_sample:
                self.comm_efficient = False if 'comm_efficient' not in params['fusion']['args'] else params['fusion']['args']['comm_efficient']


            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.kd_flag = params.get('kd_flag', False)
            if self.kd_flag:
                self.singlestudent = params['kd_flag']['singlestudent']

            self.box_align = False
            if "box_align" in params:
                self.box_align = True
                self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
                self.stage1_result = read_json(self.stage1_result_path)
                self.box_align_args = params['box_align']['args']
                


        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_pose : list, length 6
                The ego vehicle lidar pose under world coordinate.
            ego_pose_clean : list, length 6
                only used for gt box generation

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']
            
            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose) # T_ego_cav
            transformation_matrix_clean = \
                x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                        ego_pose_clean)
            
            # lidar
            if self.load_lidar_file or self.visualize:
                # process lidar
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                lidar_np_clean = copy.deepcopy(lidar_np)

                # point clouds downsample 
                if self.cav_lidar_sample:
                    points_count = lidar_np.shape[0]
                    if self.cav_lidar_sample['method'] == 'RS': # random sampling, 均匀随机采样
                        target_point_count = int(points_count* self.cav_lidar_sample['ratio']) # 随机采样20%
                        random_indices = np.random.choice(points_count, target_point_count, replace=False)
                        lidar_np_ds = lidar_np[random_indices]

                        projected_lidar_ds = \
                        box_utils.project_points_by_matrix_torch(lidar_np_ds[:, :3],
                                                                    transformation_matrix)
                        if self.proj_first:
                            lidar_np_ds[:, :3] = projected_lidar_ds                        
                    processed_lidar_ds = self.pre_processor.preprocess(lidar_np_ds)
                    selected_cav_processed.update({
                        'original_lidar_downsample': lidar_np_ds,
                        'processed_features_downsample': processed_lidar_ds})

                # project the lidar to ego space
                # x,y,z in ego space
                projected_lidar = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                                transformation_matrix)
                projected_lidar_clean = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                                transformation_matrix_clean)
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar
                    lidar_np_clean[:, :3] = projected_lidar_clean

                selected_cav_processed.update({
                    'original_lidar': lidar_np,
                    'original_lidar_clean': lidar_np_clean}) # 保存智能体的原始点云（无论先后投影）

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                if self.kd_flag:
                    lidar_proj_np = copy.deepcopy(lidar_np)
                    lidar_proj_np[:,:3] = projected_lidar

                    selected_cav_processed.update({'projected_lidar': lidar_proj_np})

                processed_lidar = self.pre_processor.preprocess(lidar_np)
                selected_cav_processed.update({'processed_features': processed_lidar})

            if self.supervise_single:
                # generate targets label single GT, note the reference pose is itself.

                object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single(
                    [selected_cav_base], transformation_matrix
                )

                label_dict = self.post_processor.generate_label(
                    gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
                )
                selected_cav_processed.update({
                                    "single_label_dict": label_dict,
                                    "single_object_bbx_center": object_bbx_center,
                                    "single_object_bbx_mask": object_bbx_mask})

            

            # anchor box
            selected_cav_processed.update({"anchor_box": self.anchor_box})
            
            # note the reference pose ego
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],transformation_matrix)

            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean
                }
            )

            return selected_cav_processed

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            # TODO 
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            
            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break
                
            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            agents_image_inputs = []
            processed_features = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            too_far = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []

            originial_lidar_for_agent_list = [] # for each agent
            originial_clean_lidar_for_agent_list = [] # for each agent

            transformation_matrix_list = []
            transformation_matrix_clean_list = []

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []
            
            if self.cav_lidar_sample:
                processed_features_downsample = []
                originial_lidar_for_agent_list_downsample = []
                if self.comm_efficient:
                    projected_lidar_stack_ce = []
                    nei_num = 0
                

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                
                # v2v4real is not list
                if isinstance(selected_cav_base['params']['lidar_pose'], list):
                    distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 
                    + (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
                else: # v2v4real
                    distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0,-1] - ego_lidar_pose[0,-1]) ** 2 
                    + (selected_cav_base['params']['lidar_pose'][1,-1] - ego_lidar_pose[1,-1]) ** 2)

                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    continue

                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                cav_id_list.append(cav_id)   

            for cav_id in too_far:
                base_data_dict.pop(cav_id)


            pairwise_t_matrix = \
                get_pairwise_transformation(base_data_dict,
                                                self.max_cav,
                                                self.proj_first)
            
            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list)
            
            
            for _i, cav_id in enumerate(cav_id_list):
                
                selected_cav_base = base_data_dict[cav_id]
                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_cav_base)
                    
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                
                transformation_matrix_list.append(selected_cav_processed['transformation_matrix'])
                transformation_matrix_clean_list.append(selected_cav_processed['transformation_matrix_clean'])

                if 'only_vis_ego' in self.params and self.params['only_vis_ego']:
                    if not selected_cav_base['ego']:
                        # 单车感知，只需要ego的数据，但需要多车的协同目标
                        continue
                
                if self.load_lidar_file:
                    if self.cav_lidar_sample:                        
                        if self.comm_efficient:
                            if cav_id == ego_id:
                                projected_lidar_stack_ce.append(selected_cav_processed['original_lidar']) # ego是原始点云
                            else:
                                if len(selected_cav_processed['processed_features_downsample']['voxel_coords']) != 0:
                                    projected_lidar_stack_ce.append(selected_cav_processed['original_lidar_downsample'])
                                    processed_features_downsample.append(selected_cav_processed['processed_features_downsample'])
                                    originial_lidar_for_agent_list_downsample.append(selected_cav_processed['original_lidar_downsample'])
                                    nei_num += 1
                            
                        else:
                            if len(selected_cav_processed['processed_features_downsample']['voxel_coords']) == 0:
                                print(selected_cav_processed['processed_features_downsample']['voxel_coords'])
                                continue
                            processed_features_downsample.append(selected_cav_processed['processed_features_downsample'])
                            originial_lidar_for_agent_list_downsample.append(selected_cav_processed['original_lidar_downsample'])

                        

                    if self.kd_flag and self.singlestudent:
                        if cav_id == ego_id:
                            processed_features.append(selected_cav_processed['processed_features'])
                            originial_lidar_for_agent_list.append(selected_cav_processed['original_lidar'])
                            originial_clean_lidar_for_agent_list.append(selected_cav_processed['original_lidar_clean'])
                    else:
                        # if self.nei_only and cav_id != ego_id:
                        processed_features.append(selected_cav_processed['processed_features'])
                        originial_lidar_for_agent_list.append(selected_cav_processed['original_lidar'])
                        originial_clean_lidar_for_agent_list.append(selected_cav_processed['original_lidar_clean'])

                if self.load_camera_file:
                    agents_image_inputs.append(
                        selected_cav_processed['image_inputs'])

                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(
                        selected_cav_processed['projected_lidar'])
                
                if self.supervise_single:
                    if self.supervise_ego:
                        if cav_id == ego_id:
                            single_label_list.append(selected_cav_processed['single_label_dict'])
                            single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                            single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])
                    else:
                        single_label_list.append(selected_cav_processed['single_label_dict'])
                        single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                        single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])
            
            

            # generate single view GT label
            if self.supervise_single:
                single_label_dicts = self.post_processor.collate_batch(single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict['ego'].update({
                    "single_label_dict_torch": single_label_dicts,
                    "single_object_bbx_center_torch": single_object_bbx_center,
                    "single_object_bbx_mask_torch": single_object_bbx_mask,
                    })

            processed_data_dict['ego'].update({
                'original_lidar': originial_lidar_for_agent_list,
                'original_lidar_clean': originial_clean_lidar_for_agent_list})

            if self.kd_flag:
                stack_lidar_np = np.vstack(projected_lidar_stack)
                stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                            self.params['preprocess'][
                                                'cav_lidar_range'])
                stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                processed_data_dict['ego'].update({'teacher_processed_lidar':
                stack_feature_processed})

            
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
            
            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_features)
                processed_data_dict['ego'].update({'processed_lidar': merged_feature_dict})

                if self.cav_lidar_sample:
                    if len(processed_features_downsample):
                        merged_feature_dict_downsample = merge_features_to_dict(processed_features_downsample)
                        processed_data_dict['ego'].update({
                            'processed_lidar_downsample': merged_feature_dict_downsample,
                            'original_lidar_downsample': originial_lidar_for_agent_list_downsample})
                    
                    if self.comm_efficient:
                        stack_lidar_np_ce = np.vstack(projected_lidar_stack_ce) # [[n1,4],[n2,4],[n3,4]] -> [n1+n2+n3,4]
                        stack_lidar_np_ce = mask_points_by_range(stack_lidar_np_ce,
                                                    self.params['preprocess'][
                                                        'cav_lidar_range'])
                        stack_feature_processed_ce = self.pre_processor.preprocess(stack_lidar_np_ce)
                        processed_data_dict['ego'].update({
                            'processed_lidar_ce': stack_feature_processed_ce,
                            'nei_num': nei_num
                        })

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
                processed_data_dict['ego'].update({'image_inputs': merged_image_inputs_dict})


            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=self.anchor_box,
                    mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,
                'transformation_matrix': transformation_matrix_list,
                'transformation_matrix_clean': transformation_matrix_clean_list})

            

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                    np.vstack(
                        projected_lidar_stack)})


            processed_data_dict['ego'].update({'sample_idx': idx,
                                                'cav_id_list': cav_id_list})

            return processed_data_dict


        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            image_inputs_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            origin_lidar = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # disconet
            teacher_processed_lidar_list = []

            originial_lidar_for_agent_list = [] # for each agent
            originial_clean_lidar_for_agent_list = [] # for each agent

            transformation_matrix_list = []
            transformation_matrix_clean_list = []

            if self.cav_lidar_sample:
                processed_lidar_list_downsample = []
                originial_lidar_for_agent_list_downsample = []
                if self.comm_efficient:
                    processed_lidar_list_ce = []
                    record_len_nei = []

            idxs = []
            
            ### 2022.10.10 single gt ####
            if self.supervise_single:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])

                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                    if self.cav_lidar_sample:
                        if 'processed_lidar_downsample' in ego_dict:
                            if 'voxel_coords' in ego_dict['processed_lidar_downsample'].keys() and len(ego_dict['processed_lidar_downsample']['voxel_coords']) > 0: 
                                processed_lidar_list_downsample.append(ego_dict['processed_lidar_downsample'])
                                originial_lidar_for_agent_list_downsample.append(ego_dict['original_lidar_downsample'])
                        if self.comm_efficient:
                            processed_lidar_list_ce.append(ego_dict['processed_lidar_ce'])
                            record_len_nei += [i] * ego_dict['nei_num']

                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
                
                record_len.append(ego_dict['cav_num'])
                # record_len.append(ego_dict['cav_num']-1) # 只保存nei数量
                label_dict_list.append(ego_dict['label_dict'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
                originial_lidar_for_agent_list.append(ego_dict['original_lidar'])
                originial_clean_lidar_for_agent_list.append(ego_dict['original_lidar_clean'])

                transformation_matrix_list.append(ego_dict['transformation_matrix'])
                transformation_matrix_clean_list.append(ego_dict['transformation_matrix_clean'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                if self.kd_flag:
                    teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])

                idxs.append(ego_dict['sample_idx'])

                ### 2022.10.10 single gt ####
                if self.supervise_single:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])
                    object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                    object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])


            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)
                processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(merged_feature_dict)
                output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})
                if self.cav_lidar_sample:
                    if len(processed_lidar_list_downsample) > 0:
                        merged_feature_dict_downsample = merge_features_to_dict(processed_lidar_list_downsample)
                        processed_lidar_torch_dict_downsample = \
                            self.pre_processor.collate_batch(merged_feature_dict_downsample)
                        output_dict['ego'].update({
                            'processed_lidar_downsample': processed_lidar_torch_dict_downsample,
                            'original_lidar_downsample': originial_lidar_for_agent_list_downsample
                            })
                    if self.comm_efficient:
                        merged_feature_dict_ce = merge_features_to_dict(processed_lidar_list_ce)
                        processed_lidar_torch_dict_ce = \
                            self.pre_processor.collate_batch(merged_feature_dict_ce)
                        output_dict['ego'].update({
                            'processed_lidar_colc': processed_lidar_torch_dict_ce,
                            'record_len_nei': torch.tensor(record_len_nei)
                            })

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

                output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            

            record_len = torch.from_numpy(np.array(record_len, dtype=int))

            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                     'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len
            
            # output_dict['ego'].update({'original_lidar': originial_lidar_for_agent_list})
            output_dict['ego'].update({
                'original_lidar': originial_lidar_for_agent_list,
                'original_lidar_clean': originial_clean_lidar_for_agent_list,
                'transformation_matrix_list': transformation_matrix_list,
                'transformation_matrix_clean_list': transformation_matrix_clean_list
            })

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'record_len': record_len,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0],
                                    'pairwise_t_matrix': pairwise_t_matrix,
                                    'anchor_box': self.anchor_box_torch})

            if self.label_sparse:
                output_dict['ego'].update({'idx':idxs})

            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.kd_flag:
                teacher_processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(teacher_processed_lidar_list)
                output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})


            if self.supervise_single:
                output_dict['ego'].update({
                    "label_dict_single":{
                            "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                            "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                            "targets": torch.cat(targets_single, dim=0),
                            # for centerpoint
                            "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                            "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                        },
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                })


            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            # check if anchor box in the batch
            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box':
                    self.anchor_box_torch})

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation is only used in post process (no use.)
            # we all predict boxes in ego coord.
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch,
                                        'transformation_matrix_clean':
                                        transformation_matrix_clean_torch,})

            # output_dict['ego'].update({
            #     "sample_idx": batch[0]['ego']['sample_idx'],
            #     "cav_id_list": batch[0]['ego']['cav_id_list']
            # })

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


    return EarlyKDV2VRealFusionDataset


