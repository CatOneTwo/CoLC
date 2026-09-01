# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils.transformation_utils import get_relative_transformation
from opencood.utils.box_utils import create_bbx, project_box3d, nms_rotated
from opencood.utils.camera_utils import indices_to_depth
from sklearn.metrics import mean_squared_error

import time

from pdb import set_trace as pause

# def inference_late_fusion(batch_data, model, dataset):
#     """
#     Model inference for late fusion.

#     Parameters
#     ----------
#     batch_data : dict
#     model : opencood.object
#     dataset : opencood.LateFusionDataset

#     Returns
#     -------
#     pred_box_tensor : torch.Tensor
#         The tensor of prediction bounding box after NMS.
#     gt_box_tensor : torch.Tensor
#         The tensor of gt bounding box.
#     """
#     output_dict = OrderedDict()

#     for cav_id, cav_content in batch_data.items():
#         output_dict[cav_id] = model(cav_content)

#     pred_box_tensor, pred_score, gt_box_tensor, box_num = \
#         dataset.post_process(batch_data,
#                              output_dict)

#     return_dict = {"pred_box_tensor" : pred_box_tensor, \
#                     "pred_score" : pred_score, \
#                     "gt_box_tensor" : gt_box_tensor,
#                     "box_num": box_num}
#     return return_dict

def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict

def inference_late_fusion_hetero(batch_data, model, nei_model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    
    for cav_id, cav_content in batch_data.items():
        if cav_id == 'ego':
            output_dict[cav_id] = model(cav_content)
        else:
            output_dict[cav_id] = nei_model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict

def inference_no_fusion(batch_data, model, dataset, single_gt=False):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    single_gt : bool
        if True, only use ego agent's label.
        else, use all agent's merged labels.
    """
    output_dict_ego = OrderedDict()
    if single_gt:
        batch_data = {'ego': batch_data['ego']}
        
    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_no_fusion(batch_data,  # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict

def inference_no_fusion_w_uncertainty(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
        dataset.post_process_no_fusion_uncertainty(batch_data, # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor, \
                    "uncertainty_tensor" : uncertainty_tensor}

    return return_dict

def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)
                             
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    
    if 'comm_rate' in output_dict['ego']:
         return_dict.update({"comm_rate" : output_dict['ego']['comm_rate']})

    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    
    if 'origin_lidar_dict' in cav_content:
        return_dict.update({"origin_lidar_dict": cav_content['origin_lidar_dict']})
    return return_dict


def inference_cecooper_fusion(batch_data, model, dataset, device):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    # neighbor FAPS
    batch_data = model.neighbor_points_selection(batch_data, dataset, device)
    
    # Ego points + seleted neighbor points 
    cav_content = batch_data['ego']
    
    ### vis debug
    """
    from opencood.visualization import simple_vis
    
    vis_save_path_root = os.path.join('debug_v2xsim_bg_green')
    if not os.path.exists(vis_save_path_root):
        os.makedirs(vis_save_path_root)
    gt_range = [-32, -32, -3, 32, 32, 2]

    nei_lidar_original_list = cav_content['nei_lidar_original_list'][0]
    nei_lidar_downsample_list = cav_content['nei_lidar_downsample_list'][0]
    # nei_lidar_box_list = cav_content['nei_lidar_box_list'][0]
    # nei_lidar_rec_list = cav_content['nei_lidar_rec_list'][0]

    
    for i in range(len(nei_lidar_original_list)):

        # lidar_original_dict = ({i+1: nei_lidar_original_list[i].cpu()})
        lidar_downsample_dict =({i+1: nei_lidar_downsample_list[i].cpu()})
        # lidar_box_dict =({"pred_box_tensor": nei_lidar_box_list[i].cpu()})
        # lidar_rec_dict = ({i+1: nei_lidar_rec_list[i].cpu()})
        

        # vis_save_path = os.path.join(vis_save_path_root, 'ori_%05d.png' % i)
    
        # simple_vis.visualize_colorful(lidar_box_dict,
        #                                 lidar_original_dict,
        #                                 gt_range,
        #                                 vis_save_path,
        #                                 method='bev',
        #                                 left_hand=False)
        vis_save_path = os.path.join(vis_save_path_root, 'sample_%05d.png' % i)

        simple_vis.visualize_colorful({},
                                        lidar_downsample_dict,
                                        gt_range,
                                        vis_save_path,
                                        method='bev',
                                        left_hand=False)
        # vis_save_path = os.path.join(vis_save_path_root, 'rec_%05d.png' % i)

        # simple_vis.visualize_colorful({},
        #                                 lidar_rec_dict,
        #                                 gt_range,
        #                                 vis_save_path,
        #                                 method='bev',
        #                                 left_hand=False)
        
    pause()
    """

    t1 = time.time()

    output_dict['ego'] = model(cav_content)
    
    # pred_box_tensor [8, 8, 3], pred_score [8]
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)
    
    t2 = time.time()                

    projected_lidar_dict = {}
    original_lidar = cav_content['original_lidar'][0]
    for i in range(len(original_lidar)):
        if i ==0:
            projected_lidar_dict.update({'ego' : original_lidar[0]})
        else:    
            projected_lidar_dict.update({i: original_lidar[i]})
    
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor, \
                    "origin_lidar_dict" : projected_lidar_dict,
                    'pillar_feature': output_dict['ego']['pillar_feature']}
    
    return_dict.update({
        'sdlc_time': t2 - t1,
        'faps_time': cav_content['faps_time'],
        'icp_time': cav_content['icp_time'],
        'enc_time': output_dict['ego']['enc_time'],
        'vq_time': output_dict['ego']['vq_time'],
        'dec_time': output_dict['ego']['dec_time'],
        'fusion_time': output_dict['ego']['fusion_time'],
        'completion_time': output_dict['ego']['completion_time']
    })

    # if model.vqvae:
    #     nei_lidar_original_list = cav_content['nei_lidar_original_list'][0]
    #     nei_lidar_downsample_list = cav_content['nei_lidar_downsample_list'][0]
    #     nei_lidar_rec_list = cav_content['nei_lidar_rec_list'][0]

    #     if len(nei_lidar_original_list) > 0:
    #         for i in range(len(nei_lidar_original_list)):
    #             lidar_original_dict = ({i: nei_lidar_original_list[i].cpu()})
    #             lidar_downsample_dict =({i: nei_lidar_downsample_list[i].cpu()})
    #             lidar_rec_dict = ({i: nei_lidar_rec_list[i].cpu()})
            
    #         return_dict.update({
    #             'lidar_original_dict': lidar_original_dict,
    #             'lidar_downsample_dict': lidar_downsample_dict,
    #             'lidar_rec_dict': lidar_rec_dict
    #         })

    return return_dict


def inference_cecooper_fusion_(batch_data, model, nei_model, dataset, device):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    # 近邻端筛选
    batch_data = nei_model.neighbor_points_selection(batch_data, dataset, device)
        
    # Ego端融合近邻端传输的点云，并进行感知 
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    projected_lidar_dict = {}
    original_lidar = cav_content['original_lidar'][0]
    for i in range(len(original_lidar)):
        if i ==0:
            projected_lidar_dict.update({'ego' : original_lidar[0]})
        else:    
            projected_lidar_dict.update({i: original_lidar[i]})
    
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor, \
                    "origin_lidar_dict" : projected_lidar_dict}
    

    if nei_model.vqvae:
        nei_lidar_original_list = cav_content['nei_lidar_original_list'][0]
        nei_lidar_downsample_list = cav_content['nei_lidar_downsample_list'][0]
        nei_lidar_rec_list = cav_content['nei_lidar_rec_list'][0]

        if len(nei_lidar_original_list) > 0:
            for i in range(len(nei_lidar_original_list)):
                lidar_original_dict = ({i: nei_lidar_original_list[i].cpu()})
                lidar_downsample_dict =({i: nei_lidar_downsample_list[i].cpu()})
                lidar_rec_dict = ({i: nei_lidar_rec_list[i].cpu()})
            
            return_dict.update({
                'lidar_original_dict': lidar_original_dict,
                'lidar_downsample_dict': lidar_downsample_dict,
                'lidar_rec_dict': lidar_rec_dict
            })

    return return_dict

def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return_dict = inference_early_fusion(batch_data, model, dataset)
    return return_dict


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)


def depth_metric(depth_items, grid_conf):
    # depth logdit: [N, D, H, W]
    # depth gt indices: [N, H, W]
    depth_logit, depth_gt_indices = depth_items
    depth_pred_indices = torch.argmax(depth_logit, 1)
    depth_pred = indices_to_depth(depth_pred_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    depth_gt = indices_to_depth(depth_gt_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    rmse = mean_squared_error(depth_gt.cpu(), depth_pred.cpu(), squared=False)
    return rmse


def fix_cavs_box(pred_box_tensor, gt_box_tensor, pred_score, batch_data):
    """
    Fix the missing pred_box and gt_box for ego and cav(s).
    Args:
        pred_box_tensor : tensor
            shape (N1, 8, 3), may or may not include ego agent prediction, but it should include
        gt_box_tensor : tensor
            shape (N2, 8, 3), not include ego agent in camera cases, but it should include
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    Returns:
        pred_box_tensor : tensor
            shape (N1+?, 8, 3)
        gt_box_tensor : tensor
            shape (N2+1, 8, 3)
    """
    if pred_box_tensor is None or gt_box_tensor is None:
        return pred_box_tensor, gt_box_tensor, pred_score, 0
    # prepare cav's boxes

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()
        
    extent = [2.45, 1.06, 0.75]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_tensor = torch.tensor(np.concatenate(box_list, axis=0), device=pred_box_tensor.device)
    
    pred_box_tensor_ = torch.cat((cav_box_tensor, pred_box_tensor), dim=0)
    gt_box_tensor_ = torch.cat((cav_box_tensor, gt_box_tensor), dim=0)

    pred_score_ = torch.cat((torch.ones(N, device=pred_score.device), pred_score))

    gt_score_ = torch.ones(gt_box_tensor_.shape[0], device=pred_box_tensor.device)
    gt_score_[N:] = 0.5

    keep_index = nms_rotated(pred_box_tensor_,
                            pred_score_,
                            0.01)
    pred_box_tensor = pred_box_tensor_[keep_index]
    pred_score = pred_score_[keep_index]

    keep_index = nms_rotated(gt_box_tensor_,
                            gt_score_,
                            0.01)
    gt_box_tensor = gt_box_tensor_[keep_index]

    return pred_box_tensor, gt_box_tensor, pred_score, N


def get_cav_box(batch_data):
    """
    Args:
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    """

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
        lidar_agent_record = batch_data['ego']['lidar_agent_record'].cpu().numpy()

    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        lidar_agent_record = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
            lidar_agent_record.append(1 if 'processed_lidar' in cav_data else 0)
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()

        

    extent = [0.2, 0.2, 0.2]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_np = np.concatenate(box_list, axis=0)


    return cav_box_np, lidar_agent_record