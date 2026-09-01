# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
3D Anchor Generator for Voxel classification in SECOND
"""
import math
import sys
from scipy.spatial import Delaunay

import numpy as np
import torch
from torch.nn.functional import sigmoid
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils
from opencood.utils.common_utils import limit_period

from pdb import set_trace as pause

class ClsVoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(ClsVoxelPostprocessor, self).__init__(anchor_params, train)
        
        self.anchor_num = self.params['anchor_args']['num']

    def generate_anchor_box(self):
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w']
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        vh = self.params['anchor_args']['vh'] # voxel_size
        vw = self.params['anchor_args']['vw']

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride']
        else:
            feature_stride = 2


        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride) # vw is not precise, vw * feature_stride / 2 should be better?
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)


        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num) # center
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        if self.params['order'] == 'hwl': # pointpillar
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1) # (50, 176, 2, 7)

        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        return anchors

    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        # (max_num, 7)
        gt_box_center = kwargs['gt_box_center']
        # (max_num)
        masks = kwargs['mask']

        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']
        D = self.params['anchor_args']['D']
        grid_size = [W,H,D]
        grid_size = np.round(grid_size).astype(np.int64)


        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]
        zrange = [self.params['anchor_args']['cav_lidar_range'][2],
                  self.params['anchor_args']['cav_lidar_range'][5]]

        vh = self.params['anchor_args']['vh'] # voxel_size
        vw = self.params['anchor_args']['vw']
        vd = self.params['anchor_args']['vd']

        x_centers = np.linspace(xrange[0] + vw/2, xrange[1] - vw/2, W)
        y_centers = np.linspace(yrange[0] + vh/2, yrange[1] - vh/2, H)
        z_centers = np.linspace(zrange[0] + vd/2, zrange[1] - vd/2, D)

        xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
        voxel_centers = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)  # (D_feat*H_feat*W_feat, 3)

        # 初始化标签
        voxel_labels = np.zeros((voxel_centers.shape[0],), dtype=np.float32)

        # (n, 7)
        gt_box_center_valid = gt_box_center[masks == 1]
        # (n, 8, 3)
        gt_box_corner_valid = box_utils.boxes_to_corners_3d(gt_box_center_valid, self.params['order'])

        # 遍历每个gt box
        for corners in gt_box_corner_valid:
            hull = Delaunay(corners)  # convex hull
            inside = hull.find_simplex(voxel_centers) >= 0  # 点在凸包内
            voxel_labels[inside] = 1.0

        voxel_labels = voxel_labels.reshape(W, H, D, 1).transpose(2, 1, 0, 3)  # (D,H,W,1)

        # print(voxel_labels.shape, voxel_labels.sum()) # (40, 640, 640, 1) 223641.0

        return {"cls_label": voxel_labels}


    def generate_label_(self, **kwargs):
        gt_box_center = kwargs['gt_box_center']
        masks = kwargs['mask']

        W = int(round(self.params['anchor_args']['W']))
        H = int(round(self.params['anchor_args']['H']))
        D = int(round(self.params['anchor_args']['D']))

        cav_range = self.params['anchor_args']['cav_lidar_range']
        vw, vh, vd = self.params['anchor_args']['vw'], self.params['anchor_args']['vh'], self.params['anchor_args']['vd']

        x_centers = np.linspace(cav_range[0] + vw / 2, cav_range[3] - vw / 2, W)
        y_centers = np.linspace(cav_range[1] + vh / 2, cav_range[4] - vh / 2, H)
        z_centers = np.linspace(cav_range[2] + vd / 2, cav_range[5] - vd / 2, D)
        xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
        voxel_centers = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)  # (W*H*D, 3)

        voxel_labels = np.zeros((voxel_centers.shape[0],), dtype=np.float32)

        gt_box_center_valid = gt_box_center[masks == 1]

        # 提取 voxel 坐标
        vx, vy, vz = voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2]

        # box: (x, y, z, dx, dy, dz, heading)
        for box in gt_box_center_valid:
            cx, cy, cz, dx, dy, dz, yaw = box
            cosa, sina = np.cos(-yaw), np.sin(-yaw)

            # 平移到 box 坐标系
            x = vx - cx
            y = vy - cy
            z = vz - cz

            # 旋转回对齐坐标
            x_rot = x * cosa - y * sina
            y_rot = x * sina + y * cosa

            inside = (
                (np.abs(x_rot) <= dx / 2) &
                (np.abs(y_rot) <= dy / 2) &
                (np.abs(z) <= dz / 2)
            )
            voxel_labels[inside] = 1.0

        voxel_labels = voxel_labels.reshape(W, H, D, 1).transpose(2, 1, 0, 3)  # (D,H,W,1)
        return {"cls_label": voxel_labels}

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        cls_label = []

        for i in range(len(label_batch_list)):
            cls_label.append(label_batch_list[i]['cls_label'])

        cls_label = torch.from_numpy(np.array(cls_label))

        return {'cls_label': cls_label}


    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix'] # no clean

            # rename variable 
            if 'psm' in output_dict[cav_id]:
                output_dict[cav_id]['cls_preds'] = output_dict[cav_id]['psm']
            if 'rm' in output_dict:
                output_dict[cav_id]['reg_preds'] = output_dict[cav_id]['rm'] 
            if 'dm' in output_dict:
                output_dict[cav_id]['dir_preds'] = output_dict[cav_id]['dm']

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box'] 

            # classification probability
            prob = output_dict[cav_id]['cls_preds'] # [1,2,h,w] 测试时b=1
            prob = F.sigmoid(prob.permute(0, 2, 3, 1)) # [1,h,w,2]

            prob = prob.reshape(1, -1) # [b,h*w*2]

            # regression map
            reg = output_dict[cav_id]['reg_preds'] # [b, h, w, 14] [1,14,h,w]

            # convert regression map back to bounding box
            if len(reg.shape) == 4: # anchor-based. PointPillars, SECOND
                batch_box3d = self.delta_to_boxes3d(reg, anchor_box) # [1,h*w*2,7]从offset转换到真实的坐标
            else: # anchor-free. CenterPoint
                batch_box3d = reg.view(1, -1, 7)

            mask = torch.gt(prob, self.params['target_args']['score_threshold']) # 大于分数阈值的框 [1,h*w*2], True or False

            mask = mask.view(1, -1) # [b,h*w*2]
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7) # [1,h*w*2,7]

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7) # [m,7] m为分数阈值筛选的框数量
            scores = torch.masked_select(prob[0], mask[0]) # [m]

            # adding dir classifier
            if 'dir_preds' in output_dict[cav_id].keys() and len(boxes3d) !=0:
                dir_offset = self.params['dir_args']['dir_offset'] # 0.7853
                num_bins = self.params['dir_args']['num_bins'] # 2


                dm  = output_dict[cav_id]['dir_preds'] # [N, H, W, 4] [b,4,h,w]
                dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2] [1,h*w*2,2]
                dir_cls_preds = dir_cls_preds[mask] # [m,2]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0 [m]
                
                period = (2 * np.pi / num_bins) # pi 3.141592653589793
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间 [m]
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi [m,7]
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi] [m,7]

            if 'iou_preds' in output_dict[cav_id].keys() and len(boxes3d) != 0:
                iou = torch.sigmoid(output_dict[cav_id]['iou_preds'].permute(0, 2, 3, 1).contiguous()).reshape(1, -1)
                iou = torch.clamp(iou, min=0.0, max=1.0)
                iou = (iou + 1) * 0.5
                scores = scores * torch.pow(iou.masked_select(mask), 4)

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3) 转换为8个角点表示的形式
                boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=self.params['order']) # [m, 8, 3]
                
                # STEP 2
                # (N, 8, 3) 投影到ego的坐标系，中期协同的结果不需要转换
                projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix) # [m, 8, 3]
                
                # convert 3d bbx to 2d, (N,4) 投影到BEV
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d) # [m,4]
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1) # [m,5]

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list) # [m,5]
        # scores
        scores = pred_box2d_list[:, -1] # [m]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list) # # [m, 8, 3]
        # remove large bbx and negative z 
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor) # [m] True or False
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor) # [m] True or False
        keep_index = torch.logical_and(keep_index_1, keep_index_2) # [m] True or False
        
        pred_box3d_tensor = pred_box3d_tensor[keep_index] # [m2, 8, 3] m2为keep的预测框数量
        scores = scores[keep_index] # [m2]

        # STEP3
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           ) # shape为（m3）只返回nms后保留的索引

        pred_box3d_tensor = pred_box3d_tensor[keep_index] # [m3, 8, 3]

        # select cooresponding score
        scores = scores[keep_index] # [m3]
        
        # filter out the prediction out of the range. with z-dim
        pred_box3d_np = pred_box3d_tensor.cpu().numpy() # 转换成numpy (m3,8,3)
        pred_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(pred_box3d_np,
                                                    self.params['gt_range'],
                                                    order=None,
                                                    return_mask=True) # (m4,8,3) (m3,)mask为True or False
        pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box3d_tensor.device) # [m4,8,3]
        scores = scores[mask] # [m4]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    def post_process_hys(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []

        # box_num = []
        
        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix'] # no clean

            # rename variable 
            if 'psm' in output_dict[cav_id]:
                output_dict[cav_id]['cls_preds'] = output_dict[cav_id]['psm']
            if 'rm' in output_dict:
                output_dict[cav_id]['reg_preds'] = output_dict[cav_id]['rm']
            if 'dm' in output_dict:
                output_dict[cav_id]['dir_preds'] = output_dict[cav_id]['dm']

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # classification probability
            prob = output_dict[cav_id]['cls_preds']
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))

            # apply confidence-aware late fusion # cosdh (2025 CVPR)
            # if confidence_beta is not None and confidence_threshold is not None and cav_id != 'ego':
            #     prob[prob < confidence_threshold] = 0.0
            #     prob *= confidence_beta
                
            prob = prob.reshape(1, -1)


            # regression map
            reg = output_dict[cav_id]['reg_preds']

            # convert regression map back to bounding box
            if len(reg.shape) == 4: # anchor-based. PointPillars, SECOND
                batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
            else: # anchor-free. CenterPoint
                batch_box3d = reg.view(1, -1, 7)

            mask = \
                torch.gt(prob, self.params['target_args']['score_threshold'])
            
            # hys add
            if 'score_threshold' in output_dict[cav_id]: 
                score_threshold = output_dict[cav_id]['score_threshold']
                mask = torch.gt(prob, score_threshold) # 大于分数阈值的框 [1,h*w*2]
                        
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])
            mask_index = mask[0].nonzero()[:,0] # [m]

            # adding dir classifier
            if 'dir_preds' in output_dict[cav_id].keys() and len(boxes3d) !=0:
                dir_offset = self.params['dir_args']['dir_offset']
                num_bins = self.params['dir_args']['num_bins']


                dm  = output_dict[cav_id]['dir_preds'] # [N, H, W, 4]
                dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
                
                period = (2 * np.pi / num_bins) # pi
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]

            if 'iou_preds' in output_dict[cav_id].keys() and len(boxes3d) != 0:
                iou = torch.sigmoid(output_dict[cav_id]['iou_preds'].permute(0, 2, 3, 1).contiguous()).reshape(1, -1)
                iou = torch.clamp(iou, min=0.0, max=1.0)
                iou = (iou + 1) * 0.5
                scores = scores * torch.pow(iou.masked_select(mask), 4)

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                
                # STEP 2
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)
                
                # if cav_id!='ego':
                #     box_num.append(len(projected_boxes3d))
            

        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            if 'pred_index' in output_dict[cav_id] and output_dict[cav_id]['pred_index']:
                return None, None, None
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        boxes3d=boxes3d[keep_index] # [m2, 7]
        mask_index=mask_index[keep_index] # [m2]

        # STEP3
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        boxes3d=boxes3d[keep_index] # [m3, 7]
        mask_index=mask_index[keep_index] # [m3]

        # select cooresponding score
        scores = scores[keep_index]
        
        # filter out the prediction out of the range. with z-dim
        # pred_box3d_np = pred_box3d_tensor.cpu().numpy()
        if pred_box3d_tensor.requires_grad:
            pred_box3d_np = pred_box3d_tensor.cpu().detach().numpy()
        else:
            pred_box3d_np = pred_box3d_tensor.cpu().numpy()

        pred_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(pred_box3d_np,
                                                    self.params['gt_range'],
                                                    order=None,
                                                    return_mask=True)
        pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box3d_tensor.device)
        scores = scores[mask]
        boxes3d=boxes3d[mask] # [m4, 7]
        mask_index=mask_index[mask] #[m4]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        if 'pred_center' in output_dict[cav_id] and output_dict[cav_id]['pred_center']:
            if 'pred_index' in output_dict[cav_id] and output_dict[cav_id]['pred_index']:
                return boxes3d, scores, mask_index
            return boxes3d, scores

        return pred_box3d_tensor, scores
        # return pred_box3d_tensor, scores, box_num

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, 14, H, W)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    show_vis,
                                                    save_path)
