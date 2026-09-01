# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.loss.point_pillar_loss import PointPillarLoss

from pdb import set_trace as pause

class PointPillarDiscoNetLoss(PointPillarLoss):
    def __init__(self, args):
        super(PointPillarDiscoNetLoss, self).__init__(args)
        self.kd = args['kd']

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        total_loss = super().forward(output_dict, target_dict)

        ########## KL loss ############
        rm = output_dict['reg_preds']  # [B, 14, 50, 176]
        psm = output_dict['cls_preds'] # [B, 2, 50, 176]
        feature = output_dict['feature']

        teacher_rm = output_dict['teacher_reg_preds']
        teacher_psm = output_dict['teacher_cls_preds']
        
        feature = output_dict['feature']
        teacher_feature = output_dict['teacher_feature']
        kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)

        kd_loss = 0.

        # if self.kd.get('feature_kd', False):
        #     N, C, H, W = teacher_feature.shape
        #     teacher_feature = teacher_feature.permute(0,2,3,1).reshape(N*H*W, C)
        #     student_feature = feature.permute(0,2,3,1).reshape(N*H*W, C)
        #     kd_loss_feature = kl_loss_mean(
        #             F.log_softmax(student_feature, dim=1), F.softmax(teacher_feature, dim=1)
        #         )
        
        #     kd_loss += kd_loss_feature

        if self.kd.get('pillar_kd', False):  
            pillar_feature = output_dict['pillar_feature']
            teacher_pillar_feature = output_dict['teacher_pillar_feature']

            N, C, H, W = teacher_pillar_feature.shape
            teacher_pillar_feature = teacher_pillar_feature.permute(0,2,3,1).reshape(N*H*W, C)
            student_pillar_feature = pillar_feature.permute(0,2,3,1).reshape(N*H*W, C)
            kd_loss_pillar_feature = kl_loss_mean(
                    F.log_softmax(student_pillar_feature, dim=1), F.softmax(teacher_pillar_feature, dim=1)
            )
            kd_loss += kd_loss_pillar_feature

            kd_loss *= self.kd['weight']
            total_loss += kd_loss

            self.loss_dict.update({'kd_loss': kd_loss.item()})
        

        if self.kd.get('pillar_kd_wt', False):  
            # 使用带温度系数的KL散度
            temperature = 2.0  # 可调参数，建议2.0-8.0
            pillar_feature = output_dict['pillar_feature']
            teacher_pillar_feature = output_dict['teacher_pillar_feature']

            N, C, H, W = teacher_pillar_feature.shape
            teacher_pillar_feature = teacher_pillar_feature.permute(0,2,3,1).reshape(N*H*W, C)
            student_pillar_feature = pillar_feature.permute(0,2,3,1).reshape(N*H*W, C)

            teacher_probs = F.softmax(teacher_pillar_feature / temperature, dim=1)
            student_log_probs = F.log_softmax(student_pillar_feature / temperature, dim=1)

            kd_loss_pillar_feature = kl_loss_mean(student_log_probs, teacher_probs)
            kd_loss_pillar_feature = kd_loss_pillar_feature * (temperature ** 2)  # 重要：尺度补偿

            kd_loss += kd_loss_pillar_feature

            kd_loss *= self.kd['weight']
            total_loss += kd_loss

            self.loss_dict.update({'kd_loss': kd_loss.item()})

            
        

        # if self.kd.get('decoder_kd', False):
        #     N, C, H, W = teacher_rm.shape
        #     teacher_rm = teacher_rm.permute(0,2,3,1).reshape(N*H*W, C)
        #     student_rm = rm.permute(0,2,3,1).reshape(N*H*W, C)
        #     kd_loss_rm = kl_loss_mean(
        #             F.log_softmax(student_rm, dim=1), F.softmax(teacher_rm, dim=1)
        #         )

        #     N, C, H, W = teacher_psm.shape
        #     teacher_psm = teacher_psm.permute(0,2,3,1).reshape(N*H*W, C)
        #     student_psm = psm.permute(0,2,3,1).reshape(N*H*W, C)
        #     kd_loss_psm = kl_loss_mean(
        #             F.log_softmax(student_psm, dim=1), F.softmax(teacher_psm, dim=1)
        #         )
            
            

        #     kd_loss += kd_loss_rm + kd_loss_psm
            

        if self.kd.get("cos_kd", False):
            pillar_feature = output_dict["pillar_feature"]  # (B,C,H,W)
            teacher_pillar_feature = output_dict["teacher_pillar_feature"]  # (B,C,H,W)

            pillar_mask = (pillar_feature.abs().sum(dim=1, keepdim=True) > 0).float()  

            # 通道归一化
            student = F.normalize(pillar_feature, p=2, dim=1)
            teacher = F.normalize(teacher_pillar_feature, p=2, dim=1)

            # (B,C,H,W) → (B*H*W, C)
            B, C, H, W = teacher.shape
            student = student.permute(0,2,3,1).reshape(B*H*W, C)
            teacher = teacher.permute(0,2,3,1).reshape(B*H*W, C)
            
            # mask 展平
            mask = pillar_mask.reshape(B*H*W) > 0  # valid pillars
            student = student[mask]
            teacher = teacher[mask]

            cos_sim = (student * teacher).sum(dim=1)  # 点积 → cos方向一致性
            cos_loss = 10 * (1 - cos_sim).mean()

            self.loss_dict.update({'cos_loss': cos_loss.item()})

            total_loss += cos_loss

        
        if self.kd.get('dist_kd', False): 
            pillar_feature = output_dict['pillar_feature']
            teacher_pillar_feature = output_dict['teacher_pillar_feature'] 

            # 3. 分布统计损失（全局特性对齐）
            pred_mean, pred_std = pillar_feature.mean(), pillar_feature.std()
            teacher_mean, teacher_std = teacher_pillar_feature.mean(), teacher_pillar_feature.std()
            dist_loss = F.smooth_l1_loss(pred_mean, teacher_mean) + F.smooth_l1_loss(pred_std, teacher_std)
            # print(kd_loss_pillar_feature.item(),cos_loss.item(),dist_loss.item())
            total_loss += 1 * dist_loss

            # total_loss += 10 * cos_loss + 1 * dist_loss

        
        self.loss_dict.update({'total_loss': total_loss.item()})


        return total_loss


    def logging(self, epoch, batch_id, batch_len, writer = None, suffix='', pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        kd_loss = self.loss_dict.get('kd_loss', 0)
        cos_loss = self.loss_dict.get('cos_loss', 0)
        
        # print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
        #       " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || KD Loss: %.4f" % (
        #           epoch, batch_id + 1, batch_len, suffix,
        #           total_loss, cls_loss, reg_loss, dir_loss, iou_loss, kd_loss))
        

        msg = "[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f || Dir Loss: %.4f || KD Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, kd_loss)

        if cos_loss > 0:
            msg = "[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f || Dir Loss: %.4f || KD Loss: %.4f || Cos Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, kd_loss, cos_loss)
        

        if pbar is None:
            print(msg)
        else:
            pbar.set_description(msg)

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss'+suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss'+suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Kd_loss'+suffix, kd_loss,
                            epoch*batch_len + batch_id)