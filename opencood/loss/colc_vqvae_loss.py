# -*- coding: utf-8 -*-
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pdb import set_trace as pause

class ColcVqvaeLoss(nn.Module):
    def __init__(self, args):
        super(ColcVqvaeLoss, self).__init__()

        self.loss_dict = {}


    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        # total_loss = super().forward(output_dict, target_dict) # 检测损失

        total_loss = 0.

        ########## VQ-VAE loss ############
            
        # (Commitment Loss)
        commitment_loss = output_dict['loss_emb']
        # (Reconstruction Loss)
        loss_lidar_rec = output_dict['loss_lidar_rec']
        # Occupancy Loss 
        loss_lidar_occ = output_dict['loss_lidar_occ']

    

        # Other metrics
        lidar_rec_diff = output_dict['lidar_rec_diff'] # Voxel reconstruction error
        lidar_rec_iou = output_dict['lidar_rec_iou'] # Voxel IoU
        code_util = output_dict['code_util'] # Codebook utilization
        code_uniformity = output_dict['code_uniformity'] # Codebook uniformity

        vq_vae_loss = commitment_loss + loss_lidar_rec + loss_lidar_occ

        if 'loss_lidar_per' in output_dict:
            loss_lidar_per = output_dict['loss_lidar_per']
            vq_vae_loss += loss_lidar_per
            self.loss_dict.update({
                'loss_lidar_per': loss_lidar_per
            })
                    
        total_loss += vq_vae_loss
        self.loss_dict.update({'vq_vae_loss': vq_vae_loss.item(),
                                'commitment_loss': commitment_loss.item(),
                                'loss_lidar_rec': loss_lidar_rec.item(),
                                'loss_lidar_occ': loss_lidar_occ.item(),
                                'lidar_rec_diff': lidar_rec_diff.item(),
                                'lidar_rec_iou': lidar_rec_iou.item(),
                                'code_util': code_util.item(),
                                'code_uniformity': code_uniformity.item()})

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

        
        vq_vae_loss = self.loss_dict.get('vq_vae_loss', 0)
        commitment_loss = self.loss_dict.get('commitment_loss', 0)
        loss_lidar_rec = self.loss_dict.get('loss_lidar_rec', 0)
        loss_lidar_occ = self.loss_dict.get('loss_lidar_occ', 0)
        
        loss_lidar_per = self.loss_dict.get('loss_lidar_per', 0)

        
        lidar_rec_diff = self.loss_dict.get('lidar_rec_diff', 0)
        lidar_rec_iou = self.loss_dict.get('lidar_rec_iou', 0)
        code_util = self.loss_dict.get('code_util', 0)
        code_uniformity = self.loss_dict.get('code_uniformity', 0)
        
        if loss_lidar_per > 0:
            msg = "[epoch %d][%d/%d]%s || VQ Loss: %.3f || comm Loss: %.3f || recon Loss: %.3f || occ Loss: %.3f || per Loss: %.3f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  vq_vae_loss, commitment_loss, loss_lidar_rec, loss_lidar_occ, loss_lidar_per)
        else:
            # codebook学习 + lidar 重建
            msg = "[epoch %d][%d/%d]%s || VQ Loss: %.3f || comm Loss: %.3f || recon Loss: %.3f || occ Loss: %.3f " % (
                    epoch, batch_id + 1, batch_len, suffix,
                    vq_vae_loss, commitment_loss, loss_lidar_rec, loss_lidar_occ)

        if pbar is None:
            print(msg)
        else:
            pbar.set_description(msg)

        if not writer is None:
            writer.add_scalar('VQ_loss'+suffix, vq_vae_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('commitment_loss'+suffix, commitment_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('loss_lidar_rec'+suffix, loss_lidar_rec,
                            epoch*batch_len + batch_id)
            writer.add_scalar('loss_lidar_occ'+suffix, loss_lidar_occ,
                            epoch*batch_len + batch_id)
            writer.add_scalar('lidar_rec_diff'+suffix, lidar_rec_diff,
                            epoch*batch_len + batch_id)
            writer.add_scalar('lidar_rec_iou'+suffix, lidar_rec_iou,
                            epoch*batch_len + batch_id)
            writer.add_scalar('code_util'+suffix, code_util,
                            epoch*batch_len + batch_id)
            writer.add_scalar('code_uniformity'+suffix, code_uniformity,
                            epoch*batch_len + batch_id)
            if loss_lidar_per > 0:
                writer.add_scalar('loss_lidar_per'+suffix, loss_lidar_per,
                            epoch*batch_len + batch_id)
