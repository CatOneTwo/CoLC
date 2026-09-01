# 用于voxel-level的分类


# -*- coding: utf-8 -*-
# Author: Yifan Lu
# Add direction classification loss
# The originally point_pillar_loss.py, can not determine if the box heading is opposite to the GT.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic

from pdb import set_trace as pause

class PointSegLoss(nn.Module):
    def __init__(self, args):
        super(PointSegLoss, self).__init__()
        self.seg_weight  = args['seg']
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.seg_weight))
        self.dice_weight = args['dice']
        
        self.loss_dict = {}

    def dice_loss(self, preds, targets, eps=1e-6):
        # preds: [B, N, 1] logits → sigmoid
        preds = torch.sigmoid(preds)
        
        # flatten
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + eps) / (preds.sum() + targets.sum() + eps)
        return 1.0 - dice  # dice loss
    
    # def forward(self, seg_preds, seg_labels, validate=False):
    #     """
    #     Parameters
    #     ----------
    #     seg_preds : B,N,1
    #     seg_labels : B,N
    #     """
    #     seg_labels = seg_labels.float()  # ensure float
        
    #     bce_loss = self.bce(seg_preds.squeeze(-1), seg_labels)

    #     if self.dice_weight > 0:
    #         dice_loss = self.dice_loss(seg_preds, seg_labels)
    #         loss = bce_loss + self.dice_weight * dice_loss
    #     else:
    #         loss = bce_loss

    #     self.loss_dict.update({'seg_loss': loss.item()})

    #     return loss


    def forward(self, seg_preds, seg_labels, validate=False):
        if isinstance(seg_preds, torch.Tensor):
            return self.forward_tensor(seg_preds, seg_labels)
        elif isinstance(seg_preds, list):
            return self.forward_list(seg_preds, seg_labels)
        else:
            raise TypeError(f"Unsupported input type: {type(seg_preds)}")


    def forward_tensor(self, seg_preds, seg_labels):
        """
        Parameters
        ----------
        seg_preds : B,N,1
        seg_labels : B,N
        """
        seg_labels = seg_labels.float()  # ensure float
        
        bce_loss = self.bce(seg_preds.squeeze(-1), seg_labels)

        if self.dice_weight > 0:
            dice_loss = self.dice_loss(seg_preds, seg_labels)
            loss = bce_loss + self.dice_weight * dice_loss
        else:
            loss = bce_loss

        self.loss_dict.update({'seg_loss': loss.item()})

        return loss
    
    def forward_list(self, seg_preds_list, seg_labels_list):
        # seg_preds_list: list of [N_i, 1], seg_labels_list: list of [N_i]
        total_loss = 0.0
        for pred, label in zip(seg_preds_list, seg_labels_list):
            label = label.float().to(pred.device)
            bce_loss = self.bce(pred.squeeze(-1), label)
            dice_loss = self.dice_loss(pred, label)
            total_loss += bce_loss + self.dice_weight * dice_loss
        total_loss = total_loss / len(seg_preds_list)
        self.loss_dict.update({'seg_loss': total_loss.item()})
        return total_loss


    def logging(self, epoch, batch_id, batch_len, writer = None, suffix="", pbar=None):
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

        seg_loss = self.loss_dict.get('seg_loss', 0)

        
        msg = "[epoch %d][%d/%d]%s || Seg Loss: %.4f" % (epoch, batch_id + 1, batch_len, suffix, seg_loss)
        
        if pbar is None:
            print(msg)
        else:
            pbar.set_description(msg)

        if not writer is None:
            writer.add_scalar('Seg_loss'+suffix, seg_loss, epoch*batch_len + batch_id)

            

