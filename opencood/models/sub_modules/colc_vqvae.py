# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia

from opencood.models.sub_modules.colc_vqvae_layers import VectorQuantizer, VQEncoder, VQDecoder

from opencood.loss.point_pillar_loss import weighted_smooth_l1_loss

import time

from pdb import set_trace as pause

class Vqvae(nn.Module):
    def __init__(self, args):
        super(Vqvae, self).__init__()

        self.vector_quantizer = VectorQuantizer(args['vector_quantizer'])
        self.pre_quant = nn.Sequential(
            nn.Linear(self.vector_quantizer.e_dim, self.vector_quantizer.e_dim), 
            nn.LayerNorm(self.vector_quantizer.e_dim))
        
        self.lidar_encoder = VQEncoder(args['lidar_encoder'])
        self.lidar_decoder = VQDecoder(args['lidar_decoder'])

        self.register_buffer("code_age", torch.zeros(self.vector_quantizer.n_e) * 10000)
        self.register_buffer("code_usage", torch.zeros(self.vector_quantizer.n_e))
        
        self.rec_mode = args['rec_mode'] if 'rec_mode' in args else 'd2d'

        if self.rec_mode =='d2d':
            self.aug = nn.Sequential(
                kornia.augmentation.RandomVerticalFlip(),
                kornia.augmentation.RandomHorizontalFlip(),
            )

        else:
            self.aug = kornia.augmentation.AugmentationSequential(
                kornia.augmentation.RandomVerticalFlip(),
                kornia.augmentation.RandomHorizontalFlip(),
                data_keys=["input"]
            )
        
    
    def forward(self, data_dict):

        if self.rec_mode == 's2d':
            losses = self.forward_s2d(data_dict)
        elif self.rec_mode == 'd2d':
            losses = self.forward_d2d(data_dict)
        return losses

    def forward_s2d(self, data_dict):

        voxels = data_dict['spatial_features'] # 1, 64, 200, 504
        voxels_sparse = data_dict['spatial_features_sparse'] # 1, 64, 200, 504

        if voxels.size()[0] != voxels_sparse.size()[0]:
            print(voxels.size(), voxels_sparse.size())

        if self.training:
            # 先统一采样一次增强参数
            params = self.aug.forward_parameters(voxels.shape)
            # 对两个voxel应用相同的参数
            voxels = self.aug(voxels, params=params)
            voxels_sparse = self.aug(voxels_sparse, params=params)

        lidar_feats = self.lidar_encoder(voxels_sparse)

        feats = self.pre_quant(lidar_feats) # [1, 6300, 512]
        lidar_quant, emb_loss, _ = self.vector_quantizer(feats, self.code_age, self.code_usage)

        # lidar_rec = self.lidar_decoder(lidar_quant)
        occ, lidar_rec = self.lidar_decoder(lidar_quant)

        voxels_gt = (voxels.abs().sum(dim=1, keepdim=True) > 0).float() # 占用gt

        # lidar_rec_loss = (F.binary_cross_entropy_with_logits(lidar_rec, voxels, reduction="none") * 100).mean()
        lidar_occ_loss = (F.binary_cross_entropy_with_logits(occ, voxels_gt, reduction="none") * 100).mean()
        
        mask = (voxels_gt > 0).float()   # [B,1,H,W]
        mask = mask.expand_as(voxels)    # [B,64,H,W]
        lidar_rec_loss = F.mse_loss(lidar_rec * mask, voxels * mask)*100

        lidar_rec_prob = occ.sigmoid().detach()
        lidar_rec_diff = (lidar_rec_prob - voxels_gt).abs().sum() / voxels_gt.shape[0]
        lidar_rec_iou = ((lidar_rec_prob >= 0.5) & (voxels_gt >= 0.5)).sum() / (
            (lidar_rec_prob >= 0.5) | (voxels_gt >= 0.5)
        ).sum()

        lidar_rec_fiou = ((lidar_rec_prob >= 0.5) & (voxels_gt < 0.5)).sum() / (
            (lidar_rec_prob >= 0.5) | (voxels_gt < 0.5)
        ).sum()

        code_util = (self.code_age < self.vector_quantizer.dead_limit).sum() / self.code_age.numel()
        code_uniformity = self.code_usage.topk(10)[0].sum() / self.code_usage.sum()

        losses = dict()

        losses.update(
            {
                "loss_lidar_occ": lidar_occ_loss,
                "loss_lidar_rec": lidar_rec_loss,
                "loss_emb": sum(emb_loss)*10,
                "lidar_rec_diff": lidar_rec_diff,
                "lidar_rec_iou": lidar_rec_iou,
                "lidar_rec_fiou": lidar_rec_fiou,
                "code_util": code_util,
                "code_uniformity": code_uniformity,
            }
        )

        occ_mask = (lidar_rec_prob > 0.5).float() 
        occ_mask = occ_mask.expand_as(lidar_rec)
        rec_feat = lidar_rec*occ_mask

        losses.update(
            {   
                "occ_mask": occ_mask,
                "voxels": voxels,
                "generated_voxel": rec_feat,
                "voxels_sparse": voxels_sparse,
            })

        return losses
    
    def forward_d2d(self, data_dict):
        
        voxels = data_dict['spatial_features']

        if self.training:
            voxels = self.aug(voxels)

        lidar_feats = self.lidar_encoder(voxels)

        feats = self.pre_quant(lidar_feats)
        lidar_quant, emb_loss, _ = self.vector_quantizer(feats, self.code_age, self.code_usage)

        occ, lidar_rec = self.lidar_decoder(lidar_quant)

        voxels_gt = (voxels.abs().sum(dim=1, keepdim=True) > 0).float() # 占用gt

        lidar_occ_loss = (F.binary_cross_entropy_with_logits(occ, voxels_gt, reduction="none") * 100).mean()
        
        mask = (voxels_gt > 0).float()   # [B,1,H,W]
        mask = mask.expand_as(voxels)    # [B,64,H,W]
        lidar_rec_loss = F.mse_loss(lidar_rec * mask, voxels * mask)*100

        lidar_rec_prob = occ.sigmoid().detach()
        lidar_rec_diff = (lidar_rec_prob - voxels_gt).abs().sum() / voxels_gt.shape[0]
        lidar_rec_iou = ((lidar_rec_prob >= 0.5) & (voxels_gt >= 0.5)).sum() / (
            (lidar_rec_prob >= 0.5) | (voxels_gt >= 0.5)
        ).sum()

        code_util = (self.code_age < self.vector_quantizer.dead_limit).sum() / self.code_age.numel()
        code_uniformity = self.code_usage.topk(10)[0].sum() / self.code_usage.sum()

        losses = dict()

        losses.update(
            {
                "loss_lidar_occ": lidar_occ_loss,
                "loss_lidar_rec": lidar_rec_loss,
                "loss_emb": sum(emb_loss)*10,
                "lidar_rec_diff": lidar_rec_diff,
                "lidar_rec_iou": lidar_rec_iou,
                "code_util": code_util,
                "code_uniformity": code_uniformity,
            }
        )

        occ_mask = (lidar_rec_prob > 0.5).float() 
        occ_mask = occ_mask.expand_as(lidar_rec)
        rec_feat = lidar_rec*occ_mask

        losses.update(
            {   
                "occ_mask": occ_mask,
                "voxels": voxels,
                "generated_voxel": rec_feat
            })

        return losses


    def s2d_completion(self, data_dict):
        
        voxels_sparse = data_dict['spatial_features_sparse']

        t1 = time.time()

        lidar_feats = self.lidar_encoder(voxels_sparse)

        feats = self.pre_quant(lidar_feats)

        t2 = time.time()

        lidar_quant, emb_loss, _ = self.vector_quantizer(feats, self.code_age, self.code_usage)

        t3 = time.time()

        occ, lidar_rec = self.lidar_decoder(lidar_quant)

        t4 = time.time()

        occ_prob = occ.sigmoid()
        occ_mask = (occ_prob > 0.5).float() # 0.5
        occ_mask = occ_mask.expand_as(lidar_rec)
        rec_feat = lidar_rec*occ_mask

        mask = (voxels_sparse.abs().sum(dim=1, keepdim=True) > 0) # [B, 1, H, W]
        rec_feat = torch.where(mask.bool(), voxels_sparse, rec_feat)
        
        output_dict = dict()

        output_dict.update({
            "generated_voxel": rec_feat,
            "occ_mask": occ_prob
        })

        output_dict.update({
            "enc_time": (t2-t1),
            "quant_time": (t3-t2),
            "dec_time": (t4-t3)
        })


        return output_dict
        





