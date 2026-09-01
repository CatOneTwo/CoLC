# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn> Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn

from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatterV2
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv

from pdb import set_trace as pause

class PointPillarV2(nn.Module):
    def __init__(self, args):
        super(PointPillarV2, self).__init__()

      
        self.scatter = PointPillarScatterV2(args['point_pillar_scatter'])

        self.mid_layer = nn.Conv2d(40, args['pillar_vfe']['num_filters'][0], kernel_size=1, stride=1, padding=0, bias=True)
        
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], # 384
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'], # 384
                                  kernel_size=1)
        
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2， # 384
        else:
            self.use_dir = False
        

    def forward(self, data_dict):
                
        if 'spatial_features' not in data_dict:
            voxel_coords = data_dict['processed_lidar']['voxel_coords'] 
            batch_dict = {'voxel_coords': voxel_coords}
            batch_dict = self.scatter(batch_dict) # batch_dict['spatial_features'] [B, 40, H, W]

        spatial_features = batch_dict['spatial_features']
        spatial_features = self.mid_layer(spatial_features) # [B, 64, H, W]
        batch_dict['spatial_features'] = spatial_features

        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'feature': spatial_features_2d,
                        'cls_preds': psm,
                       'reg_preds': rm}
                       
        if self.use_dir:
            dm = self.dir_head(spatial_features_2d)
            output_dict.update({'dir_preds': dm})

        return output_dict


