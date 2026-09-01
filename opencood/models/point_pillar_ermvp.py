import torch
import torch.nn as nn
from einops import rearrange, repeat

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.ermvp_fusion_modules import \
    ERMVPFusionEncoder
from opencood.models.fuse_modules.fuse_utils import regroup
import torch.nn.functional as F
import numpy as np
from opencood.models.sub_modules.sampler import SortSampler
from opencood.models.sub_modules.cluster import merge_tokens,cluster_dpc_knn,index_points
import math
from pdb import set_trace as pause

def get_selected_cav_feature(x, record_len,selected_cav_id_list):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    out = []
    idx = 0
    for xx in split_x:
        xx = xx[selected_cav_id_list[idx]].unsqueeze(0)
        out.append(xx)
        idx = idx + 1 
    return torch.cat(out, dim=0)

def get_ego_feature(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    out = []
    for xx in split_x:
        xx = xx[0].unsqueeze(0)
        out.append(xx)
    return torch.cat(out, dim=0)

def get_fused_ego_feature(x):
    B,N,C,H,W = x.shape
    out = []
    for b in range(B):
       xx = x[b][0].unsqueeze(0)
       out.append(xx)
    return torch.cat(out, dim=0)

class PointPillarErmvp(nn.Module):
    def __init__(self, args):
        super(PointPillarErmvp, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = ERMVPFusionEncoder(args['ermvp_fusion'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        if args['backbone_fix']:
            self.backbone_fix()        
        
        self.topk_ratio = args['comm']['topk_ratio'] # 0
        self.cluster_sample_ratio = args['comm']['cluster_sample_ratio'] # 0.2
        
        self.sampler = SortSampler(topk_ratio=self.topk_ratio, input_dim=256, score_pred_net='2layer-fc-256')

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        # spatial_correction_matrix = data_dict['spatial_correction_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        
        # [1,384,120,360]
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # [1,384,60,180]

        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        
        if 'nei_processed_lidar' in data_dict and record_len.sum() > 1:
            nei_spatial_features_2d = self.nei_model.forward_features(data_dict)
            N,C0,H0,W0 = spatial_features_2d.size()
            N,C1,H1,W1 = nei_spatial_features_2d.size()
            
            if H0 != H1 or W0 != W1:
                nei_spatial_features_2d = torch.nn.functional.interpolate(nei_spatial_features_2d, [H0,W0], mode='bilinear', align_corners=False) # [nei_num, C1, H0, W0]

            # naive channel selection
            if C0 <= C1:
                x_nei_new = nei_spatial_features_2d[:, :C0] # channel dropping [nei_num, C0, H0, W0]
            else:
                x_nei_new = torch.zeros(N ,C0, H0, W0).to(nei_spatial_features_2d.device)
                x_nei_new[:,:C1,:,:] = nei_spatial_features_2d # padding

            spatial_features_2d[1:] = x_nei_new[1:]


        N,C,H,W = spatial_features_2d.shape
        dis_priority = torch.ones([N,H,W]).to('cuda')
        idx = torch.arange(H*W).repeat(N,1,1).permute(2, 0, 1).to('cuda')
        #src:N,B,C
        src, sample_reg_loss, sort_confidence_topk, pos_embed = self.sampler(spatial_features_2d, idx, None,dis_priority)
        # # # B N C
        src = src.permute(1,0,2)
        _,s_len,_ = src.shape # s_len = H*W*self.topk_ratio
        
        cluster_num = max(math.ceil(s_len * self.cluster_sample_ratio), 1)

        idx_cluster, cluster_num = cluster_dpc_knn(src, cluster_num, 10)
        down_dict,idx = merge_tokens(src, idx_cluster, cluster_num, sort_confidence_topk.unsqueeze(2))
        idxxs = []
        for b in range(N):
            i = torch.arange(s_len)
            idxxs.append(idx[b][i])
        idxxs = torch.vstack(idxxs)

        src = index_points(down_dict,idxxs)
        src = src.permute(0,2,1)
      
        pos_embed = pos_embed.permute(1, 2, 0)
        
        batch_spatial_features = []
        for cav_idx in range(N):
            index = cav_idx
            for i, ele in enumerate(record_len):
                if index < ele:
                    break
                index = index-ele
            spatial_feature = torch.zeros(
                C,H*W,
                dtype=src.dtype,
                device=src.device)
            spatial_feature[:, pos_embed[cav_idx][0]] = src[cav_idx]

            if index==0:
                spatial_feature = spatial_features_2d[cav_idx].flatten(1)
            # print(timestamp_index)
            batch_spatial_features.append(spatial_feature)


        batch_spatial_features = \
            torch.stack(batch_spatial_features, 0)
        batch_spatial_features = \
            batch_spatial_features.view(N,C,H,W)
        
        spatial_features_2d = batch_spatial_features
        # batch_dict['spatial_features'] = batch_spatial_features

        ego_features = get_ego_feature(spatial_features_2d,record_len)
        
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        # [1,1,1,1,2]
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # [1,60,180,1,2]
        com_mask = repeat(com_mask,
                          'b h w c l -> b (h new_h) (w new_w) c l',
                          new_h=regroup_feature.shape[3],
                          new_w=regroup_feature.shape[4])
        
        fused_feature = self.fusion_net(regroup_feature, com_mask)
        
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        ego_psm = self.cls_head(ego_features)
        ego_rm = self.reg_head(ego_features)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm,
                       'cls_preds_single': ego_psm,
                       'reg_preds_single': ego_rm,
                       }

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature),
                                'dir_preds_single': self.dir_head(ego_features)})

        return output_dict
