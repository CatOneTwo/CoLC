"""
Pillar VFE, credits to OpenPCDet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as pause

# 简化的PointNet层，输入特征为10， 输出特征为64，网络是论文中提出的线性网络
class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs) # x的维度由（M, 32, 10）升维成了（M, 32, 64）
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2,
                                                  1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        # 完成pointnet的最大池化操作，找出每个pillar中最能代表该pillar的点
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max # 返回经过简化版pointnet处理pillar的结果
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

# 生成一个个Pillar，并将点云原来的4维特征( x , y , z , r ) 扩充为10维特征 (x,y,z,r, x_c,y_c,z_c,x_p,y_p,z_p)
class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm'] # true
        self.with_distance = self.model_cfg['with_distance'] # false

        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz'] # true
        num_point_features += 6 if self.use_absolute_xyz else 3 # 10
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg['num_filters'] # 64
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters) # [10, 64]

        
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                         last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers) # 加入线性层，将10维特征变为64维特征

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod # 表明一个pillar中哪些是真实数据，哪些是填充的0数据
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict):
        """encoding voxel feature using point-pillar method
        Args:
            voxel_features: [M, 32, 4] 每个体素的特征为 (x, y, z, intensity)
            voxel_num_points: [M,]
            voxel_coords: [M, 4] (batch_id, z, y, x) x in [0,704], y in [0,200]
        Returns:
            features: [M,64], after PFN
        """
        voxel_features, voxel_num_points, coords = \
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
            batch_dict['voxel_coords']

        # 算每个pillar均值, 分子[M, 32, 3]->[M, 1, 3]，分母[M,]->[M,1,1]
        points_mean = \
            voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1) # [M, 1, 3]
        
        # 每个点云数据减去该点对应pillar的平均值得到差值 xc,yc,zc
        f_cluster = voxel_features[:, :, :3] - points_mean # [M, 32, 3]

        # Find distance of x, y, and z from pillar center  点与几何中心的相对位置。
        # 创建每个点云到该pillar的坐标中心点偏移量空数据 xp,yp,zp
        f_center = torch.zeros_like(voxel_features[:, :, :3]) # [M, 32, 3]

        # coords是每个网格点的坐标，即[432, 496, 1]，需要乘以每个pillar的长宽得到点云数据中实际的长宽（单位米）
        # 同时为了获得每个pillar的中心点坐标，还需要加上每个pillar长宽的一半得到中心点坐标
        # 每个点的x、y、z减去对应pillar的坐标中心点，得到每个点到该点中心点的偏移量
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_x + self.x_offset) # 获得pillar每个点的相对坐标，还需要加上每个pillar长宽的一半得到中心点坐标
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz: # True
            features = [voxel_features, f_cluster, f_center] 
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance: # False
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2,
                                     keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1) # [M, 32, 10] 4+3+3

        voxel_count = features.shape[1] # 32
        """
        由于在生成每个pillar中，不满足最大32个点的pillar会存在由0填充的数据，
        而刚才上面的计算中，会导致这些
        由0填充的数据在计算出现xc,yc,zc和xp,yp,zp出现数值，
        所以需要将这个被填充的数据的这些数值清0,
        因此使用get_paddings_indicator计算features中哪些是需要被保留真实数据和需要被置0的填充数据
        """
        # 得到mask维度是（M， 32, 1）值为0和1
        # mask中指名了每个pillar中哪些是需要被保留的数据
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count,
                                           axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask #  将feature中被填充为0的数据的所有特征置0
        for pfn in self.pfn_layers:
            features = pfn(features) # [M, 32, 10]-> [M, 1, 64]
        features = features.squeeze()
        batch_dict['pillar_features'] = features # [M, 64] 每个pillar的特征

        return batch_dict
