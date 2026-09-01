import torch
import torch.nn as nn

from pdb import set_trace as pause

from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule

class PointNet(nn.Module):
    """
    Lightweight PointNet2 Segmentation Network for fixed-size batch input
    Input: pointcloud [B, N, 4]
    Output: [B, N, 1] logits per point (foreground/background)
    """
    def __init__(self, args):
        super().__init__()
        # -------------------
        # Set Abstraction (SA) layers
        # -------------------
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=32,
                mlp=[4, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=None,  # group all
                radius=None,
                nsample=None,
                mlp=[128, 256, 512],
                use_xyz=True,
            )
        )

        # -------------------
        # Feature Propagation (FP) layers
        # # -------------------
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[64+4, 64, 32])) # FP between result (64) and original features (4)
        self.FP_modules.append(PointnetFPModule(mlp=[128+64, 128, 64])) # FP between result (128) and sa1 (64)
        self.FP_modules.append(PointnetFPModule(mlp=[512+128, 256, 128])) # FP between sa3 (512) and sa2 (128)

        # -------------------
        # Point-wise classifier
        # -------------------
        self.classifier = nn.Sequential(
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(32, 1, 1)  # 2 classes: foreground/background
        )

    def forward(self, data_dict):
        """
        pointcloud: [B, N, 4]
        returns: [B, N, 2] logits
        """

        pointcloud = data_dict['origin_lidar']

        B, N, C = pointcloud.shape
        l_xyz = pointcloud[..., :3].contiguous()  # [B, N, 3]
        l_features = pointcloud.transpose(1, 2).contiguous()  # [B, 4, N]

        xyz_list, feature_list = [l_xyz], [l_features]

        # -------------------
        # Set Abstraction
        # -------------------
        for sa_module in self.SA_modules:
            li_xyz, li_features = sa_module(xyz_list[-1], feature_list[-1])
            xyz_list.append(li_xyz)
            feature_list.append(li_features)


        # -------------------
        # Feature Propagation
        # -------------------
        for i in range(-1, -(len(self.FP_modules)+1), -1):
            feature_list[i-1] = self.FP_modules[i](
                xyz_list[i-1], xyz_list[i], feature_list[i-1], feature_list[i]
            )

        # -------------------
        # Point-wise classification
        # -------------------
        out = self.classifier(feature_list[0])  # [B, 2, N]
        out = out.transpose(1, 2).contiguous()   # [B, N, 1]

        return out

    