# Densely Scene-Guided Semantic Enhancement (DSSE)
import torch
import torch.nn as nn
from pdb import set_trace as pause
# S2D module for PointPillar

class DSSE(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        
        num_features = model_cfg['num_features'] # 64
        downsample_size1 = model_cfg['downsample_size1'] # [128, 50, 126]
        downsample_size2 = model_cfg['downsample_size2'] # [256, 25, 63]

        self.encoder_1 = nn.Sequential(     #  N,64,468,468
            nn.MaxPool2d(2,2),           #  N,64,234,234
            nn.Conv2d(num_features,32,1,1,0),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32,32,2,2),        #  N,64,117,117
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32,128,1,1,0),
            nn.BatchNorm2d(128),
            nn.GELU(),                   # 2,64,117,117
             
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(128,128,3,2,1),    #  N,64,59,59
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.convnext_block_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            # nn.LayerNorm([256,25,63], eps=1e-6),
            nn.LayerNorm(downsample_size2, eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )
        self.convnext_block_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            # nn.LayerNorm([256,25,63], eps=1e-6),
            nn.LayerNorm(downsample_size2, eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )
        
        self.convnext_block_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            # nn.LayerNorm([256,25,63], eps=1e-6),
            nn.LayerNorm(downsample_size2, eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # nn.Upsample((50,126))              # 2,64,117,117
            nn.Upsample(tuple(downsample_size1[1:]))              # 2,64,117,117
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(128+128,num_features,3,1,1),        # 2,64,117,117
            nn.BatchNorm2d(num_features),
            nn.GELU(),
            nn.ConvTranspose2d(num_features,num_features,4,2,1),    #  N,64,234,234
            nn.BatchNorm2d(num_features),
            nn.GELU(),
            nn.Conv2d(num_features,num_features,1,1,0),             #  N,64,234,234
            nn.BatchNorm2d(num_features),
            nn.GELU(),
            nn.Upsample(scale_factor=2)         #  N,64,468,468
        )

        self.fusion_sparse = nn.Sequential(
            nn.Conv2d(num_features,num_features,1,1,0),
            nn.BatchNorm2d(num_features),
            nn.GELU(),
        )

        self.fusion_dense = nn.Sequential(
            nn.Conv2d(num_features,num_features,1,1,0),
            nn.BatchNorm2d(num_features),
            nn.GELU(),
        )

    def forward(self, batch_dict):
        
        spatial_features = batch_dict['spatial_features'] # [8, 64, 200, 504] 

        y_1 = self.encoder_1(spatial_features)    # [8, 128, 50, 126]
        y_2 = self.encoder_2(y_1)                # [8, 256, 25, 63] 
        att = self.convnext_block_1(y_2) + y_2   #                       
        att = self.convnext_block_2(att) + att   # [8, 256, 25, 63]                          
        att = self.convnext_block_3(att) + att   # [8, 256, 25, 63]                          
        y_3 = torch.cat([self.decoder_1(att) , y_1],1) # [8, 256, 50, 126]
        fg_spatial_features = self.decoder_2(y_3)  # [8, 64, 200, 504]                                
        enhanced_spatial_features = self.fusion_dense(fg_spatial_features) + self.fusion_sparse(spatial_features) # [8, 64, 200, 504]

        batch_dict['ori_spatial_features'] = spatial_features
        batch_dict['spatial_features'] = enhanced_spatial_features
        batch_dict['fg_spatial_features'] = fg_spatial_features
        
        return batch_dict

