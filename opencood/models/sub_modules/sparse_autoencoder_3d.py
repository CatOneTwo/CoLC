# sencond based auto encoder-decoder

from functools import partial
import torch
import torch.nn as nn

from pdb import set_trace as pause

import time

try: # spconv1
    from spconv import SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor
except: # spconv2
    from spconv.pytorch import  SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor

def post_act_block(in_channels, out_channels, kernel_size, norm_fn, stride=1, padding=0, indice_key=None, conv_type='subm', ):

    if conv_type == 'subm':
        conv = SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m



class SparseVoxelEncoderDecoder(nn.Module):
    def __init__(self, input_channels, grid_size, num_classes=1, use_skip_connection=True):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.sparse_shape = grid_size[::-1]  # 不添加 +[1,0,0]
        self.use_skip_connection = use_skip_connection

        # ---------------- Encoder ----------------
        self.enc1 = post_act_block(input_channels, 16, 3, norm_fn, padding=1, indice_key="subm1", conv_type="subm")
        self.enc2 = post_act_block(16, 32, 3, norm_fn, stride=2, padding=1, indice_key="down2", conv_type="spconv")
        self.enc3 = post_act_block(32, 64, 3, norm_fn, stride=2, padding=1, indice_key="down3", conv_type="spconv")

        # ---------------- Decoder ----------------
        self.dec2 = post_act_block(64, 32, 3, norm_fn, indice_key="down3", conv_type="inverseconv")
        self.dec1 = post_act_block(32, 16, 3, norm_fn, indice_key="down2", conv_type="inverseconv")

        # ---------------- Skip fusion ----------------
        if self.use_skip_connection:
            self.skip_fuse2 = post_act_block(32 + 32, 32, 1, norm_fn, indice_key="skip2", conv_type="subm")
            self.skip_fuse1 = post_act_block(16 + 16, 16, 1, norm_fn, indice_key="skip1", conv_type="subm")

        # ---------------- Classifier ----------------
        self.classifier = SparseSequential(
            SubMConv3d(16, num_classes, kernel_size=1, bias=True, indice_key="cls")
        )

    def forward(self, batch_dict):
        
        t1= time.time()
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        

        x = SparseConvTensor(voxel_features, voxel_coords.int(), self.sparse_shape, batch_size)
        # ========== Encoder ==========
        x1 = self.enc1(x) # 原始分辨率
        x2 = self.enc2(x1) # /2
        x3 = self.enc3(x2) # /4
        # ========== Decoder ==========
        d2 = self.dec2(x3)  # 恢复至 enc2 的空间分辨率
        if self.use_skip_connection:
            d2 = self._fuse_skip(d2, x2, self.skip_fuse2)

        d1 = self.dec1(d2)   # 恢复至 enc1 的空间分辨率（即原始 voxel grid）
        if self.use_skip_connection:
            d1 = self._fuse_skip(d1, x1, self.skip_fuse1)
        t2=time.time()
        print(t2-t1)
        # ========== 分类 ==========
        out = self.classifier(d1)

        return out
    
    def _fuse_skip(self, decoder_feat, encoder_feat, fuse_module):
        """在形状匹配时进行 skip 融合（concatenate + 1x1 conv）"""
        if decoder_feat.spatial_shape == encoder_feat.spatial_shape:
            fused_features = torch.cat([decoder_feat.features, encoder_feat.features], dim=1)
            fused_tensor = decoder_feat.replace_feature(fused_features)
            return fuse_module(fused_tensor)
        else:
            # 形状不匹配时跳过 skip
            print(f"[Skip] Skipped: shape mismatch {decoder_feat.spatial_shape} vs {encoder_feat.spatial_shape}")
            return decoder_feat




class VoxelFeatureEncoder(nn.Module):
    """3D voxel特征降采样编码器 - 确保形状跟踪"""
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # 存储原始网格尺寸
        self.original_grid_size = grid_size  # [X, Y, Z]
        self.sparse_shape = grid_size[::-1]  # [Z, Y, X] - 只包含空间维度
        
        print(f"Encoder input shape: {self.sparse_shape}")

        # 输入卷积 - 保持形状
        self.conv_input = SparseSequential(
            SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )

        # 第一层 - 保持形状
        self.conv1 = SparseSequential(
            post_act_block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        # 下采样层 - 2倍降采样，计算输出形状
        self.conv2 = SparseSequential(
            post_act_block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, 
                          indice_key='spconv2', conv_type='spconv'),
            post_act_block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        # 计算下采样后的形状
        self.downsampled_shape = [
            (self.sparse_shape[0]) // 2,  # Z维度
            (self.sparse_shape[1]) // 2,  # Y维度  
            (self.sparse_shape[2]) // 2   # X维度
        ]
        print(f"Encoder output shape: {self.downsampled_shape}")

        self.encoder_features_out = model_cfg.get('encoder_features_out', 32)

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        input_sp_tensor = SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,  # 原始形状
            batch_size=batch_size
        )

        # 编码路径
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)      # 保持原始形状
        x_conv2 = self.conv2(x_conv1) # 2倍下采样

        # 存储形状信息用于解码器
        batch_dict['multi_scale_features'] = {
            'x_conv1': x_conv1,           # 高分辨率特征
            'x_conv2': x_conv2,           # 低分辨率特征
            'input_shape': self.sparse_shape,        # 输入形状
            'encoded_shape': x_conv2.spatial_shape,  # 编码后形状
        }

        batch_dict['encoded_features'] = x_conv2
        batch_dict['encoded_stride'] = 2
        
        return batch_dict

class VoxelFeatureDecoder(nn.Module):
    """3D voxel特征上采样解码器 - 确保恢复到原始尺寸"""
    def __init__(self, model_cfg, encoder_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        # 存储目标形状（原始输入形状）
        self.target_shape = grid_size[::-1]  # [Z, Y, X]
        print(f"Decoder target shape: {self.target_shape}")

        # 上采样层 - 使用与编码器对称的参数
        self.deconv1 = SparseSequential(
            # 关键：使用与下采样层相同的indice_key进行反向卷积
            post_act_block(encoder_channels, 32, 3, norm_fn=norm_fn, 
                          indice_key='spconv2', conv_type='inverseconv'),
            post_act_block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm_deconv1'),
        )

        # 可选：添加形状调整层来处理可能的形状不匹配
        self.shape_adjust = self._create_shape_adjust_layer(32, 32, norm_fn)

        self.decoder_features_out = model_cfg.get('decoder_features_out', 16)
        
        self.conv_out = SparseSequential(
            SubMConv3d(32, self.decoder_features_out, 3, padding=1, 
                      bias=False, indice_key='subm_out'),
            norm_fn(self.decoder_features_out),
            nn.ReLU(),
        )

    def _create_shape_adjust_layer(self, in_channels, out_channels, norm_fn):
        """创建形状调整层来处理边界情况"""
        return SparseSequential(
            SubMConv3d(in_channels, out_channels, 1, bias=False, indice_key='adjust'),
            norm_fn(out_channels),
            nn.ReLU(),
        )

    def forward(self, batch_dict):
        encoded_features = batch_dict['encoded_features']
        
        print(f"Decoder input shape: {encoded_features.spatial_shape}")
        print(f"Decoder target shape: {self.target_shape}")

        # 上采样
        x_deconv1 = self.deconv1(encoded_features)
        print(f"After deconv shape: {x_deconv1.spatial_shape}")

        # 形状检查和调整
        if x_deconv1.spatial_shape != self.target_shape:
            print(f"Shape mismatch! Adjusting from {x_deconv1.spatial_shape} to {self.target_shape}")
            x_deconv1 = self._adjust_sparse_tensor_shape(x_deconv1, self.target_shape)

        # 最终输出
        decoded_features = self.conv_out(x_deconv1)
        
        # 最终形状验证
        if decoded_features.spatial_shape != self.target_shape:
            print(f"Warning: Final output shape {decoded_features.spatial_shape} doesn't match target {self.target_shape}")
            decoded_features = self._adjust_sparse_tensor_shape(decoded_features, self.target_shape)

        batch_dict['decoded_features'] = decoded_features
        batch_dict['decoded_stride'] = 1
        batch_dict['output_shape'] = decoded_features.spatial_shape
        
        return batch_dict

    def _adjust_sparse_tensor_shape(self, sp_tensor, target_shape):
        """调整稀疏张量的形状到目标形状"""
        # 方法1: 直接创建新张量（如果索引在范围内）
        current_indices = sp_tensor.indices
        
        # 过滤掉超出目标形状的索引
        valid_mask = torch.ones(current_indices.shape[0], dtype=torch.bool)
        for i in range(3):  # 遍历Z,Y,X维度
            valid_mask &= (current_indices[:, i+1] < target_shape[i])
            valid_mask &= (current_indices[:, i+1] >= 0)
        
        filtered_indices = current_indices[valid_mask]
        filtered_features = sp_tensor.features[valid_mask]
        
        adjusted_tensor = SparseConvTensor(
            features=filtered_features,
            indices=filtered_indices,
            spatial_shape=target_shape,
            batch_size=sp_tensor.batch_size
        )
        
        return adjusted_tensor


class VoxelClassifier(nn.Module):
    """确保形状一致的voxel分类模型"""
    def __init__(self, model_cfg, input_channels, grid_size, num_classes, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_classes = num_classes
        self.grid_size = grid_size
        
        self.encoder = VoxelFeatureEncoder(model_cfg, input_channels, grid_size, **kwargs)
        
        encoder_out_channels = getattr(self.encoder, 'encoder_features_out', 32)
        self.decoder = VoxelFeatureDecoder(model_cfg, encoder_out_channels, grid_size, **kwargs)
        
        decoder_out_channels = getattr(self.decoder, 'decoder_features_out', 16)
        self.classifier = nn.Sequential(
            nn.Linear(decoder_out_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, batch_dict):
        # 添加原始形状信息
        batch_dict['original_grid_size'] = self.grid_size
        
        # 特征编码
        batch_dict = self.encoder(batch_dict)
        
        # 特征解码
        batch_dict = self.decoder(batch_dict)
        
        # 验证最终形状
        output_shape = batch_dict['output_shape']
        target_shape = self.grid_size[::-1]
        
        if output_shape == target_shape:
            print("✓ Success: Output shape matches target shape")
        else:
            print(f"✗ Warning: Output shape {output_shape} doesn't match target {target_shape}")
        
        # 分类预测
        decoded_features = batch_dict['decoded_features']
        voxel_logits = self.classifier(decoded_features.features)
        
        batch_dict['voxel_logits'] = voxel_logits
        batch_dict['voxel_predictions'] = torch.argmax(voxel_logits, dim=1)
        
        return batch_dict
