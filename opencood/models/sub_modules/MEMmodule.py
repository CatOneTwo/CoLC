import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from mmcv.ops import DeformConv2d
from torch.nn.modules.utils import _pair, _single
# input size: (bs, 256, 100, 352)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # avg pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # max pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # channel-wise attention
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False) #kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # add
        out = avg_out + max_out
        return self.sigmoid(out)

# spatial-wise attention 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # same padding
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # avg pool
        max_out, _ = torch.max(x, dim=1, keepdim=True) # max pool
        # concatenation
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class MotionEnhancedMech(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        bottleneck_size = [384, 128]
        self.reduce_dim_z = BasicConv2d(input_size*2, bottleneck_size[0], kernel_size=1, padding=0)
        self.s_atten_z = SpatialAttention()
        self.c_atten_z = ChannelAttention(bottleneck_size[0])
        deform_groups = 32
        kernel_size = 3
        self.motion_offset = nn.Conv2d(input_size, 2*kernel_size*kernel_size*deform_groups, kernel_size=3, padding=1)
        self.modion_dconv = DeformConv2d(input_size, input_size, kernel_size=kernel_size, padding=1, deform_groups=deform_groups)

        self.fusion_offset = nn.Conv2d(input_size, 2*kernel_size*kernel_size*deform_groups, kernel_size=3, padding=1)
        self.fusion_dconv = DeformConv2d(input_size, input_size, kernel_size=kernel_size, padding=1, deform_groups=deform_groups)
        self.relu = nn.ReLU()
    def generate_attention_z(self, x):
        z = self.reduce_dim_z(x) #[1, 384, 100, 352]
        atten_s = self.s_atten_z(z.mean(dim=1, keepdim=True)).view(z.size(0), -1, z.size(2), z.size(3))
        atten_c = self.c_atten_z(z)
        z = F.sigmoid(atten_s * atten_c)
        return z, 1 - z
    def regroup(self, x, record_len):
        cum_sum_len = np.cumsum(record_len).tolist()
        split_x = torch.tensor_split(x, cum_sum_len[:-1])
        return split_x
    def forward(self, bev_features_2d, record_len, his_frames_lens):
        start_idx = [0] + record_len[:-1]
        curren_bev_features = bev_features_2d[start_idx]
        batch_size, C, H, W = curren_bev_features.shape
        # historical BEV feature
        his_bev_start_idx = sum(record_len)
        his_bev_end_idx = sum(record_len+his_frames_lens)
        his_bev = bev_features_2d[his_bev_start_idx: his_bev_end_idx]
        # [(frames1, c, h, w), (frames2, c, h, w)]
        batch_his_bev = self.regroup(his_bev, his_frames_lens)
        # input_size: (bs, num_frames, C, H, W)

        batch_temporal_fusion = []
        for i, his_bev in enumerate(batch_his_bev):
            if his_frames_lens[i] == 0:
                batch_temporal_fusion.append(curren_bev_features[i].unsqueeze(0))
                continue
            x = his_bev  # (num_frames, c, h, w)
            depth, num_channels, h, w = x.shape # depth: num_frames

            res = torch.cat((x[0].unsqueeze(0), x), dim=0)  #torch.Size([num_frames+1, 384, 100, 352])
            # res：[t1, t1, t2, t3, tn-1]
            pre = res[:-1]
            # motion:  [t1-t1, t2-t1, t3-t2, ..., tn-tn-1]
            res = x - pre
            motion_offsets = self.motion_offset(res)
            motion_features = self.relu(self.modion_dconv(pre, motion_offsets))
            h = x[0] # refined the feature of the first frame
            for t in range(depth):
                # h(refined feature):torch.Size([1, 512, 100, 352])
                con_fea = torch.cat((h, motion_features[t]), dim=0).unsqueeze(0)  #initialize t=0 -----> 0
                z_p, z_r = self.generate_attention_z(con_fea)
                h = z_r * h + z_p * motion_features[t]

                fusion_offset = self.fusion_offset(h)
                fusion_dconv = self.relu(self.fusion_dconv(h, fusion_offset))
                h = fusion_dconv.squeeze(0)
            batch_temporal_fusion.append(h.unsqueeze(0))
        fea_t = torch.cat(batch_temporal_fusion, dim=0)  #channel层concat

        return fea_t


class DeformConv2d(nn.Module):
    r"""Deformable 2D convolution.

    Applies a deformable 2D convolution over an input signal composed of
    several input planes. DeformConv2d was described in the paper
    `Deformable Convolutional Networks
    <https://arxiv.org/pdf/1703.06211.pdf>`_

    Note:
        The argument ``im2col_step`` was added in version 1.3.17, which means
        number of samples processed by the ``im2col_cuda_kernel`` per call.
        It enables users to define ``batch_size`` and ``im2col_step`` more
        flexibly and solved `issue mmcv#1440
        <https://github.com/open-mmlab/mmcv/issues/1440>`_.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size(int, tuple): Size of the convolving kernel.
        stride(int, tuple): Stride of the convolution. Default: 1.
        padding (int or tuple): Zero-padding added to both sides of the input.
            Default: 0.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input.
            channels to output channels. Default: 1.
        deform_groups (int): Number of deformable group partitions.
        bias (bool): If True, adds a learnable bias to the output.
            Default: False.
        im2col_step (int): Number of samples processed by im2col_cuda_kernel
            per call. It will work when ``batch_size`` > ``im2col_step``, but
            ``batch_size`` must be divisible by ``im2col_step``. Default: 32.
            `New in version 1.3.17.`
    """

    # @deprecated_api_warning({'deformable_groups': 'deform_groups'},
    #                         cls_name='DeformConv2d')
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding= 0,
                 dilation= 1,
                 groups= 1,
                 deform_groups= 1,
                 bias= False,
                 im2col_step= 32):
        super().__init__()

        assert not bias, \
            f'bias={bias} is not supported in DeformConv2d.'
        assert in_channels % groups == 0, \
            f'in_channels {in_channels} cannot be divisible by groups {groups}'
        assert out_channels % groups == 0, \
            f'out_channels {out_channels} cannot be divisible by groups \
              {groups}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.im2col_step = im2col_step
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        # only weight, no bias
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        # switch the initialization of `self.weight` to the standard kaiming
        # method described in `Delving deep into rectifiers: Surpassing
        # human-level performance on ImageNet classification` - He, K. et al.
        # (2015), using a uniform distribution
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x, offset):
        """Deformable Convolutional forward function.

        Args:
            x (Tensor): Input feature, shape (B, C_in, H_in, W_in)
            offset (Tensor): Offset for deformable convolution, shape
                (B, deform_groups*kernel_size[0]*kernel_size[1]*2,
                H_out, W_out), H_out, W_out are equal to the output's.

                An offset is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
                The spatial arrangement is like:

                .. code:: text

                    (x0, y0) (x1, y1) (x2, y2)
                    (x3, y3) (x4, y4) (x5, y5)
                    (x6, y6) (x7, y7) (x8, y8)

        Returns:
            Tensor: Output of the layer.
        """
        # To fix an assert error in deform_conv_cuda.cpp:128
        # input image is smaller than kernel
        input_pad = (x.size(2) < self.kernel_size[0]) or (x.size(3) <
                                                          self.kernel_size[1])
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0)
            offset = offset.contiguous()
        out = deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                            self.dilation, self.groups, self.deform_groups,
                            False, self.im2col_step)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()
        return out




if __name__ == "__main__":
    bev_features_2d = torch.ones(8, 384, 100, 352)
    record_len = [2, 3]
    his_frames_lens = [0, 3]
    model = MotionEnhancedMech(input_size=384, hidden_size=384)
    model(bev_features_2d, record_len, his_frames_lens)