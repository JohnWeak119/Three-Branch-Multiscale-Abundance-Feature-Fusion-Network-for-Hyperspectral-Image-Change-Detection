'''

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import configs.configs as cfg

big_scale = cfg.patch_size
mid_scale = cfg.patch_size - 2
sml_scale = cfg.patch_size - 4

'''
一。 RI & CBAM
'''
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out * x
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        out = out * x
        return out

class CBAM(nn.Module):
    '''
        CBAM: Convolutional Block Attention Module
    '''
    def __init__(self, in_ch):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_ch)
        self.sa = SpatialAttention(in_ch)
        self.convc = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_an = self.sa(self.ca(x))
        out = self.convc(x_an)
        return out

class ReducedInception(nn.Module):
    '''
        reduced inception (RI)
    '''
    def __init__(self, in_ch, di=4):
        super(ReducedInception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Conv2d(in_ch, in_ch // di, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        cat = torch.cat((branch1, branch2, branch3, branch4), dim=1)
        out = self.conv(cat)
        return out
    
class SpatialInception(nn.Module):
    '''
        Spatial inception (SI)
    '''
    def __init__(self, in_ch, di=4):
        super(SpatialInception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=(3,1), padding=(1,0)),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=(1,3), padding=(0,1)),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        cat = torch.cat((branch1, branch2, branch3, branch4), dim=1)
        out = self.conv(cat)
        return out

'''
三。 Res-Multi-head-cross Attention (ResMHCA)
'''
class ResMHCA(nn.Module):
    def __init__(self, in_channels, Nh=4, kernel_size=3, stride=1):
        super(ResMHCA, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dk = in_channels
        self.dv = in_channels
        self.Nh = Nh
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.qkv_conv1 = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size = self.kernel_size,
                                   stride = stride, padding = self.padding)
        self.qkv_conv2 = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size = self.kernel_size,
                                   stride = stride, padding = self.padding)
        self.qkv_conv12 = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size = self.kernel_size,
                                    stride = stride, padding = self.padding)

        self.attn_out1 = nn.Conv2d(self.dv, self.dv, kernel_size = 1, stride = 1)
        self.attn_out2 = nn.Conv2d(self.dv, self.dv, kernel_size = 1, stride = 1)
        self.attn_out12 = nn.Conv2d(self.dv, self.dv, kernel_size = 1, stride = 1)

        self.BN = nn.BatchNorm2d(self.in_channels)

    def forward(self, img1, img2, abu):
        # Input x
        # (batch_size, channels, height, width)
        batch, _, height, width = img1.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q1, flat_k1, flat_v1, q1, k1, v1 = self.compute_flat_qkv(img1, self.dk, self.dv, self.Nh, 1) # (batch, Nh, dkh, height * width)
        flat_q2, flat_k2, flat_v2, q2, k2, v2 = self.compute_flat_qkv(img2, self.dk, self.dv, self.Nh, 2)
        flat_q12, flat_k12, flat_v12, q12, k12, v12 = self.compute_flat_qkv(abu, self.dk, self.dv, self.Nh, 12)
        logits1 = torch.matmul(flat_q1.transpose(2, 3), flat_k12)   # (batch, Nh, height * width, height * width)
        logits2 = torch.matmul(flat_q2.transpose(2, 3), flat_k12)
        logits12 = torch.matmul(flat_q12.transpose(2, 3), flat_k1) - torch.matmul(flat_q12.transpose(2, 3), flat_k2)
        weights1 = nn.Softmax(dim=-1)(logits1)  # (batch, Nh, height * width, height * width)
        weights2 = nn.Softmax(dim=-1)(logits2)
        weights12 = nn.Softmax(dim=-1)(logits12)

        # attn_out
        # (batch, Nh, height * width, dvh)
        img1_out = torch.matmul(weights1, flat_v1.transpose(2, 3))
        img2_out = torch.matmul(weights2, flat_v2.transpose(2, 3))
        abu_out = torch.matmul(weights12, flat_v12.transpose(2, 3))
        img1_out = torch.reshape(img1_out, (batch, self.Nh, self.dv // self.Nh, height, width))   # (batch, Nh, dvh, height, width)
        img2_out = torch.reshape(img2_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        abu_out = torch.reshape(abu_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        img1_out = self.combine_heads_2d(img1_out)
        img2_out = self.combine_heads_2d(img2_out)
        abu_out = self.combine_heads_2d(abu_out)

        # Residual connection
        img1_out = self.attn_out1(img1_out) + img1
        img2_out = self.attn_out1(img2_out) + img2
        abu_out = self.attn_out1(abu_out)
        return img1_out, img2_out, abu_out

    def compute_flat_qkv(self, x, dk, dv, Nh, idx):
        if idx == 1:
            qkv = self.qkv_conv1(x)
        elif idx == 2:
            qkv = self.qkv_conv2(x)
        elif idx == 12:
            qkv = self.qkv_conv12(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim = 1)        # 将qkv按照dk、dk、dv分割channel, (batch, dk/dk/dv, height, width)
        q = self.split_heads_2d(q, Nh)  # 将qkv按照head个数分割channel
        k = self.split_heads_2d(k, Nh)  # (batch, Nh, dkh, height, width)
        v = self.split_heads_2d(v, Nh)
        dkh = dk // Nh
        q = q * (dkh ** -0.5)   # 缩放scale
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))  # 将空间维展开  (batch, Nh, dkh, height * width)
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()   # (batch, dk/dk/dv, height, width)
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)  # (batch, Nh, dv, height, width)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

'''
Model
ResMHCA在MHCA中加入输入与输出相加的残差
RI替换为了SI
起始特征提取也加了3*3卷积
'''
class TMAFN(nn.Module):
    def __init__(self,img_ch, abu_ch): 
        super(TMAFN, self).__init__()

        low_ch = 64
        mid_ch = 128
        deep_ch = 256

        self.img_input = nn.Conv2d(in_channels=img_ch, out_channels=low_ch, kernel_size=1)
        self.abu_input = nn.Conv2d(in_channels=abu_ch, out_channels=low_ch, kernel_size=1)

        # img网络
        self.img_1_conv = nn.Sequential(
            SpatialInception(low_ch),
            nn.Conv2d(low_ch, low_ch, kernel_size=3, padding=1),
            CBAM(low_ch)
        )       # b,low_ch, big_scale, big_scale
        self.img_2_conv = nn.Sequential(
            SpatialInception(low_ch),
            nn.Conv2d(low_ch, mid_ch, kernel_size=3),
            CBAM(mid_ch)
        )       # b,mid_ch, mid_scale, mid_scale
        self.img_3_conv = nn.Sequential(
            SpatialInception(mid_ch),
            nn.Conv2d(mid_ch, deep_ch, kernel_size=3),
            CBAM(deep_ch)
        )       # b,deep_ch, sml_scale, sml_scale

        # abu网络
        self.abu_1_conv = nn.Sequential(
            SpatialInception(low_ch),
            nn.Conv2d(low_ch, low_ch, kernel_size=3, padding=1),
            CBAM(low_ch)
        )       # b,low_ch, big_scale, big_scale
        self.abu_2_conv = nn.Sequential(
            SpatialInception(low_ch),
            nn.Conv2d(low_ch, mid_ch, kernel_size=3),
            CBAM(mid_ch)
        )       # b,mid_ch, mid_scale, mid_scale
        self.abu_3_conv = nn.Sequential(
            SpatialInception(mid_ch),
            nn.Conv2d(mid_ch, deep_ch, kernel_size=3),
            CBAM(deep_ch)
        )       # b,deep_ch, sml_scale, sml_scale

        # 1. ResMHCA
        self.ResMHCA1 = ResMHCA(in_channels = low_ch, Nh=4)
        self.ResMHCA2 = ResMHCA(in_channels = mid_ch, Nh=4)
        self.ResMHCA3 = ResMHCA(in_channels = deep_ch, Nh=4)

        # 2. abu MSF
        self.abu_convl2m = nn.Sequential(
            nn.Conv2d(low_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.abu_convm2m = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.abu_convd2m = nn.Sequential(
            nn.Conv2d(deep_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.abu_up35 = nn.ConvTranspose2d(mid_ch, mid_ch, 3)
        self.abu_up37 = nn.ConvTranspose2d(mid_ch, mid_ch, 3)
        self.abu_up_bdfr = nn.ConvTranspose2d(mid_ch, mid_ch, 3)
        self.abu_down39 = nn.Conv2d(mid_ch, mid_ch, 3)
        self.abu_down37 = nn.Conv2d(mid_ch, mid_ch, 3)
        self.abu_down_bdfr = nn.Conv2d(mid_ch, mid_ch, 3)
        self.abu_convfuse = nn.Conv2d(mid_ch*2, mid_ch, 1)
        self.abu_feature_fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 3, mid_ch, 1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        # 3. fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(mid_ch * mid_scale * mid_scale, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cfg.classes, bias=True),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img1, img2, abu): # input data size: b, ch,h,w
        if abu.dtype != torch.float32:
            abu = abu.float()
        # 第一层
        img1 = self.img_input(img1)
        img2 = self.img_input(img2)
        abu = self.abu_input(abu)

        out1 = self.img_1_conv(img1)    # b,low_ch, big_scale, big_scale
        out2 = self.img_1_conv(img2)
        out12 = self.abu_1_conv(abu)
        img1_feature1, img2_feature1, abu_feature1 = self.ResMHCA1(out1, out2, out12)

        # 第二层，添加残差连接
        out1 = self.img_2_conv(img1_feature1)   # b,mid_ch, mid_scale, mid_scale
        out2 = self.img_2_conv(img2_feature1)
        out12 = self.abu_2_conv(abu_feature1)
        img1_feature2, img2_feature2, abu_feature2 = self.ResMHCA2(out1, out2, out12)

        # 第三层
        out1 = self.img_3_conv(img1_feature2)   # b,deep_ch, sml_scale, sml_scale
        out2 = self.img_3_conv(img2_feature2)
        out12 = self.abu_3_conv(abu_feature2)
        img1_feature3, img2_feature3, abu_feature3 = self.ResMHCA3(out1, out2, out12)

        # abu_BDFR module
        abu_map_big = self.abu_convl2m(abu_feature1)    # b,mid_ch, big_scale, big_scale
        abu_map_mid = self.abu_convm2m(abu_feature2)    # b,mid_ch, mid_scale, mid_scale
        abu_map_sml = self.abu_convd2m(abu_feature3)    # b,mid_ch, sml_scale, sml_scale

        abu_d7f9 = self.abu_down39(abu_map_big) + abu_map_mid   # b,mid_ch, mid_scale, mid_scale
        abu_d5f7 = self.abu_down37(abu_d7f9) + abu_map_sml   # b,mid_ch, sml_scale, sml_scale
        abu_d7f5 = self.abu_up35(abu_map_sml) + abu_map_mid    # b,mid_ch, mid_scale, mid_scale
        abu_d9f7 = self.abu_up37(abu_d7f5) + abu_map_big  # b,mid_ch, big_scale, big_scale

        abu_bdfr1 = self.abu_down_bdfr(abu_d9f7)  # b,mid_ch, mid_scale, mid_scale
        abu_bdfr2 = self.abu_convfuse(torch.cat((abu_d7f9, abu_d7f5), dim=1))  # b,mid_ch, mid_scale, mid_scale
        abu_bdfr3 = self.abu_up_bdfr(abu_d5f7)  # b,mid_ch, mid_scale, mid_scale

        net_out = self.abu_feature_fuse(torch.cat((abu_bdfr1, abu_bdfr2, abu_bdfr3), dim=1))

        # Net output
        flatten_out = torch.flatten(net_out, 1, 3)  # b,mid_ch* mid_scale* mid_scale
        fc_out = self.fc(flatten_out)
        return fc_out