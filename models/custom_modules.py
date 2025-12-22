import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CBAM ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResCBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ResCBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# --- SCConv ---
class SRU(nn.Module):
    def __init__(self, oup_channels, group_num=16, gate_treshold=0.5):
        super(SRU, self).__init__()
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        # Use torch.sum to ensure it's a tensor operation and potentially faster/more stable
        # Add a small epsilon to avoid division by zero if weights sum to 0 (unlikely but safe)
        w_gamma = self.gn.weight / (torch.sum(self.gn.weight) + 1e-6)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweights = self.sigmoid(gn_x * w_gamma)
        
        # Gate
        info_mask = reweights >= self.gate_treshold
        non_info_mask = reweights < self.gate_treshold
        x_1 = info_mask * x
        x_2 = non_info_mask * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class CRU(nn.Module):
    def __init__(self, op_channel, alpha=1/2, squeeze_radio=2, group_size=2, group_kernel_size=3, stride=1):
        super(CRU, self).__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        
        # Up channel
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=stride, padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        
        # Low channel
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)
        self.stride = stride

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        
        # Up channel
        up = self.squeeze1(up)
        # For stride > 1, we only use GWC which has stride. PWC1 is 1x1 and doesn't support stride here easily.
        # In original SCConv, it is designed for stride=1 usually or specific block design.
        # Here we simplify: if stride > 1, we skip PWC1 addition or we need to pool PWC1 input.
        # But since we only use SCConv in conv2 of Bottleneck which is 3x3, and in ResNet Bottleneck conv2 is 3x3 with stride.
        # Wait, in standard ResNet Bottleneck:
        # conv1: 1x1, stride=1 (or stride=2 in some variants, but usually stride is on conv2)
        # conv2: 3x3, stride=stride
        # conv3: 1x1, stride=1
        # So conv2 CAN have stride=2.
        
        if self.stride > 1:
            # If stride > 1, GWC handles stride. PWC1 (1x1) needs to match output size.
            # We can just omit PWC1 for strided case or use avgpool.
            Y1 = self.GWC(up)
        else:
            Y1 = self.GWC(up) + self.PWC1(up)
        
        # Low channel
        low = self.squeeze2(low)
        Y2 = self.PWC2(low)
        
        # If stride > 1, low channel also needs to be downsampled to match Y1
        if self.stride > 1:
            Y2 = F.avg_pool2d(Y2, kernel_size=self.stride, stride=self.stride)
            low = F.avg_pool2d(low, kernel_size=self.stride, stride=self.stride)
            
        Y2 = torch.cat([Y2, low], dim=1)
        
        # Fusion
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

# Let's rewrite CRU to be simple and robust or just use it for stride=1.


class SCConv(nn.Module):
    def __init__(self, op_channel, group_num=16, gate_treshold=0.5, alpha=1/2, squeeze_radio=2, group_size=2, group_kernel_size=3, stride=1):
        super(SCConv, self).__init__()
        self.sru = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.cru = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size, group_kernel_size=group_kernel_size, stride=stride)

    def forward(self, x):
        x = self.sru(x)
        x = self.cru(x)
        return x

# --- ProtoNet ---
class ProtoNet(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=256):
        super(ProtoNet, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_dim))
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        # x: [B, C, H, W] -> [B, C] (Global Avg Pool)
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        # Cosine similarity
        x_norm = F.normalize(x, p=2, dim=1)
        p_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        logits = self.scale * F.linear(x_norm, p_norm)
        return logits

# --- CrossScaleAttention ---
class CrossScaleAttention(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(CrossScaleAttention, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, len(in_channels_list), 1),
            nn.Softmax(dim=1)
        )
        self.out_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features):
        # features: list of tensors from different scales
        # Resize all to the largest scale (usually the first one in FPN P2)
        target_size = features[0].shape[-2:]
        
        resized_features = []
        for i, feat in enumerate(features):
            feat = self.convs[i](feat)
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(feat)
            
        concat_feat = torch.cat(resized_features, dim=1)
        attn_weights = self.attention(concat_feat) # [B, N, H, W]
        
        out = 0
        for i in range(len(resized_features)):
            out += resized_features[i] * attn_weights[:, i:i+1, :, :]
            
        out = self.out_conv(out)
        return out
