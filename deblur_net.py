import torch
import torch.nn as nn

def conv(input_dim, output_dim, kernel_size, stride = 1, bias = True):
    return nn.Conv2d(input_dim, output_dim, kernel_size, stride = stride, padding=(kernel_size//2), bias = bias)

def deconv(input_dim, output_dim, kernel_size, stride = 1, bias = True):
    return nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride = stride, padding=(kernel_size//2), bias = bias)


class resblock(nn.Module):
    def __init__(self, input_dim = 32, output_dim = 32):
        super().__init__()
        self.conv1 = conv(input_dim, output_dim, 3)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv(input_dim, output_dim, 3)
    
    def forward(self, x):
        orangin = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = orangin + x
        return x

class DeblurNetwork(nn.Module):
    def __init__(self, input_channels, num_resblocks=5, gen_input_channels=6):
        super().__init__()
        # 输入通道数为原始图片通道数 + 模糊核通道数
        self.conv1 = conv(input_channels + 3, 64, kernel_size=3)  # 模糊核作为额外通道
        self.resblocks = nn.Sequential(*[resblock(64, 64) for _ in range(num_resblocks)])
        self.conv2 = conv(64, 3, kernel_size=3)  # 输出清晰图片
        self.fc = nn.Sequential(
            nn.Linear(27, 128),          # 输入: 3×3x3=27 → 输出:128
            nn.ReLU(inplace=True),
            nn.Linear(128, gen_input_channels - 3),  # 输出通道数根据生成器输入需求调整
        )
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, blur_kernel):
        # 将核展平为特征向量 [B, 27]
        kernel_features = blur_kernel.view(-1, 27)
    
       # 通过全连接层生成通道特征
        channel_features = self.fc(kernel_features)  # [B, C]
        channel_features = channel_features.view(-1, 3, 1, 1)
        # 将模糊核与模糊图片拼接
        x = torch.cat([x, channel_features.expand(-1, -1, x.size(2), x.size(3))], dim=1)  # 在通道维度拼接
        x = self.conv1(x)
        x = self.resblocks(x)
        x = self.conv2(x)
        return x