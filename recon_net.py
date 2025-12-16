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

class BlurReconstructionNetwork(nn.Module):
    def __init__(self, input_channels, num_resblocks=5):
        super().__init__()
        # 输入通道数为清晰图片通道数
        self.conv1 = conv(input_channels, 64, kernel_size=3)
        self.resblocks = nn.Sequential(*[resblock(64, 64) for _ in range(num_resblocks)])
        self.conv2 = conv(64, 3, kernel_size=3)  # 输出模糊图片

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblocks(x)
        x = self.conv2(x)
        return x