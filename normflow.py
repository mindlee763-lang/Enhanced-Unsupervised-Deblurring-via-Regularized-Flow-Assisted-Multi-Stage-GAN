import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class BlurKernelFlow(nn.Module):
    """ 单分支模糊核预测器 """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 直接预测模糊核参数
        self.kernel_predictor = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 27)  # 输出27个参数用于构建3x3x3核
        )

    def forward(self, blur_img):
        # 编码模糊图像特征
        features = self.encoder(blur_img).view(-1, 128)
        
        # 直接预测核参数
        kernel_params = self.kernel_predictor(features)
        kernel = kernel_params.view(-1, 3, 3, 3)  # 重塑为[B,3,3,3]
            
        # 核归一化条件
        kernel = kernel.softmax(dim=2).softmax(dim=3)  # 对每个3x3核独立归一化
        kernel = kernel / kernel.sum(dim=(2,3), keepdim=True)  # 保持总和为1
        return kernel