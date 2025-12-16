import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def calculate_psnr(img1, img2):
    """
    计算两张图像的 PSNR。
    img1 和 img2 应该是 Tensor，形状为 [C, H, W]，值在 [0, 1] 之间。
    """
    img1 = img1.cpu().numpy().transpose(1, 2, 0)  # 转换为 NumPy 格式，形状为 (H, W, C)
    img2 = img2.cpu().numpy().transpose(1, 2, 0)
    
    psnr = compare_psnr(img1, img2, data_range=1.0)  # PSNR 需要输入值在 [0, 1] 之间
    return psnr

def calculate_ssim(img1, img2, win_size=3):
    """
    计算两张图像的 SSIM。
    img1 和 img2 应该是 Tensor，形状为 [C, H, W]，值在 [0, 1] 之间。
    """
    img1 = img1.cpu().numpy().transpose(1, 2, 0)  # 转换为 NumPy 格式，形状为 (H, W, C)
    img2 = img2.cpu().numpy().transpose(1, 2, 0)
    
    ssim = compare_ssim(img1, img2, multichannel=True, win_size=win_size, data_range=1.0, channel_axis=2)  # SSIM 需要 multichannel=True
    return ssim