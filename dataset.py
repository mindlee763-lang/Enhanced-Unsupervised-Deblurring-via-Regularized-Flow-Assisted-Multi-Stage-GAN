import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

# 自定义 Dataset
class DeblurDataset(Dataset):
    def __init__(self, blurred_dir, sharp_dir, unpaired_dir, transform=None):
        self.blurred_dir = blurred_dir
        self.sharp_dir = sharp_dir
        self.unpaired_dir = unpaired_dir
        self.transform = transform
        self.blurred_images = sorted(os.listdir(blurred_dir))
        self.sharp_images = sorted(os.listdir(sharp_dir))
        self.unpaired_images = sorted(os.listdir(unpaired_dir))
        assert len(self.blurred_images) == len(self.sharp_images), "模糊图像与清晰图像数量不一致！"
        assert len(self.blurred_images) == len(self.unpaired_images), "模糊图像与清晰图像数量不一致！"

    def __len__(self):
        return len(self.blurred_images)

    def __getitem__(self, idx):
        blurred_img_path = os.path.join(self.blurred_dir, self.blurred_images[idx])
        sharp_img_path = os.path.join(self.sharp_dir, self.sharp_images[idx])
        unpaired_img_path = os.path.join(self.unpaired_dir, self.unpaired_images[idx])

        blurred_image = Image.open(blurred_img_path).convert("RGB")
        sharp_image = Image.open(sharp_img_path).convert("RGB")
        unpaired_image = Image.open(unpaired_img_path).convert("RGB")

        if self.transform:
            blurred_image, sharp_image, unpaired_image = self.transform(blurred_image, sharp_image, unpaired_image)

        return blurred_image, sharp_image, unpaired_image

class testDataset(Dataset):
    def __init__(self, blurred_dir, sharp_dir, transform=None):
        self.blurred_dir = blurred_dir
        self.sharp_dir = sharp_dir
        self.transform = transform
        self.blurred_images = sorted(os.listdir(blurred_dir))
        self.sharp_images = sorted(os.listdir(sharp_dir))
        assert len(self.blurred_images) == len(self.sharp_images), "模糊图像与清晰图像数量不一致！"

    def __len__(self):
        return len(self.blurred_images)

    def __getitem__(self, idx):
        blurred_img_path = os.path.join(self.blurred_dir, self.blurred_images[idx])
        sharp_img_path = os.path.join(self.sharp_dir, self.sharp_images[idx])

        blur_name = self.blurred_images[idx]
        sharp_name = self.sharp_images[idx]

        blurred_image = Image.open(blurred_img_path).convert("RGB")
        sharp_image = Image.open(sharp_img_path).convert("RGB")

        if self.transform:
            blurred_image, sharp_image = self.transform(blurred_image, sharp_image)

        return blurred_image, sharp_image, blur_name, sharp_name
    
#自定义数据增强操作  
class PairedTransforms:
    def __init__(self, resize=(256, 256), horizontal_flip=True, random_crop=True, color_jitter=True, rotate=True, max_rotation=360):
 
        self.resize = resize
        self.horizontal_flip = horizontal_flip
        self.random_crop = random_crop
        self.color_jitter = color_jitter
        self.rotate = rotate
        self.max_rotation = max_rotation

    def __call__(self, blurred_image, sharp_image, unpaired_image):
        # Resize
        blurred_image = TF.resize(blurred_image, self.resize)
        sharp_image = TF.resize(sharp_image, self.resize)
        unpaired_image = TF.resize(unpaired_image, self.resize)

        # Apply Horizontal Flip
        if self.horizontal_flip and random.random() > 0.5:
            blurred_image = TF.hflip(blurred_image)
            sharp_image = TF.hflip(sharp_image)
            unpaired_image = TF.hflip(unpaired_image)

        # Apply Random Crop
        if self.random_crop:
            i, j, h, w = transforms.RandomCrop.get_params(blurred_image, output_size=self.resize)
            blurred_image = TF.crop(blurred_image, i, j, h, w)
            sharp_image = TF.crop(sharp_image, i, j, h, w)
            unpaired_image = TF.crop(unpaired_image, i, j, h, w)

        # Apply Color Jitter
        if self.color_jitter:
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
            blurred_image = color_jitter(blurred_image)
            sharp_image = color_jitter(sharp_image)
            unpaired_image = color_jitter(unpaired_image)

        # Apply Random Rotation
        if self.rotate:
            angle = random.uniform(-self.max_rotation, self.max_rotation)  # 生成一个随机的旋转角度
            blurred_image = TF.rotate(blurred_image, angle)
            sharp_image = TF.rotate(sharp_image, angle)
            unpaired_image = TF.rotate(unpaired_image, angle)

        # Convert to Tensor
        blurred_image = TF.to_tensor(blurred_image)
        sharp_image = TF.to_tensor(sharp_image)
        unpaired_image = TF.to_tensor(unpaired_image)

        # Normalize to [-1, 1]
        blurred_image = TF.normalize(blurred_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        sharp_image = TF.normalize(sharp_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        unpaired_image = TF.normalize(unpaired_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return blurred_image, sharp_image, unpaired_image
