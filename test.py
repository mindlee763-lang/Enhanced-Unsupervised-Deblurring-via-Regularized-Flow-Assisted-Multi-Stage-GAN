import torch
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms
from model import model
import os
from PIL import Image
from torchvision.utils import save_image
from dataset import testDataset, PairedTransforms
from psnr_ssim import calculate_psnr, calculate_ssim
from tqdm import tqdm

#数据路径
data_path = 'GAN/data/test_samples'
gt_path = 'GAN/data/gt'

def data_process():
    
    # 数据预处理
    transform = PairedTransforms(resize = (256, 256), horizontal_flip = False, random_crop = False, color_jitter = False, rotate = False, max_rotation = 360)

    # 创建数据集实例
    test_dataset = testDataset(data_path, gt_path, transform = transform)
    
    #加载数据集
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    return test_loader

def denormalize(img):
    
    return (img + 1) / 2

def test(model, test_loader):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    image_names = []

    for image_name in os.listdir(data_path):
        image_names.append(image_name)
    
    avg_psnr = 0.0
    avg_ssim = 0.0
    num_samples = 0

    model.eval()
    with tqdm(total=len(image_names), desc="Testing") as pbar:
    #将梯度置为0，不进行反向传播，加快运行速度
        with torch.no_grad():  # 禁用梯度计算
            for i, (blurred_images, sharp_images, blur_name, sharp_name) in enumerate(test_loader):
                blurred_images = blurred_images.cuda()
                sharp_images = sharp_images.cuda()

                # 生成清晰图像
                d4 = model(blurred_images)[0]

                generated_images = d4

                save_image(generated_images, 'GAN/data/test_results/'+ blur_name[0], normalize = True)
                save_image(blurred_images, 'GAN/data/test_results_blur/'+ blur_name[0], normalize = True)
                save_image(sharp_images, 'GAN/data/test_results_gt/'+ sharp_name[0], normalize = True)


                for j in range(generated_images.size(0)):  # 遍历 batch 中的每一张图像
                    gen_img = generated_images[j]
                    real_img = sharp_images[j]

                    # 如果图像值范围是 [-1, 1]，则需要将其归一化到 [0, 1]
                    gen_img = denormalize(gen_img)
                    real_img = denormalize(real_img)
                
                    # 生成图像和真实图像的像素值范围应该在 [0, 1] 之间
                    gen_img = torch.clamp(gen_img, 0.0, 1.0)
                    real_img = torch.clamp(real_img, 0.0, 1.0)

                    # 计算 PSNR 和 SSIM
                    psnr_value = calculate_psnr(gen_img, real_img)
                    ssim_value = calculate_ssim(gen_img, real_img, win_size=3)

                    avg_psnr += psnr_value
                    avg_ssim += ssim_value
                    num_samples += 1
        
                pbar.update(1)

    # 计算平均 PSNR 和 SSIM
        avg_psnr /= num_samples
        avg_ssim /= num_samples


    return avg_psnr, avg_ssim



if __name__ == "__main__":
    
    #加载模型
    gan = model(num_resblocks = [1, 2, 3, 4], input_channels = [3, 6, 6, 6])
    #将最好模型的参数赋予模型
    gan.load_state_dict(torch.load('GAN/best_generator_model.pth'))
    
    #加载测试数据
    test_dataloader = data_process()

    psnr, ssim = test(gan, test_dataloader)

    print(f"avg_psnr:{psnr:.4f}, avg_ssim:{ssim:.4f}")
