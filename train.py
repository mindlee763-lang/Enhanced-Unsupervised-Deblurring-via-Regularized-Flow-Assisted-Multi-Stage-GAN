import torch.optim
from torchvision import transforms
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
from model import model
import torch
import torch.nn as nn
import time
from dataset import DeblurDataset, PairedTransforms
from model import discriminator
from model import VGGFeatureExtractor
from normflow import BlurKernelFlow
import torch.nn.functional as F
from torchvision.utils import save_image
from save_best_model import save_checkpoint
import os
import shutil
from psnr_ssim import calculate_psnr, calculate_ssim
from torch.optim.lr_scheduler import CosineAnnealingLR
from recon_net import BlurReconstructionNetwork
from deblur_net import DeblurNetwork

# 模糊和清晰图像的文件夹路径
blurred_dir = './train_data/train'
sharp_dir = './train_gt_data/gt_train'
val_blurred_dir = './val_data/val'
val_sharp_dir = './val_gt_data/gt_val'
unpaired_train_dir = './unpaired_train_data/unpaired_train'
unpaired_val_dir = './unpaired_val_data/unpaired_val'


# 加载、处理数据
def data_process():
    
    # 图像预处理操作
    train_transform = PairedTransforms(resize = (256, 256), horizontal_flip = True, random_crop = True, color_jitter = False, rotate = True, max_rotation = 360)
    val_transform = PairedTransforms(resize = (256, 256), horizontal_flip = False, random_crop = False, color_jitter = False, rotate = False, max_rotation = 360)

    # 创建数据集实例
    train_dataset = DeblurDataset(blurred_dir, sharp_dir, unpaired_train_dir, transform=train_transform)

    val_dataset = DeblurDataset(val_blurred_dir, val_sharp_dir, unpaired_val_dir, transform=val_transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    return train_loader, val_loader

def denormalize(img):
    """
    将图像从 [-1, 1] 归一化到 [0, 1]。
    """
    return (img + 1) / 2

#计算感知损失
def cul_perceptual_loss(real, fake, vgg):
    real_features = vgg(real)
    fake_features = vgg(fake)
    return F.mse_loss(real_features, fake_features)

# 验证函数
def validate(flow, generator, val_loader, vgg , f = 0):
    generator.eval()  # 切换到评估模式
    perceptual_loss_total = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    num_samples = 0

    with torch.no_grad():  # 禁用梯度计算
        for i, (blurred_images, sharp_images, unpaired_images) in enumerate(val_loader):
            blurred_images = blurred_images.cuda()
            sharp_images = sharp_images.cuda()

            # 生成清晰图像
            blur_kernel = flow(blurred_images)
            generated_images = generator(blurred_images, blur_kernel)[0]

            # 计算感知损失
            perceptual_loss = cul_perceptual_loss(sharp_images, generated_images, vgg)
            perceptual_loss_total += perceptual_loss.item()


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

    # 计算平均 PSNR 和 SSIM
    avg_psnr /= num_samples
    avg_ssim /= num_samples


    return perceptual_loss_total / len(val_loader), avg_psnr, avg_ssim

def apply_blur(sharp_images, blur_kernels):

    B, C, H, W = sharp_images.shape
    
    # 重塑核为分组卷积格式 [B*3,1,3,3]
    grouped_kernels = blur_kernels.view(B*3, 1, 3, 3)
    
    # 分组卷积参数
    groups = B*3  # 每个样本的每个通道独立处理
    
    # 执行卷积
    blurred_images = F.conv2d(
        sharp_images.view(1, B*3, H, W),  # 输入重塑为 [1, B*3, H, W]
        grouped_kernels, 
        padding=1,
        groups=groups
    )
    
    # 恢复维度 [B,3,H,W]
    blurred_images = blurred_images.view(B, 3, H, W)
    return blurred_images

def train(model, blur_kernel_generator, recon_net, disc1, disc2, disc3, train_loader, val_loader, num_epochs):
    #初始化
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    model = model.to(device) #将模型放入GPU

    flow = blur_kernel_generator.to(device)

    disc1 = disc1.to(device)

    disc2 = disc2.to(device)

    disc3 = disc3.to(device)

    # deblur_net = deblur_net.to(device)

    recon_net = recon_net.to(device)

    vgg = VGGFeatureExtractor().cuda()
    
    optimizer_G = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(disc1.parameters(), lr=1e-4, betas=(0.5, 0.999)) #指定梯度下降方法

    optimizer_G1 = torch.optim.Adam(flow.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D1 = torch.optim.Adam(disc2.parameters(), lr=1e-4, betas=(0.5, 0.999)) 

    optimizer_G2 = torch.optim.Adam(recon_net.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D2 = torch.optim.Adam(disc3.parameters(), lr=1e-4, betas=(0.5, 0.999)) 

    # 定义余弦退火并带有重启的学习率调度器
    # T_max 是一个周期的步数，在 T_max 之后，学习率将退火到 eta_min
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=2000, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=2000, eta_min=1e-6)

    scheduler_G1 = CosineAnnealingLR(optimizer_G1, T_max=2000, eta_min=1e-6)
    scheduler_D1 = CosineAnnealingLR(optimizer_D1, T_max=2000, eta_min=1e-6)

    scheduler_G2 = CosineAnnealingLR(optimizer_G2, T_max=2000, eta_min=1e-6)
    scheduler_D2 = CosineAnnealingLR(optimizer_D2, T_max=2000, eta_min=1e-6)

    criterion = nn.BCELoss() #指定损失函数，此处采用交叉熵损失

    # 初始化最佳感知损失
    best_perceptual_loss = float("inf")

    best_psnr = 0.0

    best_ssim = 0.0

    former_psnr = best_psnr

    former_ssim = best_ssim

    e = 1

    since = time.time() #保存当前时间

    #训练（反向传播）

    for epoch in range(num_epochs):
        print("-"*50)
        print("Epoch {}/{}".format(epoch+1,num_epochs))


        #从数据中取一个batch来训练，直到所有batch取完，一轮训练完成
        for i, (blurred_images, sharp_images, unpaired_images) in enumerate(train_loader):
            #将图片、标签放入device
            blurred_images = blurred_images.to(device)
            sharp_images = sharp_images.to(device)
            unpaired_images = unpaired_images.to(device)

            save_image(blurred_images, './data/ground_truth/1.png')
            save_image(sharp_images, './data/ground_truth/2.png')
            save_image(unpaired_images, './data/ground_truth/3.png')

            batch_size = blurred_images.size(0)
            real_labels = torch.ones(batch_size, 1).cuda()   # 真实标签为1
            fake_labels = torch.zeros(batch_size, 1).cuda()  # 假标签为0

            # 调整目标标签的形状，使其与 disc 的输出相同
            real_labels = real_labels.view(2, 1, 1, 1)  # 调整为 [batch_size, 1, 1, 1]
            real_labels = real_labels.expand(2, 256, 256, 1)  # 扩展为 [batch_size, 256, 256, 1]

            fake_labels = fake_labels.view(2, 1, 1, 1)  # 调整为 [batch_size, 1, 1, 1]
            fake_labels = fake_labels.expand(2, 256, 256, 1)  # 扩展为 [batch_size, 256, 256, 1]

            #设置模型为训练模式
            model.train()
            
            #训练判别器
            optimizer_D.zero_grad() #梯度清零
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()

            # 对真实图像的判别
            real_outputs = disc1(sharp_images)
            real_loss = criterion(real_outputs, real_labels)

            real_outputs1 = disc2(blurred_images)
            real_loss1 = criterion(real_outputs1, real_labels)

            real_outputs2 = disc3(sharp_images)
            real_loss2 = criterion(real_outputs2, real_labels)

            # 对生成图像的判别
            blur_kernel = flow(blurred_images)

            fake_images, d3, d2, d1, blur_kernel = model(blurred_images, blur_kernel)  # 生成清晰图像  前向传播，输出为一个batch
            fake_outputs = disc1(fake_images.detach())   # 注：detach使得生成器不参与判别器的梯度计算
            fake_loss = criterion(fake_outputs, fake_labels)

            reblurred_images = apply_blur(unpaired_images, blur_kernel)

            fake_outputs1 = disc2(reblurred_images.detach()) 
            fake_loss1 = criterion(fake_outputs1, fake_labels)

            fake_images2 = recon_net(reblurred_images, blur_kernel)
            fake_outputs2 = disc3(fake_images2.detach())
            fake_loss2 = criterion(fake_outputs2, fake_labels)

            save_image(blur_kernel, './data/ground_truth/4.png')
            save_image(reblurred_images, './data/ground_truth/5.png')
            save_image(fake_images2, './data/ground_truth/6.png')
            save_image(d1, './data/ground_truth/7.png')
            save_image(d2, './data/ground_truth/8.png')
            save_image(d3, './data/ground_truth/9.png')
            save_image(fake_images, './data/ground_truth/10.png', normalize=True)

            # 判别器的总损失
            D_loss = real_loss + fake_loss
            D_loss.backward()  #反向传播
            optimizer_D.step() #更新参数

            D_loss1 = real_loss1 + fake_loss1
            D_loss1.backward()
            optimizer_D1.step()

            D_loss2 = real_loss2 + fake_loss2
            D_loss2.backward()
            optimizer_D2.step()

            #训练生成器
            optimizer_G.zero_grad()
            optimizer_G1.zero_grad()
            optimizer_G2.zero_grad()

            # 生成图像并通过判别器，目标是让判别器认为生成图像是真实的
            fake_outputs = disc1(fake_images)
            adversarial_loss = criterion(fake_outputs, real_labels)  # 生成器希望判别器输出1

            fake_outputs1 = disc2(reblurred_images)
            adversarial_loss1 = criterion(fake_outputs1, real_labels)

            fake_outputs2 = disc3(fake_images2)
            adversarial_loss2 = criterion(fake_outputs2, real_labels)

            # 计算感知损失
            perceptual_loss_value = cul_perceptual_loss(sharp_images, fake_images, vgg)

            # perceptual_loss_value1 = cul_perceptual_loss(blurred_images, reblurred_images, vgg)

            perceptual_loss_value2 = cul_perceptual_loss(unpaired_images, fake_images2, vgg)

            # 生成器的总损失 = 感知损失 + 对抗损失
            G_loss = perceptual_loss_value + adversarial_loss
            
            G_loss1 = adversarial_loss1

            G_loss2 = perceptual_loss_value2 + adversarial_loss2

            total_loss = G_loss + G_loss1 + G_loss2
            total_loss.backward()
            optimizer_G.step()
            optimizer_G1.step()
            optimizer_G2.step()

            i += 1

             # 打印训练进度
            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i}/{len(train_dataloader)} "
                  f"Loss D: {D_loss.item():.4f}, Loss G: {G_loss.item():.4f}, "
                  f"Loss D1: {D_loss1.item():.4f}, Loss G1: {G_loss1.item():.4f}, "
                  f"Loss D2: {D_loss2.item():.4f}, Loss G2: {G_loss2.item():.4f}, "
                  f"Perceptual Loss: {perceptual_loss_value.item():.4f}," 
                  f"Perceptual Loss2: {perceptual_loss_value2.item():.4f}" )

        # 在每个 epoch 之后更新学习率
        scheduler_G.step(epoch + i / len(train_loader))
        scheduler_D.step(epoch + i / len(train_loader))

        scheduler_G1.step(epoch + i / len(train_loader))
        scheduler_D1.step(epoch + i / len(train_loader))

        scheduler_G2.step(epoch + i / len(train_loader))
        scheduler_D2.step(epoch + i / len(train_loader))
           
        # 每100个epoch保存一次生成的图像
        if epoch % 100 == 0:
            save_image(fake_images[:2], f"./outputs/fake_images_epoch_{epoch}.png", nrow=2, normalize=True)
            save_image(sharp_images[:2], f"./outputs/real_images_epoch_{epoch}.png", nrow=2, normalize=True)

        #验证
        val_perceptual_loss, avg_psnr, avg_ssim = validate(flow, model, val_loader, vgg)
        if avg_psnr > best_psnr:
            former_psnr = best_psnr
            best_psnr = avg_psnr
            e = epoch
        if avg_ssim > best_ssim:
            former_ssim = best_ssim
            best_ssim = avg_ssim
            e = epoch
        
        print(f"Valiation loss:{val_perceptual_loss:.4f}")
        print(f"Validation Results - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
        print(f"best_psnr: {best_psnr:.4f}, best_ssim: {best_ssim:.4f}, epoch: {e + 1}")

        # 如果PSNR变高，保存模型
        if best_psnr == avg_psnr:
           print(f"PSNR improved from {former_psnr:.4f} to {best_psnr:.4f}. Saving model...")
           best_perceptual_loss = val_perceptual_loss
           save_checkpoint(model, disc1, epoch, best_perceptual_loss)
           save_checkpoint(flow, disc2, epoch, best_perceptual_loss, checkpoint_dir="./checkpoints1/")
        
        #计算耗时
        time_use = time.time() - since
        print("训练和验证耗时：{:.0f}min{:.0f}s".format(time_use // 60, time_use % 60))

if __name__ == "__main__":
    #将模型实例化
    generator = model(num_resblocks = [1, 2, 3, 4], input_channels = [3, 6, 6, 6])
    #generator.load_state_dict(torch.load('./best_generator_model.pth'))

    # 正则化流模型
    blur_kernel_generator = BlurKernelFlow()

    # recon_net = BlurReconstructionNetwork(input_channels = 3)
    recon_net = DeblurNetwork(input_channels = 3)

    disc1 = discriminator(input_dim = 3)
    #disc.load_state_dict(torch.load('./best_discriminator_model.pth'))
    disc2 = discriminator(input_dim = 3)

    disc3 = discriminator(input_dim = 3)

    #加载数据集
    train_dataloader, val_dataloader = data_process()
    #训练
    train(generator, blur_kernel_generator, recon_net, disc1, disc2, disc3, train_dataloader, val_dataloader, num_epochs=2000)