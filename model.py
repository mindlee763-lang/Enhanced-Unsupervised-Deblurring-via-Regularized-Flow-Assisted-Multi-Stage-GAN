import torch
import torch.nn as nn
from torchvision.models import vgg16

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
    

class generator(nn.Module):
    def __init__(self, num_resblock, input_dim):
        super().__init__()
        # 输入通道数调整为 input_dim + 1（模糊核为单通道）
        self.layer1 = conv(input_dim + 1, 32, kernel_size=3)  # 修改输入通道数
        modules = []
        for i in range(num_resblock):
            modules.append(resblock(32, 32))
        self.layer2 = nn.Sequential(*modules)

        self.layer3 = conv(32, 64, kernel_size=3)
        modules = []
        for i in range(num_resblock):
            modules.append(resblock(64, 64))
        self.layer4 = nn.Sequential(*modules)

        self.layer5 = conv(64, 128, kernel_size=3)
        modules = []
        for i in range(num_resblock):
            modules.append(resblock(128, 128))
        self.layer6 = nn.Sequential(*modules)

        # deconv
        self.layer7 = deconv(128, 64, kernel_size=3)
        modules = []
        for i in range(num_resblock):
            modules.append(resblock(64, 64))
        self.layer8 = nn.Sequential(*modules)

        self.layer9 = deconv(64, 32, kernel_size=3)
        modules = []
        for i in range(num_resblock):
            modules.append(resblock(32, 32))
        self.layer10 = nn.Sequential(*modules)

        self.layer11 = conv(32, 3, kernel_size=3)

    def forward(self, x, f=0):
        x = self.layer1(x)
        x2 = self.layer2(x)

        x = self.layer3(x2)
        x4 = self.layer4(x)

        x = self.layer5(x4)
        x = x + f
        x6 = self.layer6(x)

        x = self.layer7(x6)
        x = x + x4
        x = self.layer8(x)

        x = self.layer9(x)
        x = x + x2
        x = self.layer10(x)

        x = self.layer11(x)

        return x, x6


class discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = conv(input_dim, 32 ,kernel_size = 3)
        self.layer2 = nn.LeakyReLU()
        self.layer3 = conv(32, 64, kernel_size = 3)
        self.layer4 = nn.LeakyReLU()
        self.layer5 = conv(64, 128, kernel_size = 3)
        self.layer6 = nn.LeakyReLU()
        self.layer7 = conv(128,256, kernel_size = 3)
        self.layer8 = nn.LeakyReLU()
        self.layer9 = nn.Linear(256, 1)
        self.layer10 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)

        return x


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        # 使用预训练的 VGG16 网络并冻结参数
        vgg = vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features[:16]))  # VGG16 前几层
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)
    

class model(nn.Module):
    def __init__(self, num_resblocks, input_channels, gen_input_channels=6):
        super().__init__()
        # 初始化生成器
        self.gen1 = generator(num_resblock=num_resblocks[0], input_dim=input_channels[0] + 2)  # 输入通道数 + 3
        self.gen2 = generator(num_resblock=num_resblocks[1], input_dim=input_channels[1] + 2)  # 输入通道数 + 3
        self.gen3 = generator(num_resblock=num_resblocks[2], input_dim=input_channels[2] + 2)  # 输入通道数 + 3
        self.gen4 = generator(num_resblock=num_resblocks[3], input_dim=input_channels[3] + 2)  # 输入通道数 + 3
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
        x_with_kernel = torch.cat([x, channel_features.expand(-1, -1, x.size(2), x.size(3))], dim=1)  # 在通道维度拼接
        d1, f1 = self.gen1(x_with_kernel)
        d2, f2 = self.gen2(torch.cat((x_with_kernel , d1), 1), f1)
        d3, f3 = self.gen3(torch.cat((x_with_kernel , d2), 1), f2)
        d4 = self.gen4(torch.cat((x_with_kernel , d3), 1),f3)[0]

        return d4, d3, d2, d1, blur_kernel