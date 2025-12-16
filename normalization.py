import numpy as np
import os
from PIL import Image

def normalize():
    floder_path = 'D:\desktop\study\code\GAN\data\train'

    #初始化累计变量
    total_pixels = 0
    sum_normalized_pixel_value = np.zeros(3)

    #遍历文件夹中的图片
    for root, dirs, files, in os.walk(floder_path):
        for filename in files:
            if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
               iamge_path = os.path.join(root, filename)
               image = Image.open(iamge_path)
               image_array = np.array(image)

            #归一化像素值到0-1之间
            normalized_image_array = image_array / 255.0

            #累计归一化后的像素值和像素数量
            total_pixels += normalized_image_array.size
            sum_normalized_pixel_value += np.sum(normalized_image_array, axis = (0,1))


#计算均值和方差
    mean = sum_normalized_pixel_value / total_pixels


    sum_squared_diff = np.zeros(3)
    for root, dirs, files, in os.walk(floder_path):
        for filename in files:
            if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
               iamge_path = os.path.join(root, filename)
               image = Image.open(iamge_path)
               image_array = np.array(image)

               #归一化像素值到0-1之间
               normalized_image_array = image_array / 255.0

               try:
                   diff = (normalized_image_array - mean) ** 2
                   sum_squared_diff += np.sum(diff, axis = (0, 1))
               except:
                   print("捕获到自定义异常")

               
    variance = sum_squared_diff / total_pixels

    print('mean:', mean)
    print('variance:', variance)

    return mean, variance

                