import os
import cv2
import numpy as np
from PIL import Image

def scale_first_label():
    # 设置输入和输出路径
    input_dir = "dataset/aug_data_val/gt/o/0/f/0"  # 修改为你的gt文件夹路径
    output_dir = "dataset/gt_scaled"  # 修改为你想保存的路径
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f == "0.png"]  # 只选择0.png
    print(image_files,'djy')
    if not image_files:
        print("0.png not found in the input directory!")
        return
    
    # 获取图像文件
    first_image = image_files[0]  # 由于列表中只有一个文件，使用索引0
    img_path = os.path.join(input_dir, first_image)
    print(img_path,'img_path')
    # 读取图像
    img = np.array(Image.open(img_path))
    
    # 打印原始图像信息
    print(f"Original image shape: {img.shape}")
    print(f"Original value range: {img.min()} - {img.max()}")
    print(f"Original unique values: {np.unique(img)}")
    
    # 方法1：线性缩放 (0-4 -> 0-255)
    scaled_img = (img * 63.75).astype(np.uint8)  # 63.75 = 255/4
    
    # 方法2：按类别映射到特定值（取消注释使用这种方法）
    # mapped_img = np.zeros_like(img, dtype=np.uint8)
    # mapped_img[img == 0] = 0    # 背景
    # mapped_img[img == 1] = 63   # 正常斑块
    # mapped_img[img == 2] = 127  # 纤维斑块
    # mapped_img[img == 3] = 191  # 脂质斑块
    # mapped_img[img == 4] = 255  # 钙化斑块
    
    # 打印缩放后的图像信息
    print(f"\nScaled value range: {scaled_img.min()} - {scaled_img.max()}")
    print(f"Scaled unique values: {np.unique(scaled_img)}")
    
    # 保存结果
    output_file = os.path.join(output_dir, f"scaled_{first_image}")
    cv2.imwrite(output_file, scaled_img)
    print(f"\nScaled image saved to: {output_file}")

if __name__ == "__main__":
    scale_first_label() 