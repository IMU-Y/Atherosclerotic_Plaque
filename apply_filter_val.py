import os
import cv2
import numpy as np
from tqdm import tqdm
from utils.image_preprocessing import CombinedFilter
from PIL import Image

# 批量处理验证集图像，应用滤波器，并输出到dataset/aug_data_val/filtered_images
def get_all_images(directory):
    """递归获取目录下所有图片文件的路径"""
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # 获取完整路径
                full_path = os.path.join(root, file)
                # 获取相对于input_dir的相对路径
                image_files.append(full_path)
    return image_files

def process_images():
    # 定义路径
    input_dir = "dataset/aug_data_val/images"
    output_dir = "dataset/aug_data_val/filtered_images"
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在: {input_dir}")
        return
        
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 递归获取所有图片文件
    image_files = get_all_images(input_dir)
    
    if not image_files:
        print("未找到任何图片文件！")
        print("支持的格式：.png, .jpg, .jpeg, .bmp, .tiff")
        return
        
    print(f"找到 {len(image_files)} 张图片需要处理")
    
    # 统计处理成功和失败的数量
    success_count = 0
    fail_count = 0
    error_log = []
    
    # 初始化滤波器
    filter = CombinedFilter(
        gaussian_kernel_size=5,
        gaussian_sigma=0.5,
        nlm_h=10,
        nlm_template_size=7,
        nlm_search_size=21
    )
    
    # 使用tqdm显示处理进度
    for img_path in tqdm(image_files, desc="处理图片"):
        try:
            # 读取图片
            img = cv2.imread(img_path)
            
            if img is None:
                error_msg = f"无法读取图片: {img_path}"
                print(error_msg)
                error_log.append(error_msg)
                fail_count += 1
                continue
            
            # 应用滤波器
            filtered_img = filter(img)
            
            # 保持原有的目录结构
            relative_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            
            # 确保输出文件的目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存处理后的图片
            save_result = cv2.imwrite(output_path, filtered_img)
            
            if save_result:
                success_count += 1
            else:
                error_msg = f"保存图片失败: {output_path}"
                print(error_msg)
                error_log.append(error_msg)
                fail_count += 1
            
        except Exception as e:
            error_msg = f"处理图片 {img_path} 时出错: {str(e)}"
            print(error_msg)
            error_log.append(error_msg)
            fail_count += 1
            continue
    
    # 打印处理统计信息
    print("\n处理完成!")
    print(f"总图片数: {len(image_files)}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {fail_count}")
    print(f"处理后的图片保存在: {output_dir}")
    
    # 如果有错误，保存错误日志
    if error_log:
        log_path = "filter_error.log"
        print(f"\n处理过程中出现了 {len(error_log)} 个错误")
        print(f"错误日志已保存到: {log_path}")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(error_log))
    
    return success_count, fail_count, error_log

def show_comparison(original_path, filtered_path):
    """显示原图和处理后的图片对比"""
    original = cv2.imread(original_path)
    filtered = cv2.imread(filtered_path)
    
    if original is None or filtered is None:
        print("无法读取图片进行对比")
        return
        
    # 水平拼接图片
    comparison = np.hstack((original, filtered))
    
    # 显示对比图
    cv2.imshow('Original vs Filtered', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 处理所有图片
    success_count, fail_count, error_log = process_images()
    
    # 如果有成功处理的图片，显示第一张对比
    if success_count > 0:
        input_dir = "dataset/aug_data_val/images"
        output_dir = "dataset/aug_data_val/filtered_images"
        
        image_files = get_all_images(input_dir)
        if image_files:
            first_image = image_files[0]
            filtered_path = first_image.replace(input_dir, output_dir)
            show_comparison(first_image, filtered_path)