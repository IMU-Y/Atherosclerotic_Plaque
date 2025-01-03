import numpy as np
from skimage.morphology import binary_dilation, binary_erosion

def get_boundary_label(mask):
    """
    从分割掩码生成边界标签
    Args:
        mask: 二值分割掩码 (0 或 1)
    Returns:
        boundary: 边界标签 (0 或 1)
    """
    # 确保输入是二值图像
    mask = mask.astype(np.bool)
    
    # 膨胀操作
    dilated = binary_dilation(mask)
    # 腐蚀操作
    eroded = binary_erosion(mask)
    
    # 边界 = 膨胀 - 腐蚀
    boundary = np.logical_xor(dilated, eroded)
    
    return boundary.astype(np.uint8)

import os
from PIL import Image
import numpy as np

def generate_edge_labels(mask_dir, edge_dir):
    """
    处理5通道标签图片的边界生成，并保存原始边界图和缩放后的图片
    """
    os.makedirs(edge_dir, exist_ok=True)
    
    target_file = "0.png"
    if target_file in os.listdir(mask_dir):
        # 读取掩码
        mask_path = os.path.join(mask_dir, target_file)
        mask = np.array(Image.open(mask_path))
        
        print(f"原始图像形状: {mask.shape}")
        print(f"原始图像类型: {mask.dtype}")
        
        # 检查是否为5通道图片
        if len(mask.shape) == 3 and mask.shape[2] == 5:
            # 创建一个与输入相同大小的5通道输出数组
            edge_result = np.zeros_like(mask)
            scaled_edge_result = np.zeros_like(mask)
            
            # 分别处理每个通道
            for channel in range(5):
                channel_mask = mask[:, :, channel]
                # 二值化处理
                if channel_mask.max() > 1:
                    channel_mask = (channel_mask > 127).astype(np.uint8)
                
                # 生成该通道的边界
                edge = get_boundary_label(channel_mask)
                edge_result[:, :, channel] = edge * 255
                
                # 生成缩放后的边界
                if edge.max() > 0:  # 避免除以零
                    scaled_edge = (edge - edge.min()) / (edge.max() - edge.min()) * 255
                else:
                    scaled_edge = edge * 255
                scaled_edge_result[:, :, channel] = scaled_edge
            
            # 保存原始边界结果
            edge_path = os.path.join(edge_dir, "edge_" + target_file)
            Image.fromarray(edge_result.astype(np.uint8)).save(edge_path)
            print(f"已保存原始边界图片: {edge_path}")
            
            # 保存缩放后的边界结果
            scaled_edge_path = os.path.join(edge_dir, "scaled_edge_" + target_file)
            Image.fromarray(scaled_edge_result.astype(np.uint8)).save(scaled_edge_path)
            print(f"已保存缩放后的边界图片: {scaled_edge_path}")
            
        else:
            print(f"警告：输入不是5通道图片，实际shape为: {mask.shape}")
    else:
        print(f"未找到文件 {target_file}")

# 使用示例
mask_dir = "dataset/aug_data_val/gt/o/0/f/0"
edge_dir = "dataset/edges"
generate_edge_labels(mask_dir, edge_dir)


