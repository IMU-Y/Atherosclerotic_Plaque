import cv2
import numpy as np
from PIL import Image

def canny(mask):
    """
    使用 Canny 算法从分割掩码生成边界标签
    Args:
        mask: 二值分割掩码图像
    Returns:
        boundary: 边界标签 (0 或 1)
    """
    # 确保输入是 uint8 类型
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8) * 255
    else:
        mask = mask.astype(np.uint8) * 255
        
    # 使用 Canny 算法检测边缘
    # 注意：这里不需要设置阈值，cv2.Canny 会自动计算合适的阈值
    edges = cv2.Canny(mask, threshold1=0, threshold2=0)
    
    # 转换为二值图像
    edges = (edges > 0).astype(np.uint8)
    
    return edges

# 批量处理整个数据集的示例代码

def generate_edge_labels_canny(mask_dir, edge_dir):
    """
    为整个数据集生成边界标签
    """
    os.makedirs(edge_dir, exist_ok=True)
    
    for filename in os.listdir(mask_dir):
        if filename.endswith(('.png', '.jpg')):
            # 读取掩码
            mask_path = os.path.join(mask_dir, filename)
            mask = np.array(Image.open(mask_path))
            
            # 生成边界标签
            edge = get_boundary_label_canny(mask)
            
            # 保存边界标签
            edge_path = os.path.join(edge_dir, filename)
            Image.fromarray(edge * 255).save(edge_path)
            
            print(f"Processed {filename}")

# 使用示例
mask_dir = "dataset/masks"
edge_dir = "dataset/edges"
generate_edge_labels_canny(mask_dir, edge_dir)
