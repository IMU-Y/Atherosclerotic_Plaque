import cv2
import numpy as np
import torch

class CombinedFilter:
    """组合滤波器：高斯+非局部均值(NLM)"""
    
    def __init__(self, 
                 gaussian_kernel_size=3,
                 gaussian_sigma=0.3,
                 nlm_h=7,
                 nlm_template_size=5,
                 nlm_search_size=15):
        """
        参数:
            gaussian_kernel_size: 高斯核大小
            gaussian_sigma: 高斯标准差
            nlm_h: NLM滤波强度
            nlm_template_size: NLM模板窗口大小
            nlm_search_size: NLM搜索窗口大小
        """
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.nlm_h = nlm_h
        self.nlm_template_size = nlm_template_size
        self.nlm_search_size = nlm_search_size
    
    def __call__(self, image):
        """
        应用组合滤波
        
        Args:
            image: 输入图像 (H,W) 或 (H,W,C)
        Returns:
            filtered_image: 滤波后的图像
        """
        # 确保图像是uint8类型
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # 处理单通道或多通道
        if len(image.shape) == 2:
            return self._filter_single_channel(image)
        else:
            filtered = np.zeros_like(image)
            for c in range(image.shape[2]):
                filtered[:,:,c] = self._filter_single_channel(image[:,:,c])
            return filtered
            
    def _filter_single_channel(self, image):
        """对单通道图像进行滤波"""
        # 1. 高斯滤波
        gaussian = cv2.GaussianBlur(image, 
                                  (self.gaussian_kernel_size, self.gaussian_kernel_size),
                                  self.gaussian_sigma)
        
        # 2. 非局部均值滤波
        nlm = cv2.fastNlMeansDenoising(gaussian,
                                      h=self.nlm_h,
                                      templateWindowSize=self.nlm_template_size,
                                      searchWindowSize=self.nlm_search_size)
        
        return nlm 