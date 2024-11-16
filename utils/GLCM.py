import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
import os
from sklearn import preprocessing
from skimage import feature as skft

root_path = "E:/3d_reconstruction/oct/fine_annotation_plaque_dataset/data_upload"
def GLCM_Feature(patch):
    # feature = []
    # image = img_as_ubyte(patch)#变成8位无符号整型

    #8位图像含有256个灰度级，导致计算灰度共生矩阵计算量过大，因此将其进行压缩成16级，将灰度进行分区
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    # 区间化表示，小于0为0,0-16为1，以此类推
    inds = np.digitize(patch, bins)

    max_value = inds.max()+1
    matrix_coocurrence = greycomatrix(inds, #需要进行共生矩阵计算的numpy矩阵
                                  [1],#步长
                                  [0, np.pi/4, np.pi/2, 3*np.pi/4],#方向角度
                                  levels=max_value, #共生矩阵阶数
                                  normed=False, symmetric=False)
    #P[i,j,d,theta]返回的是一个四维矩阵，各维代表不同的意义

    contrast_feature = greycoprops(matrix_coocurrence, 'contrast')
    correlation_feature = greycoprops(matrix_coocurrence, 'correlation')
    energy_feature = greycoprops(matrix_coocurrence, 'energy')
    homogeneity_feature = greycoprops(matrix_coocurrence, 'homogeneity')
    #计算熵entropy
    feature = contrast_feature[0].tolist()
    feature +=correlation_feature[0].tolist()
    feature += energy_feature[0].tolist()
    feature += homogeneity_feature[0].tolist()
    return feature

#获取一副图像的GLCM feature
def getFeaturePerImage(image):#从神经网络那边传过来
    # print(img_path)#aug_data/images/o/0/f/0/0.png
    # feature的存储路径改为 aug_data/features/o/0/f/0/0.png
    # 转换成灰度图像
    print("====")
    print(image.shape)
    rgb = image.transpose(1,2,0)
    # print(rgb.shape)
    gray_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    features = []
    for m in range(gray_image.shape[0]):
        for n in range(gray_image.shape[1]):
            # 获取11*11的window,需要考虑边缘部分
            x_start = m - 5 if m - 5 >= 0 else 0
            y_start = n - 5 if n - 5 >= 0 else 0
            patch = gray_image[x_start:x_start + 11, y_start:y_start + 11]
            # 计算GLCM,共16个
            glcm_feature = GLCM_Feature(patch)
            features.append(glcm_feature)
    # 在415*415个像素点中，执行归一化操作
    # 计算原始数据每行和每列的均值和方差，data是多维数据
    scaler_feature = preprocessing.StandardScaler().fit(features)
    # 标准化数据
    features_norm = scaler_feature.transform(features)
    feature_data = np.array(features_norm).reshape((415, 415, 16,1)).transpose(3,2, 0, 1)

    return feature_data



