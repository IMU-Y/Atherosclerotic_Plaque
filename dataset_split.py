import os
import numpy as np
import pydicom
import nibabel as nib

# 分割数据集 15个序列
dcm_path = ['data/cheng hailong 785592', 'data/dong dongwang 802318', 'data/gao changyang 803197',
            'data/guo liwei 792609', 'data/li man chuan 854999', 'data/liu gang 737210',
            'data/lu yaofeng 784251', 'data/lv zhimin 792504', 'data/qi yafang 734528',
            'data/ren jianxiong 759005', 'data/zhang xiaoning 787250', 'data/zhao anhong 804329',
            'data/zhao cui ying 792986', 'data/zhou chunlai 798415', 'data/zuo butian 809461']
nii_path = ['data/Cheng Hailong 785592.nii.gz', 'data/Dong Dongwang 802318.nii.gz', 'data/Gao Chaoyang 803197.nii.gz',
            'data/Guo Liwei 792609.nii.gz', 'data/Li Manchuan 854999.nii.gz', 'data/Liu Gang 737210.nii.gz',
            'data/Lu Yaofeng 784251.nii.gz', 'data/Lv Zhimin 792504.nii.gz', 'data/Qi Yafang 734528.nii.gz',
            'data/Ren Jianxioing 759005.nii.gz', 'data/Zhang Xiaoning 787250.nii.gz', 'data/Zhao Anhong 804329.nii.gz',
            'data/Zhao Cuiying 792986.nii.gz', 'data/Zhou Chunlai 798415.nii.gz', 'data/Zuo Butian 809461.nii.gz']
output_path = 'dataset/'
# output_path = 'fine_annotation_contour_dataset'
test_imgs_num = 1500


# 裁剪370*370
def crop_data(my_dcm, my_nii):
    my_dcm = my_dcm[:, 0: 384, 150: 534, :]  # (N, 370, 370, C)
    my_nii = my_nii[:, 0: 384, 150: 534]

    return my_dcm, my_nii


if __name__ == '__main__':
    # 创建临时目录
    temp_dir = 'temp_processing'
    os.makedirs(temp_dir, exist_ok=True)
    
    # 分批处理数据
    batch_size = 100  # 每次处理100张图片
    dcm_data = []
    nii_data = []
    
    for i in range(0, 15):
        try:
            print(f"处理第 {i+1} 个文件:")
            my_dcm = pydicom.dcmread(os.path.join(dcm_path[i])).pixel_array
            my_nii = nib.load(os.path.join(nii_path[i])).get_fdata().transpose((2, 0, 1))
            
            print(f"DCM shape: {my_dcm.shape}")
            print(f"NII shape: {my_nii.shape}")
            
            if my_dcm.shape[0] != my_nii.shape[0]:
                print(f"警告：文件 {i+1} 的 DCM 和 NII 维度不匹配")
                continue
            
            # 转置处理
            for j in range(my_nii.shape[0]):
                my_nii[j] = my_nii[j].transpose((1, 0))
            
            # 裁剪
            my_dcm, my_nii = crop_data(my_dcm, my_nii)
            
            # 分批保存到临时文件
            for k in range(0, my_dcm.shape[0], batch_size):
                batch_end = min(k + batch_size, my_dcm.shape[0])
                np.save(os.path.join(temp_dir, f'dcm_{i}_{k}.npy'), my_dcm[k:batch_end])
                np.save(os.path.join(temp_dir, f'nii_{i}_{k}.npy'), my_nii[k:batch_end])
                
            # 记录样本数
            dcm_data.extend([os.path.join(temp_dir, f'dcm_{i}_{k}.npy') 
                           for k in range(0, my_dcm.shape[0], batch_size)])
            nii_data.extend([os.path.join(temp_dir, f'nii_{i}_{k}.npy') 
                           for k in range(0, my_dcm.shape[0], batch_size)])
            
            # 清理内存
            del my_dcm
            del my_nii
            
        except Exception as e:
            print(f"处理文件 {i+1} 时出错: {str(e)}")
            continue
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'gt', split), exist_ok=True)
    
    # 计算总样本数并生成随机索引
    total_samples = sum([np.load(f).shape[0] for f in dcm_data])
    np.random.seed(1024)
    random_index = np.random.permutation(total_samples)
    
    # 分配数据集
    train_data_dcm = []
    val_data_dcm = []
    test_data_dcm = []
    train_data_nii = []
    val_data_nii = []
    test_data_nii = []
    
    current_idx = 0
    for dcm_file, nii_file in zip(dcm_data, nii_data):
        batch_dcm = np.load(dcm_file)
        batch_nii = np.load(nii_file)
        
        for i in range(len(batch_dcm)):
            if current_idx in random_index[test_imgs_num + 450:]:
                train_data_dcm.append(batch_dcm[i])
                train_data_nii.append(batch_nii[i])
            elif current_idx in random_index[test_imgs_num:test_imgs_num + 450]:
                val_data_dcm.append(batch_dcm[i])
                val_data_nii.append(batch_nii[i])
            else:
                test_data_dcm.append(batch_dcm[i])
                test_data_nii.append(batch_nii[i])
            current_idx += 1
    
    # 保存最终数据集
    np.save(os.path.join(output_path, 'images', 'train', 'dcm'), np.array(train_data_dcm))
    np.save(os.path.join(output_path, 'images', 'val', 'dcm'), np.array(val_data_dcm))
    np.save(os.path.join(output_path, 'images', 'test', 'dcm'), np.array(test_data_dcm))
    np.save(os.path.join(output_path, 'gt', 'train', 'nii'), np.array(train_data_nii))
    np.save(os.path.join(output_path, 'gt', 'val', 'nii'), np.array(val_data_nii))
    np.save(os.path.join(output_path, 'gt', 'test', 'nii'), np.array(test_data_nii))
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)
    
    print('finished')
