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
    dcm_data = []
    nii_data = []

    for i in range(0, 15):
        try:
            my_dcm = pydicom.dcmread(os.path.join(dcm_path[i])).pixel_array
            my_nii = nib.load(os.path.join(nii_path[i])).get_fdata().transpose((2, 0, 1))

            print(f"处理第 {i+1} 个文件:")
            print(f"DCM shape: {my_dcm.shape}")
            print(f"NII shape: {my_nii.shape}")

            # 确保数据维度匹配
            if my_dcm.shape[0] != my_nii.shape[0]:
                print(f"警告：文件 {i+1} 的 DCM 和 NII 维度不匹配")
                continue

            # my_nii.shape (375, 736, 736)
            for j in range(my_nii.shape[0]):
                # 需要转置
                my_nii[j] = my_nii[j].transpose((1, 0))  # (N,H,W)

            # 裁剪原图 370 * 370
            my_dcm, my_nii = crop_data(my_dcm, my_nii)

            # 将数据转换为列表形式添加
            for k in range(my_dcm.shape[0]):
              dcm_data.append(my_dcm[k])
              nii_data.append(my_nii[k])

        except Exception as e:
            print(f"处理文件 {i+1} 时出错: {str(e)}")
            continue

    # 将列表转换为numpy数组
    dcm_data = np.array(dcm_data)
    nii_data = np.array(nii_data)

    print(f"最终数据形状:")
    print(f"DCM shape: {dcm_data.shape}")
    print(f"NII shape: {nii_data.shape}")

    # 打乱数据
    np.random.seed(1024)
    # 对0到dcm_data.shape[0]的序列进行随机排序
    random_index = np.random.permutation(dcm_data.shape[0])
    # TODO 不打乱数据
    # random_index = np.arange(0, dcm_data.shape[0])

    # 训练集
    train_set_dcm = dcm_data[random_index[test_imgs_num + 450:]]
    train_set_nii = nii_data[random_index[test_imgs_num + 450:]]
    # 验证集
    val_set_dcm = dcm_data[random_index[test_imgs_num:test_imgs_num + 450]]
    val_set_nii = nii_data[random_index[test_imgs_num:test_imgs_num + 450]]
    # 测试集
    test_set_dcm = dcm_data[random_index[:test_imgs_num]]
    test_set_nii = nii_data[random_index[:test_imgs_num]]

    print('训练集 dcm:{}, nii:{}'.format(train_set_dcm.shape[0], train_set_nii.shape[0]))
    print('验证集 dcm:{}, nii:{}'.format(val_set_dcm.shape[0], val_set_nii.shape[0]))
    print('测试集 dcm:{}, nii:{}'.format(test_set_dcm.shape[0], test_set_nii.shape[0]))

    image_path = 'images'
    gt_path = 'gt'

    if not os.path.exists(os.path.join(output_path, image_path, 'train')):
        os.makedirs(os.path.join(output_path, image_path, 'train'))
    if not os.path.exists(os.path.join(output_path, image_path, 'val')):
        os.makedirs(os.path.join(output_path, image_path, 'val'))
    if not os.path.exists(os.path.join(output_path, image_path, 'test')):
        os.makedirs(os.path.join(output_path, image_path, 'test'))
    if not os.path.exists(os.path.join(output_path, gt_path, 'train')):
        os.makedirs(os.path.join(output_path, gt_path, 'train'))
    if not os.path.exists(os.path.join(output_path, gt_path, 'val')):
        os.makedirs(os.path.join(output_path, gt_path, 'val'))
    if not os.path.exists(os.path.join(output_path, gt_path, 'test')):
        os.makedirs(os.path.join(output_path, gt_path, 'test'))

    np.save(os.path.join(output_path, image_path, 'train', 'dcm'), train_set_dcm)
    np.save(os.path.join(output_path, image_path, 'val', 'dcm'), val_set_dcm)
    np.save(os.path.join(output_path, image_path, 'test', 'dcm'), test_set_dcm)
    np.save(os.path.join(output_path, gt_path, 'train', 'nii'), train_set_nii)
    np.save(os.path.join(output_path, gt_path, 'val', 'nii'), val_set_nii)
    np.save(os.path.join(output_path, gt_path, 'test', 'nii'), test_set_nii)
    print('finished')
