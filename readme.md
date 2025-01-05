# 环境配置

新建一个虚拟环境：python3 -m venv myenv
激活虚拟环境：启动代码时需要先进入虚拟环境：source myenv/bin/activate

# 文件说明：

1. dataset_split.py 为分割数据集，将原始的DICOM和NIfTI格式的医学图像数据分割成训练集、验证集和测试集
2. augmentation.py 和 augmentation_train.py 是同一个数据增强脚本的两个版本

- augmentation_train.py处理训练集(training set)数据
- augmentation.py 处理验证集(validation set)数据
- 数据增强后的文件会保存在dataset/aug_data_val和dataset/aug_data中
- count_files.py 为统计数据增强后的文件数量：原始图片5624/32992，训练集原始3674/增强后29392，测试集原始1500/增强后12000，验证集原始450/增强后3600

3. train_plaque_pause.py 为训练和验证代码
4. test_plaque.py 为测试代码
5. model 为网络模型文件夹
6. utils 为工具包 如加载数据集、损失函数
7. dataset.zip 为数据集，包含15个病人的oct序列，共5624张图像，解压后将数据放在一个新建的data文件夹中

说明：可以实验一下编码器解码器的分割网络等，如FCN等

# 训练和测试代码

python train_plaque_pause.py --gpu 0 --net MambaUNet
python test_plaque.py --gpu 0 --model products/MambaUNet_plaque_lr_0.05_epoch_30.pth

pip install mamba-ssm--no-build-isolation
