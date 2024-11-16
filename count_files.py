from pathlib import Path
import os

def count_files():
    # 统计验证集
    val_pair_file = Path('dataset/val_pair.lst')
    if val_pair_file.exists():
        with open(val_pair_file, 'r') as f:
            val_count = len(f.readlines())
        print(f"验证集增强后的图片数量: {val_count}")
    
    # 统计训练集
    train_pair_file = Path('dataset/train_pair.lst')
    if train_pair_file.exists():
        with open(train_pair_file, 'r') as f:
            train_count = len(f.readlines())
        print(f"训练集增强后的图片数量: {train_count}")
    
    # 统计实际文件数量
    val_dir = Path('dataset/aug_data_val')
    train_dir = Path('dataset/aug_data')
    
    if val_dir.exists():
        val_files = len(list(val_dir.rglob('*.png'))) // 2  # 除以2因为每对包含图像和标注
        print(f"验证集实际文件数量: {val_files}")
    
    if train_dir.exists():
        train_files = len(list(train_dir.rglob('*.png'))) // 2
        print(f"训练集实际文件数量: {train_files}")

if __name__ == '__main__':
    count_files()