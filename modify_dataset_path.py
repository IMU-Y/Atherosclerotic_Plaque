# 修改训练集和验证集的图像路径，将images改为filtered_images，输出为dataset/train_pair_filtered.lst
def modify_train_list():
    # 读取原始train_pair.lst
    with open('dataset/train_pair.lst', 'r') as f:
        lines = f.readlines()
    
    # 创建新的train_pair_filtered.lst
    with open('dataset/train_pair_filtered.lst', 'w') as f:
        for line in lines:
            img_path, gt_path = line.strip().split(' ')
            # 修改图像路径，将images改为filtered_images
            new_img_path = img_path.replace('images', 'filtered_images')
            # 写入新文件
            f.write(f"{new_img_path} {gt_path}\n")
    
    print("已创建新的训练集列表文件: dataset/train_pair_filtered.lst")

if __name__ == "__main__":
    modify_train_list() 