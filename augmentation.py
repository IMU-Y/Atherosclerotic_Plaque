import os
import numpy as np
from skimage import transform
from skimage import io
import cv2

# 数据增强 翻转，旋转角度
if __name__ == '__main__':
    root_dir = 'dataset'

    # train_images_set = np.load(os.path.join(root_dir, 'images', 'train', 'dcm.npy'))
    # train_gts_set = np.load(os.path.join(root_dir, 'gt', 'train', 'nii.npy'))

    val_images_set = np.load(os.path.join(root_dir, 'images', 'val', 'dcm.npy'))
    val_gts_set = np.load(os.path.join(root_dir, 'gt', 'val', 'nii.npy'))

    orients = [0, 90, 180, 270]
    flips = [0, 1, 2]

    # aug_dir = os.path.join(root_dir, 'aug_data')
    # file = open(root_dir + '/train_pair.lst', 'w')

    aug_dir_val = os.path.join(root_dir, 'aug_data_val')
    file_val = open(root_dir + '/val_pair.lst', 'w')

    for i in range(val_images_set.shape[0]):
        img = val_images_set[i].astype(np.uint8)

        # gt = train_gts_set[i].astype(np.uint8) * 255
        # 没有乘255
        gt = val_gts_set[i].astype(np.uint8)

        # H, W = img.shape[0], img.shape[1]
        for o in orients:
            # 逆时针旋转
            img_o = transform.rotate(img, o)
            gt_o = transform.rotate(gt, o)

            for f in flips:
                if f == 0:
                    # 不翻转
                    img_o_f = img_o
                    gt_o_f = gt_o
                elif f == 1 and (o == 90 or o == 0):
                    # 左右翻转
                    img_o_f = np.fliplr(img_o)
                    gt_o_f = np.fliplr(gt_o)
                elif f == 2 and (o == 90 or o == 0):
                    # 上下翻转
                    img_o_f = np.flipud(img_o)
                    gt_o_f = np.flipud(gt_o)
                else:
                    continue

                img_save_dir = os.path.join(aug_dir_val, 'images', 'o', str(o), 'f', str(f))
                gt_save_dir = os.path.join(aug_dir_val, 'gt', 'o', str(o), 'f', str(f))
                print(img_save_dir)
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                if not os.path.exists(gt_save_dir):
                    os.makedirs(gt_save_dir)

                # cv2.imwrite(os.path.join(img_save_dir, '{}.png'.format(str(i))), img_o_f)
                # cv2.imwrite(os.path.join(gt_save_dir, '{}.png'.format(str(i))), gt_o_f)

                io.imsave(os.path.join(img_save_dir, '{}.png'.format(str(i))), img_o_f)
                # # print('gt max:{}. min:{}'.format(np.max(gt_o_f), np.min(gt_o_f)))
                io.imsave(os.path.join(gt_save_dir, '{}.png'.format(str(i))), gt_o_f)
                file_val.write(os.path.join('aug_data_val', 'images', 'o', str(o), 'f', str(f),
                                        '{}.png'.format(str(i))) + ' ' + os.path.join('aug_data_val', 'gt', 'o', str(o),
                                                                                      'f', str(f),
                                                                                      '{}.png\n'.format(str(i))))
    file_val.close()

    # for i in range(train_images_set.shape[0]):
    #     img = train_images_set[i].astype(np.uint8)
    #
    #     # gt = train_gts_set[i].astype(np.uint8) * 255
    #     # 没有乘255
    #     gt = train_gts_set[i].astype(np.uint8)
    #
    #     # H, W = img.shape[0], img.shape[1]
    #     for o in orients:
    #         # 逆时针旋转
    #         img_o = transform.rotate(img, o)
    #         gt_o = transform.rotate(gt, o)
    #
    #         for f in flips:
    #             if f == 0:
    #                 # 不翻转
    #                 img_o_f = img_o
    #                 gt_o_f = gt_o
    #             elif f == 1 and (o == 90 or o == 0):
    #                 # 左右翻转
    #                 img_o_f = np.fliplr(img_o)
    #                 gt_o_f = np.fliplr(gt_o)
    #             elif f == 2 and (o == 90 or o == 0):
    #                 # 上下翻转
    #                 img_o_f = np.flipud(img_o)
    #                 gt_o_f = np.flipud(gt_o)
    #             else:
    #                 continue
    #
    #             img_save_dir = os.path.join(aug_dir, 'images', 'o', str(o), 'f', str(f))
    #             gt_save_dir = os.path.join(aug_dir, 'gt', 'o', str(o), 'f', str(f))
    #             print(img_save_dir)
    #             if not os.path.exists(img_save_dir):
    #                 os.makedirs(img_save_dir)
    #             if not os.path.exists(gt_save_dir):
    #                 os.makedirs(gt_save_dir)
    #
    #             # cv2.imwrite(os.path.join(img_save_dir, '{}.png'.format(str(i))), img_o_f)
    #             # cv2.imwrite(os.path.join(gt_save_dir, '{}.png'.format(str(i))), gt_o_f)
    #
    #             io.imsave(os.path.join(img_save_dir, '{}.png'.format(str(i))), img_o_f)
    #             # print('gt max:{}. min:{}'.format(np.max(gt_o_f), np.min(gt_o_f)))
    #             io.imsave(os.path.join(gt_save_dir, '{}.png'.format(str(i))), gt_o_f)
    #             file.write(os.path.join('aug_data', 'images', 'o', str(o), 'f', str(f),
    #                                     '{}.png'.format(str(i))) + ' ' + os.path.join('aug_data', 'gt', 'o', str(o),
    #                                                                                   'f', str(f),
    #                                                                                   '{}.png\n'.format(str(i))))
    # file.close()
    # end for
    # aug_images = np.concatenate(np.array(aug_images), axis=0)
    # aug_gts = np.concatenate(np.array(aug_gts), axis=0)
    # print(aug_images.shape)
    # print(aug_gts.shape)
    # np.savez(os.path.join(aug_dir, 'images', 'aug_dcm.npz'), aug_images)
    # np.savez(os.path.join(aug_dir, 'gt', 'aug_nii.npz'), aug_gts)
    print('finished')
