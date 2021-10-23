from psenet_ic15 import PSENET_IC15
import numpy as np
import cv2
import random
import os
import paddle
import sys

custom_dataset = PSENET_IC15(
    split='train',
    is_transform=True,
    img_size=736,
    short_size=736,
    kernel_num=7,
    min_scale=0.4,
    read_type='cv2')

train_loader = paddle.io.DataLoader(
    custom_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
)


def to_rgb(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2) * 255
    return img


def save(img_path, imgs):
    if not os.path.exists('vis/'):
        os.makedirs('vis/')

    for i in range(len(imgs)):
        imgs[i] = cv2.copyMakeBorder(imgs[i], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
    res = np.concatenate(imgs, axis=1)
    if type(img_path) != str:
        img_name = img_path[0].split('/')[-1]
    else:
        img_name = img_path.split('/')[-1]
    print('saved %s.' % img_name)
    cv2.imwrite('vis/' + img_name, res)


for batch_idx, data in enumerate(train_loader):
    if batch_idx > 100:
        break
    imgs = data[0].numpy()
    gt_texts = data[1].numpy()
    gt_kernels = data[2].numpy()
    training_masks = data[3].numpy()

    # print(type(imgs), type(gt_texts), type(gt_kernels), type(training_masks))
    # sys.exit()
    # image_name = data_loader.img_paths[batch_idx].split('/')[-1].split('.')[0]

    # print('%d/%d %s'%(batch_idx, len(train_loader), data_loader.img_paths[batch_idx]))
    print('%d/%d' % (batch_idx, len(train_loader)))

    img = imgs[0]
    img = ((img * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) +
            np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))[:, :, ::-1].copy()

    gt_text = to_rgb(gt_texts[0])
    gt_kernel_0 = to_rgb(gt_kernels[0, 0])
    gt_kernel_1 = to_rgb(gt_kernels[0, 1])
    gt_kernel_2 = to_rgb(gt_kernels[0, 2])
    gt_kernel_3 = to_rgb(gt_kernels[0, 3])
    gt_kernel_4 = to_rgb(gt_kernels[0, 4])
    gt_kernel_5 = to_rgb(gt_kernels[0, 5])
    gt_text_mask = to_rgb(training_masks[0].astype(np.uint8))

    save('%d.png' % batch_idx,
         [img, gt_text, gt_kernel_0, gt_kernel_1, gt_kernel_2, gt_kernel_3, gt_kernel_4, gt_kernel_5, gt_text_mask])
