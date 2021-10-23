import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np
import cv2
import time
from ..loss import iou, build_loss, ohem_batch
from ..post_processing import pse
import sys
import os


class PSENet_Head(nn.Layer):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_classes,
                 loss_text,
                 loss_kernel):
        super(PSENet_Head, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(hidden_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2D(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0)

        self.text_loss = build_loss(loss_text)
        self.kernel_loss = build_loss(loss_kernel)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)

        return out

    def loss(self, out, gt_texts, gt_kernels, training_masks):
        # output
        texts = out[:, 0, :, :]
        kernels = out[:, 1:, :, :]
        # text loss
        # logging.info('text', texts)
        # logging.info('kernels', kernels)
        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        # logging.info('selected_masks', selected_masks)
        loss_text = self.text_loss(texts, gt_texts, selected_masks, reduce=False)
        # logging.info('loss_text', loss_text)
        iou_text = iou((texts > 0).cast('int32'), gt_texts, training_masks, reduce=False)
        # logging.info('iou_text', iou_text)
        losses = dict(
            loss_text=loss_text,
            iou_text=iou_text
        )

        # kernel loss
        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.shape[1]):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.kernel_loss(kernel_i, gt_kernel_i, selected_masks, reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = paddle.mean(paddle.stack(loss_kernels, axis=1), axis=1)
        # logging.info('loss_kernels', loss_kernels)
        iou_kernel = iou(
            (kernels[:, -1, :, :] > 0).cast('int32'), gt_kernels[:, -1, :, :], selected_masks, reduce=False)
        # logging.info('iou_kernel', iou_kernel)
        losses.update(dict(
            loss_kernels=loss_kernels,
            iou_kernel=iou_kernel
        ))

        return losses

    def get_results(self, out, img_meta, cfg):
        outputs = dict()
        # print(out[0,0,:,:])

        score = F.sigmoid(out[:, 0, :, :]).numpy()[0]
        # print(score)

        kernels = (out.numpy() > 0).astype(np.uint8)

        # print(kernels.shape)
        # print(np.sum(kernels[0,0,:,:]))
        # print(np.sum(kernels[0,1,:,:]))
        # print(np.sum(kernels[0,2,:,:]))
        # print(np.sum(kernels[0,3,:,:]))
        # print(np.sum(kernels[0,4,:,:]))
        # print(np.sum(kernels[0,5,:,:]))
        # print(np.sum(kernels[0,6,:,:]))

        # def to_rgb(img):
        #     img = img.reshape(img.shape[0], img.shape[1], 1)
        #     img = np.concatenate((img, img, img), axis=2) * 255
        #     return img
        #
        # def save(img_path, imgs):
        #     if not os.path.exists('vis/'):
        #         os.makedirs('vis/')
        #
        #     for i in range(len(imgs)):
        #         imgs[i] = cv2.copyMakeBorder(imgs[i], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        #     res = np.concatenate(imgs, axis=1)
        #     if type(img_path) != str:
        #         img_name = img_path[0].split('/')[-1]
        #     else:
        #         img_name = img_path.split('/')[-1]
        #     print('saved %s.' % img_name)
        #     cv2.imwrite('vis/' + img_name, res)



        text_mask = kernels[:, :1, :, :]
        # print(np.sum(text_mask))
        # sys.exit()
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        # print(np.sum(kernels))
        # kernel_0 = to_rgb(kernels[0, 0])
        # kernel_1 = to_rgb(kernels[0, 1])
        # kernel_2 = to_rgb(kernels[0, 2])
        # kernel_3 = to_rgb(kernels[0, 3])
        # kernel_4 = to_rgb(kernels[0, 4])
        # kernel_5 = to_rgb(kernels[0, 5])
        # kernel_6 = to_rgb(kernels[0, 6])
        #
        # save('kernel.png', [kernel_0, kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6])
        #
        # sys.exit()

        label = pse(kernels[0], cfg.test_cfg.min_area)

        # image size
        # print('org_img_size', img_meta['org_img_size'])
        # print('img_size', img_meta['img_size'])
        # sys.exit()
        org_img_size = img_meta['org_img_size']
        img_size = img_meta['img_size']

        label_num = np.max(label) + 1
        label = cv2.resize(label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(score, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        bboxes = []
        scores = []
        # print('label_num: {}'.format(label_num))
        for i in range(1, label_num):
            # print(i)
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))

            if points.shape[0] < cfg.test_cfg.min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < cfg.test_cfg.min_score:
                label[ind] = 0
                continue

            if cfg.test_cfg.bbox_type == 'rect':
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
            elif cfg.test_cfg.bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scale

            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        outputs.update(dict(
            bboxes=bboxes,
            scores=scores
        ))

        return outputs
