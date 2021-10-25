import math
import time

import cv2
import numpy as np
#import torch
#import torch.nn as nn

import sys
import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..loss import build_loss, iou, ohem_batch
from ..post_processing import pa


class PA_Head(nn.Layer):
    def __init__(self, in_channels, hidden_dim, num_classes, loss_text,
                 loss_kernel, loss_emb):
        super(PA_Head, self).__init__()
        self.conv1 = nn.Conv2D(in_channels,
                               hidden_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2D(hidden_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2D(hidden_dim,
                               num_classes,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.text_loss = build_loss(loss_text)
        self.kernel_loss = build_loss(loss_kernel)
        self.emb_loss = build_loss(loss_emb)

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

    def get_results(self, out, img_meta, cfg):
        outputs = dict()

        # if not self.training and cfg.report_speed:
        #     paddle.cuda.synchronize()
        #     start = time.time()

        score = F.sigmoid(out[:, 0, :, :]).numpy()
        kernels = (out[:, :2, :, :].numpy() > 0).astype(np.uint8)
        text_mask = kernels[:, :1, :, :]
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        emb = out[:, 2:, :, :]
        emb = emb * paddle.to_tensor(text_mask, dtype="float32")
        

        score = score[0].astype(np.float32)
        kernels = kernels[0].astype(np.uint8)
        emb = emb.numpy()[0].astype(np.float32)

        # pa
        label = pa(kernels, emb)

        # image size
        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]

        label_num = np.max(label) + 1
        label = cv2.resize(label, (img_size[1], img_size[0]),
                           interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(score, (img_size[1], img_size[0]),
                           interpolation=cv2.INTER_NEAREST)

        # if not self.training and cfg.report_speed:
        #     paddle.cuda.synchronize()
        #     outputs.update(dict(det_post_time=time.time() - start))

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        with_rec = hasattr(cfg.model, 'recognition_head')

        if with_rec:
            bboxes_h = np.zeros((1, label_num, 4), dtype=np.int32)
            instances = [[]]

        bboxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))

            if points.shape[0] < cfg.test_cfg.min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < cfg.test_cfg.min_score:
                label[ind] = 0
                continue

            if with_rec:
                tl = np.min(points, axis=0)
                br = np.max(points, axis=0) + 1
                bboxes_h[0, i] = (tl[0], tl[1], br[0], br[1])
                instances[0].append(i)

            if cfg.test_cfg.bbox_type == 'rect':
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
            elif cfg.test_cfg.bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scale

            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        outputs.update(dict(bboxes=bboxes, scores=scores))
        if with_rec:
            outputs.update(
                dict(label=label, bboxes_h=bboxes_h, instances=instances))

        return outputs

    def loss(self, out, gt_texts, gt_kernels, training_masks, gt_instances,
             gt_bboxes):
        # output
        texts = out[:, 0, :, :]
        kernels = out[:, 1:2, :, :]
        embs = out[:, 2:, :, :]

        # text loss
        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.text_loss(texts,
                                   gt_texts,
                                   selected_masks,
                                   reduce=False)
        iou_text = iou((texts > 0).cast('int32'),
                       gt_texts,
                       training_masks,
                       reduce=False)
        losses = dict(loss_text=loss_text, iou_text=iou_text)

        # kernel loss
        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.shape[1]):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.kernel_loss(kernel_i,
                                             gt_kernel_i,
                                             selected_masks,
                                             reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = paddle.mean(paddle.stack(loss_kernels, axis=1), axis=1)
        iou_kernel = iou((kernels[:, -1, :, :] > 0).cast('int32'),
                         gt_kernels[:, -1, :, :],
                         training_masks * gt_texts,
                         reduce=False)
        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))

        # embedding loss
        loss_emb = self.emb_loss(embs,
                                 gt_instances,
                                 gt_kernels[:, -1, :, :],
                                 training_masks,
                                 gt_bboxes,
                                 reduce=False)
        losses.update(dict(loss_emb=loss_emb))

        return losses

# import math
# import time

# import cv2
# import numpy as np
# #import torch
# #import torch.nn as nn

# import sys
# import os
# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F

# from ..loss import build_loss, iou, ohem_batch
# from ..post_processing import pa


# class PA_Head(nn.Layer):
#     def __init__(self, in_channels, hidden_dim, num_classes, loss_text,
#                  loss_kernel, loss_emb):
#         super(PA_Head, self).__init__()
#         self.conv1 = nn.Conv2D(in_channels,
#                                hidden_dim,
#                                kernel_size=3,
#                                stride=1,
#                                padding=1)
#         self.bn1 = nn.BatchNorm2D(hidden_dim)
#         self.relu1 = nn.ReLU()

#         self.conv2 = nn.Conv2D(hidden_dim,
#                                num_classes,
#                                kernel_size=1,
#                                stride=1,
#                                padding=0)

#         self.text_loss = build_loss(loss_text)
#         self.kernel_loss = build_loss(loss_kernel)
#         self.emb_loss = build_loss(loss_emb)

#         for m in self.sublayers():
#             if isinstance(m, nn.Conv2D):
#                 n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
#                 v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
#                 m.weight.set_value(v)
#             elif isinstance(m, nn.BatchNorm):
#                 m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
#                 m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

#     def forward(self, f):
#         out = self.conv1(f)
#         out = self.relu1(self.bn1(out))
#         out = self.conv2(out)

#         return out

#     def get_results(self, out, img_meta, cfg):
#         outputs = dict()

#         if not self.training and cfg.report_speed:
#             paddle.cuda.synchronize()
#             start = time.time()

#         score = F.sigmoid(out[:, 0, :, :]).numpy()
#         kernels = (out[:, :2, :, :].numpy() > 0).astype(np.uint8)
#         text_mask = kernels[:, :1, :, :]
#         kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
#         emb = out[:, 2:, :, :]
#         emb = emb * text_mask.float()

#         score = score.data.cpu().numpy()[0].astype(np.float32)
#         kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
#         emb = emb.cpu().numpy()[0].astype(np.float32)

#         # pa
#         label = pa(kernels, emb)

#         # image size
#         org_img_size = img_meta['org_img_size'][0]
#         img_size = img_meta['img_size'][0]

#         label_num = np.max(label) + 1
#         label = cv2.resize(label, (img_size[1], img_size[0]),
#                            interpolation=cv2.INTER_NEAREST)
#         score = cv2.resize(score, (img_size[1], img_size[0]),
#                            interpolation=cv2.INTER_NEAREST)

#         if not self.training and cfg.report_speed:
#             paddle.cuda.synchronize()
#             outputs.update(dict(det_post_time=time.time() - start))

#         scale = (float(org_img_size[1]) / float(img_size[1]),
#                  float(org_img_size[0]) / float(img_size[0]))

#         with_rec = hasattr(cfg.model, 'recognition_head')

#         if with_rec:
#             bboxes_h = np.zeros((1, label_num, 4), dtype=np.int32)
#             instances = [[]]

#         bboxes = []
#         scores = []
#         for i in range(1, label_num):
#             ind = label == i
#             points = np.array(np.where(ind)).transpose((1, 0))

#             if points.shape[0] < cfg.test_cfg.min_area:
#                 label[ind] = 0
#                 continue

#             score_i = np.mean(score[ind])
#             if score_i < cfg.test_cfg.min_score:
#                 label[ind] = 0
#                 continue

#             if with_rec:
#                 tl = np.min(points, axis=0)
#                 br = np.max(points, axis=0) + 1
#                 bboxes_h[0, i] = (tl[0], tl[1], br[0], br[1])
#                 instances[0].append(i)

#             if cfg.test_cfg.bbox_type == 'rect':
#                 rect = cv2.minAreaRect(points[:, ::-1])
#                 bbox = cv2.boxPoints(rect) * scale
#             elif cfg.test_cfg.bbox_type == 'poly':
#                 binary = np.zeros(label.shape, dtype='uint8')
#                 binary[ind] = 1
#                 contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
#                                                cv2.CHAIN_APPROX_SIMPLE)
#                 bbox = contours[0] * scale

#             bbox = bbox.astype('int32')
#             bboxes.append(bbox.reshape(-1))
#             scores.append(score_i)

#         outputs.update(dict(bboxes=bboxes, scores=scores))
#         if with_rec:
#             outputs.update(
#                 dict(label=label, bboxes_h=bboxes_h, instances=instances))

#         return outputs

#     def loss(self, out, gt_texts, gt_kernels, training_masks, gt_instances,
#              gt_bboxes):
#         # output
#         texts = out[:, 0, :, :]
#         kernels = out[:, 1:2, :, :]
#         embs = out[:, 2:, :, :]

#         # text loss
#         selected_masks = ohem_batch(texts, gt_texts, training_masks)
#         loss_text = self.text_loss(texts,
#                                    gt_texts,
#                                    selected_masks,
#                                    reduce=False)
#         iou_text = iou((texts > 0).cast('int32'),
#                        gt_texts,
#                        training_masks,
#                        reduce=False)
#         losses = dict(loss_text=loss_text, iou_text=iou_text)

#         # kernel loss
#         loss_kernels = []
#         selected_masks = gt_texts * training_masks
#         for i in range(kernels.shape[1]):
#             kernel_i = kernels[:, i, :, :]
#             gt_kernel_i = gt_kernels[:, i, :, :]
#             loss_kernel_i = self.kernel_loss(kernel_i,
#                                              gt_kernel_i,
#                                              selected_masks,
#                                              reduce=False)
#             loss_kernels.append(loss_kernel_i)
#         loss_kernels = paddle.mean(paddle.stack(loss_kernels, dim=1), dim=1)
#         iou_kernel = iou((kernels[:, -1, :, :] > 0).cast('int32'),
#                          gt_kernels[:, -1, :, :],
#                          training_masks * gt_texts,
#                          reduce=False)
#         losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))

#         # embedding loss
#         loss_emb = self.emb_loss(embs,
#                                  gt_instances,
#                                  gt_kernels[:, -1, :, :],
#                                  training_masks,
#                                  gt_bboxes,
#                                  reduce=False)
#         losses.update(dict(loss_emb=loss_emb))

#         return losses
