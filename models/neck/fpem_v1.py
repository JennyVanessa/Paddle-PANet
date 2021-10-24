import paddle.nn.functional as F
import paddle
import paddle.nn as nn
import numpy as np


class EmbLoss_v1(nn.Layer):
    def __init__(self, feature_dim=4, loss_weight=1.0):
        super(EmbLoss_v1, self).__init__()
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.delta_v = 0.5
        self.delta_d = 1.5
        self.weights = (1.0, 1.0)

    def forward_single(self, emb, instance, kernel, training_mask, bboxes):
        training_mask = (training_mask > 0.5)
        kernel = (kernel > 0.5)
        instance = instance * training_mask
        instance_kernel = instance * kernel
        instance_kernel = paddle.reshape(instance_kernel, [-1])
        instance = paddle.reshape(instance, [-1])
        
        emb = paddle.reshape(emb, [self.feature_dim, -1])

        unique_labels, unique_ids = paddle.unique(instance_kernel,
                                                #  sorted=True,
                                                 return_inverse=True)
        num_instance = unique_labels.shape[0]
        if num_instance <= 1:
            return 0
        emb_mean = paddle.zeros(shape=[self.feature_dim, num_instance], dtype='float32')

        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue

            ind_k = instance_kernel == lb
            emb_mean[:, i] = paddle.mean(paddle.to_tensor(emb.numpy()[:, ind_k]), axis=1)

        l_agg = paddle.zeros(shape=[num_instance], dtype='float32')

        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = paddle.to_tensor(emb.numpy()[:, ind])
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, axis=0)
            dist = F.relu(dist - self.delta_v)**2
            l_agg[i] = paddle.mean(paddle.log(dist + 1.0))
        l_agg = paddle.mean(l_agg[1:])

        if num_instance > 2:
            emb_interleave = paddle.to_tensor(np.tile(paddle.transpose(emb_mean, perm=[1, 0]).numpy(), (num_instance, 1)))
            emb_band = paddle.to_tensor(np.tile(paddle.transpose(emb_mean, perm=[1, 0]).numpy(), (1, num_instance)))
            emb_band = paddle.reshape(emb_band, [-1, self.feature_dim])

            # print(seg_band)

            mask = (1 - paddle.eye(num_instance, dtype='int32'))
            mask = paddle.to_tensor(np.tile(paddle.reshape(mask, [-1, 1]).numpy(), (1, self.feature_dim)))

            mask = paddle.reshape(mask, [num_instance, num_instance, -1])

            mask[0, :, :] = 0
            mask[:, 0, :] = 0

            mask = paddle.reshape(mask, [num_instance * num_instance, -1])

            # print(mask)

            dist = emb_interleave - emb_band
            # print(dist.shape, mask.shape)
            dist = paddle.to_tensor(dist.numpy()[mask.numpy() > 0])
            dist = paddle.reshape(dist, [-1, self.feature_dim]).norm(p=2, axis=1)
            dist = F.relu(2 * self.delta_d - dist)**2
            l_dis = paddle.mean(paddle.log(dist + 1.0))
        else:
            l_dis = 0

        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = paddle.mean(paddle.log(paddle.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self,
                emb,
                instance,
                kernel,
                training_mask,
                bboxes,
                reduce=True):
        emb = paddle.to_tensor(emb)

        loss_batch = paddle.zeros(shape=[emb.shape[0]], dtype='float32')

        for i in range(loss_batch.shape[0]):
            loss_batch[i] = self.forward_single(emb[i], instance[i], kernel[i],
                                                training_mask[i], bboxes[i])

        loss_batch = self.loss_weight * loss_batch

        if reduce:
            loss_batch = paddle.mean(loss_batch)

        return loss_batch

# import paddle.nn as nn
# import paddle.nn.functional as F

# from ..utils import Conv_BN_ReLU


# class FPEM_v1(nn.Layer):
#     def __init__(self, in_channels, out_channels):
#         super(FPEM_v1, self).__init__()
#         planes = out_channels
#         self.dwconv3_1 = nn.Conv2D(planes,
#                                    planes,
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1,
#                                    groups=planes,
#                                    )
#         self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

#         self.dwconv2_1 = nn.Conv2D(planes,
#                                    planes,
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1,
#                                    groups=planes,
#                                    )
#         self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

#         self.dwconv1_1 = nn.Conv2D(planes,
#                                    planes,
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1,
#                                    groups=planes,
#                                    )
#         self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

#         self.dwconv2_2 = nn.Conv2D(planes,
#                                    planes,
#                                    kernel_size=3,
#                                    stride=2,
#                                    padding=1,
#                                    groups=planes,
#                                    )
#         self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)

#         self.dwconv3_2 = nn.Conv2D(planes,
#                                    planes,
#                                    kernel_size=3,
#                                    stride=2,
#                                    padding=1,
#                                    groups=planes,
#                                    )
#         self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)

#         self.dwconv4_2 = nn.Conv2D(planes,
#                                    planes,
#                                    kernel_size=3,
#                                    stride=2,
#                                    padding=1,
#                                    groups=planes,
#                                    )
#         self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)

#     def _upsample_add(self, x, y):
#         _, _, H, W = y.size()
#         return F.upsample(x, size=[H, W], mode='bilinear') + y

#     def forward(self, f1, f2, f3, f4):
#         f3 = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
#         f2 = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3, f2)))
#         f1 = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2, f1)))

#         f2 = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2, f1)))
#         f3 = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3, f2)))
#         f4 = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3)))

#         return f1, f2, f3, f4
