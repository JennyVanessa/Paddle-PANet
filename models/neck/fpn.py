import paddle
import paddle.nn as nn
import math
import paddle.nn.functional as F
import numpy as np
from ..utils import Conv_BN_ReLU


class FPN(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        # Top layer
        self.toplayer_ = Conv_BN_ReLU(2048, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        self.smooth2_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        self.smooth3_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1_ = Conv_BN_ReLU(1024, 256, kernel_size=1, stride=1, padding=0)

        self.latlayer2_ = Conv_BN_ReLU(512, 256, kernel_size=1, stride=1, padding=0)

        self.latlayer3_ = Conv_BN_ReLU(256, 256, kernel_size=1, stride=1, padding=0)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

    def _upsample(self, x, y, scale=1):
        # _, _, H, W = y.size()
        h = np.size(y, 2)
        w = np.size(y, 3)
        return F.upsample(x, size=(h // scale, w // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        # _, _, H, W = y.size()
        h = np.size(y, 2)
        w = np.size(y, 3)
        return F.upsample(x, size=(h, w), mode='bilinear') + y

    def forward(self, f2, f3, f4, f5):
        p5 = self.toplayer_(f5)

        f4 = self.latlayer1_(f4)
        p4 = self._upsample_add(p5, f4)
        p4 = self.smooth1_(p4)

        f3 = self.latlayer2_(f3)
        p3 = self._upsample_add(p4, f3)
        p3 = self.smooth2_(p3)

        f2 = self.latlayer3_(f2)
        p2 = self._upsample_add(p3, f2)
        p2 = self.smooth3_(p2)

        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)

        # print('p2:', paddle.shape(p2))
        # print('p3:', paddle.shape(p3))
        # print('p4:', paddle.shape(p4))
        # print('p5:', paddle.shape(p5))

        return p2, p3, p4, p5
