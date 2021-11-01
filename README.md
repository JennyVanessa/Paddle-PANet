# Paddle-PANet

## Results_Compared 

[CTW1500](dataset/README.md) 

| Method | Backbone | Fine-tuning  | Config | Precision (%) | Recall (%) | F-measure (%) | Model | Log|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |:-: |
| PaddlePaddle_PANet | ResNet18 | N  | [panet_r18_ctw.py](https://github.com/JennyVanessa/Paddle-PANet/blob/main/config/pan/pan_r18_ctw.py) | 84.51 | 78.62 | 81.46 | [Model](https://github.com/JennyVanessa/Paddle-PANet/tree/main/checkpoints/pan_r18_ctw_train) |  [Log](https://github.com/JennyVanessa/Paddle-PANet/blob/main/trainlog.txt)
| mmocr_PANet | Resnet18 | N | -- |  77.6 | 83.8 | 80.6 | -- | -- |


# 论文介绍


## 背景简介
场景文本检测是场景文本阅读系统的重要一步，随着卷积神经网络的快速发展，场景文字检测也取得了巨大的进步。尽管如此，仍存在两个主要挑战，它们阻碍文字检测部署到现实世界的应用中。第一个问题是速度和准确性之间的平衡。第二个是对任意形状的文本实例进行建模。最近，已经提出了一些方法来处理任意形状的文本检测，但是它们很少去考虑算法的运行时间和效率，这可能在实际应用环境中受到限制。

之前在CVPR 2019上发的PSENet是效果非常好的文本检测算法，但是整个网络的后处理复杂，导致其运行速度很慢。于是PSENet算法的原班作者提出了PAN网络，使其在不损失精度的情况下，极大加快了网络inference的速度，因此也可以把PAN看做是PSENet V2版本。

## 网络结构
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/ce9767d24b054c3ebf51621a8d14ee922e96e5d9a0c64f84be60f4e7f27695eb"， height=70%, width=70%></center>
上图为PAN的整个网络结构，网络主要由Backbone + Segmentation Head（FPEM + FFM） + Output(Text Region、Kernel、Similarity Vector)组成。

本文使用ResNet-18作为PAN的默认Backbone，并提出了低计算量的Segmentation Head(FPFE + FFM)以解决因为使用ResNet-18而导致的特征提取能力较弱，特征感受野较小且表征能力不足的缺点。

此外，为了精准地重建完整的文字实例(text instance)，提出了一个可学习的后处理方法——像素聚合法（PA），它能够通过预测出的相似向量来引导文字像素聚合到正确的kernel上去。

下面将详细介绍一下上面的各个部分。

## Backbone
Backbone选择的是resnet18, 提取stride为4,8,16,32的conv2,conv3,conv4,conv5的输出作为高低层特征。每层的特征图的通道数都使用1*1卷积降维至128得到轻量级的特征图Fr。

## Segmentation Head
PAN使用resNet-18作为网络的默认backbone，虽减少了计算量，但是backbone层数的减少势必会带来模型学习能力的下降。为了提高效率，作者在 resNet-18基础上提出了一个低计算量但可高效增强特征的分割头Segmentation Head。它由两个关键模块组成：特征金字塔增强模块（Feature Pyramid Enhancement Module，FPEM）、特征融合模块（Feature Fusion Module，FFM）。

### <font face="Times new roman"> FPEM </font>
Feature Pyramid Enhancement Module(FPEM)，即特征金字塔增强模块。FPEM呈级联结构且计算量小，可以连接在backbone后面让不同尺寸的特征更深、更具表征能力，结构如下：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/723fa8c02d2240e9a31aed857040d7ddc09758974baf47aca11574f5ea7fcb05"， height=50%, width=50%></center>
FPEM是一个U形模组，由两个阶段组成，up-scale增强、down-scale增强。up-scale增强作用于输入的特征金字塔，它以步长32,16,8,4像素在特征图上迭代增强。在down-scale阶段，输入的是由up-scale增强生成的特征金字塔，增强的步长从4到32，同时，down-scale增强输出的的特征金字塔就是最终FPEM的输出。
FPEM模块可以看成是一个轻量级的FPN，只不过这个FPEM计算量不大，可以不停级联以达到不停增强特征的作用。

### <font face="Times new roman"> FPEM </font>
如上图所示，RCNNHead 主要接收三个输入 fpn features, proposal boxes, proposal features，其中后面两个输入使用上述 initial 方式作为初始值，之后使用预测的 boxes 和 features 作为下一个 RCNNHead 的输入。所以这里是一个不断迭代不断修正的过程。首先使用 fpn features 和 proposal boxes 经过 roi-align 得到 roi features，然后和 proposal features 进行 instance interactive（这里比较容易理解这个名字，因为 roi features 和 proposal features 都是 num_proposals 个 proposal 的 feature。输出为 pred_class, pred_boxes, proposal_features 后两者会被送入下一个 RCNNHead。（值得注意的是 boxes 是脱离了计算图后被送入的）

### <font face="Times new roman"> FFM </font>
Feature Fusion Module(FFM)模块用于融合不同尺度的特征，其结构如下：








## Loss
Sparse R-CNN 实际上沿用的 DETR 的 loss 和正样本匹配方式即：使用 Hungarian 算法。
$$
\mathcal{L} = \lambda_{cls} \cdot \mathcal{L}_{cls} + \lambda_{L1} \cdot \mathcal{L}_{L1} + \lambda_{giou} \cdot \mathcal{L}_{giou}
$$
其中 $\lambda$ 是权重因子，上式的权重因子分别为：2.0，5.0，2.0。我觉得这样设置的原因在于 boxes 的 l1 loss 是归一化后进行计算的，如果按照百分之一的误差，那么 boxes 会降到 0.04（因为有 4 个参数的 l1 loss）。此时分类 loss 和 giou loss 肯定在 0.1 及其以上，这样的话 boxes l1 占比很小，不会作为主要优化的一项，也就不可能降到 0.1 了，便到不到百分之一的误差了。论文的 l1 loss 是计算的左上角和右下角 xyxy 与真值的绝对值之和，而 DETR 则是使用的中心点坐标加上宽高。另外论文使用了 focal loss 作为分类损失函数，DETR 使用的多类别交叉熵。

## Experiments
### 训练方面
优化器选择了 AdamW 使用了 0.0001 的权重衰减，batch-size 为 16，8 块 GPU，学习率为 0.000025， 并在 epoch 为 27 或者 33 时进行十倍的减少。预训练权重是在 ImageNet 上训练的，其余的层都使用 Xavier 进行初始化。采用了多尺度训练和预测。
### 推理方面
唯一的后处理是将无效的 boxes 进行移除，然后将 boxes 调整为适合原图大小的尺寸（因为图片进行了 resize）。eval 的时候直接全部送入 coco 里面，根据作者介绍 coco 的计算方式会匹配分数最高的 boxes ，其余的不会产生影响。在测试阶段，设定一个分数（因为只有有物体的框分数才比较高）这里 DETR 设置的 0.7。
<center><img src="https://img2020.cnblogs.com/blog/2215171/202106/2215171-20210603161406860-1573706032.png"， height=50%, width=50%></center>

可以看到其只用了 36 epoch 达到了比 DETR 500 epoch 还好的效果。（更详细介绍请见文末链接）

## Recommended environment
```
Python 3.6+
paddlepaddle-gpu 2.0.2
nccl 2.0+
mmcv 0.2.12
editdistance
Polygon3
pyclipper
opencv-python 3.4.2.17
Cython
```

## Install

### Install env
Install paddle following the official [tutorial](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html).
```shell script
pip install -r requirement.txt
./compile.sh
```
## Dataset
> Please refer to [dataset/README.md](dataset/README.md) for dataset preparation.

### Pretrain Backbone 

> download resent18 pre-train model in `pretrain/resnet18.pdparams`

> [pretrain_resnet18](https://pan.baidu.com/s/1zwmcaAfabZ8fT-KoisbR3w)
> password: j5g3

## Training
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python dist_train.py ${CONFIG_FILE}

```
For example:
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python dist_train.py config/pan/pan_r18_ctw.py
#checkpoint continue
python3.7 dist_train.py config/pan/pan_r18_ctw_train.py --nprocs 1 --resume checkpoints/pan_r18_ctw_train
```


## Evaluation
## Introduction
The evaluation scripts of CTW 1500 dataset. [CTW](https://1drv.ms/u/s!Aplwt7jiPGKilH4XzZPoKrO7Aulk)

Text detection
```shell script
./start_test.sh
```



## License
This project is developed and maintained by [IMAGINE Lab@National Key Laboratory for Novel Software Technology, Nanjing University](https://cs.nju.edu.cn/lutong/ImagineLab.html).


This project is released under the [Apache 2.0 license](https://github.com/whai362/pan_pp.pytorch/blob/master/LICENSE).
