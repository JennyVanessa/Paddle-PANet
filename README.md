# Paddle-PANet

## Results_Compared 

[CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)

| Method | Backbone | Fine-tuning  | Config | Precision (%) | Recall (%) | F-measure (%) | Model | Log|
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |:-: |
| PaddlePaddle_PANet | ResNet18 | N  | [panet_r18_ctw.py](https://github.com/JennyVanessa/Paddle-PANet/blob/main/config/pan/pan_r18_ctw.py) | 84.51 | 78.62 | 81.46 | [Model](https://github.com/JennyVanessa/Paddle-PANet/tree/main/checkpoints/pan_r18_ctw_train) |  [Log](https://github.com/JennyVanessa/Paddle-PANet/blob/main/trainlog.txt)
| mmocr_PANet | Resnet18 | N | -- |  77.6 | 83.8 | 80.6 | -- | -- |




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

### Pretrain Backbone 

download resent18 pre-train model in `pretrain/resnet18.pdparams`

> [pretrain_resnet18](https://pan.baidu.com/s/1zwmcaAfabZ8fT-KoisbR3w)
> 提取码: j5g3

## Training
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python dist_train.py ${CONFIG_FILE}
```
For example:
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python dist_train.py config/pan/pan_r18_ctw.py
```


## Evaluation
## Introduction
The evaluation scripts of CTW 1500 dataset.
## [CTW](https://github.com/Yuliang-Liu/Curve-Text-Detector)
Text detection
```shell script
./start_test.sh
```


## Benchmark 



## License
This project is developed and maintained by [IMAGINE Lab@National Key Laboratory for Novel Software Technology, Nanjing University](https://cs.nju.edu.cn/lutong/ImagineLab.html).

<img src="logo.jpg" alt="IMAGINE Lab">

This project is released under the [Apache 2.0 license](https://github.com/whai362/pan_pp.pytorch/blob/master/LICENSE).
