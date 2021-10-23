## News
- PSENet is included in [MMOCR](https://github.com/open-mmlab/mmocr).
- We have implemented PSENet using Paddle. You can find the pytorch version [here](https://github.com/whai362/PSENet).
- You can find code of PAN [here](https://github.com/whai362/pan_pp.pytorch).
- Another group did the same job. You can visit it [here](https://github.com/PaddleEdu/OCR-models-PaddlePaddle/tree/main/PSENet). You can also have a try online with all the environment ready [here](https://aistudio.baidu.com/aistudio/projectdetail/1945560).

## Introduction
Official Paddle implementations of PSENet [1].

[1] W. Wang, E. Xie, X. Li, W. Hou, T. Lu, G. Yu, and S. Shao. Shape robust text detection with progressive scale expansion network. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 9336–9345, 2019.<br>


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
Install paddle following the official [tutorial](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html).
```shell script
pip install -r requirement.txt
./compile.sh
```

## Training
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python dist_train.py ${CONFIG_FILE}
```
For example:
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python dist_train.py config/psenet/psenet_r50_ic15_736.py
```

## Test
```
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
For example:
```shell script
python test.py config/psenet/psenet_r50_ic15_736.py checkpoints/psenet_r50_ic15_736/checkpoint.pdparams
```


## Evaluation
## Introduction
The evaluation scripts of ICDAR 2015 (IC15) dataset.
## [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4)
Text detection
```shell script
./eval_ic15.sh
```


## Benchmark 
## Results 

[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4)

| Method | Backbone | Fine-tuning | Scale | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | Shorter Side: 736 | [psenet_r50_ic15_736.py](https://github.com/RoseSakurai/PSENet_paddle/blob/main/config/psenet/psenet_r50_ic15_736.py) | 82.2 | 79.4 | 80.7 | [Google Drive](https://drive.google.com/file/d/1K-TRoKh_VtIPGaflFdpO28fiO-UfbLvf/view?usp=sharing) |



## Citation
```
@inproceedings{wang2019shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```

## License
This project is developed and maintained by [IMAGINE Lab@National Key Laboratory for Novel Software Technology, Nanjing University](https://cs.nju.edu.cn/lutong/ImagineLab.html).

<img src="logo.jpg" alt="IMAGINE Lab">

This project is released under the [Apache 2.0 license](https://github.com/whai362/pan_pp.pytorch/blob/master/LICENSE).
