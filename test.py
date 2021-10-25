import paddle
import numpy as np
import argparse
import os
import os.path as osp
import sys
import time
import json
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from utils import ResultFormat, AverageMeter


# import warnings
# warnings.filterwarnings('ignore')

def test(test_loader, model, cfg):
    model.eval()

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    for idx, data_ in enumerate(test_loader):
        print('Testing %d/%d' % (idx, len(test_loader)))
        sys.stdout.flush()

        data = dict(
            imgs=data_[1],
            img_metas=dict(
                org_img_size=data_[0].shape[1:3],
                img_size=data_[1].shape[2:]
            ),
            cfg=cfg
        )


        # forward
        with paddle.no_grad():
            outputs = model(**data)

        # save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        # print('image_name', image_name)
        rf.write_result(image_name, outputs)


def main(args):
    cfg = Config.fromfile(args.config)
    # print(json.dumps(cfg._cfg_dict, indent=4))
    sys.stdout.flush()

    # data loader
    test_dataset = build_data_loader(cfg.data.test)
    test_loader = paddle.io.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # model
    model = build_model(cfg.model)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        print("Loading model from checkpoint '{}'".format(args.checkpoint))
        model.set_state_dict(paddle.load(args.checkpoint))
    else:
        print("No checkpoint found at '{}'".format(args.checkpoint))
        raise

    test(test_loader, model, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    args = parser.parse_args()

    main(args)
