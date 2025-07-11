# Refer to MMYOLO
# Copyright (c) VCIP-NKU. All rights reserved.
import os
from argparse import ArgumentParser
from pathlib import Path
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path
from mmyolo.registry import VISUALIZERS
from mmyolo.utils import switch_to_deploy
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import get_file_list, show_data_classes

from yoloms import *



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img',
                        help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-dir',
                        default='./output_sam',
                        help='Path to output file')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='Device used for inference')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--deploy',
                        action='store_true',
                        help='Switch model to deployment mode')
    parser.add_argument('--tta',
                        action='store_true',
                        help='Whether to use test time augmentation')
    parser.add_argument('--score-thr',
                        type=float,
                        default=0.3,
                        help='Bbox score threshold')
    parser.add_argument('--class-name',
                        nargs='+',
                        type=str,
                        help='Only Save those classes if set')
    parser.add_argument('--to-labelme',
                        action='store_true',
                        help='Output labelme style label file')
    parser.add_argument('--sam_model',
                        help='The path of SAM model checkpoint',
                        required=True)
    parser.add_argument('--sam_size',
                        help='The size of SAM model',
                        default='vit_h')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.to_labelme and args.show:
        raise RuntimeError('`--to-labelme` or `--show` only '
                           'can choose one at the same time.')
    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    if args.tta:
        assert 'tta_model' in config, 'Cannot find ``tta_model`` in config.' \
            " Can't use tta !"
        assert 'tta_pipeline' in config, 'Cannot find ``tta_pipeline`` ' \
            "in config. Can't use tta !"
        config.model = ConfigDict(**config.tta_model, module=config.model)
        test_data_cfg = config.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = config.tta_pipeline

    # TODO: TTA mode will error if cfg_options is not set.
    #  This is an mmdet issue and needs to be fixed later.
    # build the model from a config file and a checkpoint file
    model = init_detector(config,
                          args.checkpoint,
                          device=args.device,
                          cfg_options={})

    if args.deploy:
        switch_to_deploy(model)

    if not args.show:
        path.mkdir_or_exist(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    files, source_type = get_file_list(args.img)

    # get model class name
    dataset_classes = model.dataset_meta.get('classes')

    # ready for labelme format if it is needed
    to_label_format = LabelmeFormat(classes=dataset_classes)

    # check class name
    if args.class_name is not None:
        for class_name in args.class_name:
            if class_name in dataset_classes:
                continue
            show_data_classes(dataset_classes)
            raise RuntimeError(
                'Expected args.class_name to be one of the list, '
                f'but got "{class_name}"')

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        result = inference_detector(model, file)
        img = mmcv.imread(file)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        visualizer.add_datasample(filename,
                                  img,
                                  data_sample=result,
                                  draw_gt=False,
                                  show=args.show,
                                  wait_time=0,
                                  out_file=out_file,
                                  pred_score_thr=args.score_thr)

        progress_bar.update()

        # Get candidate predict info with score threshold
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.score_thr]
        bboxes = pred_instances.bboxes
        labels = pred_instances.labels
        
        # sam
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image_processed = cv2.imread(out_file, cv2.IMREAD_COLOR)
        sam = models.segment_anything.sam_model_registry[args.sam_size](checkpoint=args.sam_model).to(device=torch.device(args.device))
        mask_predictor = models.segment_anything.SamPredictor(sam)
        transformed_boxes = mask_predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])
        mask_predictor.set_image(image)
        masks, scores, logits = mask_predictor.predict_torch(
            boxes = transformed_boxes,
            multimask_output=False,
            point_coords=None,
            point_labels=None
        )

        # visualize
        result = None
        for i in range(len(masks)):
            label = labels[i]
            mask_map = masks.cpu().numpy()[i][0]
            mask = mask_map * np.ones(mask_map.shape) * int((int(label) / 80 * 255))
            mask = np.stack([mask] + [mask_map * 255] * 2, axis=-1).astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            if result is None:
                result = cv2.addWeighted(image_processed, 1, mask, 0.5, 0)
            else:
                result = cv2.addWeighted(result, 1, mask, 0.5, 0)
        
        cv2.imwrite(out_file, result)


if __name__ == '__main__':
    main()
