# 자주 사용하는 라이브러리 임폴트
import numpy as np
import cv2
import torch
import os
import random

import matplotlib.pyplot as plt

# 자주 사용하는 detectron2 유틸 임폴트
from pycocotools import mask

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

from ensemble_boxes import weighted_boxes_fusion
from ensemble_boxes import nms

from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


def get_polyp_dicts(data_dirs):
    '''
        Dataset 만드는 부분
        input:
            data_dirs: List
        return:
            dataset_dicts: Dictionary
    '''
    base_dir = './trainData_EndoCV2021_21_Feb2021-V2/'
    dataset_dicts = []

    cnt = 0
    for data_dir in data_dirs:
        print("LOAD " + data_dir + " STARTED.........")

        data_dir_full_path = os.path.join (base_dir, data_dir)

        dir_names = os.listdir(data_dir_full_path)
        dir_names = [x for x in dir_names if not 'bbox_image' in x]
        dir_names = sorted(dir_names)

        bbox_dir = os.path.join(data_dir_full_path, dir_names[0])
        image_dir = os.path.join(data_dir_full_path, dir_names[1])
        mask_dir = os.path.join(data_dir_full_path, dir_names[2])

        image_filenames = sorted(os.listdir(image_dir))

        for image_filename in image_filenames:
            record = {}

            image_filename_full_path = os.path.join(image_dir, image_filename)
            height, width = cv2.imread(image_filename_full_path).shape[:2]

            record["file_name"] = image_filename_full_path
            record["height"] = height
            record["width"] = width
            record["image_id"] = cnt

            cnt += 1

            fn = os.path.splitext(image_filename)[0]
            bbox_filename_full_path = os.path.join(bbox_dir, fn + "_mask.txt")
            mask_filename_full_path = os.path.join(mask_dir, fn + "_mask.jpg")

            objs = []

            _mask = cv2.imread(mask_filename_full_path)
            _mask = cv2.cvtColor(_mask, cv2.COLOR_BGR2GRAY)

            # _mask를 출력해보면 binary가 아님.
            # 0, 1, 2, ... , 8 그리고 247, 248, ... , 255 값이 들어있는 것으로 확인.
            # binarization이 필요.
            _mask[_mask < 128] = 0
            _mask[_mask > 128] = 1

            with open(bbox_filename_full_path) as f:
                contents = f.readlines()

                for anno in contents:
                    anno = anno.replace("\n", "")
                    strings = anno.split(' ')
                    # annotation 형태: polyp x_min y_min x_max y_max

                    if strings[0] == 'polyp':
                        # polyp이 있는 경우만 데이터셋에 추가
                        x_min = int(strings[1])
                        y_min = int(strings[2])
                        x_max = int(strings[3])
                        y_max = int(strings[4])

                        # bounding box와 segmenation mask이 둘 다 1인 부분 찾기
                        # 하나의 image에 여러 polyp이 있는 경우
                        _bbox_img = np.zeros((height, width))
                        _bbox_img[y_min:y_max, x_min:x_max] = 1

                        _mask_bbox = _bbox_img * _mask

                        # binary segmentation mask를 detectron2에서 요구하는 형식(COCO’s compressed RLE format) 으로 변환
                        _mask_bbox = _mask_bbox.astype('uint8')
                        _mask_dict = mask.encode(np.asarray(_mask_bbox, order="F"))

                        obj = {
                            "bbox": [x_min, y_min, x_max, y_max],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": 0,
                            "iscrowd": 0,
                            "segmentation": _mask_dict
                        }
                        objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


def setup(pth_name):
    cfg = get_cfg()
    cfg.merge_from_file("./detectron2_repo/configs/myconfig/my_mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = 'cpu'

    cfg.OUTPUT_DIR = './ensemble_weights'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, pth_name)

    cfg.DATASETS.TEST = ("polyp_val",)
    cfg.TEST.AUG.FLIP = True
    cfg.TEST.AUG.ENABLED = True

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.INPUT.FORMAT = "RGB"

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]  # rgb

    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.TEST.AUG.FLIP = True

    return cfg


if __name__ == '__main__':
    # gpu settings
    # print('Available cuda device ', torch.cuda.is_available())
    # print('Current cuda device ', torch.cuda.current_device())  # check

    # GPU_NUM = 1  # 원하는 GPU 번호 입력
    # device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(device)  # change allocation of current GPU
    # print('Current cuda device ', torch.cuda.current_device())  # check

    # validation set settings
    val_data_dirs = ['data_C3']
    val_data_dicts = get_polyp_dicts(val_data_dirs)

    for d in ["val"]:
        DatasetCatalog.register("polyp_" + d, lambda  d=d: get_polyp_dicts(val_data_dirs))
        MetadataCatalog.get("polyp_val").set(thing_classes=["polyp"])

    polyp_metadata = MetadataCatalog.get("polyp_val")

    # a list of pth files in './ensemble_weights'
    pth_names = ['1.pth',   # center 1
                 '2.pth',   # center 2
                 '3.pth'    # center 3
    ]

    # settings for weighted_boxes_fusion
    # referred at https://github.com/ZFTurbo/Weighted-Boxes-Fusion
    iou_thr = 0.5
    skip_box_thr = 0.1
    sigma = 0.1
    weights = [1] * len(pth_names)

    # show some results
    # 모든 결과를 뽑아내려면 출력 형식에 맞게 수정 필요
    for d in random.sample(val_data_dicts, 20):
        im = cv2.imread(d["file_name"])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # initialize variables for saving results
        boxes_list = []
        scores_list = []
        labels_list = []
        seg_masks = np.zeros((im.shape[0], im.shape[1]))

        # create a new canvas
        plt.figure(figsize=(20, 20))
        cnt = 0

        # ensemble을 하기 위해서 pth_names에 있는 모델의 결과들을 모은다
        for pth_name in pth_names:
            cnt += 1
            plt.subplot(3, 2, cnt)

            cfg = setup(pth_name=pth_name)
            predictor = DefaultPredictor(cfg)
            outputs = predictor(im)

            # draw results using model at pth_name
            v = Visualizer(im[:, :, ::-1],
                           scale=2.0
                           )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            v_result = v.get_image()[:, :, ::-1]
            plt.title('Center {}'.format(cnt))
            plt.imshow(v_result)

            # collect results of model at pth_name
            outputs = outputs["instances"].to("cpu")

            num_instances = len(outputs)
            h, w = outputs.image_size

            # bounding boxes
            pred_boxes = outputs.get('pred_boxes').tensor.numpy().tolist()
            pred_classes = outputs.get('pred_classes').numpy().tolist()
            scores = outputs.get('scores').numpy().tolist()

            # segmentation masks
            pred_masks = outputs.get('pred_masks').numpy().astype(int)
            pred_masks = np.sum(pred_masks, axis=0)

            # normalize coordinates of bounding boxes
            # weighted_boxes_fusion library를 사용하기 위해서는 bbox 좌표값을 0~1 사이값으로 normalization 해줘야 한다
            for pred_box in pred_boxes:
                pred_box[0] = pred_box[0] / w
                pred_box[1] = pred_box[1] / h
                pred_box[2] = pred_box[2] / w
                pred_box[3] = pred_box[3] / h

            # detect된 객체가 있는 경우에만 저장
            if num_instances > 0:
                boxes_list.append(pred_boxes)
                scores_list.append(scores)
                labels_list.append(pred_classes)

            seg_masks += pred_masks

        # pth_names에 있는 모델의 결과들을 모은 결과로부터 ensemble 하기

        # segmentation 결과 ensemble
        seg_masks = seg_masks * 1.0 / len(pth_names)
        seg_masks[seg_masks < 0.5] = 0
        seg_masks[seg_masks > 0.5] = 1
        seg_masks = seg_masks.astype(np.double)

        # bounding box 결과 ensemble
        if boxes_list is not None:
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
            )
        else:
            boxes = []
            scores = []
            labels = []

        # detectron 결과는 절대 좌표값으로 저장해야 되므로 w, h값을 곱해준다
        for box in boxes:
            box[0] = box[0] * w
            box[1] = box[1] * h
            box[2] = box[2] * w
            box[3] = box[3] * h

        # bounding box 결과 저장하기 위한 dictionary 생성 (segmentation 결과는 따로 저장한다)
        results_ensemble = {
            'pred_boxes': Boxes(torch.Tensor(np.array(boxes))),
            'scores': torch.Tensor(np.array(scores)),
            'pred_classes': torch.Tensor(np.array(labels)).type(torch.int64),
            # 'pred_masks': pred_masks
        }

        # bounding box 결과 instance 생성
        outputs_ensemble = Instances(
            (h, w),
            **results_ensemble
        )

        plt.subplot(3, 2, 4)
        plt.title('Ensemble Segmentation')
        plt.imshow(seg_masks)

        plt.subplot(3, 2, 5)
        v = Visualizer(im[:, :, ::-1],
                       scale=2.0
                       )
        v = v.draw_instance_predictions(outputs_ensemble.to("cpu"))
        v_result = v.get_image()[:, :, ::-1]
        plt.title('Ensemble Result')
        plt.imshow(v_result)

        plt.subplot(3, 2, 6)
        v = Visualizer(im[:, :, ::-1],
                       metadata=polyp_metadata,
                       scale=2.0
                       )
        v = v.draw_dataset_dict(d)
        v_result = v.get_image()[:, :, ::-1]
        plt.title('Ground Truth')
        plt.imshow(v_result)

        plt.show()
