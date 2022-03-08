# 자주 사용하는 라이브러리 임폴트
import numpy as np
import cv2
import torch
import os

# 자주 사용하는 detectron2 유틸 임폴트
from pycocotools import mask

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
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
    cfg.OUTPUT_DIR = './mask_rcnn_R_50_FPN_1x_seq_lr0.01'
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

import matplotlib.pyplot as plt
if __name__ == '__main__':
    # gpu settings
    print('Available cuda device ', torch.cuda.is_available())
    print('Current cuda device ', torch.cuda.current_device())  # check

    GPU_NUM = 1  # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())  # check

    # validation set settings
    val_data_dirs = ['data_C5']

    DatasetCatalog.register("polyp_val", lambda: get_polyp_dicts(val_data_dirs))
    MetadataCatalog.get("polyp_val").set(thing_classes=["polyp"])

    polyp_metadata = MetadataCatalog.get("polyp_val")

    pth_names = []
    for a in range(99, 10000, 100):
        if a <= 99:
            nm = f"model_00000{a}.pth"
        elif a >99 and a <= 999:
            nm = f"model_0000{a}.pth"
        elif a > 999 and a <= 9999:
            nm = f"model_000{a}.pth"
        elif a > 9999 and a <= 99999:
            nm = f"model_00{a}.pth"
        elif a > 99999 and a <= 999999:
            nm = f"model_0{a}.pth"
        else:
            nm = f"model_{a}.pth"
        pth_names.append(nm)

    # print(pth_names)
    # pth_names = ['model_0000999.pth', 'model_0001999.pth', 'model_0002999.pth', 'model_0003999.pth', 'model_0004999.pth',
    #              'model_0005999.pth', 'model_0006999.pth', 'model_0007999.pth', 'model_0008999.pth',
    #              'model_0009999.pth', 'model_0010999.pth', 'model_0011999.pth', 'model_0012999.pth', 'model_0013999.pth',
    #              'model_0014999.pth', 'model_0015999.pth', 'model_0016999.pth', 'model_0017999.pth', 'model_0018999.pth',
    #              'model_0019999.pth', 'model_0020999.pth', 'model_0021999.pth', 'model_0022999.pth', 'model_0023999.pth',
    #              'model_0024999.pth', 'model_0025999.pth', 'model_0026999.pth', 'model_0027999.pth', 'model_0028999.pth',
    #              'model_0029999.pth', 'model_0030999.pth', 'model_0031999.pth', 'model_0032999.pth', 'model_0033999.pth',
    #              'model_0034999.pth', 'model_0035999.pth', 'model_0036999.pth', 'model_0037999.pth', 'model_0038999.pth',
    #              'model_0039999.pth', 'model_0040999.pth', 'model_0041999.pth', 'model_0042999.pth', 'model_0043999.pth',
    #              'model_0044999.pth', 'model_0045999.pth', 'model_0046999.pth', 'model_0047999.pth', 'model_0048999.pth',
    #              'model_0049999.pth',
    #              'model_0054999.pth',
    #              'model_0059999.pth',
    #              'model_0064999.pth',
    #              'model_0069999.pth',
    #              'model_0074999.pth',
    #              'model_0079999.pth',
    #              'model_0084999.pth',
    #              'model_0089999.pth',
    #              'model_0094999.pth',
    #              'model_0099999.pth'
    # ]
    # configuration settings

    bbox_ap = []
    seg_ap = []

    def fc_1(AP, AP50, AP75, APs, APm, APl):
        return AP

    def fc(bbox, segm):
        bbox_ap.append(fc_1(**bbox))
        seg_ap.append(fc_1(**segm))

    for pth_name in pth_names:
        cfg = setup(pth_name=pth_name)
        predictor = DefaultPredictor(cfg)

        # evaluator = COCOEvaluator("polyp_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
        evaluator = COCOEvaluator("polyp_val", ("bbox", "segm"), True, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, "polyp_val")

        res = inference_on_dataset(predictor.model, val_loader, evaluator)
        print(pth_name)
        print(res)
        fc(**res)

    fig = plt.figure() ## 캔버스 생성
    fig.set_facecolor('white') ## 캔버스 색상 설정
    ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성

    ax.plot(bbox_ap,marker='o',label='bbox') ## 선그래프 생성
    ax.plot(seg_ap,marker='o',label='seg')

    ax.legend() ## 범례
    plt.show()

