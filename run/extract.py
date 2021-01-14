# extract features
import detectron2
import os, json, cv2, random
import tqdm
import torch
import sys

import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import pdb

config_file = "./configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
model_file = "./models/COCO-InstanceSegmentation/X-101-32x8d.pkl"

data_path = sys.argv[1]
feat_path = sys.argv[2]
conf_th = float(sys.argv[3])

cfg = get_cfg()
cfg.merge_from_file(config_file)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_th
cfg.MODEL.WEIGHTS = model_file

predictor = DefaultPredictor(cfg)
vocab = predictor.metadata.thing_classes + ["__background__"]

with open(f"{feat_path}/vocab.txt", 'w') as f:
    for obj in vocab:
        f.write(obj + '\n')

save_dir = os.path.join(feat_path, f"feat_th{conf_th}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_fnames = os.listdir(data_path)
for fname in tqdm.tqdm(img_fnames):
    im = cv2.imread(os.path.join(data_path, fname))

    if predictor.input_format == "RGB":
        im = im[:, :, ::-1]
    h, w = im.shape[:2]
    image = predictor.aug.get_transform(im).apply_image(im)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    inputs = {"image": image, "height": h, "width": w}
    outputs = predictor.model([inputs])[0]["instances"]

    images = predictor.model.preprocess_image([inputs])
    features = predictor.model.backbone(images.tensor)
    features = [features[k] for k in predictor.model.roi_heads.box_in_features]

    # add image as feature
    img_features = predictor.model.roi_heads.box_pooler(
        features, [detectron2.structures.boxes.Boxes(
            torch.tensor([[0., 0., w, h]]).to(outputs.pred_boxes.tensor.device)
        )]
    )
    img_features = predictor.model.roi_heads.box_head(img_features).detach().cpu().numpy()
    # boxes
    box_features = predictor.model.roi_heads.box_pooler(features, [outputs.pred_boxes])
    box_features = predictor.model.roi_heads.box_head(box_features).detach().cpu().numpy()

    pred_classes = outputs.pred_classes.cpu().numpy()
    pred_boxes = outputs.pred_boxes.tensor.detach().cpu().numpy()
    pred_scores = outputs.scores.detach().cpu().numpy()
    pred_masks = outputs.pred_masks.cpu().numpy()

    # cls_probs = outputs.cls_probs.detach().cpu().numpy()

    rcnn_data = {
        "box_features": box_features,
        "img_features": img_features,
        "pred_boxes": pred_boxes,
        "pred_scores": pred_scores,
        "pred_masks": pred_masks,
        "pred_classes": pred_classes,
        "w": w, "h": h,
        # "cls_probs": cls_probs
    }

    np.save(os.path.join(save_dir, fname.replace("jpg", "npy")), rcnn_data)


