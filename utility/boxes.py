import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_convert


ANCHORS = [(12,16), (19,36), (40,28), (36,75), (76,55), (72,146), (142,110), (192,243), (459,401)]
IMAGES_DIMS = {'H': 32, 'W': 32}


def filter_boxes(boxes_offsets, objectness_scores, classes_pred, threshold=0.6):
    """
    Input:
    boxes_offsets shape     (batch, feature map size H, feature map size H, boxes number, 4)
    objectness_scores shape (batch, feature map size H, feature map size H, boxes number, 1)
    classes_pred shape      (batch, feature map size H, feature map size H, boxes number, classes number)
    
    Output:
    Remaining boxes, consisting of their offsets (None,4), scores (objectness score * class prediction) (None,) and the class predictions (None,)
    """
    
    box_scores = objectness_scores * classes_pred
    
    best_boxes, idx_best_boxes = torch.max(box_scores, dim=4)
    
    filter_mask = best_boxes > threshold
    
    return [boxes_offsets[filter_mask], best_boxes[filter_mask], idx_best_boxes[filter_mask]]
    

def non_max_suppression(boxes_offsets, scores, classes_pred, iou_threshold=0.6, max_boxes=10):
    """
    boxes_offsets - tensor of shape (None,4) holding the boxes in format (x1, y1, x2, y2) i.e. two main diagonal corners
    scores        - tensor of shape (None,) holding the score of each box
    classes_pred  - tensor of shape (None,) holding the class prediction for each box
    """
    keep_indices = nms(boxes_offsets, scores, iou_threshold)
    
    return (boxes_offsets[keep_indices[:max_boxes]], scores[keep_indices[:max_boxes]], classes_pred[keep_indices[:max_boxes]])


def scale_boxes(boxes_offsets):
    scaling_factor = torch.Tensor([[IMAGES_DIMS['H'], IMAGES_DIMS['W'], IMAGES_DIMS['H'], IMAGES_DIMS['H']]])
    return boxes_offsets * scaling_factor


""" Convert boxes between xywh and xyxy formats"""
def xywh_to_xyxy(boxes_offsets):
    return box_convert(boxes_offsets, in_fmt='xywh', out_fmt='xyxy')
def xyxy_to_xywh(boxes_offsets):
    return box_convert(boxes_offsets, in_fmt='xyxy', out_fmt='xywh')