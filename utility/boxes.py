import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_convert


ANCHORS = {'s_scale': [(12,16),     (19,36),  (40,28) ],
           'm_scale': [(36,75),     (76,55),  (72,146)], 
           'l_scale': [(142,110), (192,243), (459,401)]
          }
IMAGES_DIMS = {'W': 32, 'H': 32}


def prediction_to_boxes(pred, scale):
    """
    Input:
    pred (batch, no_boxes, 4, feature_map_w, feature_map_h)
    """
    output = torch.empty(pred.shape)
    
    anchors = ANCHORS[scale]
    cx = torch.ones(1,1,1,pred.shape[-2],pred.shape[-1]) * torch.arange(start=0, end=pred.shape[-2]).reshape(1,1,1,1,-1)
    cy = torch.ones(1,1,1,pred.shape[-2],pred.shape[-1]) * torch.arange(start=0, end=pred.shape[-1]).reshape(1,1,1,-1,1)
    
    for box in range(output.shape[1]):
        # Box coordinates
        output[:,box,0,:,:] = pred[:,box,0,:,:] + cx
        output[:,box,1,:,:] = pred[:,box,1,:,:] + cy
        # Box dimensions
        output[:,box,2:4,:,:] = pred[:,box,2:4,:,:] * torch.Tensor([[[[[ anchors[box][0] ]],[[ anchors[box][1] ]]]]])
        
    return output
    
    

def filter_boxes(boxes_offsets, objectness_scores, classes_pred, threshold=0.6):
    """
    Input:
    boxes_offsets shape     (batch, feature map size W, feature map size H, boxes number, 4)
    objectness_scores shape (batch, feature map size W, feature map size H, boxes number, 1)
    classes_pred shape      (batch, feature map size W, feature map size H, boxes number, classes number)
    
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
    scaling_factor = torch.Tensor([[IMAGES_DIMS['W'], IMAGES_DIMS['H'], IMAGES_DIMS['W'], IMAGES_DIMS['H']]])
    return boxes_offsets * scaling_factor


"""Convert YOLO box predictions to bounding box corners"""
def yolo_boxes_to_corners(box_xy, box_wh):
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    
    return torch.cat([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


""" Convert boxes between xywh and xyxy formats"""
def xywh_to_xyxy(boxes_offsets):
    return box_convert(boxes_offsets, in_fmt='xywh', out_fmt='xyxy')
def xyxy_to_xywh(boxes_offsets):
    return box_convert(boxes_offsets, in_fmt='xyxy', out_fmt='xywh')