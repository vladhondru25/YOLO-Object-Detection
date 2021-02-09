import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_convert


ANCHORS = {'l_scale': [(12,16),     (19,36),  (40,28) ],
           'm_scale': [(36,75),     (76,55),  (72,146)], 
           's_scale': [(142,110), (192,243), (459,401)]
          }
SCALE_FACTOR = {'l_scale': 8, 'm_scale': 16, 's_scale': 32}


def prediction_to_boxes(pred, scale, device):
    """
    Transform the predictions into YOLO boxes:
        b_x = sigmoid(t_x) + c_x
        b_y = sigmoid(t_y) + c_y
        b_w = p_w * e^(t_w)
        b_h = p_h * e^(t_h)
        
    , and then scale each parameter (b_x, b_y by the feature map size; b_w, b_h by the input dimensions).
    
    Keyword arguments:
    pred -- (batch, feature_map_h, feature_map_w, no_boxes, 4)
    scale -- which of the three scales of yolo is used: s_scale, m_scale or l_scale
    
    Output:
    pred_boxes: (batch, feature_map_h, feature_map_w, no_boxes, 4)
    """
    feature_map_h = pred.shape[1]
    feature_map_w = pred.shape[2]
    
    anchors = ANCHORS[scale]
    scale_f = SCALE_FACTOR[scale]
    
    cx = torch.ones(1, feature_map_h, feature_map_w, 1) * torch.arange(start=0, end=feature_map_w).reshape(1,1,-1,1)
    cy = torch.ones(1, feature_map_h, feature_map_w, 1) * torch.arange(start=0, end=feature_map_h).reshape(1,-1,1,1)
    
    scaled_anchor_w = torch.Tensor([[[[anchors[0][0]/scale_f, anchors[1][0]/scale_f, anchors[2][0]/scale_f]]]]).to(device = device)
    scaled_anchor_h = torch.Tensor([[[[anchors[0][1]/scale_f, anchors[1][1]/scale_f, anchors[2][1]/scale_f]]]]).to(device = device)

    pred_boxes = torch.empty(pred.shape)
    
    # Compute box coordinates b_x and b_y
    pred_boxes[:,:,:,:,0] = pred[:,:,:,:,0] + cx.to(device = device)
    pred_boxes[:,:,:,:,1] = pred[:,:,:,:,1] + cy.to(device = device)
    # Computer box width b_w and height b_h
    pred_boxes[:,:,:,:,2] = torch.exp(pred[:,:,:,:,2]) * scaled_anchor_w
    pred_boxes[:,:,:,:,3] = torch.exp(pred[:,:,:,:,3]) * scaled_anchor_h
        
    return pred_boxes
    

def filter_boxes(boxes_coords, objectness_scores, classes_pred, threshold=0.6):
    """
    This function is not used in Yolo v4, as the outputs are not mutually exclusive.
    Input:
    boxes_coords shape      (batch, feature map size H, feature map size W, boxes number, 4)
    objectness_scores shape (batch, feature map size H, feature map size W, boxes number, 1)
    classes_pred shape      (batch, feature map size H, feature map size W, boxes number, classes number)
    
    Output:
    Remaining boxes, consisting of their offsets (None,4), scores (objectness score * class prediction) (None,) and the class predictions (None,)
    """
    box_scores = objectness_scores.unsqueeze(-1) * classes_pred
    
    best_boxes, idx_best_boxes = torch.max(box_scores, dim=4)

    filter_mask = (best_boxes >= threshold)
    
    return [boxes_coords[filter_mask], best_boxes[filter_mask], idx_best_boxes[filter_mask]]
    

def non_max_suppression(boxes_offsets, scores, classes_pred, fm_size, iou_threshold=0.6, max_boxes=5):
    """
    boxes_offsets - tensor of shape (None,4) holding the boxes in format (x1, y1, x2, y2) i.e. two main diagonal corners
    scores        - tensor of shape (None,) holding the score of each box
    classes_pred  - tensor of shape (None,) holding the class prediction for each box
    """
    boxes_offsets = cxcywh_to_xyxy(boxes_offsets, *fm_size)
    
    keep_indices = nms(boxes_offsets, scores, iou_threshold)
    
    return (xyxy_to_xywh(boxes_offsets[keep_indices[:max_boxes]]), scores[keep_indices[:max_boxes]], classes_pred[keep_indices[:max_boxes]])


# def scale_boxes(boxes_coords, input_width, input_height):
#     scaling_factor = torch.Tensor([[[[[input_width]], [[input_height]], [[input_width]], [[input_height]]]]])
#     return boxes_coords * scaling_factor


""" Convert boxes between xywh and xyxy formats"""
def xywh_to_xyxy(boxes_offsets, fm_h, fm_w):
    new_boxes = box_convert(boxes_offsets, in_fmt='xywh', out_fmt='xyxy')
    
    new_boxes[:,0:2].clamp_(min=0)
    new_boxes[:,2].clamp_(max=fm_w)
    new_boxes[:,3].clamp_(max=fm_h)
    
    return new_boxes
def cxcywh_to_xyxy(boxes_offsets, fm_h, fm_w):
    new_boxes = box_convert(boxes_offsets, in_fmt='cxcywh', out_fmt='xyxy')
    
    new_boxes[:,0:2].clamp_(min=0)
    new_boxes[:,2].clamp_(max=fm_w)
    new_boxes[:,3].clamp_(max=fm_h)
    
    return new_boxes
def xyxy_to_xywh(boxes_offsets):
    return box_convert(boxes_offsets, in_fmt='xyxy', out_fmt='xywh')