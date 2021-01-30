import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_convert


ANCHORS = {'s_scale': [(12,16),     (19,36),  (40,28) ],
           'm_scale': [(36,75),     (76,55),  (72,146)], 
           'l_scale': [(142,110), (192,243), (459,401)]
          }
SCALE_FACTOR = {'s_scale': 8, 'm_scale': 16, 'l_scale': 32}


def prediction_to_boxes(pred, scale):
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
    
    cx = torch.ones(1, feature_map_h, feature_map_w, 1) * torch.arange(start=0, end=feature_map_h).reshape(1,1,-1,1)
    cy = torch.ones(1, feature_map_h, feature_map_w, 1) * torch.arange(start=0, end=feature_map_w).reshape(1,-1,1,1)
    
    scaled_anchor_w = torch.Tensor([[[[anchors[0][0]/scale_f, anchors[1][0]/scale_f, anchors[2][0]/scale_f]]]])
    scaled_anchor_h = torch.Tensor([[[[anchors[0][1]/scale_f, anchors[1][1]/scale_f, anchors[2][1]/scale_f]]]])

    pred_boxes = torch.empty(pred.shape)
    
    # Compute box coordinates b_x and b_y
    pred_boxes[:,:,:,:,0] = pred[:,:,:,:,0] + cx
    pred_boxes[:,:,:,:,1] = pred[:,:,:,:,1] + cy
    # Computer box width b_w and height b_h
    pred_boxes[:,:,:,:,2] = torch.exp(pred[:,:,:,:,2]) * scaled_anchor_w
    pred_boxes[:,:,:,:,3] = torch.exp(pred[:,:,:,:,3]) * scaled_anchor_h
        
    return pred_boxes
    
    
def boxes_center_to_corners(boxes):
    """
    Transform the format of the boxes' coordinates, from centre to corner.
    x, y, w, h  ->  x0, y0, x1, y1
    
    Input:
    boxes_center (no_boxes, 4), where the last dimension (size 4) contains x, y, w, h
    """
    boxes_corners = torch.empty(boxes.shape)
    
    # Transform the coordinates
    # x_0, y_0
    boxes_corners[:,0:2] = boxes[:,0:2] - boxes[:,2:4] / 2.0
    # x_1, y_1
    boxes_corners[:,2:4] = boxes[:,0:2] + boxes[:,2:4] / 2.0
    
    # Clip the bounding boxes to fit the input image size
    boxes_corners.clamp_(min=0, max=1)
    
    return boxes_corners

def boxes_offsets_to_corners(boxes_offsets):
    """
    Transform the format of the boxes' coordinates, from centre to corner.
    x, y, w, h  ->  x0, y0, x1, y1
    
    Input:
    boxes_offsets (batch, no_boxes, 4, feature_map_h, feature_map_w)
    """
    feature_map_h = boxes_offsets.shape[-2]
    feature_map_w = boxes_offsets.shape[-1] 
    
    boxes_corners = torch.empty(boxes_offsets.shape)
    
    # Transform the coordinates
    # x_0, y_0
    boxes_corners[:,:,0,:,:] = boxes_offsets[:,:,0,:,:] - boxes_offsets[:,:,2,:,:] / 2.0
    boxes_corners[:,:,1,:,:] = boxes_offsets[:,:,1,:,:] - boxes_offsets[:,:,3,:,:] / 2.0
    # x_1, y_1
    boxes_corners[:,:,2,:,:] = boxes_offsets[:,:,0,:,:] + boxes_offsets[:,:,2,:,:] / 2.0 
    boxes_corners[:,:,3,:,:] = boxes_offsets[:,:,1,:,:] + boxes_offsets[:,:,3,:,:] / 2.0
    
    # Clip the bounding boxes to fit the input image size
    boxes_corners[:,:,0,:,:].clamp_(min=0, max=1)
    boxes_corners[:,:,1,:,:].clamp_(min=0, max=1)
    boxes_corners[:,:,2,:,:].clamp_(min=0, max=1) 
    boxes_corners[:,:,3,:,:].clamp_(min=0, max=1)
    
    return boxes_corners
    

def filter_boxes(boxes_coords, objectness_scores, classes_pred, threshold=0.6):
    """
    #TODO
    This function is not used in Yolo v4, as the outputs are not mutually exclusive.
    Input:
    boxes_coords shape      (batch, boxes number,              4, feature map size W, feature map size H)
    objectness_scores shape (batch, boxes number,              1, feature map size W, feature map size H)
    classes_pred shape      (batch, boxes number, classes number, feature map size W, feature map size H)
    
    Output:
    Remaining boxes, consisting of their offsets (None,4), scores (objectness score * class prediction) (None,) and the class predictions (None,)
    """
    
    box_scores = objectness_scores * classes_pred
    
    best_boxes, idx_best_boxes = torch.max(box_scores, dim=2)
    
    filter_mask = best_boxes > threshold
    filter_mask = filter_mask.unsqueeze(2)

    result = torch.masked_select(boxes_coords,filter_mask)
    
    return [boxes_coords[filter_mask], best_boxes[filter_mask], idx_best_boxes[filter_mask]]
    

def non_max_suppression(boxes_offsets, scores, classes_pred, iou_threshold=0.6, max_boxes=10):
    """
    #TODO
    boxes_offsets - tensor of shape (None,4) holding the boxes in format (x1, y1, x2, y2) i.e. two main diagonal corners
    scores        - tensor of shape (None,) holding the score of each box
    classes_pred  - tensor of shape (None,) holding the class prediction for each box
    """
    keep_indices = nms(boxes_offsets, scores, iou_threshold)
    
    return (boxes_offsets[keep_indices[:max_boxes]], scores[keep_indices[:max_boxes]], classes_pred[keep_indices[:max_boxes]])


def scale_boxes(boxes_coords, input_width, input_height):
    scaling_factor = torch.Tensor([[[[[input_width]], [[input_height]], [[input_width]], [[input_height]]]]])
    return boxes_coords * scaling_factor


""" Convert boxes between xywh and xyxy formats"""
def xywh_to_xyxy(boxes_offsets):
    return box_convert(boxes_offsets, in_fmt='xywh', out_fmt='xyxy')
def cxcywh_to_xyxy(boxes_offsets):
    return box_convert(boxes_offsets, in_fmt='cxcywh', out_fmt='xyxy')
def xyxy_to_xywh(boxes_offsets):
    return box_convert(boxes_offsets, in_fmt='xyxy', out_fmt='xywh')