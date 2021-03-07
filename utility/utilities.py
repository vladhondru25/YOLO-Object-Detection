import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.boxes import *


_EPS = 10e-9

    
def split_output(pred, device, no_boxes=3):
    """
    Input:
    pred: (batch, 255, feature_map_h, feature_map_w)
    
    Output:
    boxes_offsets:     (batch, feature_map_h, feature_map_w, no_boxes, 4)
    objectness_scores: (batch, feature_map_h, feature_map_w, no_boxes)
    classes_pred:      (batch, feature_map_h, feature_map_w, no_boxes, 80)
    """
    no_classes = pred.shape[1] / no_boxes - 5
    if not no_classes.is_integer():
        raise RuntimeError("Incorrect no_of_boxes")
    no_classes = int(no_classes)
    
    batch_size = pred.shape[0]
    feature_map_h = pred.shape[2]
    feature_map_w = pred.shape[3]
    
    pred = pred.view(batch_size, no_boxes, no_classes+5, feature_map_h, feature_map_w)
    
    boxes_offsets     = pred[:,:, 0:4,:,:].permute(0,3,4,1,2).to(device = device)
    objectness_scores = pred[:,:, 4,  :,:].permute(0,2,3,1).to(device = device)
    classes_pred      = pred[:,:, 5:, :,:].permute(0,3,4,1,2).to(device = device)
        
    return [boxes_offsets, objectness_scores, classes_pred]


def build_target(pred_boxes, pred_class, target, scale, device, ignore_thres=0.5):
    """
    Compute the target, as well as the masks for the loss function
    
    Keyword arguments:
    pred_boxes -- (batch, feature_map_h, feature_map_w, no_boxes, 4)
    pred_class -- (batch, feature_map_h, feature_map_w, no_boxes, 80)
    target -- (idx_batch, class_label, x, y, w, h)
    scale -- which of the three scales of yolo is used: s_scale, m_scale or l_scale
    ignore_thres -- the value above which IoUs are not included in the loss
    
    Output:
    object_mask -- (batch, feature_map_h, feature_map_w, no_boxes) - mask if object exists or not
    no_object_mask -- (batch, feature_map_h, feature_map_w, no_boxes) - mask if object does not exist or does
    class_mask -- (batch, feature_map_h, feature_map_w, no_boxes) - mask only for bboxes's classes that must be included in the loss calculation
    ious_pred_target -- (batch, feature_map_h, feature_map_w, no_boxes) - IoU between predicted bboxes and target
    target_x -- (Total no. of target boxes in batch) - target boxes center x
    target_y -- (Total no. of target boxes in batch) - target boxes center y
    target_w -- (Total no. of target boxes in batch) - target boxes width
    target_h -- (Total no. of target boxes in batch) - target boxes height
    target_obj -- (batch, feature_map_h, feature_map_w, no_boxes) - target objectness
    target_class_1hot -- (batch, feature_map_h, feature_map_w, no_boxes, 80) - target labels encoded as one-hot
    """
    nB = pred_boxes.shape[0]
    nH = pred_boxes.shape[1]
    nW = pred_boxes.shape[2]
    nA = pred_boxes.shape[3]
    nC = pred_class.shape[4]
    
    scale_f = SCALE_FACTOR[scale]
    anchors = torch.Tensor(ANCHORS[scale]).to(device = device) / scale_f
    
    # Scale target boxes to the corresponding feature map
    target_boxes = target[:,2:] / scale_f
    target_boxes = xywh_to_cxcywh(target_boxes, nH, nW)
    # print("target_boxes: ", target_boxes)
    # print("anchors: ", anchors)
    # Get the batche index and the class(label) prediction
    target_b, target_c = target[:,:2].long().t()
    # Get the pixel coordinates of the object
    target_x_idx, target_y_idx = target_boxes[:,:2].long().t()
    
    ious_target_anchors = iou_xywh(target_boxes[:,2:], anchors) # torch.Size([3, 50])
    
    # best_bboxes_idx contains the index of the anchor which is the best in terms of IoU
    best_bboxes, best_bboxes_idx = torch.max(ious_target_anchors, dim=0)
    
    # Create masks if object is present, respetively not present in grid
    object_mask = torch.zeros(size=(nB,nH,nW,nA)).bool().to(device = device)
    no_object_mask = torch.ones(size=(nB,nH,nW,nA)).bool().to(device = device)
    
    # Set the object_mask where there is an object, respectively clear the no_object_mask 
    object_mask[target_b,target_y_idx,target_x_idx,best_bboxes_idx] = 1
    # print(object_mask[target_b,target_y_idx,target_x_idx,best_bboxes_idx])
    no_object_mask[target_b,target_y_idx,target_x_idx,best_bboxes_idx] = 0
    # Set to 0 the no_object_mask for IoUs greater than a threshold in order to ignore them in the loss calculation
    for i, t_a_ious in enumerate(ious_target_anchors.t()):
        no_object_mask[target_b[i], target_y_idx[i], target_x_idx[i], t_a_ious > ignore_thres] = 0
        
    # Compute the target t_x, t_y, t_w, t_h by inverting the equations
    target_x = torch.zeros(size=(nB,nH,nW,nA)).float().to(device = device)
    target_y = torch.zeros(size=(nB,nH,nW,nA)).float().to(device = device)
    target_w = torch.zeros(size=(nB,nH,nW,nA)).float().to(device = device)
    target_h = torch.zeros(size=(nB,nH,nW,nA)).float().to(device = device)
    
    # Target boxes coordinates, in format x, y(of corners), w, h
    target_x[target_b,target_y_idx,target_x_idx,best_bboxes_idx], \
    target_y[target_b,target_y_idx,target_x_idx,best_bboxes_idx] = (target_boxes[:,:2].t() - target_boxes[:,:2].floor().t()).float()
    
    target_w[target_b,target_y_idx,target_x_idx,best_bboxes_idx], \
    target_h[target_b,target_y_idx,target_x_idx,best_bboxes_idx] = (torch.log(target_boxes[:,2:] / anchors[best_bboxes_idx] + _EPS).t()).float()
    
    # Compute the target objectness
    target_obj = object_mask.float()
    
    # One-hot encode the target labels. NOTE: Only consider the bboxes that have the highest IoU 
    target_class_1hot = torch.zeros(size=(nB,nH,nW,nA,nC)).to(device = device)
    target_class_1hot[target_b,target_y_idx,target_x_idx,best_bboxes_idx,target_c] = 1
    
    # print("target_c: ", target_c)
    # print("target_y_idx: ", target_y_idx)
    # print("target_x_idx: ", target_x_idx)
    # print("best_bboxes_idx: ", best_bboxes_idx)
    
    # Set a class mask only for bboxes's classes that must be included in the loss calculation
    class_mask = torch.zeros(size=(nB,nH,nW,nA)).to(device = device)
    # print("Pred class shape: ", pred_class.shape)
    # print("pred_class[target_b,target_y_idx,target_x_idx,best_bboxes_idx,:] shape: ", pred_class[target_b,target_y_idx,target_x_idx,best_bboxes_idx,:].shape)
    # print("pred_class[target_b,target_y_idx,target_x_idx,best_bboxes_idx,:]: ", pred_class[target_b,target_y_idx,target_x_idx,best_bboxes_idx,:].argmax(1))
    # print("target_c: ", target_c)
    class_mask[target_b,target_y_idx,target_x_idx,best_bboxes_idx] = (pred_class[target_b,target_y_idx,target_x_idx,best_bboxes_idx,:].argmax(1) == target_c).float()
    
    # Compute IoU of prediction and target
    ious_pred_target = torch.zeros(size=(nB,nH,nW,nA)).to(device = device)
    ious_pred_target[target_b, target_y_idx, target_x_idx, best_bboxes_idx] = iou_xyxy(pred_boxes[target_b,target_y_idx,target_x_idx,best_bboxes_idx,:], \
                                                                                     target_boxes, (nH, nW), device).float()
    
    return object_mask, no_object_mask, class_mask, ious_pred_target, target_x, target_y, target_w, target_h, target_obj, target_class_1hot

    
def iou_xywh(target_wh, anchors):
    """
    This function is used to calculate the IoU between the target boxes and the anchors, in order to associate one anchor to each target box, 
    based on the highest IoU
    """
    target_w, target_h = target_wh.t()
    
    ious = []
    
    for a in anchors:
        inter_area = torch.min(target_w, a[0]) * torch.min(target_h, a[1])
        union_area = target_w * target_h + a[0] * a[1] - inter_area + _EPS
        ious.append(inter_area/union_area)
        
    return torch.stack(ious)

def iou_xyxy(pred_boxes, target_boxes, fm_sizes, device, flag_CIoU=True):
    """
    This function is used to calculate the IoU between the predicted boxes and the target boxes. The last paramter is a function, which is used to transform
    the boxes format (from center points to corners).
    
    Keyword arguments:
    pred_boxes -- (bx, by, bw, bh), transform them to common ground (x0, y0, x1, y1)
    target_boxes -- (cx, cy, w, h), transform them to common ground (x0, y0, x1, y1)
    """
    pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes, *fm_sizes).to(device=device)
    target_boxes_xyxy = cxcywh_to_xyxy(target_boxes, *fm_sizes).to(device=device)
    
    area_pred = (pred_boxes_xyxy[:,2] - pred_boxes_xyxy[:,0]) * (pred_boxes_xyxy[:,3]-pred_boxes_xyxy[:,1])
    area_target = (target_boxes_xyxy[:,2] - target_boxes_xyxy[:,0]) * (target_boxes_xyxy[:,3]-target_boxes_xyxy[:,1])
    
    
    inter_x = ( torch.minimum(pred_boxes_xyxy[:,2], target_boxes_xyxy[:,2]) - torch.maximum(pred_boxes_xyxy[:,0], target_boxes_xyxy[:,0]) ).clamp(min=0)
    inter_y = ( torch.minimum(pred_boxes_xyxy[:,3], target_boxes_xyxy[:,3]) - torch.maximum(pred_boxes_xyxy[:,1], target_boxes_xyxy[:,1]) ).clamp(min=0)
    inter_area = inter_x * inter_y
    
    S = inter_area / (area_pred + area_target - inter_area + _EPS)
    D = 0
    V = 0
    
    if flag_CIoU:
        # Distance
        cx1, cy1 = pred_boxes[:,0:2].t()
        cx2, cy2 = target_boxes[:,0:2].t()
        d = torch.pow(cx2-cx1, 2) + torch.pow(cy2-cy1, 2)
        
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes_xyxy.t()
        target_x1, target_y1, target_x2, target_y2 = target_boxes_xyxy.t()
        
        min_x1 = torch.min(pred_x1, target_x1)
        min_y1 = torch.min(pred_y1, target_y1)
        max_x2 = torch.max(pred_x2, target_x2)
        max_y2 = torch.max(pred_y2, target_y2)
        
        c = torch.pow(max_x2 - min_x1, 2) + torch.pow(max_y2 - min_y1, 2)
        
        D = d / c
        
        # Aspect ratio
        pred_w, pred_h = pred_boxes[:,2:4].t()
        target_w, target_h = target_boxes[:,2:4].t()
        
        v = 4 * torch.pow(torch.atan(target_w/target_h) - torch.atan(pred_w/pred_h), 2) / (3.14159 ** 2)
        
        condition = (S >= 0.5).float()
        alpha = condition * ( v / (1 - S + v) )
        
        V = alpha * v

    return S - D - V
                 
        