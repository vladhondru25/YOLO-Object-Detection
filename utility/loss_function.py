import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(pred, target):
    return F.mse_loss(input=pred, target=target, reduction='sum') # reduction='none' in order to mask

def binary_cross_entropy(pred, target, weight=None):
    return F.binary_cross_entropy(input=pred, target=target, weight=weight, reduction='sum')

def iou_loss(pred, target):
    """
    Each tensor is (batch, no_boxes,  4, feature_map_w, feature_map_h), where the coordinates are x1,y1,x2,y2
    """
    # Area of prediction
    area_pred = (pred[:,:,2,:,:] - pred[:,:,0,:,:]) * (pred[:,:,3,:,:] - pred[:,:,1,:,:])
    # Area of target
    area_target = (target[:,:,2,:,:] - target[:,:,0,:,:]) * (target[:,:,3,:,:] - target[:,:,1,:,:])
    #
    i_x0 = torch.maximum(pred[:,:,0,:,:], target[:,:,0,:,:])
    i_y0 = torch.maximum(pred[:,:,1,:,:], target[:,:,1,:,:])
    i_x1 = torch.minimum(pred[:,:,2,:,:], target[:,:,2,:,:])
    i_y1 = torch.minimum(pred[:,:,3,:,:], target[:,:,3,:,:])
    
    x_intersection = torch.maximum(i_x1 - i_x0, torch.zeros(i_x0.shape))
    y_intersection = torch.maximum(i_y1 - i_y0, torch.zeros(i_y0.shape))
    area_intersection = x_intersection * y_intersection
    
    area_union = area_pred + area_target - area_intersection
    
    iou = - torch.log(area_intersection / area_union)
    
    return iou

def loss_function(pred, target):
    """
    #TODO
    pred and target are arrays containing:
        (batch, no_boxes,  4, feature_map_w, feature_map_h) 
        (batch, no_boxes,  1, feature_map_w, feature_map_h)  
        (batch, no_boxes, 80, feature_map_w, feature_map_h)
    """
    # xy_loss = mse_loss(pred[0][:,:,0:2,:,:], target[0][:,:,0:2,:,:])
    # wh_loss = mse_loss(pred[0][:,:,2:4,:,:], target[0][:,:,2:4,:,:])
    
    # obj_loss = binary_cross_entropy(pred[1], target[1])
    
    iouLoss= iou_loss(pred[0], target[0])
    
    # print(xy_loss.item())
    # print(wh_loss.item())
    # print(obj_loss.item())
    
    # print(iouLoss.item())
    