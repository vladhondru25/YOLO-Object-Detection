import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 10e-9


def mse_loss(pred, target):
    return F.mse_loss(input=pred, target=target, reduction='sum') # reduction='none' in order to mask

def binary_cross_entropy(pred, target, weight=None):
    return F.binary_cross_entropy(input=pred, target=target, weight=weight, reduction='sum')

def iou_loss(pred, target):
    """
    #TODO
    Each tensor is (No. of boxes selected,  4), where the coordinates are x1,y1,x2,y2
    """
    print(pred.shape)
    print(target.shape)
    
    # Area of prediction
    area_pred = (pred[:,:,2,:,:] - pred[:,:,0,:,:]) * (pred[:,:,3,:,:] - pred[:,:,1,:,:])
    # Area of target
    area_target = (target[:,:,2,:,:] - target[:,:,0,:,:]) * (target[:,:,3,:,:] - target[:,:,1,:,:])
    #
    i_x0 = torch.maximum(pred[:,:,0,:,:], target[:,:,0,:,:])
    i_y0 = torch.maximum(pred[:,:,1,:,:], target[:,:,1,:,:])
    i_x1 = torch.minimum(pred[:,:,2,:,:], target[:,:,2,:,:])
    i_y1 = torch.minimum(pred[:,:,3,:,:], target[:,:,3,:,:])
    
    x_intersection = (i_x1 - i_x0).clamp(min=0)
    y_intersection = (i_y1 - i_y0).clamp(min=0)
    area_intersection = x_intersection * y_intersection
    
    area_union = area_pred + area_target - area_intersection + _EPS
    
    iou = area_intersection / area_union

    return 1-iou

def loss_function2(preds, masks_and_target, device):
    """
    preds is an array containing:
        (batch, no_boxes, feature_map_w, feature_map_h,  4) 
        (batch, no_boxes, feature_map_w, feature_map_h)  
        (batch, no_boxes, feature_map_w, feature_map_h, 80)
        
    NOTE: Boxes do not neet to be scaled, as IOU will be approximately the same.
    """
    # iouLoss= iou_loss(pred[0], target[0]).mean()
    object_mask, no_object_mask, class_mask, ious_pred_target, \
        target_x, target_y, target_w, target_h, target_obj, target_class_1hot = masks_and_target    
        
    # Bounding box prediction loss
    loss_bbox = (1 - ious_pred_target).mean()
    
    # Objectness loss
    loss_objectness = 5.0 * binary_cross_entropy(preds[1][object_mask], target_obj[object_mask]) + \
                      0.5 * binary_cross_entropy(preds[1][no_object_mask], target_obj[no_object_mask])
    
    # Class prediction loss
    loss_class = binary_cross_entropy(preds[2][object_mask], target_class_1hot[object_mask])
    
    return loss_bbox + loss_objectness + loss_class



def loss_function(preds, masks_and_target, device):
    """
    preds is an array containing:
        (batch, no_boxes, feature_map_w, feature_map_h,  4) 
        (batch, no_boxes, feature_map_w, feature_map_h)  
        (batch, no_boxes, feature_map_w, feature_map_h, 80)
        
    NOTE: Boxes do not neet to be scaled, as IOU will be approximately the same.
    """
    # iouLoss= iou_loss(pred[0], target[0]).mean()
    object_mask, no_object_mask, class_mask, ious_pred_target, \
        target_x, target_y, target_w, target_h, target_obj, target_class_1hot = masks_and_target
        
    # print(preds[0][object_mask][:,0])
    # print(target_x[object_mask])
    # print(preds[1][object_mask])
        
    # Bounding box prediction loss
    # print(preds[0][object_mask][:,2])
    # print(target_w[object_mask])
    loss_tx = mse_loss(preds[0][object_mask][:,0], target_x[object_mask])
    loss_ty = mse_loss(preds[0][object_mask][:,1], target_y[object_mask])
    loss_tw = mse_loss(preds[0][object_mask][:,2], target_w[object_mask])
    loss_th = mse_loss(preds[0][object_mask][:,3], target_h[object_mask])
    
    # print(preds[1][object_mask])
    # print(target_obj[object_mask])
    
    # Objectness loss
    loss_objectness = 1.0 * binary_cross_entropy(preds[1][object_mask], target_obj[object_mask]) + \
                      100 * binary_cross_entropy(preds[1][no_object_mask], target_obj[no_object_mask])
                      
    print("OBJ:", binary_cross_entropy(preds[1][object_mask], target_obj[object_mask]).item())
    print("NO_OBJ: ", binary_cross_entropy(preds[1][no_object_mask], target_obj[no_object_mask]).item())
    
    # Class prediction loss
    # print("target_class_1hot[object_mask]: ", target_class_1hot[object_mask])
    loss_class = binary_cross_entropy(preds[2][object_mask], target_class_1hot[object_mask])
    
    # print("IOU loss: {}".format( (1 - ious_pred_target).mean().item() ))
    print("Loss tx: {}".format(loss_tx.item()))
    print("Loss ty: {}".format(loss_ty.item()))
    print("Loss tw: {}".format(loss_tw.item()))
    print("Loss th: {}".format(loss_th.item()))
    print("Loss obj: {}".format(loss_objectness.item()))
    print("Loss class: {}".format(loss_class.item()))
    
    return loss_tx + loss_ty + loss_tw + loss_th + loss_objectness + loss_class
    