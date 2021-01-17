import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



# def process_output(pred):
    
def split_output(pred, device, no_boxes=3):
    """
    Input:
    pred: (batch, 255, feature_map_h, feature_map_w)
    
    Output:
    boxes_offsets:     (batch, no_boxes, 4,  feature_map_w, feature_map_h)
    objectness_scores: (batch, no_boxes, 1,  feature_map_w, feature_map_h)
    classes_pred:      (batch, no_boxes, 80, feature_map_w, feature_map_h)
    """
    no_classes = pred.shape[1]/3 - 5
    if not no_classes.is_integer():
        raise RuntimeError("Incorrect no_of_boxes")
    no_classes = int(no_classes)
    
    boxes_offsets     = torch.empty(pred.shape[0],no_boxes,4,pred.shape[2],pred.shape[3]).to(device = device)
    objectness_scores = torch.empty(pred.shape[0],no_boxes,1,pred.shape[2],pred.shape[3]).to(device = device)
    classes_pred      = torch.empty(pred.shape[0],no_boxes,no_classes,pred.shape[2],pred.shape[3]).to(device = device)
    
    total_box_size = 5+no_classes
    for box in range(no_boxes):
        boxes_offsets[:,box,:,:,:]     = pred[:, (box*total_box_size):(box*total_box_size+4),:,:]
        objectness_scores[:,box,:,:,:] = pred[:, (box*total_box_size+4),:,:].unsqueeze(1)
        classes_pred[:,box,:,:,:]      = pred[:, (box*total_box_size+5):((box+1)*total_box_size),:,:]
        
    return [boxes_offsets, objectness_scores, classes_pred]