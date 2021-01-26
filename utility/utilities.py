import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

    
def split_output(pred, device, no_boxes=3):
    """
    Input:
    pred: (batch, 255, feature_map_h, feature_map_w)
    
    Output:
    boxes_offsets:     (batch, no_boxes, 4,  feature_map_h, feature_map_w)
    objectness_scores: (batch, no_boxes, 1,  feature_map_h, feature_map_w)
    classes_pred:      (batch, no_boxes, 80, feature_map_h, feature_map_w)
    """
    no_classes = pred.shape[1] / no_boxes - 5
    if not no_classes.is_integer():
        raise RuntimeError("Incorrect no_of_boxes")
    no_classes = int(no_classes)
    
    batch_size = pred.shape[0]
    feature_map_h = pred.shape[2]
    feature_map_w = pred.shape[3]
      
    # boxes_offsets     = torch.empty(batch_size, no_boxes, 4, feature_map_h, feature_map_w).to(device = device)
    # objectness_scores = torch.empty(batch_size, no_boxes, 1, feature_map_h, feature_map_w).to(device = device)
    # classes_pred      = torch.empty(batch_size, no_boxes, no_classes, feature_map_h, feature_map_w).to(device = device)
    
    pred = pred.view(batch_size, no_boxes, no_classes+5, feature_map_h, feature_map_w)
    
    boxes_offsets     = pred[:,:, 0:4,:,:]
    objectness_scores = pred[:,:, 4,  :,:]
    classes_pred      = pred[:,:, 5:, :,:]
        
    return [boxes_offsets, objectness_scores, classes_pred]


def process_target():
    """
    #TODO
    """
    pass