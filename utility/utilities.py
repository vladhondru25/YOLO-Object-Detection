import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_output(pred, device, no_boxes=3):
    no_classes = pred.shape[-1]/3 - 5
    if not no_classes.is_integer():
        raise RuntimeError("Incorrect no_of_boxes")
    no_classes = int(no_classes)
    
    boxes_offsets     = torch.empty(pred.shape[0],pred.shape[1],pred.shape[2],no_boxes,4).to(device = device)
    objectness_scores = torch.empty(pred.shape[0],pred.shape[1],pred.shape[2],no_boxes,1).to(device = device)
    classes_pred      = torch.empty(pred.shape[0],pred.shape[1],pred.shape[2],no_boxes,no_classes).to(device = device)
    
    total_box_size = 5+no_classes
    for box in range(no_boxes):
        boxes_offsets[:,:,:,box,:]     = pred[:,:,:, (box*total_box_size):(box*total_box_size+4)]
        objectness_scores[:,:,:,box,:] = pred[:,:,:, (box*total_box_size+4)].unsqueeze(3)
        classes_pred[:,:,:,box,:]      = pred[:,:,:, (box*total_box_size+5):((box+1)*total_box_size)]
        
    return (boxes_offsets, objectness_scores, classes_pred)
    
    


if __name__ == "__main__":
    device = torch.device('cpu')
    predTest = torch.rand((32,7,7,30))
    split_output(predTest, device)