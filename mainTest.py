import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.yolov4 import CSPDarknet53_SPP_PAN
from models.common import *
from utility.loss_function import *
from utility.utilities import *
from utility.boxes import *
from utility.dataset import *
from utility.dataset_api import *
from utility.display import *

from pyinstrument import Profiler
    

if __name__ == "__main__":
    # profiler = Profiler()
    # profiler.start()
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
    device = torch.device('cpu')
    
    # ds = CocoDatasetTrain()
    ds = CocoDatasetAPITrain()
    dl = load_dataloader(ds, batch_size=32)
    
    modelTest = CSPDarknet53_SPP_PAN()
    modelTest = modelTest.to(device=device)
    
    optimiser = optim.Adam(modelTest.parameters())
    
    for i, batch in enumerate(dl):
        images, targets = batch
        
        preds = modelTest(images)[-1]
        
        preds = split_output(preds, device)
        
        preds[0][:,:,:,:,0:2] = ACTIVATIONS['sigmoid'](preds[0][:,:,:,:,0:2])
        preds[1] = ACTIVATIONS['sigmoid'](preds[1])
        preds[2] = ACTIVATIONS['sigmoid'](preds[2])
        
        pred_boxes = prediction_to_boxes(preds[0], 's_scale')
        
        masks_and_target = build_target(pred_boxes, preds[2], targets, 's_scale')
        
        loss = loss_function(preds, masks_and_target)
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        break
    
    ####################################################################
    # device = torch.device('cpu')
    
    # predTest = torch.rand(32,255,7,7) # pred
    # yTest = torch.rand(50,6)          # y
    
    # # 1. Retrieve the outputs as 3 tensors: 
    # # boxes_offsets - torch.Size([32, 7, 7, 3, 4])
    # # objectness_scores and - torch.Size([32, 7, 7, 3])
    # # classes_pred - torch.Size([32, 7, 7, 3, 80])
    # preds = split_output(predTest, device)
    
    # # 2. Apply Sigmoid function
    # preds[0][:,:,:,:,0:2] = ACTIVATIONS['sigmoid'](preds[0][:,:,:,:,0:2])
    # preds[1] = ACTIVATIONS['sigmoid'](preds[1])
    # preds[2] = ACTIVATIONS['sigmoid'](preds[2])
    
    # # 3. Compute bounding boxes from the predictions
    # pred_boxes = prediction_to_boxes(preds[0], 's_scale')
    
    # # 4. Built target
    # masks_and_target = build_target(pred_boxes, preds[2], yTest, 's_scale')
        
    # # 5. Calculate loss
    # loss = loss_function(preds, masks_and_target)
    ####################################################################
    