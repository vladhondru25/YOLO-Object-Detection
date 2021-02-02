import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolov4 import CSPDarknet53_SPP_PAN
from models.common import *
from utility.loss_function import *
from utility.utilities import *
from utility.boxes import *
from utility.dataset import *

from pyinstrument import Profiler

if __name__ == "__main__":
    ds = CocoDatasetTrain()
    dl = load_dataloader(ds, batch_size=32)
    
    # profiler = Profiler()
    # profiler.start()
    
    for i, batch in enumerate(dl):
        images, targets = batch
        
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))

# if __name__ == "__main__":
    # xTest = torch.rand((32,3,224,224))
    # modelTest = CSPDarknet53_SPP_PAN()
    
    # device = torch.device('cpu')
    # y_Test = split_output( torch.rand((32,255,7,7)), device ) #modelTest(xTest)[-1]
    # y_Target = split_output( torch.rand((32,255,7,7)), device )
    
    # y_Test[0] = prediction_to_boxes(y_Test[0], 's_scale', (224,224))
    # y_Target[0] = prediction_to_boxes(y_Target[0], 's_scale', (224,224))
    
    # y_Test[0] = boxes_center_to_corners(y_Test[0])
    # y_Target[0] = boxes_center_to_corners(y_Target[0])
    
    # loss_function(y_Test, y_Target)
    
    # # torch.Size([32, 3, 4, 7, 7])
    # print(y_Target[0][0,0,0,3,3])
    # print(y_Target[0][0,0,2,3,3])
    
    
    ####################################################################
    device = torch.device('cpu')
    
    predTest = torch.rand(32,255,7,7) # pred
    yTest = torch.rand(50,6)          # y
    
    # 1. Retrieve the outputs as 3 tensors: 
    # boxes_offsets - torch.Size([32, 7, 7, 3, 4])
    # objectness_scores and - torch.Size([32, 7, 7, 3])
    # classes_pred - torch.Size([32, 7, 7, 3, 80])
    preds = split_output(predTest, device)
    
    # 2. Apply Sigmoid function
    preds[0][:,:,:,:,0:2] = ACTIVATIONS['sigmoid'](preds[0][:,:,:,:,0:2])
    preds[1] = ACTIVATIONS['sigmoid'](preds[1])
    preds[2] = ACTIVATIONS['sigmoid'](preds[2])
    
    # 3. Compute bounding boxes from the predictions
    pred_boxes = prediction_to_boxes(preds[0], 's_scale')
    
    # 4. Built target
    masks_and_target = build_target(pred_boxes, preds[2], yTest, 's_scale')
        
    # 5. Calculate loss
    loss = loss_function(preds, masks_and_target)
    ####################################################################
    

    # for y in modelTest(xTest):
    #     print(y.shape)
    
    # device = torch.device('cpu')
    # predTest = torch.rand((32,255,7,7))
    
    # # 1. Retrieve the outputs
    # outputs = split_output(predTest, device)
    
    # # 2. Get boxes from prediction
    # outputs[0] = prediction_to_boxes(outputs[0], 's_scale', (224,224))
    # # 3. Center boxes to corners
    # outputs[0] = boxes_center_to_corners(outputs[0])
    
    # # Filter the boxes
    # filtered_outputs = filter_boxes(*outputs)
    # print(filtered_outputs.shape)
    
    # # 4. Scale the boxes
    # # outputs[0] = scale_boxes(outputs[0], 224, 224)
    
    # # # Apply nms
    # # final_outputs = non_max_suppression(*outputs)