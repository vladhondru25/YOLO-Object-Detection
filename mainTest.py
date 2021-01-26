import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolov4 import CSPDarknet53_SPP_PAN
from utility.loss_function import *
from utility.utilities import *
from utility.boxes import *


if __name__ == "__main__":
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
    
    device = torch.device('cpu')
    
    predTest = torch.rand(32,255,7,7)
    preds = split_output(predTest, device)
    
    
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