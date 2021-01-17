import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolov4 import CSPDarknet53_SPP_PAN
from utility.utilities import *
from utility.boxes import *


if __name__ == "__main__":
    # xTest = torch.rand((32,3,224,224))
    # modelTest = CSPDarknet53_SPP_PAN()
    # for y in modelTest(xTest):
    #     print(y.shape)
    
    device = torch.device('cpu')
    predTest = torch.rand((32,255,7,7))
    
    # Retrieve the outputs
    outputs = split_output(predTest, device)
    
    # Get boxes from prediction
    # TO DO
    outputs[0] = prediction_to_boxes(outputs[0], 's_scale')
    
    
    
    
    # # Filter the boxes
    # filtered_outputs = filter_boxes(*outputs)
    
    
    # filtered_outputs[0] = xywh_to_xyxy(filtered_outputs[0])
    
    # # Scale the boxes
    # filtered_outputs[0] = scale_boxes(filtered_outputs[0])
    
    # # Apply nms
    # final_outputs = non_max_suppression(*filtered_outputs)