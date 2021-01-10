import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolov4 import CSPDarknet53_SPP_PAN


if __name__ == "__main__":
    xTest = torch.rand((32,3,224,224))
    
    modelTest = CSPDarknet53_SPP_PAN()
    
    yTest = modelTest(xTest)
    
    for i in range(3):
        print(yTest[i].shape)