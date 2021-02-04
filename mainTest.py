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
    